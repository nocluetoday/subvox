import AppKit
import Observation

struct DecodedWord: Identifiable, Hashable {
    let id = UUID()
    var prediction: Prediction
    /// A phrase's output text, shown and spoken in place of the phrase.
    var output: String?
    var text: String { output ?? prediction.best?.label ?? "?" }
}

@MainActor @Observable
final class AppModel {
    enum Screen: String, CaseIterable, Identifiable {
        case sensors = "Sensors"
        case train = "Train"
        case decode = "Decode"
        var id: String { rawValue }
        var symbol: String {
            switch self {
            case .sensors: "waveform.path.ecg"
            case .train: "square.stack.3d.up"
            case .decode: "text.bubble"
            }
        }
    }

    enum DecodeUnit: String, CaseIterable, Identifiable {
        case words = "Words"
        case phrases = "Phrases"
        var id: String { rawValue }
    }

    enum Decoder: String, CaseIterable, Identifiable {
        case dtw = "DTW templates"
        case sequenceModel = "Core ML model"
        var id: String { rawValue }
    }

    let motion = HeadphoneMotionService()
    let camera = FaceCameraService()
    let voice = VoiceOutput()
    let audio = AudioService()
    let transcriber = MicTranscriber()
    let capture = CaptureController()
    let recorder = SessionRecorder()

    var screen: Screen = .train {
        didSet { syncCaptureMode() }
    }

    // MARK: Training

    private(set) var dataset = TemplateStore.load()
    var selectedWord: String?
    var featureSet: FeatureSet = .all {
        didSet { rebuildClassifier() }
    }
    private(set) var classifier: TemplateClassifier?
    private(set) var phraseDecoder: PhraseDecoder?
    private(set) var evaluation: Evaluation?
    private(set) var isEvaluating = false
    private(set) var lastTemplateID: Template.ID?
    private(set) var sequenceModel = SequenceModel.load()
    private(set) var isTrainingModel = false
    private(set) var modelError: String?

    // MARK: Decoding

    var decodeMode: CaptureController.Mode = .hold {
        didSet { syncCaptureMode() }
    }
    var decodeUnit: DecodeUnit = .words {
        didSet { syncCaptureMode() }
    }
    var decoder: Decoder = .dtw
    var maxDistance: Double = 1.5
    var minConfidence: Double = 0.05
    /// Core ML decoder: reject words whose top probability is below this.
    var minProbability: Double = 0.5
    private(set) var decoded: [DecodedWord] = []
    private(set) var lastSegment: [MotionSample] = []
    private(set) var lastSegmentLips: [LipSample] = []

    // MARK: Silent call

    var avatarGreenScreen = false

    @ObservationIgnored private var keyMonitor: Any?
    @ObservationIgnored private var started = false

    init() {
        selectedWord = dataset.vocabulary.first
        rebuildClassifier()
        if sequenceModel != nil { decoder = .sequenceModel }
    }

    func start() {
        guard !started else { return }
        started = true
        motion.onSample = { [weak self] sample in self?.ingest(sample) }
        capture.onSegment = { [weak self] segment in self?.handleSegment(segment) }
        camera.onSample = { [weak self] lip in self?.recorder.record(lip) }
        motion.start()
        audio.refreshDevices()
        voice.refresh()
        installKeyMonitor()
    }

    private func ingest(_ sample: MotionSample) {
        capture.ingest(sample)
        recorder.record(sample)
    }

    private func syncCaptureMode() {
        capture.autoEnabled = screen == .decode && decodeMode == .auto
        capture.phraseMode = screen == .decode && decodeUnit == .phrases
    }

    private func handleSegment(_ segment: [MotionSample]) {
        guard camera.isRunning, let first = segment.first, let last = segment.last else {
            process(segment, lips: nil)
            return
        }
        // Vision trails the motion stream by a frame or two; wait for the segment's last frames.
        let wait = min(max(camera.latency + 0.05, 0.1), 0.5)
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(wait))
            guard let self else { return }
            let lips = self.camera.lips(in: (first.t - 0.1)...(last.t + 0.1))
            self.process(segment, lips: lips.isEmpty ? nil : lips)
        }
    }

    private func process(_ segment: [MotionSample], lips: [LipSample]?) {
        lastSegment = segment
        lastSegmentLips = lips ?? []
        switch screen {
        case .train:
            guard let word = selectedWord else { return }
            addTemplate(Template(label: word, samples: segment, lips: lips))
            recorder.logEvent(segment, kind: "train", label: word, confidence: nil)
        case .decode:
            Task {
                guard let prediction = await predict(segment, lips: lips) else { return }
                let output = decodeUnit == .phrases
                    ? dataset.phrases.first { $0.text == prediction.best?.label }?.spoken
                    : nil
                let word = DecodedWord(prediction: prediction, output: output)
                decoded.append(word)
                recorder.logEvent(segment, kind: prediction.accepted ? "decode" : "reject",
                                  label: prediction.best?.label ?? "", confidence: prediction.confidence)
                if prediction.accepted, voice.speaksDecodedWords {
                    voice.speak(word.text)
                }
            }
        case .sensors:
            recorder.logEvent(segment, kind: "segment", label: recorder.prompt, confidence: nil)
        }
    }

    // MARK: Space bar = hold to capture

    private func installKeyMonitor() {
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard event.keyCode == 49 else { return event }
            let isDown = event.type == .keyDown
            let isRepeat = isDown && event.isARepeat
            let consumed = MainActor.assumeIsolated { () -> Bool in
                guard let self, self.spaceCapturesMotion else { return false }
                if !isDown {
                    self.capture.end()
                } else if !isRepeat {
                    self.capture.begin()
                }
                return true
            }
            return consumed ? nil : event
        }
    }

    private var spaceCapturesMotion: Bool {
        // Let text fields keep the space bar.
        if NSApp.keyWindow?.firstResponder is NSText { return false }
        switch screen {
        case .train: return selectedWord != nil
        case .decode: return decodeMode == .hold && readyToDecode
        case .sensors: return recorder.isRecording
        }
    }

    // MARK: Decoders

    var canDecode: Bool {
        if decodeUnit == .phrases { return phraseDecoder != nil }
        switch decoder {
        case .dtw: return classifier != nil
        case .sequenceModel: return sequenceModel != nil
        }
    }

    /// True when the active decoder fuses lip channels, so the camera should be running.
    var decoderUsesLips: Bool {
        if decodeUnit == .phrases { return phraseDecoder?.featureSet.contains(.lips) ?? false }
        switch decoder {
        case .dtw: return classifier?.featureSet.contains(.lips) ?? false
        case .sequenceModel: return sequenceModel?.info.featureSet.contains(.lips) ?? false
        }
    }

    /// A lip-fused decoder with the camera off would read a still mouth and guess; refuse instead.
    var decodeBlockedByCamera: Bool { decoderUsesLips && !camera.isRunning }

    var readyToDecode: Bool { canDecode && !decodeBlockedByCamera }

    private func predict(_ segment: [MotionSample], lips: [LipSample]?) async -> Prediction? {
        // Also refuse segments where the camera was on but saw no face.
        guard readyToDecode, lips != nil || !decoderUsesLips else { return nil }
        // Phrases are matched against the word takes with DTW, whichever word decoder is picked.
        // Every phrase is a full connected-word search, so run it off the main thread.
        if decodeUnit == .phrases, let phraseDecoder {
            let (maxDistance, minConfidence) = (maxDistance, minConfidence)
            return await Task.detached(priority: .userInitiated) {
                phraseDecoder.predict(segment, lips: lips, maxDistance: maxDistance, minConfidence: minConfidence)
            }.value
        }
        switch decoder {
        case .dtw:
            return classifier?.predict(segment, lips: lips, maxDistance: maxDistance, minConfidence: minConfidence)
        case .sequenceModel:
            return sequenceModel?.predict(segment, lips: lips, minProbability: minProbability, minConfidence: minConfidence)
        }
    }

    var sequenceModelIsStale: Bool {
        sequenceModel?.isStale(templates: dataset.templates, featureSet: featureSet) ?? false
    }

    func trainSequenceModel() {
        guard !isTrainingModel else { return }
        isTrainingModel = true
        modelError = nil
        let templates = dataset.templates
        let featureSet = featureSet
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Result { try SequenceModel.train(templates: templates, featureSet: featureSet) }
            }.value
            switch result {
            case .success(let model):
                self.sequenceModel = model
                self.decoder = .sequenceModel
            case .failure(let error):
                self.modelError = error.localizedDescription
            }
            self.isTrainingModel = false
        }
    }

    func deleteSequenceModel() {
        SequenceModel.delete()
        sequenceModel = nil
        decoder = .dtw
    }

    // MARK: Dataset editing

    func templates(for word: String) -> [Template] {
        dataset.templates.filter { $0.label == word }
    }

    func addWord(_ raw: String) {
        let word = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !word.isEmpty, !dataset.vocabulary.contains(word) else { return }
        dataset.vocabulary.append(word)
        selectedWord = word
        persist()
    }

    func removeWord(_ word: String) {
        dataset.vocabulary.removeAll { $0 == word }
        dataset.templates.removeAll { $0.label == word }
        if selectedWord == word { selectedWord = dataset.vocabulary.first }
        persist()
    }

    /// Adds a phrase and puts any new words in the vocabulary so they can be trained.
    func addPhrase(_ raw: String, output: String) {
        let words = raw.lowercased()
            .split { !$0.isLetter && !$0.isNumber && $0 != "'" }
            .map(String.init)
        guard !words.isEmpty else { return }
        let text = words.joined(separator: " ")
        guard !dataset.phrases.contains(where: { $0.text == text }) else { return }
        for word in words where !dataset.vocabulary.contains(word) {
            dataset.vocabulary.append(word)
        }
        dataset.phrases.append(Phrase(text: text, output: output.trimmingCharacters(in: .whitespacesAndNewlines)))
        persist()
    }

    func removePhrase(_ id: Phrase.ID) {
        dataset.phrases.removeAll { $0.id == id }
        persist()
    }

    func setPhraseOutput(_ id: Phrase.ID, _ output: String) {
        guard let index = dataset.phrases.firstIndex(where: { $0.id == id }) else { return }
        dataset.phrases[index].output = output.trimmingCharacters(in: .whitespacesAndNewlines)
        // Outputs are looked up at decode time; no need to rebuild the decoders.
        TemplateStore.save(dataset)
    }

    /// Words of `phrase` that have no takes yet.
    func untrainedWords(in phrase: Phrase) -> [String] {
        phrase.words.filter { templates(for: $0).isEmpty }
    }

    func addTemplate(_ template: Template) {
        dataset.templates.append(template)
        lastTemplateID = template.id
        persist()
    }

    func deleteTemplate(_ id: Template.ID) {
        dataset.templates.removeAll { $0.id == id }
        persist()
    }

    private func persist() {
        TemplateStore.save(dataset)
        rebuildClassifier()
    }

    private func rebuildClassifier() {
        classifier = TemplateClassifier(templates: dataset.templates, featureSet: featureSet)
        phraseDecoder = PhraseDecoder(templates: dataset.templates, featureSet: featureSet, phrases: dataset.phrases)
        evaluation = nil
    }

    func evaluate() {
        guard !isEvaluating else { return }
        isEvaluating = true
        let templates = dataset.templates
        let featureSet = featureSet
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Evaluation.run(templates: templates, featureSet: featureSet)
            }.value
            self.evaluation = result
            self.isEvaluating = false
        }
    }

    // MARK: Decode transcript

    var transcript: String {
        decoded.filter(\.prediction.accepted).map(\.text).joined(separator: " ")
    }

    func clearDecoded() {
        decoded.removeAll()
        lastSegment = []
        lastSegmentLips = []
    }

    // MARK: Camera

    func toggleCamera() async {
        if camera.isRunning {
            camera.stop()
        } else {
            await camera.start()
        }
    }

    // MARK: Mic

    func toggleMic() async {
        if audio.isRunning {
            transcriber.stop()
            audio.stop()
        } else {
            await audio.start()
            if audio.isRunning { await transcriber.start(sink: audio.sink) }
        }
    }
}
