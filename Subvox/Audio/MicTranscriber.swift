import AVFoundation
import Observation
import Speech

/// One recognized word with its time on the host clock (seconds since boot), so it can be
/// matched to motion and lip samples.
struct WordTiming: Hashable, Sendable {
    var text: String
    var start: TimeInterval
    var duration: TimeInterval
    /// 0…1 from the recognizer; 0 when it didn't report one.
    var confidence: Double
}

/// Continuous speech-to-text from the microphone. Used as the reference ("ground truth")
/// next to the motion decoder, and to test whether whispering is enough for input.
@MainActor @Observable
final class MicTranscriber {
    private(set) var isRunning = false
    private(set) var text = ""
    private(set) var isOnDevice = false
    private(set) var error: String?

    /// Called with each final result's words and their host-clock times.
    @ObservationIgnored var onFinalWords: (([WordTiming]) -> Void)?

    @ObservationIgnored private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    @ObservationIgnored private let box = RequestBox()
    @ObservationIgnored private var task: SFSpeechRecognitionTask?
    @ObservationIgnored private var sinkID: UUID?
    @ObservationIgnored private weak var sink: AudioSink?
    @ObservationIgnored private var finalized = ""
    @ObservationIgnored private var generation = 0

    func start(sink: AudioSink) async {
        guard !isRunning else { return }
        error = nil

        let status = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard status == .authorized else {
            error = "Speech recognition not authorized."
            return
        }
        guard let recognizer, recognizer.isAvailable else {
            error = "Speech recognizer unavailable."
            return
        }

        self.sink = sink
        sinkID = sink.add { [box] buffer, time in box.append(buffer, time) }
        isRunning = true
        startTask()
    }

    func stop() {
        isRunning = false
        generation += 1
        if let sinkID { sink?.remove(sinkID) }
        sinkID = nil
        box.set(nil)
        task?.cancel()
        task = nil
    }

    func clear() {
        finalized = ""
        text = ""
    }

    private func startTask() {
        guard isRunning, let recognizer else { return }
        generation += 1

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.addsPunctuation = true
        isOnDevice = recognizer.supportsOnDeviceRecognition
        request.requiresOnDeviceRecognition = isOnDevice
        box.set(request)

        task?.cancel()
        task = recognizer.recognitionTask(with: request, resultHandler: Self.resultHandler(owner: self, generation: generation))
    }

    private func handle(generation: Int, partial: String?, isFinal: Bool, failed: Bool, segments: [Segment]) {
        guard generation == self.generation, isRunning else { return }
        // Segment times are relative to the request's first audio buffer.
        if isFinal, !segments.isEmpty, let origin = box.startSeconds {
            onFinalWords?(segments.map {
                WordTiming(text: $0.text, start: origin + $0.timestamp, duration: $0.duration, confidence: $0.confidence)
            })
        }
        if let partial {
            text = finalized + partial
        }
        if isFinal || failed {
            if let partial, !partial.isEmpty { finalized += partial + " " }
            text = finalized
            // Recognition tasks end on silence/timeout; keep listening with a fresh one.
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(failed ? 500 : 50))
                self?.startTask()
            }
        }
    }

    nonisolated private static func resultHandler(owner: MicTranscriber, generation: Int)
        -> (SFSpeechRecognitionResult?, Error?) -> Void
    {
        { [weak owner] result, error in
            let partial = result?.bestTranscription.formattedString
            let isFinal = result?.isFinal ?? false
            let failed = error != nil
            let segments = isFinal
                ? (result?.bestTranscription.segments ?? []).map {
                    Segment(text: $0.substring, timestamp: $0.timestamp, duration: $0.duration, confidence: Double($0.confidence))
                }
                : []
            Task { @MainActor in
                owner?.handle(generation: generation, partial: partial, isFinal: isFinal, failed: failed, segments: segments)
            }
        }
    }
}

private struct Segment: Sendable {
    var text: String
    var timestamp: TimeInterval
    var duration: TimeInterval
    var confidence: Double
}

/// The current recognition request, appended to from the audio thread. Remembers the host
/// time of the request's first buffer, which is where its segment timestamps start.
private final class RequestBox: @unchecked Sendable {
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var firstHostSeconds: TimeInterval?

    var startSeconds: TimeInterval? { lock.withLock { firstHostSeconds } }

    func set(_ request: SFSpeechAudioBufferRecognitionRequest?) {
        lock.withLock {
            self.request?.endAudio()
            self.request = request
            firstHostSeconds = nil
        }
    }

    func append(_ buffer: AVAudioPCMBuffer, _ time: AVAudioTime) {
        lock.withLock {
            guard let request else { return }
            if firstHostSeconds == nil, time.isHostTimeValid {
                firstHostSeconds = AVAudioTime.seconds(forHostTime: time.hostTime)
            }
            request.append(buffer)
        }
    }
}
