import SwiftUI

struct TrainView: View {
    @Environment(AppModel.self) private var model
    @State private var newWord = ""

    var body: some View {
        HSplitView {
            vocabulary
                .frame(minWidth: 200, idealWidth: 220, maxWidth: 280)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    capturePanel
                    takes
                    EvaluationPanel()
                    SequenceModelPanel()
                    PhrasesPanel()
                }
                .padding(20)
            }
            .frame(minWidth: 520)
        }
    }

    private var vocabulary: some View {
        @Bindable var model = model
        return VStack(spacing: 0) {
            List(selection: $model.selectedWord) {
                Section("Vocabulary") {
                    ForEach(model.dataset.vocabulary, id: \.self) { word in
                        HStack {
                            Text(word)
                            Spacer()
                            Text("\(model.templates(for: word).count)")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        .tag(word)
                        .contextMenu {
                            Button("Delete “\(word)” and its takes", role: .destructive) { model.removeWord(word) }
                        }
                    }
                }
            }
            HStack {
                TextField("Add word", text: $newWord)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(addWord)
                Button("Add", systemImage: "plus", action: addWord)
                    .labelStyle(.iconOnly)
                    .disabled(newWord.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(10)
        }
    }

    private var capturePanel: some View {
        let word = model.selectedWord ?? "—"
        return VStack(alignment: .leading, spacing: 10) {
            Text("Silently mouth “\(word)” while holding Space. Keep your head still; aim for 8–10 takes per word.")
                .foregroundStyle(.secondary)
            HoldToCaptureButton(
                title: "Mouth “\(word)”",
                isCapturing: model.capture.isCapturing,
                isEnabled: model.selectedWord != nil && model.motion.status == .streaming,
                onBegin: model.capture.begin,
                onEnd: model.capture.end)
            CameraStatusRow()
            if !model.camera.isRunning {
                Text("Start the camera to store lip and jaw landmarks with each take.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.motion.status != .streaming {
                Label(model.motion.status.label, systemImage: "airpodspro")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder
    private var takes: some View {
        if let word = model.selectedWord {
            let takes = model.templates(for: word).reversed()
            GroupBox("Takes for “\(word)” (\(takes.count))") {
                if takes.isEmpty {
                    Text("No takes yet.")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, minHeight: 60)
                } else {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10) {
                        ForEach(Array(takes)) { take in
                            TakeTile(template: take, isNew: take.id == model.lastTemplateID) {
                                model.deleteTemplate(take.id)
                            }
                        }
                    }
                    .padding(6)
                }
            }
        }
    }

    private func addWord() {
        model.addWord(newWord)
        newWord = ""
    }
}

private struct TakeTile: View {
    var template: Template
    var isNew: Bool
    var onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            SignalPlot(title: "", samples: template.samples, channels: SignalPlot.rotation,
                       minScale: 0.05, height: 44, showsHeader: false)
            HStack {
                Text(String(format: "%.2f s · %d", template.duration, template.samples.count))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
                if template.hasLips {
                    Image(systemName: "camera.fill")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .help("Has camera lip data")
                }
                Spacer()
                Button("Delete", systemImage: "trash", role: .destructive, action: onDelete)
                    .labelStyle(.iconOnly)
                    .buttonStyle(.borderless)
                    .font(.caption)
            }
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 8).strokeBorder(isNew ? Color.accentColor : .clear, lineWidth: 1.5))
    }
}

private struct EvaluationPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        GroupBox("Is there signal? (leave-one-out + permutation test)") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Features:")
                    FeatureToggle(title: "Acceleration", option: .acceleration)
                    FeatureToggle(title: "Rotation", option: .rotation)
                    FeatureToggle(title: "Attitude", option: .attitude)
                    FeatureToggle(title: "Lips (camera)", option: .lips)
                    Spacer()
                    Button(model.isEvaluating ? "Evaluating…" : "Evaluate") { model.evaluate() }
                        .disabled(model.isEvaluating || model.featureSet.isEmpty)
                }
                if let result = model.evaluation {
                    EvaluationResultView(result: result)
                    if model.featureSet.contains(.lips), !result.featureSet.contains(.lips) {
                        Text("Lips were skipped: fewer than two words have takes recorded with the camera on.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    } else if result.featureSet.contains(.lips) {
                        Text("Lip channels included; only takes with camera data were used.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("Needs ≥ 2 words with ≥ 2 takes each. Each take is classified against all the others; the same is repeated with shuffled labels to show what chance looks like.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}

/// Trains the Core ML sequence model on the current takes and channels.
private struct SequenceModelPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        GroupBox("Sequence model (Core ML, trained on your takes)") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("A convolution + LSTM classifier over the same channels as above. It learns how much to trust motion versus lips.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    if model.sequenceModel != nil {
                        Button("Delete", role: .destructive, action: model.deleteSequenceModel)
                            .disabled(model.isTrainingModel)
                    }
                    Button(model.isTrainingModel ? "Training…" : (model.sequenceModel == nil ? "Train Model" : "Retrain"),
                           action: model.trainSequenceModel)
                        .disabled(model.isTrainingModel || model.featureSet.isEmpty)
                }
                if model.isTrainingModel {
                    ProgressView().controlSize(.small)
                }
                if let info = model.sequenceModel?.info {
                    HStack(spacing: 24) {
                        metric("Validation", info.validationAccuracy.map(percent) ?? "—")
                        metric("Training", percent(info.trainingAccuracy))
                        metric("Takes", "\(info.templateIDs.count)")
                        metric("Channels", "\(info.featureSet.channelNames.count)\(info.featureSet.contains(.lips) ? " incl. lips" : "")")
                    }
                    Text(info.validationAccuracy == nil
                         ? "No validation score: each word needs 5+ takes so one in five can be held out."
                         : "Validation accuracy is on \(info.validationCount) held-out takes the model never saw.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if model.sequenceModelIsStale {
                        Label("Takes or channels changed since training. Retrain to include them.", systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                } else if !model.isTrainingModel {
                    Text("Needs 2+ words with 3+ takes each; 10+ takes per word recommended.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if let error = model.modelError {
                    Text(error).font(.caption).foregroundStyle(.red)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.monospacedDigit())
        }
    }

    private func percent(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }
}

/// The phrase list for Decode › Phrases. Phrases are built from trained words, so adding one
/// needs no new recordings once its words have takes.
private struct PhrasesPanel: View {
    @Environment(AppModel.self) private var model
    @State private var newPhrase = ""
    @State private var newOutput = ""

    var body: some View {
        GroupBox("Phrases") {
            VStack(alignment: .leading, spacing: 10) {
                Text("Type a phrase made of words you train above; new words are added to the vocabulary. In Decode › Phrases, mouth the whole phrase in one hold. Only phrases on this list can come out. The output, if set, is shown and spoken instead.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ForEach(model.dataset.phrases) { phrase in
                    PhraseRow(phrase: phrase)
                }
                HStack {
                    TextField("Phrase, e.g. call you back", text: $newPhrase)
                        .frame(width: 200)
                    TextField("Output (optional), e.g. I'll call you back in five minutes", text: $newOutput)
                        .onSubmit(add)
                    Button("Add", systemImage: "plus", action: add)
                        .labelStyle(.iconOnly)
                        .disabled(newPhrase.trimmingCharacters(in: .whitespaces).isEmpty)
                }
                .textFieldStyle(.roundedBorder)
                if !model.dataset.phrases.isEmpty, model.phraseDecoder == nil {
                    Text("Phrase decoding needs at least 2 phrases whose words all have takes.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private func add() {
        model.addPhrase(newPhrase, output: newOutput)
        newPhrase = ""
        newOutput = ""
    }
}

private struct PhraseRow: View {
    @Environment(AppModel.self) private var model
    var phrase: Phrase
    @State private var output = ""

    var body: some View {
        let missing = model.untrainedWords(in: phrase)
        HStack {
            Text(phrase.text)
                .frame(width: 200, alignment: .leading)
            TextField("Output (optional) — press Return to save", text: $output)
                .textFieldStyle(.roundedBorder)
                .onSubmit { model.setPhraseOutput(phrase.id, output) }
            if missing.isEmpty {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .help("All words have takes")
            } else {
                Label("Train: \(missing.joined(separator: ", "))", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            Button("Delete", systemImage: "trash", role: .destructive) { model.removePhrase(phrase.id) }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
        }
        .onAppear { output = phrase.output }
    }
}

private struct FeatureToggle: View {
    @Environment(AppModel.self) private var model
    var title: String
    var option: FeatureSet

    var body: some View {
        Toggle(title, isOn: Binding(
            get: { model.featureSet.contains(option) },
            set: { on in
                if on { model.featureSet.insert(option) } else { model.featureSet.remove(option) }
            }))
        .toggleStyle(.checkbox)
    }
}

private struct EvaluationResultView: View {
    var result: Evaluation

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 24) {
                metric("Accuracy", percent(result.accuracy), highlight: true)
                metric("Chance", percent(result.chance))
                metric("Shuffled 95th pct", percent(result.permutationP95))
                metric("p-value", result.pValue < 0.001 ? "< 0.001" : String(format: "%.3f", result.pValue))
            }
            Text(verdict)
                .font(.callout)
                .foregroundStyle(result.pValue < 0.05 ? .green : .orange)

            Grid(horizontalSpacing: 2, verticalSpacing: 2) {
                GridRow {
                    Text("actual ↓ / predicted →").font(.caption2).foregroundStyle(.secondary)
                    ForEach(result.labels, id: \.self) { Text($0).font(.caption2).lineLimit(1) }
                }
                ForEach(result.labels.indices, id: \.self) { row in
                    let total = max(result.confusion[row].reduce(0, +), 1)
                    GridRow {
                        Text(result.labels[row]).font(.caption2).gridColumnAlignment(.trailing)
                        ForEach(result.labels.indices, id: \.self) { col in
                            let count = result.confusion[row][col]
                            Text("\(count)")
                                .font(.caption.monospacedDigit())
                                .frame(minWidth: 36, minHeight: 24)
                                .background((row == col ? Color.green : Color.red)
                                    .opacity(Double(count) / Double(total) * 0.7))
                        }
                    }
                }
            }
        }
    }

    private var verdict: String {
        if result.pValue < 0.05 {
            "Motion carries word information beyond chance across \(result.templateCount) takes."
        } else {
            "Not distinguishable from chance yet — record more takes or try other feature sets."
        }
    }

    private func metric(_ title: String, _ value: String, highlight: Bool = false) -> some View {
        VStack(alignment: .leading) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(highlight ? .title2.bold().monospacedDigit() : .title3.monospacedDigit())
        }
    }

    private func percent(_ value: Double) -> String {
        String(format: "%.0f%%", value * 100)
    }
}
