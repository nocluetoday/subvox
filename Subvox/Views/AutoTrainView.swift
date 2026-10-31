import SwiftUI

/// Collects takes from your own speech and retrains the Core ML model without supervision.
struct AutoTrainView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                CollectPanel()
                PoolPanel()
                TrainingPanel()
            }
            .padding(20)
        }
    }
}

private struct CollectPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let trainer = model.autoTrainer
        GroupBox("1 · Collect from your speech") {
            VStack(alignment: .leading, spacing: 10) {
                Text("While you talk or whisper at the Mac (calls, dictation, reading aloud), the speech recognizer marks each vocabulary word and its timing. The AirPods motion and lip landmarks over that time are stored as a take. Nothing to press.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Toggle("Collect takes from my speech", isOn: Binding(
                    get: { trainer.isCollecting },
                    set: { on in
                        if on { Task { await model.startCollecting() } } else { model.stopCollecting() }
                    }))
                .toggleStyle(.switch)

                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
                    requirement("AirPods motion", ok: model.motion.status == .streaming, detail: model.motion.status.label)
                    requirement("Microphone + recognizer", ok: model.transcriber.isRunning,
                                detail: model.transcriber.isRunning
                                    ? (model.transcriber.isOnDevice ? "on-device" : "server recognition")
                                    : (model.transcriber.error ?? model.audio.error ?? "off"))
                    requirement("Camera (optional)", ok: model.camera.status == .tracking, detail: model.camera.status.label,
                                optional: true)
                }
                .font(.callout)
                if !model.camera.isRunning {
                    Button("Start Camera") { Task { await model.toggleCamera() } }
                }
                Text("Use the Mac's built-in mic, not the AirPods mic. Only your microphone is used; other people on a call are not recorded.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private func requirement(_ title: String, ok: Bool, detail: String, optional: Bool = false) -> some View {
        GridRow {
            Image(systemName: ok ? "checkmark.circle.fill" : (optional ? "circle" : "xmark.circle.fill"))
                .foregroundStyle(ok ? .green : (optional ? .secondary : .red))
            Text(title)
            Text(detail).foregroundStyle(.secondary)
        }
    }
}

private struct PoolPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let trainer = model.autoTrainer
        GroupBox("2 · Collected takes (\(trainer.pool.count))") {
            VStack(alignment: .leading, spacing: 10) {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 110), spacing: 8)], alignment: .leading, spacing: 6) {
                    ForEach(model.dataset.vocabulary, id: \.self) { word in
                        HStack {
                            Text(word)
                            Spacer()
                            Text("\(trainer.poolCount(for: word))")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
                    }
                }
                if !trainer.recent.isEmpty {
                    Text("Recent").font(.caption.weight(.semibold))
                    ForEach(trainer.recent.prefix(12)) { item in
                        HStack(spacing: 8) {
                            Image(systemName: item.rejection == nil ? "plus.circle.fill" : "minus.circle")
                                .foregroundStyle(item.rejection == nil ? .green : .secondary)
                            Text(item.word)
                            if item.rejection == nil, item.hasLips {
                                Image(systemName: "camera.fill").font(.caption2).foregroundStyle(.secondary)
                            }
                            Text(item.rejection.map { "dropped: \($0)" } ?? "kept")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Spacer()
                            Text(item.date, style: .time).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                HStack {
                    Text("Only vocabulary words are collected. Add words in Train to collect them.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Clear Collected Takes", role: .destructive, action: trainer.clearPool)
                        .disabled(trainer.pool.isEmpty)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}

private struct TrainingPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var trainer = model.autoTrainer
        let tests = model.dataset.templates.filter(SequenceModel.isTestTake)
        GroupBox("3 · Retrain and test") {
            VStack(alignment: .leading, spacing: 10) {
                Text("A challenger model is trained on your takes plus the collected ones. It replaces the current model only if it scores better on your silent test takes (about 1 in 5 of your Train takes, never trained on). You currently have \(tests.count).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack {
                    Toggle("Retrain automatically", isOn: $trainer.isScheduled)
                        .toggleStyle(.switch)
                    Text("about once a day, when the Mac is plugged in and idle for 10 min, with \(AutoTrainer.minNewTakes)+ new takes. Subvox must be open.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button(trainer.isRunning ? "Training…" : "Run Now") {
                        Task { await model.runAutoTraining() }
                    }
                    .disabled(trainer.isRunning)
                }
                if let status = trainer.status {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text(status).font(.caption)
                    }
                }
                if !trainer.runs.isEmpty {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                        GridRow {
                            ForEach(["When", "Pool", "Tests", "Champion", "Challenger", "Result"], id: \.self) {
                                Text($0).font(.caption.weight(.semibold))
                            }
                        }
                        ForEach(trainer.runs.prefix(10)) { run in
                            GridRow {
                                Text(run.date.formatted(date: .abbreviated, time: .shortened)) + Text(run.scheduled ? " ⏱" : "")
                                Text("\(run.poolTakes)").monospacedDigit()
                                Text("\(run.testTakes)").monospacedDigit()
                                Text(percent(run.championAccuracy)).monospacedDigit()
                                Text(percent(run.challengerAccuracy)).monospacedDigit()
                                Text(run.adopted ? "adopted" : "kept old")
                                    .foregroundStyle(run.adopted ? .green : .secondary)
                                    .help(run.note)
                            }
                            .font(.caption)
                        }
                    }
                    if let note = trainer.runs.first?.note {
                        Text(note).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }

    private func percent(_ value: Double?) -> String {
        value.map { String(format: "%.0f%%", $0 * 100) } ?? "—"
    }
}
