import CoreML
import CreateML
import Foundation
import TabularData

/// Core ML sequence classifier over fused motion + lip frames, trained in the app with
/// Create ML (an activity classifier: convolution + LSTM over the frame sequence) on the
/// user's own takes. Every take is time-normalized to `frameCount` frames, so one prediction
/// window is exactly one word. Learns how much to trust each channel, which DTW can't.
struct SequenceModel: @unchecked Sendable {
    struct Info: Codable, Hashable, Sendable {
        var labels: [String]
        var featureSet: FeatureSet
        var scales: [Double]
        var trainedAt: Date
        var templateIDs: Set<UUID>
        var trainingAccuracy: Double
        var validationAccuracy: Double?
        var validationCount: Int
        /// Takes collected automatically from speech that went into training.
        var autoTakeCount: Int?
    }

    enum TrainingError: LocalizedError {
        case notEnoughData(String)
        var errorDescription: String? {
            switch self {
            case .notEnoughData(let message): message
            }
        }
    }

    static let frameCount = 32
    static let defaultIterations = 150

    let model: MLModel
    let info: Info
    /// The compiled model on disk; a temporary folder until `save()`.
    private let location: URL

    static var directory: URL {
        URL.applicationSupportDirectory.appending(path: "Subvox", directoryHint: .isDirectory)
    }
    private static var modelURL: URL { directory.appending(path: "WordModel.mlmodelc", directoryHint: .isDirectory) }
    private static var infoURL: URL { directory.appending(path: "WordModel.json") }

    static func load() -> SequenceModel? {
        guard let data = try? Data(contentsOf: infoURL),
              let info = try? JSONDecoder().decode(Info.self, from: data),
              let model = try? MLModel(contentsOf: modelURL) else { return nil }
        return SequenceModel(model: model, info: info, location: modelURL)
    }

    /// Makes this the saved model, e.g. when auto-training adopts a challenger.
    func save() throws {
        let fm = FileManager.default
        try fm.createDirectory(at: Self.directory, withIntermediateDirectories: true)
        if location.standardizedFileURL != Self.modelURL.standardizedFileURL {
            if fm.fileExists(atPath: Self.modelURL.path) { try fm.removeItem(at: Self.modelURL) }
            try fm.copyItem(at: location, to: Self.modelURL)
        }
        try JSONEncoder().encode(info).write(to: Self.infoURL, options: .atomic)
    }

    /// The fixed silent test set: about 1 in 5 of the takes you recorded yourself, picked by
    /// take ID so membership never changes. No model trains on these, so any two models can
    /// be compared on them fairly.
    static func isTestTake(_ template: Template) -> Bool {
        template.isManual && template.id.uuid.0 % 5 == 0
    }

    static func delete() {
        try? FileManager.default.removeItem(at: modelURL)
        try? FileManager.default.removeItem(at: infoURL)
    }

    /// True when the takes or channels changed since training.
    func isStale(templates: [Template], featureSet: FeatureSet) -> Bool {
        let (set, used) = featureSet.resolved(for: templates)
        return set != info.featureSet || Set(used.filter(\.isManual).map(\.id)) != info.templateIDs
    }

    // MARK: Training

    /// Trains on `templates` (your takes) plus `extra` (takes collected from speech) and, when
    /// `persist` is set, replaces the saved model. The silent test set is always held out.
    /// Blocks for seconds to a minute; call it off the main thread.
    static func train(templates all: [Template], extra: [Template] = [], featureSet requested: FeatureSet,
                      maxIterations: Int = defaultIterations, persist: Bool = true) throws -> SequenceModel {
        let (featureSet, templates) = requested.resolved(for: all + extra)
        guard !featureSet.isEmpty else { throw TrainingError.notEnoughData("Pick at least one feature channel.") }
        let counts = Dictionary(grouping: templates, by: \.label).mapValues(\.count)
        guard counts.count >= 2, counts.values.allSatisfy({ $0 >= 3 }) else {
            throw TrainingError.notEnoughData("Needs at least 2 words with 3 or more takes each.")
        }

        let raw = templates.map { $0.frames(featureSet) }
        let scales = MotionFeatures.channelScales(raw)
        let windows = raw.map { MotionFeatures.timeNormalized(MotionFeatures.scaled($0, by: scales), to: frameCount) }
        let names = featureSet.channelNames

        let held = templates.map(isTestTake)
        var trainCounts: [String: Int] = [:]
        for (template, isHeld) in zip(templates, held) where !isHeld {
            trainCounts[template.label, default: 0] += 1
        }
        if let thin = counts.keys.sorted().first(where: { trainCounts[$0, default: 0] < 2 }) {
            throw TrainingError.notEnoughData("“\(thin)” needs more takes (about 1 in 5 is kept for testing).")
        }
        let training = table(windows, templates, names, include: held.map { !$0 })
        let validation = held.contains(true) ? table(windows, templates, names, include: held) : nil

        let trainCount = held.filter { !$0 }.count
        let parameters = MLActivityClassifier.ModelParameters(
            validation: validation.map { .dataSource(.dataFrame($0)) } ?? .none,
            batchSize: min(32, trainCount),
            maximumIterations: maxIterations,
            predictionWindowSize: frameCount)
        let classifier = try MLActivityClassifier(
            trainingData: .dataFrame(training), featureColumns: names,
            labelColumn: "label", recordingFileColumn: "take", parameters: parameters)

        let info = Info(
            labels: counts.keys.sorted(),
            featureSet: featureSet,
            scales: scales,
            trainedAt: Date(),
            templateIDs: Set(templates.filter(\.isManual).map(\.id)),
            trainingAccuracy: accuracy(classifier.trainingMetrics) ?? 0,
            validationAccuracy: validation == nil ? nil : accuracy(classifier.validationMetrics),
            validationCount: held.filter { $0 }.count,
            autoTakeCount: templates.filter { !$0.isManual }.count)

        let fm = FileManager.default
        let source = fm.temporaryDirectory.appending(path: "Subvox-\(UUID().uuidString).mlmodel")
        try classifier.write(to: source)
        defer { try? fm.removeItem(at: source) }
        let compiled = try MLModel.compileModel(at: source)
        let model = SequenceModel(model: try MLModel(contentsOf: compiled), info: info, location: compiled)
        if persist { try model.save() }
        return model
    }

    /// Accuracy on `takes`, plus the mean probability given to the right word (a tie-breaker).
    func score(on takes: [Template]) -> (accuracy: Double, meanProbability: Double) {
        guard !takes.isEmpty else { return (0, 0) }
        var correct = 0
        var probability = 0.0
        for take in takes {
            let ranked = rank(take.samples, lips: take.lips)
            if ranked.first?.label == take.label { correct += 1 }
            probability += ranked.first { $0.label == take.label }?.probability ?? 0
        }
        let n = Double(takes.count)
        return (Double(correct) / n, probability / n)
    }

    private static func accuracy(_ metrics: MLClassifierMetrics) -> Double? {
        metrics.isValid ? 1 - metrics.classificationError : nil
    }

    /// One row per take; each feature column holds that take's `frameCount` values.
    /// (This is the shape `DataSource.dataFrame` expects, not one row per frame.)
    private static func table(_ windows: [[[Double]]], _ templates: [Template], _ names: [String],
                              include: [Bool]) -> DataFrame {
        var takes: [String] = []
        var labels: [String] = []
        var columns = [[[Double]]](repeating: [], count: names.count)
        for (i, window) in windows.enumerated() where include[i] {
            takes.append("take-\(i)")
            labels.append(templates[i].label)
            for d in names.indices { columns[d].append(window.map { $0[d] }) }
        }
        var frame = DataFrame()
        frame.append(column: Column(name: "take", contents: takes))
        frame.append(column: Column(name: "label", contents: labels))
        for (d, name) in names.enumerated() {
            frame.append(column: Column(name: name, contents: columns[d]))
        }
        return frame
    }

    // MARK: Inference

    func rank(_ samples: [MotionSample], lips: [LipSample]? = nil) -> [RankedLabel] {
        let frames = MotionFeatures.timeNormalized(
            MotionFeatures.scaled(
                MotionFeatures.frames(MotionFeatures.trim(samples), lips: lips, set: info.featureSet),
                by: info.scales),
            to: Self.frameCount)
        guard frames.count == Self.frameCount, let probabilities = try? probabilities(frames) else { return [] }
        return probabilities
            .map { RankedLabel(label: $0.key, distance: 1 - $0.value, probability: $0.value) }
            .sorted { $0.distance < $1.distance }
    }

    func predict(_ samples: [MotionSample], lips: [LipSample]? = nil,
                 minProbability: Double, minConfidence: Double) -> Prediction {
        let ranked = rank(samples, lips: lips)
        let p = ranked.map { $0.probability ?? 0 }
        let confidence = p.count >= 2 ? p[0] - p[1] : (p.first ?? 0)
        let accepted = (p.first ?? 0) >= minProbability && confidence >= minConfidence
        return Prediction(ranked: ranked, confidence: confidence, accepted: accepted)
    }

    /// Feeds one window. Each feature column is a `frameCount`-long array; any other input
    /// (the LSTM's `stateIn`) starts at zero, since every word is classified on its own.
    private func probabilities(_ frames: [[Double]]) throws -> [String: Double] {
        let names = info.featureSet.channelNames
        var inputs: [String: MLFeatureValue] = [:]
        for (name, description) in model.modelDescription.inputDescriptionsByName {
            guard let constraint = description.multiArrayConstraint else { continue }
            let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
            for i in 0..<array.count { array[i] = 0 }
            if let d = names.firstIndex(of: name) {
                for k in 0..<min(array.count, frames.count) { array[k] = NSNumber(value: frames[k][d]) }
            }
            inputs[name] = MLFeatureValue(multiArray: array)
        }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        let key = model.modelDescription.predictedProbabilitiesName ?? "labelProbability"
        guard let values = output.featureValue(for: key)?.dictionaryValue else { return [:] }
        var result: [String: Double] = [:]
        for (label, value) in values {
            if let label = label as? String { result[label] = value.doubleValue }
        }
        return result
    }
}
