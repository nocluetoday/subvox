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

    static var directory: URL {
        URL.applicationSupportDirectory.appending(path: "Subvox", directoryHint: .isDirectory)
    }
    private static var modelURL: URL { directory.appending(path: "WordModel.mlmodelc", directoryHint: .isDirectory) }
    private static var infoURL: URL { directory.appending(path: "WordModel.json") }

    static func load() -> SequenceModel? {
        guard let data = try? Data(contentsOf: infoURL),
              let info = try? JSONDecoder().decode(Info.self, from: data),
              let model = try? MLModel(contentsOf: modelURL) else { return nil }
        return SequenceModel(model: model, info: info)
    }

    static func delete() {
        try? FileManager.default.removeItem(at: modelURL)
        try? FileManager.default.removeItem(at: infoURL)
    }

    /// True when the takes or channels changed since training.
    func isStale(templates: [Template], featureSet: FeatureSet) -> Bool {
        let (set, used) = featureSet.resolved(for: templates)
        return set != info.featureSet || Set(used.map(\.id)) != info.templateIDs
    }

    // MARK: Training

    /// Trains and, when `persist` is set, replaces the saved model. Blocks for seconds to a
    /// minute; call it off the main thread.
    static func train(templates all: [Template], featureSet requested: FeatureSet,
                      maxIterations: Int = defaultIterations, persist: Bool = true) throws -> SequenceModel {
        let (featureSet, templates) = requested.resolved(for: all)
        guard !featureSet.isEmpty else { throw TrainingError.notEnoughData("Pick at least one feature channel.") }
        let counts = Dictionary(grouping: templates, by: \.label).mapValues(\.count)
        guard counts.count >= 2, counts.values.allSatisfy({ $0 >= 3 }) else {
            throw TrainingError.notEnoughData("Needs at least 2 words with 3 or more takes each.")
        }

        let raw = templates.map { $0.frames(featureSet) }
        let scales = MotionFeatures.channelScales(raw)
        let windows = raw.map { MotionFeatures.timeNormalized(MotionFeatures.scaled($0, by: scales), to: frameCount) }
        let names = featureSet.channelNames

        // Hold out every fifth take of each word that has at least five, for an honest score.
        var seen: [String: Int] = [:]
        let held = templates.map { template -> Bool in
            let k = seen[template.label, default: 0]
            seen[template.label] = k + 1
            return counts[template.label, default: 0] >= 5 && k % 5 == 4
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
            templateIDs: Set(templates.map(\.id)),
            trainingAccuracy: accuracy(classifier.trainingMetrics) ?? 0,
            validationAccuracy: validation == nil ? nil : accuracy(classifier.validationMetrics),
            validationCount: held.filter { $0 }.count)

        let fm = FileManager.default
        let source = fm.temporaryDirectory.appending(path: "Subvox-\(UUID().uuidString).mlmodel")
        try classifier.write(to: source)
        defer { try? fm.removeItem(at: source) }
        var compiled = try MLModel.compileModel(at: source)
        if persist {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            if fm.fileExists(atPath: modelURL.path) { try fm.removeItem(at: modelURL) }
            try fm.moveItem(at: compiled, to: modelURL)
            compiled = modelURL
            try JSONEncoder().encode(info).write(to: infoURL, options: .atomic)
        }
        return SequenceModel(model: try MLModel(contentsOf: compiled), info: info)
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
