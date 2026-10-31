import Foundation

/// A phrase the user can mouth in one go, built from trained words.
struct Phrase: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    /// Lowercased words separated by single spaces; each needs takes in the training set.
    var text: String
    /// What to show and speak when the phrase is decoded. Empty means the phrase itself.
    var output = ""

    var words: [String] { text.split(separator: " ").map(String.init) }

    var spoken: String {
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? text : trimmed
    }
}

/// Recognizes a whole mouthed phrase as a sequence of trained words, and only accepts phrases
/// from the user's list. One-pass connected-word DTW: the segment is matched against each
/// phrase's words in order, each word against any of its takes, with optional short pauses
/// between words. New phrases need no retraining as long as their words have takes.
struct PhraseDecoder: Sendable {
    /// About 1 s at the AirPods rate. Removes slow drift (head attitude, resting lip shape)
    /// over the span of a word, so a word looks the same alone and inside a phrase.
    static let centeringWindow = 25

    let featureSet: FeatureSet
    /// Phrases whose words all have takes; the others can't be decoded yet.
    let phrases: [Phrase]
    private let scales: [Double]
    private let models: [String: [[[Double]]]]

    init?(templates: [Template], featureSet: FeatureSet, phrases: [Phrase]) {
        let (featureSet, templates) = featureSet.resolved(for: templates)
        guard !featureSet.isEmpty else { return nil }
        let raw = templates.map { MotionFeatures.locallyCentered($0.frames(featureSet), window: Self.centeringWindow) }
        let scales = MotionFeatures.channelScales(raw)
        var models: [String: [[[Double]]]] = [:]
        for (template, frames) in zip(templates, raw) where !frames.isEmpty {
            models[template.label, default: []].append(MotionFeatures.scaled(frames, by: scales))
        }
        let decodable = phrases.filter { phrase in
            !phrase.words.isEmpty && phrase.words.allSatisfy { models[$0] != nil }
        }
        // With one phrase there is nothing to compare against, so confidence is meaningless.
        guard decodable.count >= 2 else { return nil }
        self.featureSet = featureSet
        self.phrases = decodable
        self.scales = scales
        self.models = models
    }

    func rank(_ samples: [MotionSample], lips: [LipSample]? = nil) -> [RankedLabel] {
        let frames = MotionFeatures.frames(MotionFeatures.trim(samples), lips: lips, set: featureSet)
        let query = MotionFeatures.scaled(MotionFeatures.locallyCentered(frames, window: Self.centeringWindow), by: scales)
        guard !query.isEmpty else { return [] }
        return phrases
            .map { phrase in
                RankedLabel(label: phrase.text,
                            distance: Self.connectedDistance(query, phrase.words.compactMap { models[$0] }))
            }
            .sorted { $0.distance < $1.distance }
    }

    func predict(_ samples: [MotionSample], lips: [LipSample]? = nil, maxDistance: Double, minConfidence: Double) -> Prediction {
        let ranked = rank(samples, lips: lips)
        let confidence = TemplateClassifier.margin(ranked.map(\.distance))
        let accepted = (ranked.first?.distance ?? .infinity) <= maxDistance && confidence >= minConfidence
        return Prediction(ranked: ranked, confidence: confidence, accepted: accepted)
    }

    // MARK: Connected-word DTW

    private struct Cell {
        var cost = Double.infinity
        /// Path length in (query + template) steps, as in `DTW.distance`'s normalization.
        var length = 0

        static let none = Cell()
        var normalized: Double { length > 0 ? cost / Double(length) : (cost == 0 ? 0 : .infinity) }

        func step(_ n: Int) -> Cell { Cell(cost: cost, length: length + n) }

        /// Picks the path with the lower per-step cost, so short takes aren't favored.
        func better(_ other: Cell) -> Cell {
            other.cost.isFinite && (!cost.isFinite || other.normalized < normalized) ? other : self
        }
    }

    /// `words[p][take][frame]`: the takes of each word in phrase order. Returns the per-step
    /// cost of the best path through all words, or infinity if the query is too short.
    static func connectedDistance(_ query: [[Double]], _ words: [[[[Double]]]]) -> Double {
        let k = words.count
        guard k > 0, !query.isEmpty else { return .infinity }

        // Flat state layout: every frame of every take of every word, then one pause state
        // between each pair of words.
        var bases: [[Int]] = []
        var count = 0
        for takes in words {
            bases.append(takes.map { take in defer { count += take.count }; return count })
        }
        let pauseBase = count
        count += k - 1

        func wordEnd(_ cells: [Cell], _ p: Int) -> Cell {
            var best = Cell.none
            for (m, take) in words[p].enumerated() where !take.isEmpty {
                best = best.better(cells[bases[p][m] + take.count - 1])
            }
            return best
        }

        var previous = [Cell](repeating: .none, count: count)
        var current = previous
        for (i, q) in query.enumerated() {
            for s in current.indices { current[s] = .none }
            // A pause costs the frame's distance from a still face/head (zero after centering).
            let stillCost = q.reduce(0) { $0 + $1 * $1 }.squareRoot()

            for p in 0..<k {
                var entry = Cell.none
                if p == 0 {
                    if i == 0 { entry = Cell(cost: 0, length: 0) }
                } else if i > 0 {
                    entry = wordEnd(previous, p - 1).better(previous[pauseBase + p - 1])
                }
                for (m, take) in words[p].enumerated() {
                    let base = bases[p][m]
                    for j in take.indices {
                        var best = previous[base + j].step(1)                       // query advances
                        if j == 0 {
                            best = best.better(entry.step(2))                       // enter the word
                        } else {
                            best = best.better(previous[base + j - 1].step(2))      // diagonal
                            best = best.better(current[base + j - 1].step(1))       // template advances
                        }
                        if best.cost.isFinite {
                            current[base + j] = Cell(cost: best.cost + DTW.euclidean(q, take[j]), length: best.length)
                        }
                    }
                }
            }
            if i > 0 {
                for p in 0..<(k - 1) {
                    let best = previous[pauseBase + p].better(wordEnd(previous, p)).step(1)
                    if best.cost.isFinite {
                        current[pauseBase + p] = Cell(cost: best.cost + stillCost, length: best.length)
                    }
                }
            }
            swap(&previous, &current)
        }
        return wordEnd(previous, k - 1).normalized
    }
}
