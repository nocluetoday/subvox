import Foundation

/// Turns your own speech into labeled takes with no button presses. The speech recognizer
/// says which word was said and when; the collector cuts the motion and lip samples over
/// that time window and keeps the take if it passes the quality checks.
struct SpeechCollector {
    enum Outcome: Hashable, Sendable {
        case kept(Template)
        case rejected(word: String, reason: String)
    }

    /// Final results can arrive long after the words were said; keep enough history to cover them.
    static let bufferSeconds: TimeInterval = 120
    /// Motion and lips are cut a little wider than the word; `MotionFeatures.trim` tightens it.
    static let padding: TimeInterval = 0.12
    static let minConfidence = 0.5
    static let durationRange: ClosedRange<TimeInterval> = 0.12...2.0
    static let maxMotionGap: TimeInterval = 0.2
    static let maxLipGap: TimeInterval = 0.25

    private(set) var motion: [MotionSample] = []
    private(set) var lips: [LipSample] = []

    mutating func ingest(_ sample: MotionSample) {
        motion.append(sample)
        if let first = motion.first, sample.t - first.t > Self.bufferSeconds * 1.25 {
            motion.removeAll { $0.t < sample.t - Self.bufferSeconds }
        }
    }

    mutating func ingest(_ sample: LipSample) {
        lips.append(sample)
        if let first = lips.first, sample.t - first.t > Self.bufferSeconds * 1.25 {
            lips.removeAll { $0.t < sample.t - Self.bufferSeconds }
        }
    }

    /// One outcome per recognized word that is in `vocabulary`; other words are skipped.
    func cut(_ words: [WordTiming], vocabulary: Set<String>) -> [Outcome] {
        words.compactMap { word in
            let label = Self.normalized(word.text)
            guard vocabulary.contains(label) else { return nil }
            if word.confidence > 0, word.confidence < Self.minConfidence {
                return .rejected(word: label, reason: "unclear speech")
            }
            guard Self.durationRange.contains(word.duration) else {
                return .rejected(word: label, reason: word.duration < Self.durationRange.lowerBound ? "too short" : "too long")
            }
            let window = (word.start - Self.padding)...(word.start + word.duration + Self.padding)
            let samples = motion.filter { window.contains($0.t) }
            guard Self.covers(samples.map(\.t), window, maxGap: Self.maxMotionGap) else {
                return .rejected(word: label, reason: "no AirPods motion")
            }
            // Lips are optional: keep a motion-only take when the face wasn't tracked.
            let lipSamples = lips.filter { window.contains($0.t) }
            let paired = Self.covers(lipSamples.map(\.t), window, maxGap: Self.maxLipGap)
            return .kept(Template(label: label, samples: samples, lips: paired ? lipSamples : nil, source: .speech))
        }
    }

    static func normalized(_ text: String) -> String {
        String(text.lowercased().filter { $0.isLetter || $0.isNumber || $0 == "'" })
    }

    /// True when `times` spans the window with no hole longer than `maxGap`.
    static func covers(_ times: [TimeInterval], _ window: ClosedRange<TimeInterval>, maxGap: TimeInterval) -> Bool {
        guard times.count >= 4, let first = times.first, let last = times.last else { return false }
        guard first - window.lowerBound <= maxGap, window.upperBound - last <= maxGap else { return false }
        return zip(times, times.dropFirst()).allSatisfy { $1 - $0 <= maxGap }
    }
}
