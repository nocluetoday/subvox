import Foundation

/// Which channels feed the classifier. Toggling these is the ablation experiment:
/// does articulation show up in acceleration, rotation, slow head attitude, or the lips?
struct FeatureSet: OptionSet, Codable, Hashable, Sendable {
    let rawValue: Int

    static let acceleration = FeatureSet(rawValue: 1 << 0)
    static let rotation = FeatureSet(rawValue: 1 << 1)
    static let attitude = FeatureSet(rawValue: 1 << 2)
    /// Camera lip and jaw landmarks, resampled onto the motion timestamps.
    static let lips = FeatureSet(rawValue: 1 << 3)
    static let motion: FeatureSet = [.acceleration, .rotation, .attitude]
    static let all: FeatureSet = [.motion, .lips]

    /// Column names in frame order, matching `MotionFeatures.frames`.
    var channelNames: [String] {
        var names: [String] = []
        if contains(.acceleration) { names += ["ax", "ay", "az"] }
        if contains(.rotation) { names += ["rx", "ry", "rz"] }
        if contains(.attitude) { names += ["pitch", "roll", "yaw"] }
        if contains(.lips) { names += LipSample.channelNames }
        return names
    }

    /// Lip channels need camera data on every take. If fewer than two words have camera takes,
    /// fall back to motion-only over all takes; otherwise keep only the takes with camera data.
    func resolved(for templates: [Template]) -> (set: FeatureSet, templates: [Template]) {
        guard contains(.lips) else { return (self, templates) }
        let paired = templates.filter(\.hasLips)
        if Set(paired.map(\.label)).count >= 2 { return (self, paired) }
        return (subtracting(.lips), templates)
    }
}

enum MotionFeatures {
    /// Activity score used for segmentation. Rotation (rad/s) and acceleration (g) live on
    /// different scales; the 5× weight puts typical jaw-motion values in the same range.
    static func energy(_ s: MotionSample) -> Double {
        let rotation = (s.rx * s.rx + s.ry * s.ry + s.rz * s.rz).squareRoot()
        let accel = (s.ax * s.ax + s.ay * s.ay + s.az * s.az).squareRoot()
        return rotation + 5 * accel
    }

    /// Drops quiet lead-in/lead-out frames so templates align on the articulation itself.
    /// Uses a threshold relative to the segment so it behaves the same for training and decoding.
    static func trim(_ samples: [MotionSample], padding: Int = 2) -> [MotionSample] {
        guard samples.count > 6 else { return samples }
        let energies = samples.map(energy)
        let sorted = energies.sorted()
        let floor = sorted[sorted.count / 5]
        let peak = sorted[sorted.count - 1]
        guard peak > floor * 1.5 else { return samples }
        let threshold = floor + 0.15 * (peak - floor)
        guard let first = energies.firstIndex(where: { $0 > threshold }),
              let last = energies.lastIndex(where: { $0 > threshold }) else { return samples }
        let lo = max(0, first - padding)
        let hi = min(samples.count - 1, last + padding)
        guard hi - lo + 1 >= 5 else { return samples }
        return Array(samples[lo...hi])
    }

    /// Per-frame feature vectors, mean-centered per channel (removes posture/heading bias).
    /// Lip channels are interpolated at each motion timestamp, so motion is the master clock;
    /// they stay zero when no camera data was captured.
    static func frames(_ samples: [MotionSample], lips: [LipSample]? = nil, set: FeatureSet) -> [[Double]] {
        guard let origin = samples.first else { return [] }
        let lipFrames = set.contains(.lips) ? resampled(lips ?? [], at: samples.map(\.t)) : []
        var frames = samples.enumerated().map { i, s -> [Double] in
            var v: [Double] = []
            if set.contains(.acceleration) { v += [s.ax, s.ay, s.az] }
            if set.contains(.rotation) { v += [s.rx, s.ry, s.rz] }
            if set.contains(.attitude) {
                v += [wrapped(s.pitch - origin.pitch), wrapped(s.roll - origin.roll), wrapped(s.yaw - origin.yaw)]
            }
            if set.contains(.lips) { v += lipFrames[i] }
            return v
        }
        guard let dims = frames.first?.count, dims > 0 else { return frames }
        for d in 0..<dims {
            let mean = frames.reduce(0) { $0 + $1[d] } / Double(frames.count)
            for i in frames.indices { frames[i][d] -= mean }
        }
        return frames
    }

    /// Per-channel standard deviation across a corpus, so no channel dominates the distance.
    static func channelScales(_ sequences: [[[Double]]]) -> [Double] {
        guard let dims = sequences.first?.first?.count else { return [] }
        var sum = [Double](repeating: 0, count: dims)
        var sumSq = [Double](repeating: 0, count: dims)
        var n = 0.0
        for sequence in sequences {
            for frame in sequence {
                for d in 0..<dims {
                    sum[d] += frame[d]
                    sumSq[d] += frame[d] * frame[d]
                }
                n += 1
            }
        }
        guard n > 1 else { return [Double](repeating: 1, count: dims) }
        return (0..<dims).map { d in
            let mean = sum[d] / n
            return max((sumSq[d] / n - mean * mean).squareRoot(), 1e-6)
        }
    }

    static func scaled(_ frames: [[Double]], by scales: [Double]) -> [[Double]] {
        frames.map { frame in zip(frame, scales).map { $0 / $1 } }
    }

    /// Linear interpolation of the lip stream at `times`, clamped at the ends.
    static func resampled(_ lips: [LipSample], at times: [TimeInterval]) -> [[Double]] {
        let zero = [Double](repeating: 0, count: LipSample.channelNames.count)
        guard !lips.isEmpty else { return times.map { _ in zero } }
        let sorted = lips.sorted { $0.t < $1.t }
        var j = 0
        return times.map { t in
            while j + 1 < sorted.count, sorted[j + 1].t <= t { j += 1 }
            let a = sorted[j]
            guard t > a.t, j + 1 < sorted.count else { return a.vector }
            let b = sorted[j + 1]
            let w = (t - a.t) / max(b.t - a.t, 1e-9)
            return zip(a.vector, b.vector).map { $0 + ($1 - $0) * w }
        }
    }

    /// Stretches or squeezes a sequence to exactly `count` frames, for fixed-window models.
    static func timeNormalized(_ frames: [[Double]], to count: Int) -> [[Double]] {
        guard let last = frames.indices.last, count > 0 else { return frames }
        return (0..<count).map { k in
            let x = count == 1 ? 0 : Double(k) * Double(last) / Double(count - 1)
            let i = Int(x), w = x - Double(i)
            guard i < last else { return frames[last] }
            return zip(frames[i], frames[i + 1]).map { $0 + ($1 - $0) * w }
        }
    }

    private static func wrapped(_ angle: Double) -> Double {
        atan2(sin(angle), cos(angle))
    }
}
