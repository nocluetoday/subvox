import CoreGraphics
import Foundation

/// Mouth and jaw geometry from one camera frame. Distances are in face units: divided by the
/// inter-ocular distance and measured in a frame aligned to the eye line, so head roll and
/// distance from the camera don't leak into the features.
/// `t` is seconds since boot, the same clock as `MotionSample.t`.
struct LipSample: Codable, Hashable, Sendable {
    var t: TimeInterval
    /// Lip corner to lip corner.
    var width: Double
    /// Outer lip extent, top to bottom.
    var outerOpen: Double
    /// Inner lip aperture, top to bottom.
    var innerOpen: Double
    /// Upper lip midpoint above the line through the corners.
    var upperLip: Double
    /// Lower lip midpoint below the corner line.
    var lowerLip: Double
    /// Chin below the corner line (jaw drop).
    var jaw: Double
    /// Inner lip polygon area, in face units squared.
    var area: Double

    static let channelNames = ["lip_width", "lip_outer", "lip_inner", "lip_upper", "lip_lower", "jaw", "lip_area"]

    var vector: [Double] { [width, outerOpen, innerOpen, upperLip, lowerLip, jaw, area] }

    static let csvHeader = "t,width,outerOpen,innerOpen,upperLip,lowerLip,jaw,area"

    func csvRow(relativeTo t0: TimeInterval) -> String {
        ([t - t0] + vector).map { String(format: "%.6f", $0) }.joined(separator: ",")
    }
}

/// Lip outlines in face units (eye midpoint at the origin, +y toward the mouth), for drawing.
struct MouthShape: Hashable, Sendable {
    var outer: [CGPoint]
    var inner: [CGPoint]
}

enum LipGeometry {
    /// Builds a `LipSample` from landmark points in any image coordinate system.
    /// Returns nil when the eyes are too close together or the lip outline is missing.
    static func measure(t: TimeInterval, leftEye: CGPoint, rightEye: CGPoint,
                        outerLips: [CGPoint], innerLips: [CGPoint], contour: [CGPoint]) -> (LipSample, MouthShape)? {
        let ex = Double(rightEye.x - leftEye.x), ey = Double(rightEye.y - leftEye.y)
        let iod = (ex * ex + ey * ey).squareRoot()
        guard iod > 1e-6, outerLips.count >= 4 else { return nil }
        let origin = CGPoint(x: (leftEye.x + rightEye.x) / 2, y: (leftEye.y + rightEye.y) / 2)
        let xAxis = (ex / iod, ey / iod)
        // Pick the perpendicular that points from the eyes toward the mouth, so the frame
        // works for Vision's bottom-left origin and for top-left image coordinates alike.
        var yAxis = (-xAxis.1, xAxis.0)
        let mouth = centroid(outerLips)
        if Double(mouth.x - origin.x) * yAxis.0 + Double(mouth.y - origin.y) * yAxis.1 < 0 {
            yAxis = (-yAxis.0, -yAxis.1)
        }
        func face(_ p: CGPoint) -> CGPoint {
            let dx = Double(p.x - origin.x), dy = Double(p.y - origin.y)
            return CGPoint(x: (dx * xAxis.0 + dy * xAxis.1) / iod, y: (dx * yAxis.0 + dy * yAxis.1) / iod)
        }

        let outer = outerLips.map(face)
        let inner = innerLips.map(face)
        let chin = contour.map(face).max { $0.y < $1.y }

        guard let left = outer.min(by: { $0.x < $1.x }), let right = outer.max(by: { $0.x < $1.x }) else { return nil }
        let cornerY = Double(left.y + right.y) / 2
        let midX = Double(left.x + right.x) / 2
        let width = Double(right.x - left.x)
        guard width > 1e-6 else { return nil }

        // Lip midpoints: the outline points nearest the midline, above and below the corners.
        func nearestMidline(_ points: [CGPoint]) -> CGPoint? {
            points.min { abs(Double($0.x) - midX) < abs(Double($1.x) - midX) }
        }
        let upper = nearestMidline(outer.filter { Double($0.y) < cornerY })
        let lower = nearestMidline(outer.filter { Double($0.y) > cornerY })
        let outerYs = outer.map { Double($0.y) }
        let innerYs = inner.map { Double($0.y) }

        let sample = LipSample(
            t: t,
            width: width,
            outerOpen: (outerYs.max() ?? 0) - (outerYs.min() ?? 0),
            innerOpen: inner.isEmpty ? 0 : (innerYs.max() ?? 0) - (innerYs.min() ?? 0),
            upperLip: upper.map { cornerY - Double($0.y) } ?? 0,
            lowerLip: lower.map { Double($0.y) - cornerY } ?? 0,
            jaw: chin.map { Double($0.y) - cornerY } ?? 0,
            area: polygonArea(inner))
        return (sample, MouthShape(outer: outer, inner: inner))
    }

    private static func centroid(_ points: [CGPoint]) -> CGPoint {
        let n = CGFloat(max(points.count, 1))
        return CGPoint(x: points.reduce(0) { $0 + $1.x } / n, y: points.reduce(0) { $0 + $1.y } / n)
    }

    private static func polygonArea(_ points: [CGPoint]) -> Double {
        guard points.count >= 3 else { return 0 }
        var sum = 0.0
        for i in points.indices {
            let a = points[i], b = points[(i + 1) % points.count]
            sum += Double(a.x * b.y - b.x * a.y)
        }
        return abs(sum) / 2
    }
}
