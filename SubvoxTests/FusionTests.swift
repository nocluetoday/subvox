import AVFoundation
import XCTest
@testable import Subvox

final class FusionTests: XCTestCase {
    /// Motion that is the same for every word, so only the lips can tell words apart.
    private func motion(frames: Int = 30) -> [MotionSample] {
        (0..<frames).map { i in
            let v = 0.2 * sin(Double.pi * Double(i) / Double(frames - 1)) + Double.random(in: -0.01...0.01)
            return MotionSample(t: Double(i) / 25, rotation: (v, 0, 0))
        }
    }

    /// 30 fps lip track: the mouth opens `cycles` times over the take.
    private func lips(cycles: Double, duration: Double = 1.16) -> [LipSample] {
        stride(from: -0.05, through: duration + 0.05, by: 1.0 / 30).map { t in
            let open = 0.15 * max(0, sin(2 * .pi * cycles * t / duration)) + Double.random(in: -0.005...0.005)
            return LipSample(t: t, width: 0.9 - open * 0.5, outerOpen: 0.25 + open, innerOpen: open,
                             upperLip: 0.1 + open * 0.3, lowerLip: 0.15 + open * 0.7, jaw: 0.9 + open, area: open * 0.1)
        }
    }

    func testLipGeometryIgnoresHeadRollAndDistance() throws {
        func face(roll: Double, scale: Double, open: Double) -> LipSample? {
            func p(_ x: Double, _ y: Double) -> CGPoint {
                let (s, c) = (sin(roll), cos(roll))
                return CGPoint(x: 300 + scale * (x * c - y * s), y: 200 + scale * (x * s + y * c))
            }
            let outer = [p(-0.45, 1.2), p(0, 1.1 - open / 2), p(0.45, 1.2), p(0, 1.3 + open / 2)]
            let inner = [p(-0.35, 1.2), p(0, 1.2 - open / 2), p(0.35, 1.2), p(0, 1.2 + open / 2)]
            return LipGeometry.measure(t: 0, leftEye: p(-0.5, 0), rightEye: p(0.5, 0),
                                       outerLips: outer, innerLips: inner, contour: [p(-0.8, 1), p(0, 2), p(0.8, 1)])?.0
        }
        let upright = try XCTUnwrap(face(roll: 0, scale: 100, open: 0.2))
        let tilted = try XCTUnwrap(face(roll: 0.5, scale: 240, open: 0.2))
        for (a, b) in zip(upright.vector, tilted.vector) { XCTAssertEqual(a, b, accuracy: 1e-9) }
        XCTAssertEqual(upright.width, 0.9, accuracy: 1e-9)
        XCTAssertEqual(upright.innerOpen, 0.2, accuracy: 1e-9)
        XCTAssertEqual(upright.jaw, 0.8, accuracy: 1e-9)
        let wider = try XCTUnwrap(face(roll: 0, scale: 100, open: 0.4))
        XCTAssertGreaterThan(wider.innerOpen, upright.innerOpen)
        XCTAssertGreaterThan(wider.area, upright.area)
    }

    func testLipsResampleOntoMotionClock() {
        let a = LipSample(t: 1.0, width: 1, outerOpen: 0, innerOpen: 0, upperLip: 0, lowerLip: 0, jaw: 0, area: 0)
        var b = a
        b.t = 1.1
        b.width = 2
        let frames = MotionFeatures.resampled([b, a], at: [0.9, 1.05, 1.2])
        XCTAssertEqual(frames.map { $0[0] }, [1, 1.5, 2])
        XCTAssertEqual(MotionFeatures.resampled([], at: [0, 1]), [[Double]](repeating: [0, 0, 0, 0, 0, 0, 0], count: 2))
    }

    func testLipChannelsSeparateWordsMotionCannot() throws {
        var templates: [Template] = []
        for _ in 0..<8 {
            templates.append(Template(label: "yes", samples: motion(), lips: lips(cycles: 1)))
            templates.append(Template(label: "no", samples: motion(), lips: lips(cycles: 3)))
        }
        let fused = try XCTUnwrap(TemplateClassifier(templates: templates, featureSet: [.rotation, .lips]))
        XCTAssertTrue(fused.featureSet.contains(.lips))
        XCTAssertEqual(fused.predict(motion(), lips: lips(cycles: 3), maxDistance: 10, minConfidence: 0).best?.label, "no")
        XCTAssertEqual(fused.predict(motion(), lips: lips(cycles: 1), maxDistance: 10, minConfidence: 0).best?.label, "yes")

        let result = try XCTUnwrap(Evaluation.run(templates: templates, featureSet: [.rotation, .lips], permutations: 200))
        XCTAssertGreaterThan(result.accuracy, 0.9)
        XCTAssertLessThan(result.pValue, 0.01)
    }

    func testLipsFallBackToMotionWithoutCameraTakes() throws {
        let templates = (0..<4).flatMap { _ in
            [Template(label: "yes", samples: motion()), Template(label: "no", samples: motion())]
        }
        let classifier = try XCTUnwrap(TemplateClassifier(templates: templates, featureSet: .all))
        XCTAssertEqual(classifier.featureSet, .motion)
        XCTAssertNil(TemplateClassifier(templates: templates, featureSet: .lips))
    }

    func testTimeNormalizationKeepsEndpoints() {
        let frames = (0..<10).map { [Double($0)] }
        let normalized = MotionFeatures.timeNormalized(frames, to: 19)
        XCTAssertEqual(normalized.count, 19)
        XCTAssertEqual(normalized.first?[0], 0)
        XCTAssertEqual(normalized.last?[0], 9)
        XCTAssertEqual(normalized[1][0], 0.5, accuracy: 1e-9)
    }

    func testSequenceModelLearnsFusedWords() throws {
        var templates: [Template] = []
        for _ in 0..<10 {
            templates.append(Template(label: "yes", samples: motion(), lips: lips(cycles: 1)))
            templates.append(Template(label: "no", samples: motion(), lips: lips(cycles: 3)))
        }
        let model = try SequenceModel.train(templates: templates, featureSet: [.rotation, .lips],
                                            maxIterations: 60, persist: false)
        XCTAssertEqual(Set(model.info.labels), ["yes", "no"])
        XCTAssertEqual(model.info.validationCount, 4)
        XCTAssertFalse(model.isStale(templates: templates, featureSet: [.rotation, .lips]))
        XCTAssertTrue(model.isStale(templates: templates, featureSet: .motion))

        let yes = model.predict(motion(), lips: lips(cycles: 1), minProbability: 0, minConfidence: 0)
        let no = model.predict(motion(), lips: lips(cycles: 3), minProbability: 0, minConfidence: 0)
        XCTAssertEqual(yes.best?.label, "yes")
        XCTAssertEqual(no.best?.label, "no")
        XCTAssertEqual(yes.ranked.compactMap(\.probability).reduce(0, +), 1, accuracy: 0.01)
    }

    func testMouthTrackFollowsSpeechAudio() throws {
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 22050, channels: 1))
        func buffer(_ sample: (Int) -> Float) throws -> AVAudioPCMBuffer {
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 2205))
            buffer.frameLength = 2205
            for i in 0..<2205 { buffer.floatChannelData![0][i] = sample(i) }
            return buffer
        }
        let silence = MouthTrack.analyze(try buffer { _ in 0 })
        let vowel = MouthTrack.analyze(try buffer { Float(0.5 * sin(2 * .pi * 180 * Double($0) / 22050)) })
        let hiss = MouthTrack.analyze(try buffer { _ in Float.random(in: -0.3...0.3) })
        XCTAssertEqual(silence.count, 10)
        XCTAssertEqual(silence.map(\.open).max() ?? 1, 0, accuracy: 1e-9)
        XCTAssertGreaterThan(vowel.map(\.open).min() ?? 0, 0.6)
        XCTAssertLessThan(vowel.map(\.spread).max() ?? 1, 0.1)
        XCTAssertGreaterThan(hiss.map(\.spread).min() ?? 0, 0.8)

        let track = MouthTrack()
        track.append(start: 100, frames: vowel)
        XCTAssertGreaterThan(track.frame(at: 100.05).open, 0.6)
        XCTAssertEqual(track.frame(at: 50), .rest)
    }
}
