import XCTest
@testable import Subvox

final class AutoTrainTests: XCTestCase {
    /// A synthetic word: a rotation burst on one axis.
    private func word(_ frequency: Double, axis: Int, frames: Int = 30, start: TimeInterval = 0) -> [MotionSample] {
        (0..<frames).map { i in
            let envelope = sin(Double.pi * Double(i) / Double(frames - 1))
            let v = 0.2 * envelope * sin(2 * .pi * frequency * Double(i) / Double(frames))
            let jitter = { Double.random(in: -0.01...0.01) }
            var rotation = (jitter(), jitter(), jitter())
            if axis == 0 { rotation.0 += v } else { rotation.1 += v }
            return MotionSample(t: start + Double(i) / 25, rotation: rotation)
        }
    }

    /// A take whose ID puts it in, or out of, the fixed silent test set.
    private func take(_ label: String, test: Bool, source: TakeSource? = nil) -> Template {
        var bytes = UUID().uuid
        bytes.0 = test ? 0 : 1
        let samples = label == "yes" ? word(2, axis: 0) : word(3, axis: 1)
        return Template(id: UUID(uuid: bytes), label: label, samples: samples, source: source)
    }

    func testCollectorCutsVocabularyWordsAndChecksQuality() throws {
        var collector = SpeechCollector()
        for i in 0..<125 where !(75..<100).contains(i) {     // 0–5 s at 25 Hz, no motion 3–4 s
            collector.ingest(MotionSample(t: Double(i) / 25, rotation: (0.1, 0, 0)))
        }
        for i in 0..<45 {                                    // lips only 0–1.5 s
            collector.ingest(LipSample(t: Double(i) / 30, width: 1, outerOpen: 0, innerOpen: 0,
                                       upperLip: 0, lowerLip: 0, jaw: 0, area: 0))
        }
        let outcomes = collector.cut([
            WordTiming(text: "Yes,", start: 0.5, duration: 0.4, confidence: 0.9),
            WordTiming(text: "hello", start: 1.2, duration: 0.3, confidence: 0.9),
            WordTiming(text: "no", start: 2.0, duration: 0.4, confidence: 0),
            WordTiming(text: "no", start: 2.5, duration: 0.3, confidence: 0.2),
            WordTiming(text: "stop", start: 3.3, duration: 0.4, confidence: 0.9),
            WordTiming(text: "stop", start: 4.2, duration: 0.05, confidence: 0.9),
        ], vocabulary: ["yes", "no", "stop"])

        XCTAssertEqual(outcomes.count, 5, "the non-vocabulary word is skipped")
        guard case .kept(let yes) = outcomes[0], case .kept(let no) = outcomes[1] else {
            return XCTFail("expected two kept takes, got \(outcomes)")
        }
        XCTAssertEqual(yes.label, "yes")
        XCTAssertEqual(yes.source, .speech)
        XCTAssertFalse(yes.isManual)
        XCTAssertTrue(yes.hasLips)
        XCTAssertEqual(yes.samples.first?.t ?? 0, 0.4, accuracy: 0.05)
        XCTAssertEqual(no.label, "no")
        XCTAssertFalse(no.hasLips, "camera data ended before this word")
        XCTAssertEqual(outcomes[2], .rejected(word: "no", reason: "unclear speech"))
        XCTAssertEqual(outcomes[3], .rejected(word: "stop", reason: "no AirPods motion"))
        XCTAssertEqual(outcomes[4], .rejected(word: "stop", reason: "too short"))
    }

    func testTestSetIsStableAndManualOnly() {
        let manual = take("yes", test: true)
        XCTAssertTrue(SequenceModel.isTestTake(manual))
        XCTAssertFalse(SequenceModel.isTestTake(take("yes", test: false)))
        XCTAssertFalse(SequenceModel.isTestTake(take("yes", test: true, source: .speech)))
    }

    func testChallengerIsGatedOnSilentTestTakes() throws {
        var manual: [Template] = []
        var pool: [Template] = []
        for label in ["yes", "no"] {
            manual += (0..<5).map { _ in take(label, test: false) } + (0..<3).map { _ in take(label, test: true) }
            pool += (0..<8).map { _ in take(label, test: true, source: .speech) }
        }

        let first = AutoTrainer.compete(manual: manual, pool: pool, featureSet: .motion, champion: nil)
        XCTAssertEqual(first.run.testTakes, 6)
        XCTAssertTrue(first.run.adopted, first.run.note)
        let champion = try XCTUnwrap(first.adopted)
        XCTAssertEqual(champion.info.autoTakeCount, 16)
        XCTAssertEqual(champion.info.validationCount, 6, "speech takes are never test takes")
        XCTAssertFalse(champion.isStale(templates: manual, featureSet: .motion), "collected takes don't make it stale")

        let second = AutoTrainer.compete(manual: manual, pool: pool, featureSet: .motion, champion: champion)
        XCTAssertNotNil(second.run.championAccuracy)
        XCTAssertEqual(second.run.adopted, second.adopted != nil)

        let noTests = manual.filter { !SequenceModel.isTestTake($0) }
        let blocked = AutoTrainer.compete(manual: noTests, pool: pool, featureSet: .motion, champion: champion)
        XCTAssertFalse(blocked.run.adopted)
        XCTAssertNil(blocked.adopted)
        XCTAssertTrue(blocked.run.note.contains("test takes"), blocked.run.note)
    }
}
