import XCTest
@testable import Subvox

final class PhraseTests: XCTestCase {
    /// A synthetic word: a rotation burst on one axis, as in `DecodingTests`.
    private func word(_ frequency: Double, axis: Int, frames: Int = 30, noise: Double = 0.01) -> [MotionSample] {
        (0..<frames).map { i in
            let envelope = sin(Double.pi * Double(i) / Double(frames - 1))
            let v = 0.2 * envelope * sin(2 * .pi * frequency * Double(i) / Double(frames))
            var rotation = (Double.random(in: -noise...noise), Double.random(in: -noise...noise), Double.random(in: -noise...noise))
            switch axis {
            case 0: rotation.0 += v
            case 1: rotation.1 += v
            default: rotation.2 += v
            }
            return MotionSample(t: Double(i) / 25, rotation: rotation)
        }
    }

    private let yes = (2.0, 0), no = (3.0, 1), stop = (2.0, 2)

    /// Words back to back on one clock, with `gap` still frames between them.
    private func utterance(_ words: [(Double, Int)], gap: Int = 0, frames: Int = 30) -> [MotionSample] {
        var samples: [MotionSample] = []
        for (n, w) in words.enumerated() {
            if n > 0 {
                for _ in 0..<gap { samples.append(MotionSample(t: 0)) }
            }
            samples += word(w.0, axis: w.1, frames: frames)
        }
        return samples.enumerated().map { i, s in
            var s = s
            s.t = Double(i) / 25
            return s
        }
    }

    private func decoder() throws -> PhraseDecoder {
        var templates: [Template] = []
        for _ in 0..<5 {
            templates.append(Template(label: "yes", samples: word(yes.0, axis: yes.1)))
            templates.append(Template(label: "no", samples: word(no.0, axis: no.1)))
            templates.append(Template(label: "stop", samples: word(stop.0, axis: stop.1)))
        }
        let phrases = ["yes no", "no yes", "yes stop", "stop no yes", "no stop"].map { Phrase(text: $0) }
        return try XCTUnwrap(PhraseDecoder(templates: templates, featureSet: .motion, phrases: phrases))
    }

    func testDecodesPhraseFromWordTakes() throws {
        let decoder = try decoder()
        XCTAssertEqual(decoder.rank(utterance([yes, no])).first?.label, "yes no")
        XCTAssertEqual(decoder.rank(utterance([no, yes])).first?.label, "no yes")
        XCTAssertEqual(decoder.rank(utterance([stop, no, yes])).first?.label, "stop no yes")
    }

    func testPhraseToleratesPausesAndTempo() throws {
        let decoder = try decoder()
        XCTAssertEqual(decoder.rank(utterance([yes, stop], gap: 10)).first?.label, "yes stop")
        // Said about 30% faster than the takes.
        let prediction = decoder.predict(utterance([no, stop], frames: 21), maxDistance: 10, minConfidence: 0)
        XCTAssertEqual(prediction.best?.label, "no stop")
        XCTAssertTrue(prediction.accepted)
    }

    func testPhrasesNeedTrainedWords() {
        let templates = (0..<3).flatMap { _ in
            [Template(label: "yes", samples: word(yes.0, axis: yes.1)), Template(label: "no", samples: word(no.0, axis: no.1))]
        }
        let phrases = [Phrase(text: "yes no"), Phrase(text: "yes later")]
        XCTAssertNil(PhraseDecoder(templates: templates, featureSet: .motion, phrases: phrases))
        let ok = PhraseDecoder(templates: templates, featureSet: .motion, phrases: phrases + [Phrase(text: "no yes")])
        XCTAssertEqual(ok?.phrases.map(\.text), ["yes no", "no yes"])
    }

    func testOutputDefaultsToPhraseText() {
        XCTAssertEqual(Phrase(text: "call back").spoken, "call back")
        XCTAssertEqual(Phrase(text: "call back", output: " I'll call you back. ").spoken, "I'll call you back.")
    }

    func testOldDatasetWithoutPhrasesStillLoads() throws {
        let json = #"{"vocabulary":["yes"],"templates":[]}"#
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(json.utf8))
        XCTAssertEqual(dataset.vocabulary, ["yes"])
        XCTAssertTrue(dataset.phrases.isEmpty)
    }
}
