import AVFoundation
import CoreAudio
import Observation

/// Speaks decoded text for silent calls. Uses the user's Personal Voice (their on-device voice
/// clone) when authorized, else a system voice. Audio plays on a chosen output device, e.g. a
/// virtual cable that a call app uses as its microphone. Each buffer is also analyzed into a
/// mouth track that drives the lip-synced avatar.
@MainActor @Observable
final class VoiceOutput {
    private(set) var voices: [AVSpeechSynthesisVoice] = []
    var voiceID: String?
    private(set) var personalVoiceStatus = AVSpeechSynthesizer.personalVoiceAuthorizationStatus
    private(set) var outputDevices: [AudioDevices.Device] = []
    /// `nil` means "system default output".
    var outputDeviceID: AudioDeviceID? {
        didSet { if oldValue != outputDeviceID { tearDown() } }
    }
    var speaksDecodedWords = false
    private(set) var isSpeaking = false
    private(set) var lastSpoken: String?
    private(set) var error: String?

    @ObservationIgnored let mouth = MouthTrack()
    @ObservationIgnored private let synthesizer = AVSpeechSynthesizer()
    @ObservationIgnored private var engine: AVAudioEngine?
    @ObservationIgnored private var player: AVAudioPlayerNode?
    @ObservationIgnored private var playerFormat: AVAudioFormat?
    @ObservationIgnored private var scheduledEnd: TimeInterval = 0
    @ObservationIgnored private var pending = 0

    var selectedVoice: AVSpeechSynthesisVoice? { voices.first { $0.identifier == voiceID } }
    var hasPersonalVoice: Bool { voices.contains { $0.voiceTraits.contains(.isPersonalVoice) } }

    func refresh() {
        personalVoiceStatus = AVSpeechSynthesizer.personalVoiceAuthorizationStatus
        voices = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.voiceTraits.contains(.isPersonalVoice) || $0.language.hasPrefix("en") }
            .sorted { a, b in
                let pa = a.voiceTraits.contains(.isPersonalVoice), pb = b.voiceTraits.contains(.isPersonalVoice)
                if pa != pb { return pa }
                if a.quality != b.quality { return a.quality.rawValue > b.quality.rawValue }
                return a.name < b.name
            }
        if selectedVoice == nil { voiceID = voices.first?.identifier }
        outputDevices = AudioDevices.outputDevices()
    }

    func requestPersonalVoice() async {
        personalVoiceStatus = await withCheckedContinuation { continuation in
            AVSpeechSynthesizer.requestPersonalVoiceAuthorization { continuation.resume(returning: $0) }
        }
        voiceID = nil
        refresh()
    }

    func speak(_ raw: String) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        error = nil
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = selectedVoice
        lastSpoken = text
        pending += 1
        isSpeaking = true
        synthesizer.write(utterance, toBufferCallback: Self.bufferHandler(owner: self))
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        player?.stop()
        mouth.clear()
        scheduledEnd = 0
        pending = 0
        isSpeaking = false
    }

    private func enqueue(_ buffer: AVAudioPCMBuffer) {
        // An empty buffer marks the end of an utterance.
        guard buffer.frameLength > 0 else {
            finishUtterance()
            return
        }
        guard let pcm = Self.floatBuffer(buffer), let player = playerReady(for: pcm.format), let engine else { return }
        let now = ProcessInfo.processInfo.systemUptime
        let start = max(now + engine.outputNode.presentationLatency, scheduledEnd)
        scheduledEnd = start + Double(pcm.frameLength) / pcm.format.sampleRate
        mouth.append(start: start, frames: MouthTrack.analyze(pcm))
        player.scheduleBuffer(pcm)
        if !player.isPlaying { player.play() }
    }

    private func finishUtterance() {
        pending = max(0, pending - 1)
        let remaining = max(0, scheduledEnd - ProcessInfo.processInfo.systemUptime)
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(remaining))
            guard let self, self.pending == 0 else { return }
            self.isSpeaking = false
        }
    }

    private func playerReady(for format: AVAudioFormat) -> AVAudioPlayerNode? {
        if engine == nil {
            let engine = AVAudioEngine()
            if let outputDeviceID, let unit = engine.outputNode.audioUnit {
                var id = outputDeviceID
                let status = AudioUnitSetProperty(
                    unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                    &id, UInt32(MemoryLayout<AudioDeviceID>.size))
                if status != noErr {
                    error = "Couldn't switch output device (\(status))."
                    return nil
                }
            }
            let player = AVAudioPlayerNode()
            engine.attach(player)
            self.engine = engine
            self.player = player
            playerFormat = nil
        }
        guard let engine, let player else { return nil }
        // Voices differ in sample rate; reconnect when the format changes.
        if playerFormat != format {
            player.stop()
            engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            playerFormat = format
        }
        if !engine.isRunning {
            do {
                engine.prepare()
                try engine.start()
            } catch {
                self.error = error.localizedDescription
                return nil
            }
        }
        return player
    }

    private func tearDown() {
        stop()
        engine?.stop()
        engine = nil
        player = nil
        playerFormat = nil
    }

    /// Synthesizer voices may emit 16-bit buffers; the player wants deinterleaved float.
    private static func floatBuffer(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let format = buffer.format
        if format.commonFormat == .pcmFormatFloat32, !format.isInterleaved { return buffer }
        guard let target = AVAudioFormat(standardFormatWithSampleRate: format.sampleRate, channels: format.channelCount),
              let converter = AVAudioConverter(from: format, to: target),
              let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: buffer.frameLength) else { return nil }
        do {
            try converter.convert(to: out, from: buffer)
        } catch {
            return nil
        }
        return out
    }

    // Built outside main-actor isolation: the synthesizer calls back on its own queue.
    nonisolated private static func bufferHandler(owner: VoiceOutput) -> (AVAudioBuffer) -> Void {
        { [weak owner] buffer in
            guard let pcm = buffer as? AVAudioPCMBuffer else { return }
            let box = BufferBox(buffer: pcm)
            Task { @MainActor in owner?.enqueue(box.buffer) }
        }
    }
}

private struct BufferBox: @unchecked Sendable {
    let buffer: AVAudioPCMBuffer
}

/// Mouth pose at one instant: `open` is jaw/lip opening, `spread` is lip width (high for
/// "s"/"ee"-like sounds). Both 0…1.
struct MouthFrame: Hashable, Sendable {
    var open: Double
    var spread: Double

    static let rest = MouthFrame(open: 0, spread: 0.3)
}

/// Timeline of mouth poses for scheduled speech audio, keyed by host time (seconds since
/// boot). Written on the main thread, read by the avatar's render loop.
final class MouthTrack: @unchecked Sendable {
    static let frameSeconds = 0.01

    private struct Clip {
        var start: TimeInterval
        var frames: [MouthFrame]
        var end: TimeInterval { start + Double(frames.count) * MouthTrack.frameSeconds }
    }

    private let lock = NSLock()
    private var clips: [Clip] = []

    func append(start: TimeInterval, frames: [MouthFrame]) {
        lock.withLock {
            clips.removeAll { $0.end < start - 5 }
            clips.append(Clip(start: start, frames: frames))
        }
    }

    func clear() {
        lock.withLock { clips.removeAll() }
    }

    /// Pose at host time `t`, averaged over ±30 ms so the mouth doesn't flicker.
    func frame(at t: TimeInterval) -> MouthFrame {
        lock.withLock {
            var open = 0.0, spread = 0.0, n = 0.0
            for offset in stride(from: -0.03, through: 0.03, by: Self.frameSeconds) {
                let time = t + offset
                guard let clip = clips.last(where: { $0.start <= time && time < $0.end }) else { continue }
                let index = min(clip.frames.count - 1, Int((time - clip.start) / Self.frameSeconds))
                open += clip.frames[index].open
                spread += clip.frames[index].spread
                n += 1
            }
            guard n > 0 else { return .rest }
            // Edge frames outside any clip count as a closed mouth.
            let total = 7.0
            return MouthFrame(open: open / total, spread: (spread + MouthFrame.rest.spread * (total - n)) / total)
        }
    }

    /// Per-10 ms loudness → opening and zero-crossing rate → spread. Fricatives (high ZCR)
    /// get a narrower jaw and wider lips; vowels (loud, low ZCR) open the jaw.
    static func analyze(_ buffer: AVAudioPCMBuffer) -> [MouthFrame] {
        guard let channel = buffer.floatChannelData?[0] else { return [] }
        let count = Int(buffer.frameLength)
        let hop = max(1, Int(buffer.format.sampleRate * frameSeconds))
        // A trailing scrap under half a frame is dropped rather than read as its own frame.
        return stride(from: 0, to: count - hop / 2, by: hop).map { start in
            let end = min(count, start + hop)
            var sum: Float = 0
            var crossings = 0
            for i in start..<end {
                sum += channel[i] * channel[i]
                if i > start, (channel[i] >= 0) != (channel[i - 1] >= 0) { crossings += 1 }
            }
            let n = Float(end - start)
            let db = 20 * log10(max((sum / n).squareRoot(), 1e-7))
            let loudness = Double(max(0, min(1, (db + 45) / 35)))
            let zcr = Double(crossings) / Double(max(end - start - 1, 1))
            let spread = max(0, min(1, (zcr - 0.05) / 0.25))
            return MouthFrame(open: loudness * (1 - 0.5 * spread), spread: loudness > 0.05 ? spread : MouthFrame.rest.spread)
        }
    }
}
