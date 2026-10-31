import AVFoundation
import CoreAudio
import SwiftUI

/// Decode-screen panel for silent calls: decoded words → Personal Voice speech on a chosen
/// output device, plus a lip-synced avatar to screen-share as your video.
struct CallPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var model = model
        @Bindable var voice = model.voice
        GroupBox("Silent call (voice + avatar)") {
            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Voice", selection: $voice.voiceID) {
                        ForEach(voice.voices, id: \.identifier) { v in
                            Text(v.voiceTraits.contains(.isPersonalVoice) ? "\(v.name) (Personal Voice)" : v.name)
                                .tag(Optional(v.identifier))
                        }
                    }
                    personalVoiceRow
                    Picker("Output", selection: $voice.outputDeviceID) {
                        Text("System default").tag(AudioDeviceID?.none)
                        ForEach(voice.outputDevices) { device in
                            Text(device.name).tag(Optional(device.id))
                        }
                    }
                    Text("To speak into Zoom, FaceTime or Meet, pick a virtual cable (e.g. BlackHole) here and select it as the call's microphone.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Toggle("Speak each decoded word or phrase", isOn: $voice.speaksDecodedWords)
                    HStack {
                        Button("Speak Transcript") { voice.speak(model.transcript) }
                            .disabled(model.transcript.isEmpty)
                        Button("Stop", action: voice.stop)
                            .disabled(!voice.isSpeaking)
                        Button("Refresh", systemImage: "arrow.clockwise", action: voice.refresh)
                            .labelStyle(.iconOnly)
                    }
                    if let error = voice.error {
                        Text(error).font(.caption).foregroundStyle(.red)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 8) {
                    AvatarView(track: voice.mouth, greenScreen: model.avatarGreenScreen)
                        .frame(width: 200, height: 150)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    HStack {
                        Button("Open Avatar Window") { openWindow(id: "avatar") }
                        Toggle("Green", isOn: $model.avatarGreenScreen)
                            .toggleStyle(.checkbox)
                            .help("Green background for chroma-keying in OBS")
                    }
                    .font(.caption)
                }
            }
            .padding(6)
        }
    }

    @ViewBuilder
    private var personalVoiceRow: some View {
        let voice = model.voice
        switch voice.personalVoiceStatus {
        case .authorized:
            if !voice.hasPersonalVoice {
                Text("No Personal Voice found. Create one in System Settings › Accessibility › Personal Voice.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        case .notDetermined:
            Button("Use My Personal Voice…") { Task { await voice.requestPersonalVoice() } }
        case .denied:
            Text("Personal Voice access denied. Allow it in System Settings › Privacy & Security › Personal Voice.")
                .font(.caption).foregroundStyle(.orange)
        default:
            Text("Personal Voice isn't supported on this Mac; using a system voice.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
