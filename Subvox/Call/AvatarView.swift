import SwiftUI

/// Stylized face whose mouth follows the speech audio's mouth track. Drawn every display
/// frame; the window can be screen-shared into a call, or keyed out on green in OBS.
struct AvatarView: View {
    var track: MouthTrack
    var greenScreen = false

    var body: some View {
        TimelineView(.animation) { _ in
            let t = ProcessInfo.processInfo.systemUptime
            let mouth = track.frame(at: t)
            Canvas { context, size in
                draw(in: &context, size: size, mouth: mouth, t: t)
            }
        }
        .background(greenScreen ? AnyShapeStyle(Color(red: 0, green: 0.85, blue: 0))
                                : AnyShapeStyle(LinearGradient(colors: [.indigo.opacity(0.5), .black.opacity(0.8)],
                                                               startPoint: .top, endPoint: .bottom)))
    }

    private func draw(in context: inout GraphicsContext, size: CGSize, mouth: MouthFrame, t: TimeInterval) {
        let side = min(size.width, size.height)
        let cx = size.width / 2, cy = size.height / 2
        let headW = side * 0.62, headH = side * 0.78
        // A slow idle sway and a speech-linked nod keep the face from looking frozen.
        let sway = sin(t * 0.7) * side * 0.006
        let nod = mouth.open * side * 0.008
        let head = CGRect(x: cx - headW / 2 + sway, y: cy - headH / 2 + nod, width: headW, height: headH)

        context.fill(Path(ellipseIn: head), with: .color(Color(red: 0.93, green: 0.78, blue: 0.66)))
        context.stroke(Path(ellipseIn: head), with: .color(.black.opacity(0.25)), lineWidth: side * 0.006)

        // Eyes, with a blink every ~4 s.
        let blink = t.truncatingRemainder(dividingBy: 4.2) < 0.13
        let eyeY = head.minY + headH * 0.40
        let eyeH = blink ? side * 0.006 : side * 0.05
        for dx in [-0.18, 0.18] {
            let eye = CGRect(x: head.midX + headW * dx - side * 0.03, y: eyeY - eyeH / 2, width: side * 0.06, height: eyeH)
            context.fill(Path(ellipseIn: eye), with: .color(.black.opacity(0.8)))
        }

        // Mouth: jaw opening sets height, spread sets width.
        let mouthW = headW * 0.32 * (0.85 + 0.35 * mouth.spread)
        let mouthH = max(side * 0.008, headH * 0.17 * mouth.open)
        let mouthY = head.minY + headH * 0.70
        let mouthRect = CGRect(x: head.midX - mouthW / 2, y: mouthY - mouthH / 2, width: mouthW, height: mouthH)
        let lips = Path(roundedRect: mouthRect, cornerSize: CGSize(width: mouthW / 2, height: mouthH / 2))
        context.fill(lips, with: .color(Color(red: 0.35, green: 0.08, blue: 0.10)))
        if mouth.open > 0.25 {
            let teeth = CGRect(x: mouthRect.minX + mouthW * 0.2, y: mouthRect.minY, width: mouthW * 0.6, height: mouthH * 0.22)
            context.fill(Path(roundedRect: teeth, cornerRadius: mouthH * 0.08), with: .color(.white.opacity(0.9)))
        }
        context.stroke(lips, with: .color(Color(red: 0.70, green: 0.30, blue: 0.32)), lineWidth: side * 0.012)
    }
}

/// The detachable avatar window (Window ▸ Avatar), sized for a video tile.
struct AvatarWindow: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        AvatarView(track: model.voice.mouth, greenScreen: model.avatarGreenScreen)
            .frame(minWidth: 320, minHeight: 240)
            .ignoresSafeArea()
    }
}
