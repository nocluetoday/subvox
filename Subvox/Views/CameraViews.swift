import SwiftUI

/// Sensors-screen card: camera preview, tracked mouth outline and live lip features.
struct CameraCard: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let camera = model.camera
        let span = camera.latest.map { ($0.t - 8)...$0.t }
        GroupBox("Front camera (lips + jaw)") {
            VStack(alignment: .leading, spacing: 10) {
                CameraStatusRow()
                if camera.isRunning {
                    HStack(alignment: .top, spacing: 12) {
                        CameraPreview(session: camera.session)
                            .frame(width: 240, height: 135)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        MouthScope(shape: camera.mouth)
                            .frame(width: 135, height: 135)
                        VStack(alignment: .leading, spacing: 4) {
                            LabeledContent("Frame rate", value: String(format: "%.1f fps", camera.frameRate))
                            LabeledContent("Latency", value: String(format: "%.0f ms", camera.latency * 1000))
                            if let lip = camera.latest {
                                LabeledContent("Aperture", value: String(format: "%.3f", lip.innerOpen))
                                LabeledContent("Width", value: String(format: "%.3f", lip.width))
                                LabeledContent("Jaw", value: String(format: "%.3f", lip.jaw))
                            }
                        }
                        .font(.callout.monospacedDigit())
                    }
                    LipPlot(samples: camera.history, span: span)
                }
                Text("Lip and jaw landmarks share the AirPods clock, so each take stores both. Face the camera in even light.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(6)
        }
    }
}

/// One-line camera status with a start/stop button, for the Train and Decode screens.
struct CameraStatusRow: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let camera = model.camera
        HStack(spacing: 8) {
            Circle().fill(color(camera.status)).frame(width: 8, height: 8)
            Text(camera.status.label).font(.callout)
            if let name = camera.cameraName, camera.isRunning {
                Text("· \(name)").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button(camera.isRunning ? "Stop Camera" : "Start Camera") {
                Task { await model.toggleCamera() }
            }
        }
    }

    private func color(_ status: FaceCameraService.Status) -> Color {
        switch status {
        case .tracking: .green
        case .starting, .noFace: .orange
        case .off: .secondary
        default: .red
        }
    }
}

/// The tracked lip outlines in face units, scaled to fit.
struct MouthScope: View {
    var shape: MouthShape?

    var body: some View {
        Canvas { context, size in
            guard let shape, !shape.outer.isEmpty else { return }
            let all = shape.outer + shape.inner
            let minX = all.map(\.x).min()!, maxX = all.map(\.x).max()!
            let minY = all.map(\.y).min()!, maxY = all.map(\.y).max()!
            // Fixed scale (face units) so opening the mouth visibly grows the outline.
            let scale = size.width * 0.8 / max(maxX - minX, 0.6)
            let center = CGPoint(x: (minX + maxX) / 2, y: (minY + maxY) / 2)
            func map(_ p: CGPoint) -> CGPoint {
                CGPoint(x: size.width / 2 + (p.x - center.x) * scale, y: size.height / 2 + (p.y - center.y) * scale)
            }
            for (points, color) in [(shape.outer, Color.pink), (shape.inner, Color.red)] where points.count > 2 {
                var path = Path()
                path.addLines(points.map(map))
                path.closeSubpath()
                context.stroke(path, with: .color(color), lineWidth: 1.5)
            }
        }
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        .overlay {
            if shape == nil {
                Text("No face").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

/// Live plot of the main lip channels, same look as `SignalPlot`.
struct LipPlot: View {
    var samples: [LipSample]
    var span: ClosedRange<TimeInterval>?
    var height: CGFloat = 70

    private static let channels: [(String, Color, KeyPath<LipSample, Double>)] = [
        ("aperture", .red, \.innerOpen),
        ("width", .blue, \.width),
        ("jaw", .orange, \.jaw),
    ]

    var body: some View {
        let range = span ?? ((samples.first?.t ?? 0)...(samples.last?.t ?? 1))
        let visible = samples.filter { range.contains($0.t) }
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                Text("Lip features (face units, mean-removed)").font(.caption.weight(.semibold))
                ForEach(Self.channels, id: \.0) { channel in
                    Label(channel.0, systemImage: "circle.fill")
                        .labelStyle(.titleAndIcon)
                        .font(.caption2)
                        .foregroundStyle(channel.1)
                }
            }
            Canvas { context, size in
                let width = max(range.upperBound - range.lowerBound, 1e-3)
                for channel in Self.channels {
                    let values = visible.map { $0[keyPath: channel.2] }
                    guard !values.isEmpty else { continue }
                    let mean = values.reduce(0, +) / Double(values.count)
                    let scale = max(values.map { abs($0 - mean) }.max() ?? 0, 0.02)
                    var path = Path()
                    for (i, s) in visible.enumerated() {
                        let point = CGPoint(
                            x: (s.t - range.lowerBound) / width * size.width,
                            y: size.height / 2 - (values[i] - mean) / scale * (size.height / 2 - 2))
                        if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
                    }
                    context.stroke(path, with: .color(channel.1), lineWidth: 1.2)
                }
            }
            .frame(height: height)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        }
    }
}
