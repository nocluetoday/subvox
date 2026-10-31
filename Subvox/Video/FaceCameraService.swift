import AVFoundation
import Observation
import SwiftUI
import Vision

/// Streams lip and jaw landmarks from the Mac's front camera. Each frame's presentation time
/// is converted to the host clock, so `LipSample.t` lines up with `MotionSample.t`.
@MainActor @Observable
final class FaceCameraService {
    enum Status: Equatable {
        case off
        case denied
        case unavailable
        case starting
        case noFace
        case tracking
        case error(String)

        var label: String {
            switch self {
            case .off: "Camera off"
            case .denied: "Camera access denied"
            case .unavailable: "No camera found"
            case .starting: "Starting camera…"
            case .noFace: "No face in view"
            case .tracking: "Tracking lips"
            case .error(let message): message
            }
        }
    }

    static let historySeconds: TimeInterval = 10

    private(set) var status: Status = .off
    private(set) var isRunning = false
    private(set) var cameraName: String?
    private(set) var latest: LipSample?
    private(set) var mouth: MouthShape?
    private(set) var history: [LipSample] = []
    private(set) var frameRate: Double = 0
    /// Seconds from frame capture until its features reach the main thread.
    private(set) var latency: Double = 0

    var onSample: ((LipSample) -> Void)?

    @ObservationIgnored let session = AVCaptureSession()
    @ObservationIgnored private let processor = FrameProcessor()
    @ObservationIgnored private var configured = false
    @ObservationIgnored private var rateWindow: [TimeInterval] = []

    func start() async {
        guard !isRunning else { return }
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            status = .denied
            return
        }
        if !configured {
            guard configure() else { return }
        }
        status = .starting
        isRunning = true
        processor.onResult = { [weak self] result, t in
            Task { @MainActor in self?.handle(result, t) }
        }
        let session = UncheckedSession(session: session)
        // startRunning blocks until the camera is up; keep it off the main thread.
        await Task.detached { session.session.startRunning() }.value
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        processor.onResult = nil
        let session = UncheckedSession(session: session)
        Task.detached { session.session.stopRunning() }
        status = .off
        frameRate = 0
        latest = nil
        mouth = nil
        rateWindow.removeAll()
    }

    /// Lip samples captured within `range` (seconds since boot).
    func lips(in range: ClosedRange<TimeInterval>) -> [LipSample] {
        history.filter { range.contains($0.t) }
    }

    private func configure() -> Bool {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .unspecified)
            ?? AVCaptureDevice.default(for: .video) else {
            status = .unavailable
            return false
        }
        do {
            session.beginConfiguration()
            defer { session.commitConfiguration() }
            // 720p gives the lip outline enough pixels; Vision still runs well under frame time.
            if session.canSetSessionPreset(.hd1280x720) { session.sessionPreset = .hd1280x720 }
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                status = .error("Can't use \(device.localizedName)")
                return false
            }
            session.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(processor, queue: processor.queue)
            guard session.canAddOutput(output) else {
                status = .error("Can't read frames from \(device.localizedName)")
                return false
            }
            session.addOutput(output)
        } catch {
            status = .error(error.localizedDescription)
            return false
        }
        processor.session = session
        cameraName = device.localizedName
        configured = true
        return true
    }

    private func handle(_ result: (LipSample, MouthShape)?, _ t: TimeInterval) {
        guard isRunning else { return }
        rateWindow.append(t)
        rateWindow.removeAll { $0 < t - 1 }
        if let first = rateWindow.first, t > first {
            frameRate = Double(rateWindow.count - 1) / (t - first)
        }
        latency = ProcessInfo.processInfo.systemUptime - t

        guard let (sample, shape) = result else {
            status = .noFace
            mouth = nil
            return
        }
        status = .tracking
        latest = sample
        mouth = shape
        history.append(sample)
        let cutoff = sample.t - Self.historySeconds
        if let firstKept = history.firstIndex(where: { $0.t >= cutoff }), firstKept > 0 {
            history.removeFirst(firstKept)
        }
        onSample?(sample)
    }
}

private struct UncheckedSession: @unchecked Sendable {
    let session: AVCaptureSession
}

/// Runs Vision face landmarks on the capture queue.
private final class FrameProcessor: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let queue = DispatchQueue(label: "subvox.camera", qos: .userInteractive)
    weak var session: AVCaptureSession?
    var onResult: (((LipSample, MouthShape)?, TimeInterval) -> Void)?

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let onResult, let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t = hostSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

        let request = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixels, orientation: .up)
        try? handler.perform([request])

        let size = CGSize(width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels))
        // The largest face is the user sitting at the Mac.
        let face = request.results?.max {
            $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height
        }
        onResult(face.flatMap { Self.measure($0, t: t, imageSize: size) }, t)
    }

    private func hostSeconds(_ pts: CMTime) -> TimeInterval {
        guard let clock = session?.synchronizationClock else { return CMTimeGetSeconds(pts) }
        return CMTimeGetSeconds(CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock()))
    }

    private static func measure(_ face: VNFaceObservation, t: TimeInterval, imageSize: CGSize) -> (LipSample, MouthShape)? {
        guard let landmarks = face.landmarks,
              let outer = landmarks.outerLips?.pointsInImage(imageSize: imageSize),
              let leftEye = eyeCenter(landmarks.leftPupil, landmarks.leftEye, imageSize),
              let rightEye = eyeCenter(landmarks.rightPupil, landmarks.rightEye, imageSize) else { return nil }
        return LipGeometry.measure(
            t: t, leftEye: leftEye, rightEye: rightEye, outerLips: outer,
            innerLips: landmarks.innerLips?.pointsInImage(imageSize: imageSize) ?? [],
            contour: landmarks.faceContour?.pointsInImage(imageSize: imageSize) ?? [])
    }

    private static func eyeCenter(_ pupil: VNFaceLandmarkRegion2D?, _ eye: VNFaceLandmarkRegion2D?, _ size: CGSize) -> CGPoint? {
        let points = (pupil ?? eye)?.pointsInImage(imageSize: size) ?? []
        guard !points.isEmpty else { return nil }
        let n = CGFloat(points.count)
        return CGPoint(x: points.reduce(0) { $0 + $1.x } / n, y: points.reduce(0) { $0 + $1.y } / n)
    }
}

/// Live camera preview for framing the face.
struct CameraPreview: NSViewRepresentable {
    var session: AVCaptureSession

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer = layer
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
