import AppKit
import Foundation
import IOKit.ps
import Observation
import UserNotifications

/// Trains the Core ML model on its own. Takes collected from your speech go into a pool.
/// About once a day, when the Mac is idle and plugged in, a challenger model is trained on
/// your takes plus the pool. It replaces the current model (the champion) only if it scores
/// better on your silent test takes, which neither model ever trains on.
@MainActor @Observable
final class AutoTrainer {
    struct Run: Codable, Identifiable, Hashable, Sendable {
        var id = UUID()
        var date: Date
        var scheduled: Bool
        var poolTakes: Int
        var testTakes: Int
        var championAccuracy: Double?
        var challengerAccuracy: Double?
        var adopted: Bool
        var note: String
    }

    struct Collected: Identifiable, Hashable, Sendable {
        let id = UUID()
        var date: Date
        var word: String
        /// nil when kept; otherwise why it was dropped.
        var rejection: String?
        var hasLips: Bool
    }

    static let maxPoolPerWord = 150
    static let maxRuns = 30
    /// A scheduled run waits for this many new pool takes since the last run.
    static let minNewTakes = 10
    static let minTestTakes = 6
    static let minIdleSeconds: TimeInterval = 600
    static let minHoursBetweenRuns: TimeInterval = 20

    var isCollecting: Bool = UserDefaults.standard.bool(forKey: "autoTrain.collecting") {
        didSet { UserDefaults.standard.set(isCollecting, forKey: "autoTrain.collecting") }
    }
    var isScheduled: Bool = UserDefaults.standard.bool(forKey: "autoTrain.scheduled") {
        didSet {
            UserDefaults.standard.set(isScheduled, forKey: "autoTrain.scheduled")
            updateSchedule()
        }
    }

    private(set) var pool: [Template] = []
    private(set) var runs: [Run] = []
    private(set) var recent: [Collected] = []
    private(set) var isRunning = false
    private(set) var status: String?

    /// Set by `AppModel`: runs training with the current takes and model.
    @ObservationIgnored var onScheduledRun: (() async -> Void)?
    @ObservationIgnored private(set) var collector = SpeechCollector()
    @ObservationIgnored private var scheduler: NSBackgroundActivityScheduler?

    init() {
        pool = Self.read([Template].self, from: Self.poolURL) ?? []
        runs = Self.read([Run].self, from: Self.runsURL) ?? []
    }

    // MARK: Collecting

    func ingest(_ sample: MotionSample) {
        guard isCollecting else { return }
        collector.ingest(sample)
    }

    func ingest(_ sample: LipSample) {
        guard isCollecting else { return }
        collector.ingest(sample)
    }

    func collect(_ words: [WordTiming], vocabulary: [String]) {
        guard isCollecting else { return }
        let outcomes = collector.cut(words, vocabulary: Set(vocabulary))
        guard !outcomes.isEmpty else { return }
        for outcome in outcomes {
            switch outcome {
            case .kept(let take):
                pool.append(take)
                recent.insert(Collected(date: Date(), word: take.label, rejection: nil, hasLips: take.hasLips), at: 0)
            case .rejected(let word, let reason):
                recent.insert(Collected(date: Date(), word: word, rejection: reason, hasLips: false), at: 0)
            }
        }
        recent = Array(recent.prefix(30))
        // Keep the newest takes of each word so the pool, and training time, stay bounded.
        var counts: [String: Int] = [:]
        pool = pool.reversed().filter { take in
            counts[take.label, default: 0] += 1
            return counts[take.label]! <= Self.maxPoolPerWord
        }.reversed()
        Self.write(pool, to: Self.poolURL)
    }

    func poolCount(for word: String) -> Int {
        pool.filter { $0.label == word }.count
    }

    func clearPool() {
        pool.removeAll()
        recent.removeAll()
        Self.write(pool, to: Self.poolURL)
    }

    // MARK: Champion vs. challenger

    /// Trains a challenger and keeps it only if it beats the champion on the silent test set.
    /// Returns the challenger when it was adopted.
    func run(manual: [Template], featureSet: FeatureSet, champion: SequenceModel?, scheduled: Bool) async -> SequenceModel? {
        guard !isRunning else { return nil }
        isRunning = true
        status = "Training challenger on \(manual.filter(\.isManual).count) of your takes + \(pool.count) from speech…"
        defer {
            isRunning = false
            status = nil
        }
        let pool = self.pool
        let outcome = await Task.detached(priority: .utility) {
            Self.compete(manual: manual, pool: pool, featureSet: featureSet, champion: champion)
        }.value

        var run = outcome.run
        run.scheduled = scheduled
        if let challenger = outcome.adopted {
            do {
                try challenger.save()
            } catch {
                run.adopted = false
                run.note = "Couldn't save the new model: \(error.localizedDescription)"
            }
        }
        record(run)
        notify(run)
        return run.adopted ? outcome.adopted : nil
    }

    /// The comparison itself; pure and off the main thread.
    nonisolated static func compete(manual: [Template], pool: [Template], featureSet: FeatureSet,
                                    champion: SequenceModel?) -> (run: Run, adopted: SequenceModel?) {
        var run = Run(date: Date(), scheduled: false, poolTakes: pool.count, testTakes: 0,
                      championAccuracy: nil, challengerAccuracy: nil, adopted: false, note: "")
        let challenger: SequenceModel
        do {
            challenger = try SequenceModel.train(templates: manual, extra: pool, featureSet: featureSet, persist: false)
        } catch {
            run.note = error.localizedDescription
            return (run, nil)
        }

        // Score both on the same test takes. If either model uses lips, only takes with
        // camera data are fair to both.
        let needsLips = challenger.info.featureSet.contains(.lips) || (champion?.info.featureSet.contains(.lips) ?? false)
        let tests = manual.filter { SequenceModel.isTestTake($0) && (!needsLips || $0.hasLips) }
        run.testTakes = tests.count
        guard tests.count >= minTestTakes, Set(tests.map(\.label)).count >= 2 else {
            run.note = "Needs \(minTestTakes)+ silent test takes across 2+ words. About 1 in 5 of your Train takes becomes a test take; record more."
            return (run, nil)
        }

        let new = challenger.score(on: tests)
        run.challengerAccuracy = new.accuracy
        if let champion {
            let old = champion.score(on: tests)
            run.championAccuracy = old.accuracy
            run.adopted = new.accuracy > old.accuracy
                || (new.accuracy == old.accuracy && new.meanProbability > old.meanProbability)
            run.note = run.adopted ? "Challenger scored better; it is now the model." : "Champion kept; the challenger was not better."
        } else {
            // No model yet: adopt if it beats always guessing the most common word.
            let counts = Dictionary(grouping: tests, by: \.label).mapValues(\.count)
            let chance = Double(counts.values.max() ?? 0) / Double(tests.count)
            run.adopted = new.accuracy > chance
            run.note = run.adopted ? "First model; it beats chance on the test takes." : "Not better than chance; nothing saved."
        }
        return (run, run.adopted ? challenger : nil)
    }

    private func record(_ run: Run) {
        runs.insert(run, at: 0)
        runs = Array(runs.prefix(Self.maxRuns))
        Self.write(runs, to: Self.runsURL)
    }

    var newTakesSinceLastRun: Int {
        guard let last = runs.first?.date else { return pool.count }
        return pool.filter { $0.createdAt > last }.count
    }

    // MARK: Schedule

    func startScheduleIfEnabled() {
        updateSchedule()
    }

    private func updateSchedule() {
        scheduler?.invalidate()
        scheduler = nil
        guard isScheduled else { return }
        requestNotificationPermission()
        // Check hourly; `scheduledRunIsDue` limits real runs to about once a day.
        let scheduler = NSBackgroundActivityScheduler(identifier: "com.joeblau.subvox.autotrain")
        scheduler.repeats = true
        scheduler.interval = 3600
        scheduler.tolerance = 600
        scheduler.qualityOfService = .utility
        scheduler.schedule(Self.activity(owner: self))
        self.scheduler = scheduler
    }

    private func scheduledCheck() async -> NSBackgroundActivityScheduler.Result {
        guard scheduledRunIsDue else { return .finished }
        await onScheduledRun?()
        return .finished
    }

    /// Plugged in, idle for 10 min, 20 h since the last run, and enough new takes.
    var scheduledRunIsDue: Bool {
        guard !isRunning, Self.onACPower, Self.idleSeconds >= Self.minIdleSeconds,
              newTakesSinceLastRun >= Self.minNewTakes else { return false }
        if let last = runs.first?.date, Date().timeIntervalSince(last) < Self.minHoursBetweenRuns * 3600 { return false }
        return true
    }

    nonisolated private static func activity(owner: AutoTrainer)
        -> @Sendable (@escaping NSBackgroundActivityScheduler.CompletionHandler) -> Void
    {
        { [weak owner] completion in
            Task { @MainActor in
                let result = await owner?.scheduledCheck() ?? .finished
                completion(result)
            }
        }
    }

    nonisolated static var onACPower: Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() else { return true }
        return (type as String) == kIOPSACPowerValue
    }

    nonisolated static var idleSeconds: TimeInterval {
        // ~0 is "any input event".
        CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: CGEventType(rawValue: ~0)!)
    }

    // MARK: Report

    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert]) { _, _ in }
    }

    private func notify(_ run: Run) {
        let content = UNMutableNotificationContent()
        content.title = run.adopted ? "Subvox model improved" : "Subvox auto-training"
        func percent(_ value: Double?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
        content.body = run.challengerAccuracy == nil
            ? run.note
            : "Test accuracy \(percent(run.championAccuracy)) → \(percent(run.challengerAccuracy)). \(run.note)"
        let request = UNNotificationRequest(identifier: run.id.uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request) { _ in }
    }

    // MARK: Storage

    private static var directory: URL {
        SequenceModel.directory.appending(path: "AutoTrain", directoryHint: .isDirectory)
    }
    private static var poolURL: URL { directory.appending(path: "pool.json") }
    private static var runsURL: URL { directory.appending(path: "runs.json") }

    private static func read<T: Decodable>(_ type: T.Type, from url: URL) -> T? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private static func write<T: Encodable>(_ value: T, to url: URL) {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try JSONEncoder().encode(value).write(to: url, options: .atomic)
        } catch {
            print("Subvox: failed to save \(url.lastPathComponent): \(error)")
        }
    }
}
