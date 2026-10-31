import Foundation

struct Dataset: Codable {
    var vocabulary: [String]
    var templates: [Template]
    var phrases: [Phrase] = []

    static let starter = Dataset(vocabulary: ["yes", "no", "next", "back", "stop"], templates: [])
}

extension Dataset {
    // Older dataset.json files have no phrases.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabulary = try container.decode([String].self, forKey: .vocabulary)
        templates = try container.decode([Template].self, forKey: .templates)
        phrases = try container.decodeIfPresent([Phrase].self, forKey: .phrases) ?? []
    }
}

/// Persists the training set as JSON in Application Support.
enum TemplateStore {
    static var url: URL {
        let dir = URL.applicationSupportDirectory.appending(path: "Subvox", directoryHint: .isDirectory)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appending(path: "dataset.json")
    }

    static func load() -> Dataset {
        guard let data = try? Data(contentsOf: url),
              let dataset = try? JSONDecoder().decode(Dataset.self, from: data) else { return .starter }
        return dataset
    }

    static func save(_ dataset: Dataset) {
        do {
            let data = try JSONEncoder().encode(dataset)
            try data.write(to: url, options: .atomic)
        } catch {
            print("Subvox: failed to save dataset: \(error)")
        }
    }
}
