import Foundation

final class DataLogger {
    private let logsURL: URL

    init(baseDirectory: URL) {
        self.logsURL = baseDirectory.appendingPathComponent("generation_jobs.jsonl")
    }

    func append(_ record: GenerationLogRecord) throws {
        let data = try JSONEncoder().encode(record)
        let line = data + Data("\n".utf8)
        if FileManager.default.fileExists(atPath: logsURL.path) {
            let handle = try FileHandle(forWritingTo: logsURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
        } else {
            try line.write(to: logsURL)
        }
    }
}
