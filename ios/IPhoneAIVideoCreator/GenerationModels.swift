import Foundation
import CoreGraphics

enum GenerationMode: String, CaseIterable, Identifiable {
    case textToImageVideo = "Text to Image / Video"
    case videoFrameToImageVideo = "Video Frame to Image / Video"
    case videoToVideo = "Video to Video"

    var id: String { rawValue }
}

enum OutputKind: String, CaseIterable, Identifiable {
    case image = "Image"
    case video = "Video"

    var id: String { rawValue }
}

enum DurationPreset: Int, CaseIterable, Identifiable {
    case sec5 = 5
    case sec6 = 6
    case sec8 = 8
    case sec10 = 10
    case sec15 = 15
    case sec20 = 20
    case sec30 = 30
    case sec60 = 60

    var id: Int { rawValue }
    var label: String { "\(rawValue)s" }
}

enum ResolutionPreset: String, CaseIterable, Identifiable {
    case p480 = "480p"
    case p720 = "720p"
    case p1080 = "1080p"
    case p4k = "4K"

    var id: String { rawValue }

    var size: CGSize {
        switch self {
        case .p480: return CGSize(width: 854, height: 480)
        case .p720: return CGSize(width: 1280, height: 720)
        case .p1080: return CGSize(width: 1920, height: 1080)
        case .p4k: return CGSize(width: 3840, height: 2160)
        }
    }
}

struct GenerationRequest {
    var prompt: String
    var mode: GenerationMode
    var output: OutputKind
    var duration: DurationPreset
    var resolution: ResolutionPreset
    var fps: Int
    var sourceVideoURL: URL?
    var sourceFrameSecond: Double
}

struct GenerationLogRecord: Codable {
    let timestamp: String
    let mode: String
    let output: String
    let prompt: String
    let durationSeconds: Int
    let resolution: String
    let fps: Int
    let sourceVideoPath: String
    let outputPath: String
    let gpuName: String
    let elapsedSeconds: Double
}
