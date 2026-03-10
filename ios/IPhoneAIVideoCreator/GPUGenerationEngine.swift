import Foundation
import Metal
import UIKit
import AVFoundation

final class GPUGenerationEngine {
    private let device: MTLDevice

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "GPUGenerationEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "No Metal GPU device found."])
        }
        self.device = device
    }

    var gpuName: String { device.name }

    func generateImage(prompt: String, size: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: size, format: format)

        return renderer.image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))

            let colorA = UIColor.systemIndigo
            let colorB = UIColor.systemTeal
            let cg = ctx.cgContext
            let colors = [colorA.cgColor, colorB.cgColor] as CFArray
            let space = CGColorSpaceCreateDeviceRGB()
            if let gradient = CGGradient(colorsSpace: space, colors: colors, locations: [0, 1]) {
                cg.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: size.width, y: size.height), options: [])
            }

            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: max(18, size.width * 0.03)),
                .foregroundColor: UIColor.white
            ]
            let text = NSString(string: prompt)
            let rect = CGRect(x: 24, y: size.height * 0.75, width: size.width - 48, height: size.height * 0.2)
            text.draw(in: rect, withAttributes: attrs)
        }
    }

    func frameFromVideo(url: URL, second: Double) throws -> UIImage {
        let asset = AVAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        let time = CMTime(seconds: second, preferredTimescale: 600)
        let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
        return UIImage(cgImage: cgImage)
    }

    func stylize(image: UIImage, prompt: String, size: CGSize) -> UIImage {
        // Placeholder for Core ML Stable Diffusion / custom model pipeline.
        // Keeps requested iPhone-GPU flow architecture and can be swapped with real model inference.
        let base = UIGraphicsImageRenderer(size: size).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        return overlayPrompt(base, prompt: prompt)
    }

    func generateFrames(request: GenerationRequest) throws -> [UIImage] {
        let totalFrames = request.duration.rawValue * request.fps
        let size = request.resolution.size

        switch request.mode {
        case .textToImageVideo:
            return (0..<totalFrames).map { idx in
                generateImage(prompt: "\(request.prompt) | frame \(idx + 1)", size: size)
            }

        case .videoFrameToImageVideo:
            guard let url = request.sourceVideoURL else {
                throw NSError(domain: "GPUGenerationEngine", code: -2, userInfo: [NSLocalizedDescriptionKey: "Source video is required."])
            }
            let frame = try frameFromVideo(url: url, second: request.sourceFrameSecond)
            let stylized = stylize(image: frame, prompt: request.prompt, size: size)
            return Array(repeating: stylized, count: totalFrames)

        case .videoToVideo:
            guard let url = request.sourceVideoURL else {
                throw NSError(domain: "GPUGenerationEngine", code: -2, userInfo: [NSLocalizedDescriptionKey: "Source video is required."])
            }
            let asset = AVAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true

            let durationSec = CMTimeGetSeconds(asset.duration)
            guard durationSec.isFinite, durationSec > 0 else {
                throw NSError(domain: "GPUGenerationEngine", code: -3, userInfo: [NSLocalizedDescriptionKey: "Invalid source video duration."])
            }

            return try (0..<totalFrames).map { idx in
                let sec = (Double(idx) / Double(max(1, totalFrames - 1))) * durationSec
                let cg = try generator.copyCGImage(at: CMTime(seconds: sec, preferredTimescale: 600), actualTime: nil)
                let image = UIImage(cgImage: cg)
                return stylize(image: image, prompt: request.prompt, size: size)
            }
        }
    }

    private func overlayPrompt(_ image: UIImage, prompt: String) -> UIImage {
        let size = image.size
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: max(16, size.width * 0.02), weight: .semibold),
                .foregroundColor: UIColor.white
            ]
            NSString(string: prompt).draw(in: CGRect(x: 20, y: 20, width: size.width - 40, height: 80), withAttributes: attrs)
        }
    }
}
