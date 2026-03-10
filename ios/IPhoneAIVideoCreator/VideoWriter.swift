import Foundation
import AVFoundation
import UIKit

final class VideoWriter {
    func writeMP4(frames: [UIImage], fps: Int, outputURL: URL, size: CGSize) throws {
        guard !frames.isEmpty else {
            throw NSError(domain: "VideoWriter", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames to write."])
        }

        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: size.width,
            AVVideoHeightKey: size.height
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
            kCVPixelBufferWidthKey as String: size.width,
            kCVPixelBufferHeightKey as String: size.height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)

        guard writer.canAdd(input) else {
            throw NSError(domain: "VideoWriter", code: -2, userInfo: [NSLocalizedDescriptionKey: "Cannot add writer input."])
        }
        writer.add(input)

        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
        var frameCount: Int64 = 0

        for frame in frames {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            guard let pixelBuffer = pixelBuffer(from: frame, size: size) else {
                throw NSError(domain: "VideoWriter", code: -3, userInfo: [NSLocalizedDescriptionKey: "Could not create pixel buffer."])
            }
            let presentation = CMTimeMultiply(frameDuration, multiplier: Int32(frameCount))
            adaptor.append(pixelBuffer, withPresentationTime: presentation)
            frameCount += 1
        }

        input.markAsFinished()
        writer.finishWriting {}
    }

    private func pixelBuffer(from image: UIImage, size: CGSize) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height), kCVPixelFormatType_32ARGB, attrs, &buffer)
        guard let pxBuffer = buffer else { return nil }

        CVPixelBufferLockBaseAddress(pxBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pxBuffer, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pxBuffer),
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pxBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }

        ctx.clear(CGRect(origin: .zero, size: size))
        ctx.draw(image.cgImage!, in: CGRect(origin: .zero, size: size))
        return pxBuffer
    }
}
