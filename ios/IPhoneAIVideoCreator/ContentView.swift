import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var prompt = "cinematic mountain sunrise"
    @State private var mode: GenerationMode = .textToImageVideo
    @State private var output: OutputKind = .video
    @State private var duration: DurationPreset = .sec5
    @State private var resolution: ResolutionPreset = .p720
    @State private var fps = 8
    @State private var frameSecond = 1.0
    @State private var sourceURL: URL?
    @State private var status = "Ready"
    @State private var outputURL: URL?

    var body: some View {
        NavigationView {
            Form {
                Section("Prompt") {
                    TextField("Describe your scene", text: $prompt, axis: .vertical)
                        .lineLimit(3...6)
                }

                Section("Mode") {
                    Picker("Workflow", selection: $mode) {
                        ForEach(GenerationMode.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Picker("Output", selection: $output) {
                        ForEach(OutputKind.allCases) { Text($0.rawValue).tag($0) }
                    }
                }

                Section("Video Options") {
                    Picker("Duration", selection: $duration) {
                        ForEach(DurationPreset.allCases) { Text($0.label).tag($0) }
                    }
                    Picker("Resolution", selection: $resolution) {
                        ForEach(ResolutionPreset.allCases) { Text($0.rawValue).tag($0) }
                    }
                    Stepper("FPS: \(fps)", value: $fps, in: 4...30)
                    Stepper("Frame second: \(String(format: "%.1f", frameSecond))", value: $frameSecond, in: 0...120, step: 0.5)
                }

                Section("Source Video") {
                    if let sourceURL {
                        Text(sourceURL.lastPathComponent)
                    } else {
                        Text("No source selected")
                    }
                    Button("Pick Video") { status = "Use PhotosPicker/UIDocumentPicker integration in app target." }
                }

                Section {
                    Button("Generate on iPhone GPU") {
                        Task { await generate() }
                    }
                    .buttonStyle(.borderedProminent)
                }

                Section("Status") {
                    Text(status)
                    if let outputURL {
                        Text("Saved: \(outputURL.lastPathComponent)")
                    }
                }
            }
            .navigationTitle("iPhone AI Video Creator")
        }
    }

    @MainActor
    private func generate() async {
        status = "Starting..."
        do {
            let engine = try GPUGenerationEngine()
            let writer = VideoWriter()
            let outputDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let logger = DataLogger(baseDirectory: outputDir)

            let req = GenerationRequest(
                prompt: prompt,
                mode: mode,
                output: output,
                duration: duration,
                resolution: resolution,
                fps: fps,
                sourceVideoURL: sourceURL,
                sourceFrameSecond: frameSecond
            )

            let started = Date()
            let frames = try engine.generateFrames(request: req)

            if output == .image {
                let file = outputDir.appendingPathComponent("image_\(Int(Date().timeIntervalSince1970)).jpg")
                try frames[0].jpegData(compressionQuality: 0.92)?.write(to: file)
                outputURL = file
            } else {
                let file = outputDir.appendingPathComponent("video_\(Int(Date().timeIntervalSince1970)).mp4")
                try writer.writeMP4(frames: frames, fps: fps, outputURL: file, size: resolution.size)
                outputURL = file
            }

            let elapsed = Date().timeIntervalSince(started)
            try logger.append(GenerationLogRecord(
                timestamp: ISO8601DateFormatter().string(from: Date()),
                mode: mode.rawValue,
                output: output.rawValue,
                prompt: prompt,
                durationSeconds: duration.rawValue,
                resolution: resolution.rawValue,
                fps: fps,
                sourceVideoPath: sourceURL?.path ?? "",
                outputPath: outputURL?.path ?? "",
                gpuName: engine.gpuName,
                elapsedSeconds: elapsed
            ))

            status = "Done on GPU: \(engine.gpuName)"
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}
