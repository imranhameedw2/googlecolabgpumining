# iPhone AI Video Creator (iOS + iPhone GPU)

This repository now provides an **iOS SwiftUI app scaffold** designed to run generation workflows using the **iPhone GPU (Metal)**.

## Supported workflows
- **Text to image / video**
- **Video frame to image / video**
- **Video to video**

## Presets implemented
- Duration presets: **5s, 6s, 8s, 10s, 15s, 20s, 30s, 60s**
- Resolution presets: **480p, 720p, 1080p, 4K**
- Adjustable FPS: **4...30**

## Data collection
Generation metadata is logged as JSONL to app Documents:
- `generation_jobs.jsonl`

## iOS project files
Swift source files are in:
- `ios/IPhoneAIVideoCreator/`

Main files:
- `IPhoneAIVideoCreatorApp.swift`
- `ContentView.swift`
- `GenerationModels.swift`
- `GPUGenerationEngine.swift`
- `VideoWriter.swift`
- `DataLogger.swift`

## Build in Xcode
1. Create a new **iOS App** project in Xcode (SwiftUI lifecycle).
2. Replace generated Swift files with the files from `ios/IPhoneAIVideoCreator/`.
3. Add required frameworks (usually auto-linked):
   - SwiftUI
   - Metal
   - AVFoundation
   - UIKit
4. Run on iPhone.

## Note on AI model integration
`GPUGenerationEngine` is structured for iPhone GPU workflows and currently includes placeholder stylization/rendering logic.
To use full on-device AI generation, plug in a Core ML text/image/video model pipeline (for example a converted Stable Diffusion or custom model).
