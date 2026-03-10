# Windows AI Video Creator (GPU)

Desktop application to generate AI images and videos using the system graphic card.

## Features
- Modes:
  - **Text to Image / Video**
  - **Video Frame to Image / Video**
  - **Video to Video**
- Duration presets: **5, 6, 8, 10, 15, 20, 30, 60 seconds**.
- Output resolutions: **480p, 720p, 1080p, 4K**.
- Uses GPU automatically when CUDA is available.
- Data collection: generation metadata is logged in `outputs/generation_jobs.jsonl`.

## Setup (Windows)
1. Install Python 3.10+.
2. Install CUDA-enabled PyTorch for your GPU (from pytorch.org).
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run app:
   ```powershell
   python ai_video_creator.py
   ```

## Notes
- First run downloads the model (`runwayml/stable-diffusion-v1-5`) from Hugging Face.
- 4K generation is VRAM intensive; if GPU memory is limited, use 720p or 1080p.
- For video modes, keep FPS low (e.g. 6-12) for faster rendering.

## Output
Generated media is saved in `outputs/`.

