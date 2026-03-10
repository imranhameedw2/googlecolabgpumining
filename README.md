# Ubuntu CPU AI Video Creator (Web Interface)

This application now runs on **Ubuntu** as a **web interface** and generates image/video outputs using **CPU**.

## Features
- Modes:
  - **Text to Image / Video**
  - **Video Frame to Image / Video**
  - **Video to Video**
- Duration presets: **5, 6, 8, 10, 15, 20, 30, 60 seconds**
- Resolution presets: **480p, 720p, 1080p, 4K**
- CPU generation flow with optional Diffusers CPU pipeline
- Data collection to `outputs/generation_jobs.jsonl`

## Ubuntu setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Notes
- If Diffusers/Torch are unavailable, the app still works using a CPU fallback frame renderer.
- Generated media files are written to `outputs/`.
- Uploaded source videos are stored in `uploads/`.
