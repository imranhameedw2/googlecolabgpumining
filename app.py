import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageDraw

try:
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
except Exception:  # Optional at runtime; app falls back to procedural generation
    torch = None
    StableDiffusionPipeline = None
    StableDiffusionImg2ImgPipeline = None


DURATIONS = [5, 6, 8, 10, 15, 20, 30, 60]
RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}
MODES = ["Text to Image/Video", "Video Frame to Image/Video", "Video to Video"]
OUTPUT_TYPES = ["Image", "Video"]


@dataclass
class JobRecord:
    timestamp: str
    mode: str
    output_type: str
    prompt: str
    duration_seconds: int
    resolution: str
    fps: int
    source_video: str
    output_path: str
    device: str
    model: str
    elapsed_seconds: float


class DataCollector:
    def __init__(self, out_dir: Path):
        self.log_path = out_dir / "generation_jobs.jsonl"

    def log(self, record: JobRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


class CPUVideoEngine:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.device = "cpu"
        self.model_id = model_id
        self._txt2img = None
        self._img2img = None

    @property
    def has_ai_pipeline(self) -> bool:
        return StableDiffusionPipeline is not None and torch is not None

    def _load_txt2img(self):
        if not self.has_ai_pipeline:
            return None
        if self._txt2img is None:
            self._txt2img = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
            ).to("cpu")
        return self._txt2img

    def _load_img2img(self):
        if not self.has_ai_pipeline:
            return None
        if self._img2img is None:
            self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                safety_checker=None,
            ).to("cpu")
        return self._img2img

    def _fallback_image(self, prompt: str, size: Tuple[int, int]) -> Image.Image:
        w, h = size
        img = Image.new("RGB", size)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(20, 220, w, dtype=np.uint8)
        arr[:, :, 1] = np.linspace(30, 180, h, dtype=np.uint8)[:, None]
        arr[:, :, 2] = 90
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, h - 110, w - 10, h - 10), fill=(0, 0, 0))
        draw.text((20, h - 95), prompt[:140], fill=(255, 255, 255))
        return img

    def text_to_image(self, prompt: str, size: Tuple[int, int]) -> Image.Image:
        pipe = self._load_txt2img()
        if pipe is None:
            return self._fallback_image(prompt, size)
        return pipe(prompt=prompt, width=size[0], height=size[1], num_inference_steps=25, guidance_scale=7.0).images[0]

    def image_to_image(self, source: Image.Image, prompt: str, size: Tuple[int, int]) -> Image.Image:
        pipe = self._load_img2img()
        source = source.convert("RGB").resize(size, Image.LANCZOS)
        if pipe is None:
            return self._fallback_image(f"{prompt} | transformed", size)
        return pipe(prompt=prompt, image=source, strength=0.45, guidance_scale=7.5, num_inference_steps=20).images[0]

    def frame_from_video(self, video_path: str, second: float) -> Image.Image:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open source video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, fps * second)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not read frame from source video")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def text_to_video(self, prompt: str, size: Tuple[int, int], duration: int, fps: int) -> list[Image.Image]:
        return [self.text_to_image(f"{prompt}, shot {i+1}", size) for i in range(duration * fps)]

    def video_to_video(self, video_path: str, prompt: str, size: Tuple[int, int], duration: int, fps: int) -> list[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open source video")
        total_target = duration * fps
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or total_target
        out = []
        for i in range(total_target):
            src_idx = int(i * src_total / total_target)
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok:
                break
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.append(self.image_to_image(pil, prompt, size))
        cap.release()
        if not out:
            raise RuntimeError("No frames generated")
        return out

    @staticmethod
    def make_video(frames: list[Image.Image], output_path: Path, fps: int) -> None:
        first = cv2.cvtColor(np.array(frames[0].convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for f in frames:
                bgr = cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()


app = Flask(__name__)
app.secret_key = "ubuntu-cpu-video-app"

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs"
UPLOAD_DIR = BASE / "uploads"
OUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

engine = CPUVideoEngine(model_id=os.getenv("AI_MODEL_ID", "runwayml/stable-diffusion-v1-5"))
collector = DataCollector(OUT_DIR)


@app.get("/")
def index():
    return render_template(
        "index.html",
        modes=MODES,
        output_types=OUTPUT_TYPES,
        durations=DURATIONS,
        resolutions=list(RESOLUTIONS.keys()),
        ai_enabled=engine.has_ai_pipeline,
    )


@app.post("/generate")
def generate():
    prompt = request.form.get("prompt", "").strip()
    mode = request.form.get("mode", MODES[0])
    output_type = request.form.get("output_type", OUTPUT_TYPES[0])
    duration = int(request.form.get("duration", DURATIONS[0]))
    resolution = request.form.get("resolution", "720p")
    fps = int(request.form.get("fps", 8))
    frame_second = float(request.form.get("frame_second", 1.0))

    if not prompt:
        flash("Prompt is required", "error")
        return redirect(url_for("index"))

    source_path = ""
    file = request.files.get("source_video")
    if file and file.filename:
        name = f"src_{int(time.time())}_{file.filename}"
        save_to = UPLOAD_DIR / name
        file.save(save_to)
        source_path = str(save_to)

    if mode != MODES[0] and not source_path:
        flash("Source video is required for selected mode", "error")
        return redirect(url_for("index"))

    size = RESOLUTIONS[resolution]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "png" if output_type == "Image" else "mp4"
    out_name = f"{mode.replace(' ', '_').lower()}_{ts}.{ext}"
    out_file = OUT_DIR / out_name

    start = time.time()
    try:
        if mode == MODES[0]:
            if output_type == "Image":
                img = engine.text_to_image(prompt, size)
                img.save(out_file)
            else:
                frames = engine.text_to_video(prompt, size, duration, fps)
                engine.make_video(frames, out_file, fps)

        elif mode == MODES[1]:
            frame = engine.frame_from_video(source_path, frame_second)
            styled = engine.image_to_image(frame, prompt, size)
            if output_type == "Image":
                styled.save(out_file)
            else:
                engine.make_video([styled] * (duration * fps), out_file, fps)

        else:
            frames = engine.video_to_video(source_path, prompt, size, duration, fps)
            if output_type == "Image":
                frames[0].save(out_file)
            else:
                engine.make_video(frames, out_file, fps)

        elapsed = round(time.time() - start, 2)
        collector.log(
            JobRecord(
                timestamp=datetime.now().isoformat(),
                mode=mode,
                output_type=output_type,
                prompt=prompt,
                duration_seconds=duration,
                resolution=resolution,
                fps=fps,
                source_video=source_path,
                output_path=str(out_file),
                device="cpu",
                model=engine.model_id if engine.has_ai_pipeline else "fallback-renderer",
                elapsed_seconds=elapsed,
            )
        )
        flash(f"Done in {elapsed}s. Output: {out_name}", "success")
        return redirect(url_for("download_output", filename=out_name))
    except Exception as exc:
        flash(f"Generation failed: {exc}", "error")
        return redirect(url_for("index"))


@app.get("/outputs/<path:filename>")
def download_output(filename: str):
    return send_from_directory(OUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
