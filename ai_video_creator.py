import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import tkinter as tk
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from tkinter import filedialog, messagebox, ttk


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
    gpu_name: str
    elapsed_seconds: float


class DataCollector:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "generation_jobs.jsonl"

    def log(self, record: JobRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


class AIVideoEngine:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._txt2img = None
        self._img2img = None

    @property
    def gpu_name(self) -> str:
        if self.device == "cuda":
            return torch.cuda.get_device_name(0)
        return "No CUDA GPU detected"

    def _load_txt2img(self):
        if self._txt2img is None:
            self._txt2img = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
            ).to(self.device)
        return self._txt2img

    def _load_img2img(self):
        if self._img2img is None:
            self._img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
            ).to(self.device)
        return self._img2img

    def text_to_image(self, prompt: str, size: Tuple[int, int], steps: int = 30, guidance: float = 7.5) -> Image.Image:
        pipe = self._load_txt2img()
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=size[0],
            height=size[1],
        ).images[0]
        return image

    def image_to_image(self, image: Image.Image, prompt: str, size: Tuple[int, int], strength: float = 0.45) -> Image.Image:
        pipe = self._load_img2img()
        resized = image.convert("RGB").resize(size, Image.LANCZOS)
        output = pipe(
            prompt=prompt,
            image=resized,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=30,
        ).images[0]
        return output

    @staticmethod
    def _make_video(frames: list[Image.Image], output_path: Path, fps: int) -> None:
        if not frames:
            raise ValueError("No frames were produced.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2BGR)
        h, w, _ = first.shape
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        try:
            for frame in frames:
                bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

    def text_to_video(self, prompt: str, size: Tuple[int, int], duration: int, fps: int) -> list[Image.Image]:
        frame_count = duration * fps
        frames = []
        for i in range(frame_count):
            variation_prompt = f"{prompt}, cinematic shot {i + 1}, coherent style"
            frame = self.text_to_image(variation_prompt, size=size, steps=20, guidance=6.8)
            frames.append(frame)
        return frames

    def frame_from_video(self, video_path: str, second: float) -> Image.Image:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to read video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        frame_no = int(max(0, second * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not extract frame at selected timestamp.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def video_to_video(self, video_path: str, prompt: str, size: Tuple[int, int], duration: int, fps: int) -> list[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to read video: {video_path}")
        raw_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        target_frames = duration * fps
        frames = []
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or target_frames

        for i in range(target_frames):
            src_idx = int(i * src_total / target_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            stylized = self.image_to_image(
                pil_img,
                prompt=f"{prompt}, frame transfer, preserve scene continuity",
                size=size,
                strength=0.35,
            )
            frames.append(stylized)

        cap.release()
        if not frames:
            raise RuntimeError("No output frames created from source video.")
        return frames


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Windows AI Video Creator (GPU)")
        self.geometry("900x700")

        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

        self.engine = AIVideoEngine()
        self.collector = DataCollector(self.output_dir)

        self.mode_var = tk.StringVar(value=MODES[0])
        self.output_var = tk.StringVar(value=OUTPUT_TYPES[0])
        self.duration_var = tk.IntVar(value=DURATIONS[0])
        self.resolution_var = tk.StringVar(value="720p")
        self.fps_var = tk.IntVar(value=8)
        self.video_path_var = tk.StringVar()
        self.frame_second_var = tk.DoubleVar(value=1.0)
        self.status_var = tk.StringVar(value=f"Device: {self.engine.device} | GPU: {self.engine.gpu_name}")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        ttk.Label(self, text="Prompt").pack(anchor="w", **pad)
        self.prompt_entry = tk.Text(self, height=5, width=100)
        self.prompt_entry.pack(fill="x", **pad)

        row1 = ttk.Frame(self)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(row1, textvariable=self.mode_var, values=MODES, state="readonly", width=28).grid(row=0, column=1, padx=6)
        ttk.Label(row1, text="Output").grid(row=0, column=2, sticky="w")
        ttk.Combobox(row1, textvariable=self.output_var, values=OUTPUT_TYPES, state="readonly", width=12).grid(row=0, column=3, padx=6)
        ttk.Label(row1, text="Duration (s)").grid(row=0, column=4, sticky="w")
        ttk.Combobox(row1, textvariable=self.duration_var, values=DURATIONS, state="readonly", width=8).grid(row=0, column=5, padx=6)

        row2 = ttk.Frame(self)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Resolution").grid(row=0, column=0, sticky="w")
        ttk.Combobox(row2, textvariable=self.resolution_var, values=list(RESOLUTIONS.keys()), state="readonly", width=10).grid(row=0, column=1, padx=6)
        ttk.Label(row2, text="FPS").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(row2, from_=4, to=30, textvariable=self.fps_var, width=7).grid(row=0, column=3, padx=6)
        ttk.Label(row2, text="Frame second (for frame mode)").grid(row=0, column=4, sticky="w")
        ttk.Entry(row2, textvariable=self.frame_second_var, width=8).grid(row=0, column=5, padx=6)

        row3 = ttk.Frame(self)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Source Video").grid(row=0, column=0, sticky="w")
        ttk.Entry(row3, textvariable=self.video_path_var, width=80).grid(row=0, column=1, padx=6)
        ttk.Button(row3, text="Browse", command=self._pick_video).grid(row=0, column=2)

        controls = ttk.Frame(self)
        controls.pack(fill="x", **pad)
        ttk.Button(controls, text="Generate", command=self._start_generate).pack(side="left")
        ttk.Button(controls, text="Open Output Folder", command=self._open_output_folder).pack(side="left", padx=6)

        ttk.Separator(self, orient="horizontal").pack(fill="x", **pad)
        self.log_text = tk.Text(self, height=18, width=110)
        self.log_text.pack(fill="both", expand=True, **pad)
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", **pad)

    def _pick_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
        if path:
            self.video_path_var.set(path)

    def _open_output_folder(self):
        folder = str(self.output_dir.resolve())
        os.startfile(folder) if os.name == "nt" else os.system(f'xdg-open "{folder}"')

    def _log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def _start_generate(self):
        th = threading.Thread(target=self._generate, daemon=True)
        th.start()

    def _generate(self):
        prompt = self.prompt_entry.get("1.0", "end").strip()
        if not prompt:
            messagebox.showerror("Validation", "Prompt is required.")
            return

        mode = self.mode_var.get()
        out_type = self.output_var.get()
        duration = self.duration_var.get()
        resolution_name = self.resolution_var.get()
        size = RESOLUTIONS[resolution_name]
        fps = self.fps_var.get()
        source_video = self.video_path_var.get().strip()

        if mode != MODES[0] and not source_video:
            messagebox.showerror("Validation", "Source video is required for selected mode.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "png" if out_type == "Image" else "mp4"
        output_path = self.output_dir / f"{mode.replace(' ', '_').lower()}_{ts}.{ext}"

        started = time.time()
        self._log(f"Starting {mode} -> {out_type} at {resolution_name}...")

        try:
            if mode == MODES[0]:
                if out_type == "Image":
                    result = self.engine.text_to_image(prompt, size=size)
                    result.save(output_path)
                else:
                    frames = self.engine.text_to_video(prompt, size=size, duration=duration, fps=fps)
                    self.engine._make_video(frames, output_path=output_path, fps=fps)

            elif mode == MODES[1]:
                frame = self.engine.frame_from_video(source_video, self.frame_second_var.get())
                styled = self.engine.image_to_image(frame, prompt=prompt, size=size)
                if out_type == "Image":
                    styled.save(output_path)
                else:
                    frames = [styled] * (duration * fps)
                    self.engine._make_video(frames, output_path=output_path, fps=fps)

            else:  # Video to Video
                frames = self.engine.video_to_video(
                    video_path=source_video,
                    prompt=prompt,
                    size=size,
                    duration=duration,
                    fps=fps,
                )
                if out_type == "Image":
                    frames[0].save(output_path)
                else:
                    self.engine._make_video(frames, output_path=output_path, fps=fps)

            elapsed = round(time.time() - started, 2)
            self._log(f"Done in {elapsed}s -> {output_path}")
            self.collector.log(
                JobRecord(
                    timestamp=datetime.now().isoformat(),
                    mode=mode,
                    output_type=out_type,
                    prompt=prompt,
                    duration_seconds=duration,
                    resolution=resolution_name,
                    fps=fps,
                    source_video=source_video,
                    output_path=str(output_path),
                    device=self.engine.device,
                    gpu_name=self.engine.gpu_name,
                    elapsed_seconds=elapsed,
                )
            )
        except Exception as ex:
            self._log(f"Error: {ex}")
            messagebox.showerror("Generation failed", str(ex))


if __name__ == "__main__":
    app = App()
    app.mainloop()
