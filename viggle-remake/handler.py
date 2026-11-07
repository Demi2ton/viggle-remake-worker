# SPDX-License-Identifier: Apache-2.0
"""RunPod Serverless handler for Viggle-Remake.

Adapts Viggle/Viggle-Remake inference/sample.py for Serverless.
Input JSON:
  {
    "model": "viggle-remake",
    "mode": "swap",                  # "swap" or "edit"
    "video": "https://.../source.mp4",   # source video URL (required)
    "audio": "https://.../soundtrack.mp3", # optional soundtrack URL
    "prompt": "make it snow",        # inline prompt text (required)
    "prompt_file": null,             # or path to prompt file (ignored if prompt given)
    "refs": ["https://.../ref1.png", "https://.../ref2.png"],  # reference images (required)
    "pins": [12, 96],                # optional keyframe pins, one per ref
    "frames": 124,                   # 17n+5, recommended up to 294
    "short_edge": 480,               # 480 or 736
    "seed": 0,
    "lora": "swap",                  # "swap" or "edit" (overrides mode)
  }

Output:
  {
    "status": "succeeded",
    "output": {
      "video_path": "/workspace/output/result_xxxx.mp4",
      "mode": "swap",
      "frames": 124,
      "resolution": "832x480"
    }
  }

Weights are expected in WEIGHTS_DIR (Network Volume):
  /workspace/weights/
    base_model/        <- MiniMaxAI/MiniMax-H3 (downloaded on cold start if missing)
    lora/swap/         <- Viggle/Viggle-Remake lora/swap
    lora/edit/         <- Viggle/Viggle-Remake lora/edit
"""

import os
import sys
import json
import subprocess
import tempfile
import traceback
import uuid
from pathlib import Path
import urllib.request

import torch
import runpod
from huggingface_hub import snapshot_download

# ============================================================================
# КОНФИГУРАЦИЯ (через --env при создании Serverless Endpoint)
# ============================================================================

BASE_MODEL_ID   = os.getenv("BASE_MODEL_ID", "MiniMaxAI/MiniMax-H3")
LORA_REPO_ID    = os.getenv("LORA_REPO_ID", "Viggle/Viggle-Remake")
HF_TOKEN        = os.getenv("HF_TOKEN", None)
WEIGHTS_DIR     = os.getenv("WEIGHTS_DIR", "/workspace/weights")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "/workspace/output")
MODEL_REVISION  = os.getenv("MODEL_REVISION", None)
USE_FP16        = os.getenv("USE_FP16", "true").lower() == "true"

SAMPLE_PY_PATH  = "/app/inference/sample.py"  # скопировано в образ

pipe_ready = False

# ============================================================================
# СКАЧИВАНИЕ ВЕСОВ
# ============================================================================

def download_file(url: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(url, dst)

def ensure_weights():
    """Скачивает базовую модель и LoRA в WEIGHTS_DIR, если их там нет."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_path = os.path.join(WEIGHTS_DIR, "base_model")
    if not os.path.exists(os.path.join(base_path, "modular_model_index.json")):
        print(f"[Viggle-Remake] Скачивание базовой модели {BASE_MODEL_ID} ({BASE_MODEL_ID}) -> {base_path}")
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            local_dir=base_path,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
            revision=MODEL_REVISION,
            allow_patterns=[
                "modular_model_index.json",
                "transformer/*",
                "text_encoder/*",
                "tokenizer/*",
                "processor/*",
                "vae/*",
                "scheduler/*",
                "audio_vae/*",
                "audio_scheduler/*",
                "docs/*",
            ],
        )
    else:
        print(f"[Viggle-Remake] Базовая модель уже в Volume: {base_path}")

    # LoRAs (swap и edit)
    for lora_name in ("swap", "edit"):
        lora_path = os.path.join(WEIGHTS_DIR, "lora", lora_name)
        if not os.path.exists(os.path.join(lora_path, "diffusion_pytorch_model.safetensors")):
            print(f"[Viggle-Remake] Скачивание LoRA {lora_name} -> {lora_path}")
            snapshot_download(
                repo_id=LORA_REPO_ID,
                local_dir=lora_path,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
                revision=MODEL_REVISION,
                allow_patterns=[f"lora/{lora_name}/*"],
            )
        else:
            print(f"[Viggle-Remake] LoRA {lora_name} уже в Volume: {lora_path}")

    return base_path

# ============================================================================
# ПОДГОТОВКА ВХОДНЫХ ФАЙЛОВ
# ============================================================================

def prepare_inputs(job_input):
    """Скачивает видео, аудио и рефы во временную директорию. Возвращает пути."""
    tmp = tempfile.mkdtemp(prefix="viggle_")

    video_url = job_input.get("video", "")
    if not video_url:
        raise ValueError("Поле 'video' обязательно (URL исходного видео)")
    video_path = os.path.join(tmp, "source.mp4")
    download_file(video_url, video_path)

    audio_url = job_input.get("audio", None)
    audio_path = None
    if audio_url:
        audio_path = os.path.join(tmp, "audio.mp4")
        download_file(audio_url, audio_path)

    refs = job_input.get("refs", [])
    if not refs:
        raise ValueError("Поле 'refs' обязательно (список URL reference-изображений)")
    ref_paths = []
    for i, ref_url in enumerate(refs):
        ext = Path(ref_url).suffix or ".png"
        ref_path = os.path.join(tmp, f"ref_{i}{ext}")
        download_file(ref_url, ref_path)
        ref_paths.append(ref_path)

    # Промпт
    prompt = job_input.get("prompt", "")
    if not prompt:
        raise ValueError("Поле 'prompt' обязательно (текст запроса)")
    prompt_path = os.path.join(tmp, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    return {
        "tmp": tmp,
        "video": video_path,
        "audio": audio_path,
        "prompt": prompt_path,
        "refs": ref_paths,
    }

# ============================================================================
# ЗАПУСК ИНФЕРЕНСА
# ============================================================================

def run_inference(job_input, base_path):
    global pipe_ready

    mode = job_input.get("mode", "swap")
    lora = job_input.get("lora", mode)
    if lora not in ("swap", "edit"):
        lora = "swap"

    frames = int(job_input.get("frames", 124))
    short_edge = int(job_input.get("short_edge", 480))
    seed = int(job_input.get("seed", 0))
    pins = job_input.get("pins", [])

    inputs = prepare_inputs(job_input)

    out_name = f"result_{uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    # Сборка аргументов sample.py
    cmd = [
        sys.executable, SAMPLE_PY_PATH,
        "--mode", mode,
        "--video", inputs["video"],
        "--prompt", inputs["prompt"],
        "--frames", str(frames),
        "--short-edge", str(short_edge),
        "--seed", str(seed),
        "--base", base_path,
        "--lora-repo", LORA_REPO_ID,
        "--out", out_path,
    ]

    if inputs["audio"]:
        cmd += ["--audio", inputs["audio"]]

    for ref in inputs["refs"]:
        cmd += ["--ref", ref]

    for pin in pins:
        cmd += ["--pin", str(int(pin))]

    print(f"[Viggle-Remake] Запуск: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN or ""
    env["CUDA_VISIBLE_DEVICES"] = "0"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-4000:], flush=True)
        print("STDERR:", result.stderr[-4000:], flush=True)
        raise RuntimeError(f"sample.py завершился с кодом {result.returncode}: {result.stderr[-1000:]}")

    print(result.stdout[-2000:], flush=True)

    pipe_ready = True
    return {
        "video_path": out_path,
        "mode": mode,
        "lora": lora,
        "frames": frames,
        "short_edge": short_edge,
        "seed": seed,
    }

# ============================================================================
# HANDLER
# ============================================================================

def handler(job):
    job_input = job.get("input", {})
    model = job_input.get("model", "viggle-remake")

    if model not in ("viggle-remake", "remake", ""):
        return {"error": f"Неизвестная модель: {model}"}

    try:
        base_path = ensure_weights()
        out = run_inference(job_input, base_path)
        return {
            "status": "succeeded",
            "output": out,
            "message": "Генерация видео завершена",
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "trace": traceback.format_exc(),
        }

runpod.serverless.start({"handler": handler})
