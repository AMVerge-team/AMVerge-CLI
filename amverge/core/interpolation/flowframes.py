from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from ..infra.config import get_amverge_config_dir

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

FLOWFRAMES_VERSION = "1.42.0"

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def _find_flowframes_exe() -> Optional[str]:
    config_path = os.path.join(get_amverge_config_dir(), "flowframes_path.txt")
    if os.path.exists(config_path):
        candidate = Path(config_path).read_text().strip()
        if candidate and os.path.exists(candidate):
            return candidate

    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata:
        guess = os.path.join(localappdata, "Flowframes", "Flowframes.exe")
        if os.path.exists(guess):
            return guess

    return None


def _set_flowframes_path(exe_path: str) -> None:
    config_dir = get_amverge_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    Path(os.path.join(config_dir, "flowframes_path.txt")).write_text(exe_path)


def flowframes_available() -> bool:
    return _find_flowframes_exe() is not None


def _build_clean_env() -> dict:
    env = {}
    for key, val in os.environ.items():
        if key.lower() not in ("nodefaultcurrentdirectoryinexepath",):
            env[key] = val
    return env


def _kill_existing(timeout: float = 10.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", "Flowframes.exe"],
                capture_output=True, creationflags=CREATE_NO_WINDOW,
            )
        except Exception:
            pass
        try:
            r = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Flowframes.exe", "/NH"],
                capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
            )
            if "Flowframes.exe" not in r.stdout:
                return True
        except Exception:
            return True
        time.sleep(0.4)
    return False


def _find_session_log(logs_dir: str, known_sessions: set[str]) -> Optional[str]:
    try:
        entries = os.listdir(logs_dir)
    except OSError:
        return None
    for entry in entries:
        if entry not in known_sessions:
            candidate = os.path.join(logs_dir, entry, "sessionlog.txt")
            if os.path.exists(candidate):
                return candidate
    return None


def _find_newest_output(output_dir: str, start_time: float) -> Optional[str]:
    best = None
    best_mtime = 0
    try:
        for entry in os.listdir(output_dir):
            ext = os.path.splitext(entry)[1].lower()
            if ext not in MEDIA_EXTENSIONS:
                continue
            full = os.path.join(output_dir, entry)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if not st.st_size:
                continue
            if st.st_mtime >= start_time - 2 and st.st_mtime >= best_mtime:
                best_mtime = st.st_mtime
                best = full
    except OSError:
        pass
    return best


def run_flowframes(
    input_path: str,
    output_dir: str,
    factor: int = 2,
    ai: str = "RifeNcnn",
    model: str = "RIFE 4.26",
    output_format: str = "Mp4",
    encoder: str = "X264",
    pix_fmt: str = "Yuv420P",
    quality: Optional[float] = None,
    max_fps: Optional[float] = None,
    max_height: Optional[int] = None,
    scene_change: bool = False,
    scene_sensitivity: Optional[float] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    timeout: float = 36000.0,
) -> str:
    exe = _find_flowframes_exe()
    if not exe:
        raise RuntimeError(
            "Flowframes.exe not found. Set with: amverge flowframes-path <path>"
        )

    os.makedirs(output_dir, exist_ok=True)

    logs_dir = os.path.join(os.path.dirname(exe), "FlowframesData", "logs")

    known_sessions: set[str] = set()
    try:
        for entry in os.listdir(logs_dir):
            known_sessions.add(entry)
    except OSError:
        pass

    args = [
        exe, "-a", "-nc", "-mdc",
        "-f", str(factor),
        "-ai", ai,
        "-m", model,
        "-vf", output_format,
        "-ve", encoder,
        "-pf", pix_fmt,
        "-o", output_dir,
        input_path,
    ]

    if quality is not None:
        args += ["-q", str(int(quality))]
    if max_fps is not None and max_fps > 0:
        args += ["-fps", str(int(max_fps))]
    if max_height is not None and max_height > 0:
        args += ["-mh", str(int(max_height))]
    if scene_change:
        args.append("-scn")
        if scene_sensitivity is not None:
            args += ["-scnv", str(int(scene_sensitivity))]

    if not _kill_existing():
        raise RuntimeError(
            "Could not terminate existing Flowframes instance. "
            "Close it manually or run: taskkill /F /T /IM Flowframes.exe"
        )

    start_time = time.monotonic()
    env = _build_clean_env()

    proc = subprocess.Popen(
        args, env=env, creationflags=CREATE_NO_WINDOW,
    )

    if progress_cb:
        progress_cb(0, "Flowframes starting...")

    session_log: Optional[str] = None
    last_line_count = 0
    ai_complete = False
    last_pct = 0
    output_file: Optional[str] = None
    stable_count = 0
    last_output_size = 0
    error_message: Optional[str] = None
    poll_interval = 1.5

    try:
        while proc.poll() is None:
            if cancel_check and cancel_check():
                _kill_existing(timeout=2.0)
                raise RuntimeError("Cancelled by user")

            if time.monotonic() - start_time > timeout:
                _kill_existing(timeout=2.0)
                raise RuntimeError(f"Flowframes timed out after {timeout:.0f}s")

            if session_log is None:
                session_log = _find_session_log(logs_dir, known_sessions)

            if session_log and not ai_complete:
                try:
                    content = Path(session_log).read_text(errors="replace")
                    lines = content.splitlines()
                except OSError:
                    lines = []

                for j in range(last_line_count, len(lines)):
                    line = lines[j]
                    if not line:
                        continue

                    if log_cb:
                        cleaned = re.sub(r"^\[[^\]]*\]\s*\[[^\]]*\]:\s*", "", line)
                        log_cb(cleaned)

                    pct = _parse_progress_line(line)
                    if pct is not None:
                        ai_complete = ai_complete or pct >= 99.5
                        pct = min(100, pct)
                        if pct > last_pct:
                            last_pct = pct
                            if progress_cb:
                                progress_cb(int(pct), f"Interpolating... {int(pct)}%")

                    completion = _parse_completion_line(line)
                    if completion:
                        ai_complete = True

                    error_match = _parse_error_line(line)
                    if error_match:
                        error_message = error_match

                last_line_count = len(lines)

            if ai_complete:
                candidate = _find_newest_output(output_dir, start_time)
                if candidate:
                    try:
                        size = os.path.getsize(candidate)
                    except OSError:
                        size = 0
                    if size == last_output_size and size > 0:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_output_size = size
                    output_file = candidate
                    if stable_count >= 3:
                        break
                elif output_file and error_message:
                    break

            time.sleep(poll_interval)

        proc.wait()

    except Exception:
        try:
            _kill_existing(timeout=2.0)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        raise

    if error_message and not output_file:
        raise RuntimeError(f"Flowframes error: {error_message}")

    if not output_file:
        raise RuntimeError("Flowframes did not produce an output file")

    if progress_cb:
        progress_cb(100, "Complete")

    return output_file


def _parse_progress_line(line: str) -> Optional[float]:
    m = re.search(r"Interpolated\s+(\d+)\s*\/\s*(\d+)\s*Frames?", line, re.IGNORECASE)
    if m:
        done = float(m.group(1))
        total = float(m.group(2))
        if total > 0:
            return done / total * 100

    return None


def _parse_completion_line(line: str) -> bool:
    patterns = [
        r"Done interpolating",
        r"Interpolation done",
        r"Frame interpolation (?:took|done)",
        r"Output video",
        r"Encoding finished",
        r"\[Done\]",
    ]
    for pat in patterns:
        if re.search(pat, line, re.IGNORECASE):
            return True
    return False


def _parse_error_line(line: str) -> Optional[str]:
    patterns = [
        (r"Failed to initialize MediaFile", "Failed to initialize MediaFile"),
        (r"No frames left", "No frames left"),
        (r"Interpolation failed", "Interpolation failed"),
        (r"Error occured", "Error occurred"),
        (r"ran out of memory", "Out of memory"),
        (r"vkAllocateMemory failed", "Vulkan out of memory"),
        (r"vkWaitForFences failed", "Vulkan device lost"),
        (r"No valid AI model has been selected", "No valid AI model selected"),
    ]
    for pat, msg in patterns:
        if re.search(pat, line, re.IGNORECASE):
            return msg
    return None


def cancel_flowframes() -> bool:
    return _kill_existing(timeout=5.0)


def set_flowframes_path(exe_path: str) -> None:
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Flowframes.exe not found at: {exe_path}")
    _set_flowframes_path(exe_path)


def get_flowframes_path() -> Optional[str]:
    return _find_flowframes_exe()


def _get_flowframes_pkgs_dir() -> Optional[str]:
    localappdata = os.environ.get("LOCALAPPDATA", "")
    if localappdata:
        pkgs = os.path.join(localappdata, "Flowframes", "FlowframesData", "pkgs")
        if os.path.exists(pkgs):
            return pkgs
    return None


FLOWFRAMES_MODELS = {
    "rife4.25-ncnn": {
        "name": "RIFE 4.25 (NCNN)",
        "engine": "RifeNcnn",
        "model_name": "RIFE 4.25",
        "subdir": "rife-ncnn",
        "credit": "by hzwer / megvii-research",
        "description": "RIFE v4.25 NCNN variant (Flowframes default)",
    },
    "rife4.26-ncnn": {
        "name": "RIFE 4.26 (NCNN)",
        "engine": "RifeNcnn",
        "model_name": "RIFE 4.26",
        "subdir": "rife-ncnn",
        "credit": "by hzwer / megvii-research",
        "description": "RIFE v4.26 NCNN variant (Flowframes default)",
    },
    "rife4.13-cuda": {
        "name": "RIFE 4.13 (CUDA)",
        "engine": "RifeCuda",
        "model_name": "RIFE 4.13.2",
        "subdir": "rife-cuda",
        "credit": "by hzwer / megvii-research",
        "description": "RIFE v4.13 CUDA variant",
    },
    "rife4.14-cuda": {
        "name": "RIFE 4.14 (CUDA)",
        "engine": "RifeCuda",
        "model_name": "RIFE 4.14",
        "subdir": "rife-cuda",
        "credit": "by hzwer / megvii-research",
        "description": "RIFE v4.14 CUDA variant",
    },
    "rife4.25-vs": {
        "name": "RIFE 4.25 (NCNN VS)",
        "engine": "RifeNcnnVs",
        "model_name": "RIFE 4.25",
        "subdir": "rife-ncnn-vs",
        "credit": "by hzwer / megvii-research",
        "description": "RIFE v4.25 NCNN VapourSynth variant",
    },
    "rife4.26-vs": {
        "name": "RIFE 4.26 (NCNN VS)",
        "engine": "RifeNcnnVs",
        "model_name": "RIFE 4.26",
        "subdir": "rife-ncnn-vs",
        "credit": "by hzwer / megvii-research",
        "description": "RIFE v4.26 NCNN VapourSynth variant",
    },
    "dain-ncnn": {
        "name": "DAIN (NCNN)",
        "engine": "DainNcnn",
        "model_name": "DAIN",
        "subdir": "dain-ncnn",
        "credit": "by baohengtao / DAIN",
        "description": "Depth-Aware Video Frame Interpolation NCNN",
    },
    "flavr-cuda": {
        "name": "FLAVR (CUDA)",
        "engine": "FlavrCuda",
        "model_name": "FLAVR",
        "subdir": "flavr-cuda",
        "credit": "by tarun005 / FLAVR",
        "description": "Flow-Agnostic Video Representations for Fast Frame Interpolation",
    },
    "xvfi-cuda": {
        "name": "XVFI (CUDA)",
        "engine": "XvfiCuda",
        "model_name": "XVFI",
        "subdir": "xvfi-cuda",
        "credit": "by JihyongOh / XVFI",
        "description": "eXtreme Video Frame Interpolation CUDA",
    },
}


def is_flowframes_model_installed(key: str) -> bool:
    entry = FLOWFRAMES_MODELS.get(key)
    if not entry:
        return False
    pkgs_dir = _get_flowframes_pkgs_dir()
    if not pkgs_dir:
        return False
    subdir = os.path.join(pkgs_dir, entry["subdir"])
    if not os.path.isdir(subdir):
        return False
    if key.startswith("rife"):
        return any(
            d.startswith("rife-v") for d in os.listdir(subdir)
            if os.path.isdir(os.path.join(subdir, d))
        )
    return True
