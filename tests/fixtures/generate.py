from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .specs import FIXTURES, MediaSpec

GENERATOR_VERSION = "1"
BARCODE_BITS = 12
DEFAULT_CACHE = Path(__file__).parent / ".media"


def cache_dir() -> Path:
    return Path(os.environ.get("AMVERGE_TEST_MEDIA", DEFAULT_CACHE))


def barcode_geometry(width: int, height: int) -> tuple[int, int, int]:
    strip_h = max(16, height // 6)
    block_w = width // BARCODE_BITS
    return height - strip_h, strip_h, block_w


def color_probe_box(width: int, height: int) -> tuple[int, int, int, int]:
    return 0, 0, width // 4, height // 6


def _video_graph(spec: MediaSpec, timecode: bool) -> str:
    size = f"{spec.width}x{spec.height}"
    chains = []
    labels = []
    for i, frames in enumerate(spec.segments):
        chains.append(
            f"color=c={spec.segment_color(i)}:s={size}:r={spec.fps},"
            f"trim=end_frame={frames},setpts=PTS-STARTPTS[s{i}]"
        )
        labels.append(f"[s{i}]")

    filters = [f"{''.join(labels)}concat=n={len(spec.segments)}:v=1:a=0"]
    if timecode:
        filters.append(
            f"drawtext=timecode='00\\:00\\:00\\:00':rate={spec.fps}"
            f":fontsize={max(16, spec.height // 8)}:fontcolor=white"
            f":box=1:boxcolor=black@0.6:x=(w-tw)/2:y=(h-th)/2"
        )
    y0, strip_h, block_w = barcode_geometry(spec.width, spec.height)
    filters.append(f"drawbox=x=0:y={y0}:w={spec.width}:h={strip_h}:color=black:t=fill")
    for bit in range(BARCODE_BITS):
        filters.append(
            f"drawbox=x={bit * block_w}:y={y0}:w={block_w}:h={strip_h}"
            f":color=white:t=fill:enable='mod(floor(n/{2 ** bit}),2)'"
        )
    filters.append(f"format={spec.pix_fmt}")
    chains.append(",".join(filters) + "[v]")
    return ";".join(chains)


def _audio_graph(spec: MediaSpec) -> list[str]:
    layouts = {1: "mono", 2: "stereo"}
    return [
        f"sine=frequency={track.frequency}:sample_rate={track.sample_rate}:duration={spec.duration:.6f},"
        f"aformat=channel_layouts={layouts[track.channels]}[a{i}]"
        for i, track in enumerate(spec.audio)
    ]


def _force_keyframes(spec: MediaSpec) -> list[str]:
    if spec.keyframes not in ("cuts", "cuts_gop"):
        return []
    terms = "+".join(f"eq(n,{f})" for f in [0, *spec.cut_frames])
    return ["-force_key_frames", f"expr:{terms}"]


def _video_encoder(spec: MediaSpec) -> list[str]:
    total = spec.total_frames + 1
    keyint = {"cuts": total, "gop": spec.gop, "cuts_gop": spec.gop, "sparse": total}.get(spec.keyframes)
    if spec.codec == "h264":
        args = [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-bf", str(spec.bframes), "-g", str(keyint), "-keyint_min", str(keyint),
            "-forced-idr", "1",
            "-x264-params", "scenecut=0:open-gop=0",
        ]
    elif spec.codec == "h265":
        args = [
            "-c:v", "libx265", "-preset", "fast", "-crf", "20", "-forced-idr", "1",
            "-x265-params",
            f"keyint={keyint}:min-keyint={keyint}:scenecut=0:open-gop={int(spec.open_gop)}"
            f":bframes={spec.bframes}:log-level=error",
        ]
        if spec.container != "mkv":
            args += ["-tag:v", "hvc1"]
    elif spec.codec == "av1":
        args = [
            "-c:v", "libsvtav1", "-preset", "10", "-crf", "30",
            "-svtav1-params", f"keyint={keyint}:scd=0",
        ]
    elif spec.codec in ("huffyuv", "utvideo", "magicyuv"):
        args = ["-c:v", spec.codec]
    else:
        args = ["-c:v", "prores_ks", "-vendor", "apl0"]
    if spec.profile:
        args += ["-profile:v", spec.profile]
    return args + _force_keyframes(spec)


def _audio_encoder(spec: MediaSpec) -> list[str]:
    args: list[str] = []
    for i, track in enumerate(spec.audio):
        args += [f"-c:a:{i}", track.codec]
        if track.codec == "aac":
            args += [f"-b:a:{i}", "128k"]
        if track.language:
            args += [f"-metadata:s:a:{i}", f"language={track.language}"]
    return args


def build_command(spec: MediaSpec, out: Path, timecode: bool = True) -> list[str]:
    graph = ";".join([_video_graph(spec, timecode), *_audio_graph(spec)])
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-filter_complex", graph, "-map", "[v]",
    ]
    for i in range(len(spec.audio)):
        cmd += ["-map", f"[a{i}]"]
    cmd += ["-r", spec.fps, "-fps_mode", "cfr"]
    cmd += _video_encoder(spec) + _audio_encoder(spec)
    if spec.container == "mp4":
        cmd += ["-movflags", "+faststart"]
    if spec.track_timescale and spec.container != "mkv":
        cmd += ["-video_track_timescale", str(spec.track_timescale)]
    cmd.append(str(out))
    return cmd


def fixture_path(spec: MediaSpec, root: Path | None = None) -> Path:
    digest = hashlib.sha1(f"{GENERATOR_VERSION}:{spec!r}".encode()).hexdigest()[:10]
    return (root or cache_dir()) / f"{spec.name}-{digest}.{spec.extension}"


def generate(spec: MediaSpec, root: Path | None = None) -> Path:
    out = fixture_path(spec, root)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".tmp-{os.getpid()}-{out.name}")
    result = subprocess.run(build_command(spec, tmp), capture_output=True, text=True)
    if result.returncode != 0 and "drawtext" in result.stderr + result.stdout:
        result = subprocess.run(build_command(spec, tmp, timecode=False), capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"fixture {spec.name} failed: {result.stderr[-1500:]}")
    tmp.replace(out)
    return out


def generate_all(root: Path | None = None, workers: int = 4) -> dict[str, Path]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(lambda s: generate(s, root), FIXTURES))
    return {spec.name: path for spec, path in zip(FIXTURES, paths)}


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    for name, path in generate_all(target).items():
        print(f"{name}: {path}")
