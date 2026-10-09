from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .generate import BARCODE_BITS, barcode_geometry, color_probe_box


@dataclass(frozen=True)
class Frame:
    number: int
    color: str
    time: float


def ffprobe(path: str | Path, *args: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", *args, str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def streams(path: str | Path, kind: str) -> list[dict]:
    data = ffprobe(path, "-show_streams", "-select_streams", kind)
    return data.get("streams", [])


def video_stream(path: str | Path) -> dict:
    return streams(path, "v:0")[0]


def audio_streams(path: str | Path) -> list[dict]:
    return streams(path, "a")


def frame_rate(path: str | Path) -> Fraction:
    return Fraction(video_stream(path)["avg_frame_rate"])


def format_duration(path: str | Path) -> float:
    return float(ffprobe(path, "-show_format")["format"]["duration"])


def stream_duration(stream: dict) -> float:
    return float(stream["duration"])


def packet_durations(path: str | Path) -> dict[str, list[float]]:
    """Span of each video/audio stream measured from its packets, for
    containers (AVI) whose streams carry no duration. Video is its packet
    count over the frame rate: AVI gives B-frame packets no pts."""
    spans: dict[int, tuple] = {}
    counts: dict[int, int] = {}
    with av.open(str(path)) as container:
        rates = {s.index: s.average_rate for s in container.streams.video}
        kinds = {s.index: s.type for s in container.streams}
        for packet in container.demux():
            index = packet.stream.index
            if kinds.get(index) == "video" and packet.size:
                counts[index] = counts.get(index, 0) + 1
            elif kinds.get(index) == "audio" and packet.pts is not None:
                tb = packet.time_base
                start, end = packet.pts * tb, (packet.pts + (packet.duration or 0)) * tb
                lo, hi = spans.get(index, (start, end))
                spans[index] = (min(lo, start), max(hi, end))
    return {
        "video": [float(n / rates[i]) for i, n in counts.items()],
        "audio": [float(hi - lo) for lo, hi in spans.values()],
    }


def packet_count(path: str | Path) -> int:
    data = ffprobe(path, "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets")
    return int(data["streams"][0]["nb_read_packets"])


def keyframe_times(path: str | Path) -> list[float]:
    data = ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags")
    return sorted(
        float(p["pts_time"]) for p in data.get("packets", [])
        if "K" in p.get("flags", "") and p.get("pts_time") not in (None, "N/A")
    )


def _classify(rgb: np.ndarray) -> str:
    r, g, b = (float(c) for c in rgb.reshape(-1, 3).mean(axis=0))
    if b > 128 and r < 96 and g < 96:
        return "blue"
    if r > 128 and b < 96 and g < 96:
        return "red"
    return "unknown"


def _barcode(rgb: np.ndarray) -> int:
    h, w = rgb.shape[:2]
    y0, strip_h, block_w = barcode_geometry(w, h)
    cy = y0 + strip_h // 2
    value = 0
    for bit in range(BARCODE_BITS):
        cx = bit * block_w + block_w // 2
        patch = rgb[cy - 2:cy + 3, cx - 2:cx + 3].astype(np.float32)
        if patch.mean() > 127:
            value |= 1 << bit
    return value


def read_frames(path: str | Path) -> list[Frame]:
    frames: list[Frame] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            x, y, w, h = color_probe_box(rgb.shape[1], rgb.shape[0])
            frames.append(Frame(
                number=_barcode(rgb),
                color=_classify(rgb[y:y + h, x:x + w]),
                time=float(frame.time) if frame.time is not None else -1.0,
            ))
    return frames


def frame_numbers(path: str | Path) -> list[int]:
    return [f.number for f in read_frames(path)]


def runs(values: list) -> list[tuple[object, int]]:
    out: list[tuple[object, int]] = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1] = (v, out[-1][1] + 1)
        else:
            out.append((v, 1))
    return out


def jpeg_color(path: str | Path) -> str:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    x, y, w, h = color_probe_box(rgb.shape[1], rgb.shape[0])
    return _classify(rgb[y:y + h, x:x + w])
