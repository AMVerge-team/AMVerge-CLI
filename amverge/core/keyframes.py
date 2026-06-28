"""Keyframe timestamp extraction via PyAV."""
from __future__ import annotations

import time
from typing import Callable

import av


ProgressCb = Callable[[int, str], None]


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _pts_to_seconds(pts, time_base) -> float | None:
    try:
        if pts is None or time_base is None:
            return None
        return float(pts * time_base)
    except Exception:
        return None


def _looks_pathological(times: list[float], duration_s: float | None) -> bool:
    if len(times) < 2:
        return True

    times_sorted = sorted(times)

    for prev, curr in zip(times_sorted, times_sorted[1:]):
        if curr <= prev:
            return True

    if duration_s and duration_s > 0:
        if (len(times_sorted) / duration_s) > 10.0:
            return True

    deltas = sorted(curr - prev for prev, curr in zip(times_sorted, times_sorted[1:]))
    return deltas[len(deltas) // 2] < 0.05


def _decode_keyframe_times(container, stream, emit_liveness) -> list[float]:
    times: list[float] = []

    try:
        stream.codec_context.skip_frame = "NONKEY"
    except Exception:
        pass

    for frame_index, frame in enumerate(container.decode(stream), 1):
        ts = _pts_to_seconds(frame.pts, stream.time_base)
        if ts is not None:
            times.append(ts)
        if frame_index % 250 == 0:
            emit_liveness("decode", len(times), ts)

    return times


def generate_keyframes(
    video_path: str,
    progress_cb: ProgressCb | None = None,
    *,
    progress_base: int = 10,
    progress_range: int = 30,
    progress_interval_s: float = 1.0,
) -> list[float]:
    start_time = time.monotonic()
    last_emit = start_time - 9999.0

    def safe_progress(percent: int, message: str) -> None:
        if progress_cb:
            try:
                progress_cb(int(percent), str(message))
            except Exception:
                pass

    def percent_for_time(ts: float | None, dur: float | None) -> int:
        base = int(progress_base)
        span = max(0, int(progress_range))
        if not dur or dur <= 0 or ts is None or ts < 0:
            return base
        fraction = max(0.0, min(1.0, ts / dur))
        return _clamp_int(base + int(span * fraction), base, base + span)

    def maybe_emit(stage: str, count: int, ts: float | None, dur: float | None) -> None:
        nonlocal last_emit
        now = time.monotonic()
        if (now - last_emit) < float(progress_interval_s):
            return
        last_emit = now
        elapsed = now - start_time
        percent = percent_for_time(ts, dur)
        if dur and ts is not None and ts >= 0:
            msg = f"Extracting keyframes [{stage}] found={count} at={ts:.1f}s/{dur:.1f}s elapsed={elapsed:.0f}s"
        else:
            msg = f"Extracting keyframes [{stage}] found={count} elapsed={elapsed:.0f}s"
        safe_progress(percent, msg)

    keyframes: list[float] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]

        duration_s: float | None = None
        try:
            if container.duration is not None:
                duration_s = float(container.duration) / 1_000_000.0
        except Exception:
            pass

        maybe_emit("open", 0, 0.0, duration_s)

        try:
            for packet_index, packet in enumerate(container.demux(stream), 1):
                ts = _pts_to_seconds(packet.pts, stream.time_base)
                if packet.is_keyframe and ts is not None:
                    keyframes.append(ts)
                if packet_index % 500 == 0:
                    maybe_emit("demux", len(keyframes), ts, duration_s)
        except Exception:
            keyframes = []

        if not keyframes or _looks_pathological(keyframes, duration_s):
            try:
                with av.open(video_path) as dc:
                    ds = dc.streams.video[0]
                    maybe_emit("decode", 0, 0.0, duration_s)
                    keyframes = _decode_keyframe_times(
                        dc, ds,
                        lambda stage, count, ts: maybe_emit(stage, count, ts, duration_s),
                    )
            except Exception:
                return []

    normalized = sorted(set(round(t, 6) for t in keyframes if t is not None and t >= 0.0))
    done_pct = int(progress_base) + max(0, int(progress_range))
    safe_progress(done_pct, f"Keyframes done - found {len(normalized)}")

    return normalized
