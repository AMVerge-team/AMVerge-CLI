from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..video.probe_utils import probe_video_fps
from ._encode import build_stats, encode_selected, probe_frame_count

ADVANCED_AVAILABLE = False
try:
    import cv2
    import numpy as np
    ADVANCED_AVAILABLE = True
except ImportError:
    pass

_ANALYZE_MAX_WIDTH = 640
_GRID = 4
_LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(3, 10, 0.03)) if ADVANCED_AVAILABLE else {}


def detect_cadence(keep_indices: List[int]) -> Dict[str, float]:
    """Dominant gap between kept frames (anime cadence: 2=on-twos, 3=on-threes)."""
    if len(keep_indices) < 3:
        return {"cadence": 0, "confidence": 0.0}
    gaps: Dict[int, int] = {}
    for a, b in zip(keep_indices, keep_indices[1:]):
        g = b - a
        gaps[g] = gaps.get(g, 0) + 1
    total = sum(gaps.values())
    period, count = max(gaps.items(), key=lambda kv: kv[1])
    return {"cadence": int(period), "confidence": round(count / total, 3)}


if ADVANCED_AVAILABLE:
    def _region_max_meandiff(a: "np.ndarray", b: "np.ndarray") -> float:
        h, w = a.shape
        rh, rw = h // _GRID, w // _GRID
        if rh == 0 or rw == 0:
            return float(cv2.absdiff(a, b).mean())
        worst = 0.0
        for gy in range(_GRID):
            for gx in range(_GRID):
                y0, x0 = gy * rh, gx * rw
                ra = a[y0:y0 + rh, x0:x0 + rw]
                rb = b[y0:y0 + rh, x0:x0 + rw]
                worst = max(worst, float(cv2.absdiff(ra, rb).mean()))
        return worst

    def _edge_change_pct(a: "np.ndarray", b: "np.ndarray") -> float:
        ea = cv2.Canny(a, 80, 160)
        eb = cv2.Canny(b, 80, 160)
        return float(np.count_nonzero(cv2.absdiff(ea, eb))) / ea.size * 100.0

    def _hist_corr(a: "np.ndarray", b: "np.ndarray") -> float:
        ha = cv2.calcHist([a], [0], None, [64], [0, 256])
        hb = cv2.calcHist([b], [0], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))

    def _flow_motion(a: "np.ndarray", b: "np.ndarray") -> float:
        pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8)
        if pts is None or len(pts) < 5:
            return 0.0
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **_LK_PARAMS)
        if nxt is None or st is None:
            return 0.0
        good_new = nxt[st.flatten() == 1]
        good_old = pts[st.flatten() == 1]
        if len(good_new) < 5:
            return 0.0
        disp = np.linalg.norm(good_new.reshape(-1, 2) - good_old.reshape(-1, 2), axis=1)
        return float(np.median(disp))


def analyze_advanced(
    video_path: str,
    sensitivity: float = 1.0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    progress_hi: int = 95,
) -> Tuple[List[int], int, float, Dict[str, float]]:
    """Multi-signal dead/duplicate-frame analysis.

    Combines a 4x4 region grid (localized motion the global average misses),
    sparse Lucas-Kanade optical flow (cheaper than dense Farneback for a
    same/different decision), Canny edge change (line-art shifts) and histogram
    correlation (robust to small misalignment). A frame is KEPT when any signal
    fires - false-keep is cheaper than removing a real frame. Thresholds adapt
    to a per-clip noise floor; ``sensitivity`` scales them (higher = keeps more).

    Returns (keep_indices, frames_in, fps, cadence_info).
    """
    if not ADVANCED_AVAILABLE:
        raise ImportError(
            "Advanced dedup requires opencv. Run: pip install amverge[dedup]"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    scale = min(1.0, _ANALYZE_MAX_WIDTH / max(1, w))

    def _prep(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        return gray

    success, frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError("No frames in video")

    if progress_cb:
        progress_cb(0, "Sampling baseline...")

    prev_gray = _prep(frame)
    sample_count = min(30, total_frames - 1)
    region_samples: List[float] = []
    edge_samples: List[float] = []
    for _ in range(sample_count):
        success, frame = cap.read()
        if not success:
            break
        curr_gray = _prep(frame)
        region_samples.append(_region_max_meandiff(prev_gray, curr_gray))
        edge_samples.append(_edge_change_pct(prev_gray, curr_gray))
        prev_gray = curr_gray

    def _median(xs, default):
        if not xs:
            return default
        s = sorted(xs)
        return s[len(s) // 2]

    region_thr = max(3.0, _median(region_samples, 3.0) * 1.8) * sensitivity
    edge_thr = max(0.5, _median(edge_samples, 0.5) * 1.8) * sensitivity
    motion_thr = 0.6 * sensitivity
    hist_thr = 0.9990

    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen: {video_path}")

    success, frame = cap.read()
    keep_indices: List[int] = [0]
    prev_gray = _prep(frame)
    frame_idx = 0
    last_pct = -1

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        curr_gray = _prep(frame)
        region = _region_max_meandiff(prev_gray, curr_gray)
        keep = region > region_thr
        if not keep:
            keep = _edge_change_pct(prev_gray, curr_gray) > edge_thr
        if not keep:
            keep = _flow_motion(prev_gray, curr_gray) > motion_thr
        if not keep:
            keep = _hist_corr(prev_gray, curr_gray) < hist_thr

        if keep:
            keep_indices.append(frame_idx)
            prev_gray = curr_gray

        if progress_cb:
            pct = min(progress_hi - 1, int((frame_idx / max(1, total_frames - 1)) * progress_hi))
            if pct != last_pct:
                progress_cb(pct, f"Dedup (advanced)... {len(keep_indices)}/{frame_idx + 1}")
                last_pct = pct

    cap.release()
    frames_in = frame_idx + 1

    probe_n = probe_frame_count(video_path)
    if probe_n > 0 and abs(probe_n - frames_in) > max(2, int(0.01 * probe_n)):
        raise RuntimeError(
            f"Frame count mismatch (decoded {frames_in}, container {probe_n}) - "
            "source is likely VFR. Use the ffmpeg method or re-encode to CFR first."
        )

    return keep_indices, frames_in, probe_video_fps(video_path), detect_cadence(keep_indices)


def dedup_advanced(
    video_path: str,
    output_path: str,
    sensitivity: float = 1.0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    codec: Optional[str] = None,
    crf: int = 18,
) -> Tuple[str, dict]:
    """Multi-signal dead-frame removal (region grid + LK flow + edges + cadence).

    Returns (output_path, stats); stats includes ``cadence`` and ``confidence``.
    """
    keep_indices, frames_in, _, cadence = analyze_advanced(
        video_path, sensitivity, progress_cb
    )

    if progress_cb:
        progress_cb(95, "Encoding output...")

    encode_selected(video_path, output_path, keep_indices, crf=crf, codec=codec,
                    progress_cb=progress_cb, progress_lo=95, progress_hi=99)
    stats = build_stats(frames_in, len(keep_indices))
    stats.update(cadence)

    if progress_cb:
        progress_cb(100, f"Complete ({len(keep_indices)}/{frames_in} frames kept)")

    return output_path, stats
