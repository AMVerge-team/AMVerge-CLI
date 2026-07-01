from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from typing import Callable, Optional, Dict, Tuple, List

from ..infra.binaries import get_ffmpeg

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

ADVANCED_AVAILABLE = False
try:
    import cv2
    import numpy as np
    ADVANCED_AVAILABLE = True
except ImportError:
    pass


if ADVANCED_AVAILABLE:

    class _CadenceDetector:
        def __init__(self, window_size=24):
            self.window_size = window_size
            self.pattern = deque(maxlen=window_size)

        def add_frame(self, is_dup):
            self.pattern.append(1 if is_dup else 0)

        def detect(self):
            if len(self.pattern) < self.window_size // 2:
                return None
            p = list(self.pattern)
            for period in [2, 3, 4, 5, 6]:
                if len(p) >= period * 3:
                    matches = sum(1 for i in range(len(p) - period) if p[i] == p[i + period])
                    total = len(p) - period
                    if total > 0 and matches / total > 0.75:
                        return period
            return None


    class _AdvancedDedup:
        def __init__(
            self,
            threshold=0.95,
            region_grid=(4, 4),
            min_changed_regions=1,
            use_optical_flow=True,
            camera_motion_compensation=True,
            remove_static_subject=True,
        ):
            self.threshold = threshold
            self.region_grid = region_grid
            self.min_changed_regions = min_changed_regions
            self.use_optical_flow = use_optical_flow
            self.camera_motion_compensation = camera_motion_compensation
            self.remove_static_subject = remove_static_subject
            self.cadence = _CadenceDetector()

        def _region_analysis(self, gray1, gray2):
            h, w = gray1.shape
            rows, cols = self.region_grid
            rh, rw = h // rows, w // cols
            changed = []
            total_diff = 0.0
            for i in range(rows):
                for j in range(cols):
                    r1 = gray1[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
                    r2 = gray2[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
                    mean_diff = float(np.mean(cv2.absdiff(r1, r2)))
                    if mean_diff > 3.0 or np.max(cv2.absdiff(r1, r2)) > 50:
                        changed.append((i, j))
                        total_diff += mean_diff
            total_regions = rows * cols
            return changed, total_diff / max(1, total_regions), total_diff

        def _optical_flow(self, gray1, gray2):
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_mag = float(np.mean(mag))
            std_mag = float(np.std(mag))

            h, w = gray1.shape
            m = 0.25
            cx1, cx2 = int(w*m), int(w*(1-m))
            cy1, cy2 = int(h*m), int(h*(1-m))
            center_mag = float(np.mean(mag[cy1:cy2, cx1:cx2]))
            edge_mag = float(np.mean(np.concatenate([
                mag[:cy1, :].flatten(), mag[cy2:, :].flatten(),
                mag[cy1:cy2, :cx1].flatten(), mag[cy1:cy2, cx2:].flatten(),
            ])))

            is_camera = False
            is_bg_only = False
            if mean_mag > 0.5:
                uniformity = 1.0 - (std_mag / max(1e-6, mean_mag))
                is_camera = uniformity > 0.6 and mean_mag > 1.0
                if edge_mag > 0.8 and center_mag < edge_mag * 0.7:
                    is_bg_only = True
                if mean_mag > 0.8 and center_mag < 0.5:
                    is_bg_only = True

            mean_dx = float(np.mean(flow[..., 0]))
            mean_dy = float(np.mean(flow[..., 1]))

            return {
                "magnitude": mean_mag,
                "is_camera": is_camera,
                "is_bg_only": is_bg_only,
                "dx": mean_dx,
                "dy": mean_dy,
                "center_mag": center_mag,
            }

        def _center_similarity(self, gray1, gray2, dx=0.0, dy=0.0):
            h, w = gray1.shape
            m = 0.25
            x1, x2 = int(w*m), int(w*(1-m))
            y1, y2 = int(h*m), int(h*(1-m))
            if x2 <= x1 or y2 <= y1:
                return False, 0.0

            if abs(dx) > 0.1 or abs(dy) > 0.1:
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                gray2 = cv2.warpAffine(gray2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

            c1 = gray1[y1:y2, x1:x2]
            c2 = gray2[y1:y2, x1:x2]
            mean_diff = float(np.mean(cv2.absdiff(c1, c2)))
            max_diff_val = float(np.max(cv2.absdiff(c1, c2)))

            hist1 = cv2.calcHist([c1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([c2], [0], None, [256], [0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            corr = max(0.0, min(1.0, (corr + 1) / 2.0))

            edges1 = cv2.Canny(c1, 50, 150)
            edges2 = cv2.Canny(c2, 50, 150)
            edge_changed = np.sum(cv2.absdiff(edges1, edges2) > 0) / max(1, np.sum(edges1 > 0))

            is_static = mean_diff < 10.0 and max_diff_val < 70 and corr > 0.85 and edge_changed < 0.30
            similarity = 1.0 - min(1.0, mean_diff / 20.0)

            return is_static, similarity

        def is_duplicate(self, prev_frame, curr_frame):
            scale = min(1.0, 640 / max(prev_frame.shape[1], prev_frame.shape[0]))
            p = cv2.resize(prev_frame, None, fx=scale, fy=scale) if scale < 1.0 else prev_frame
            c = cv2.resize(curr_frame, None, fx=scale, fy=scale) if scale < 1.0 else curr_frame

            gray1 = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if len(p.shape) == 3 else p
            gray2 = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if len(c.shape) == 3 else c

            changed_regions, region_score, _ = self._region_analysis(gray1, gray2)
            flow_info = self._optical_flow(gray1, gray2) if self.use_optical_flow else {"magnitude": 0, "is_camera": False, "is_bg_only": False, "dx": 0, "dy": 0}

            if self.remove_static_subject and self.camera_motion_compensation:
                is_static, _ = self._center_similarity(gray1, gray2, flow_info["dx"], flow_info["dy"])
                is_global = flow_info["is_camera"] or flow_info["is_bg_only"]
                has_motion = region_score > 0.02 or flow_info["magnitude"] > 0.3
                if is_static and (is_global or has_motion):
                    return True
                if is_global:
                    return False

            if len(changed_regions) >= self.min_changed_regions:
                return False

            if flow_info["magnitude"] > 1.5:
                return False

            return region_score < (1.0 - self.threshold)


def dedup_advanced(
    video_path: str,
    output_path: str,
    threshold: float = 0.95,
    region_sensitivity: int = 1,
    use_optical_flow: bool = True,
    camera_motion_compensation: bool = True,
    remove_static_subject: bool = True,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, Dict]:
    if not ADVANCED_AVAILABLE:
        raise ImportError("Advanced dedup requires opencv. Run: pip install opencv-python")

    ffmpeg = get_ffmpeg()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps_val <= 0 or not np.isfinite(fps_val):
        fps_val = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    dedup_engine = _AdvancedDedup(
        threshold=threshold,
        region_grid=(4, 4),
        min_changed_regions=max(1, min(4, region_sensitivity)),
        use_optical_flow=use_optical_flow,
        camera_motion_compensation=camera_motion_compensation,
        remove_static_subject=remove_static_subject,
    )

    ffmpeg_cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps_val), "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=CREATE_NO_WINDOW)

    import threading
    def _drain():
        try:
            for _ in ffmpeg_proc.stderr:
                pass
        except Exception:
            pass
    threading.Thread(target=_drain, daemon=True).start()

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        raise RuntimeError("No frames in video")

    ffmpeg_proc.stdin.write(prev_frame.tobytes())
    unique_count = 1
    dup_count = 0
    cam_only_removed = 0
    frame_idx = 0
    last_pct = -1

    flow_info_prev = None

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if progress_cb:
            pct = min(99, int((frame_idx / max(1, total_frames - 1)) * 100))
            if pct != last_pct:
                progress_cb(pct, f"Dedup (advanced)... {unique_count}/{frame_idx + 1}")
                last_pct = pct

        is_dup = dedup_engine.is_duplicate(prev_frame, curr_frame)
        dedup_engine.cadence.add_frame(is_dup)

        if not is_dup:
            ffmpeg_proc.stdin.write(curr_frame.tobytes())
            unique_count += 1
            prev_frame = curr_frame.copy()
        else:
            dup_count += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    cadence_period = dedup_engine.cadence.detect()

    stats = {
        "frames_in": frame_idx + 1,
        "frames_out": unique_count,
        "frames_removed": dup_count,
        "pct_removed": round((dup_count / max(1, frame_idx + 1)) * 100, 1),
        "cadence": cadence_period,
    }

    if progress_cb:
        extra = f" (cadence every {cadence_period} frames)" if cadence_period else ""
        progress_cb(100, f"Complete ({unique_count}/{frame_idx + 1} frames kept, {stats['pct_removed']}% removed{extra})")

    return output_path, stats
