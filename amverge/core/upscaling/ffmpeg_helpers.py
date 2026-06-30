import os
import subprocess
import sys

from ..infra.binaries import get_ffmpeg, get_ffprobe

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def ensure_ffmpeg():
    from ..infra.ffmpeg_bootstrap import ensure_ffmpeg as _bootstrap, is_portable_ffmpeg_installed
    if not is_portable_ffmpeg_installed():
        _bootstrap()


def encode_thread_count(out_w, out_h):
    cpu_cores = os.cpu_count() or 4
    return max(2, min(cpu_cores, 4 if out_w * out_h >= 3840 * 2160 else 8))


def get_video_dims_ffprobe(input_path):
    ffprobe = get_ffprobe()
    try:
        w = int(subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width", "-of", "default=nw=1:nk=1",
             str(input_path)],
            text=True, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
        ).strip())
        h = int(subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "default=nw=1:nk=1",
             str(input_path)],
            text=True, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
        ).strip())
        return w, h
    except FileNotFoundError:
        raise RuntimeError("FFprobe not found. Install FFmpeg: https://ffmpeg.org/download.html")


def get_color_args(input_path):
    ffprobe = get_ffprobe()
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_primaries,color_transfer,color_space,color_range",
             "-of", "default=nw=1", str(input_path)],
            text=True, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return []
    vals = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v.strip()

    def ok(x):
        return x and x.lower() not in ("unknown", "n/a", "reserved", "")

    args = []
    if ok(vals.get("color_primaries")):
        args += ["-color_primaries", vals["color_primaries"]]
    if ok(vals.get("color_transfer")):
        args += ["-color_trc", vals["color_transfer"]]
    if ok(vals.get("color_space")):
        args += ["-colorspace", vals["color_space"]]
    if ok(vals.get("color_range")):
        args += ["-color_range", vals["color_range"]]
    return args


def build_ffmpeg_pipe(out_w, out_h, fps_val, crf, x264_preset, tune, output_path, extra_vf=None, color_args=None):
    enc_threads = encode_thread_count(out_w, out_h)
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps_val), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high",
    ]
    if tune:
        cmd += ["-tune", tune]
    if extra_vf:
        cmd += ["-vf", extra_vf]
    cmd += [
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
    ]
    if color_args:
        cmd += color_args
    cmd += [str(output_path)]
    return cmd


def mux_audio(video_path, audio_source_path):
    ffprobe = get_ffprobe()
    probe_cmd = [
        ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=codec_type", "-of", "csv=p=0", str(audio_source_path),
    ]
    has_audio = False
    try:
        r = subprocess.run(probe_cmd, capture_output=True, text=True,
                           timeout=10, creationflags=CREATE_NO_WINDOW)
        if "audio" in r.stdout.lower():
            has_audio = True
    except Exception:
        pass

    if not has_audio:
        return False

    ffmpeg = get_ffmpeg()
    tmp = str(video_path) + ".tmp.mp4"

    def _mux(audio_codec_args):
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(audio_source_path),
            "-c:v", "copy", *audio_codec_args,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-movflags", "+faststart", tmp,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, creationflags=CREATE_NO_WINDOW)
            if r.returncode == 0:
                os.replace(tmp, str(video_path))
                return True
        except Exception:
            pass
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False

    # Copy the original audio untouched when the codec fits MP4; only
    # re-encode to AAC as a fallback for incompatible source codecs.
    if _mux(["-c:a", "copy"]):
        return True
    return _mux(["-c:a", "aac", "-b:a", "192k"])
