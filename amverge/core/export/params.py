"""FFmpeg argument + compatibility helpers for export."""

from __future__ import annotations

from pathlib import Path

_CODEC_ALIASES = {
    "h264": "h264_high",
    "h265": "h265_main",
    "hevc": "h265_main",
    "av1": "av1_main",
}


def normalize_codec(codec: str) -> str:
    return _CODEC_ALIASES.get(codec, codec)


# Per-codec video encode parameters. ``cpu`` is always present; ``gpu`` is set
# only where an NVENC path exists (else None → falls back to CPU). Each value is
# ``(encoder, [extra ffmpeg args])``.
VIDEO_PARAMS: dict[str, dict[str, tuple[str, list[str]] | None]] = {
    "h264_main": {
        "cpu": ("libx264", ["-profile:v", "main", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18"]),
        "gpu": ("h264_nvenc", ["-profile:v", "main", "-pix_fmt", "yuv420p", "-cq", "19"]),
    },
    "h264_high": {
        "cpu": ("libx264", ["-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18"]),
        "gpu": ("h264_nvenc", ["-profile:v", "high", "-pix_fmt", "yuv420p", "-cq", "19"]),
    },
    "h264_high10": {
        "cpu": ("libx264", ["-profile:v", "high10", "-pix_fmt", "yuv420p10le", "-preset", "slow", "-crf", "19"]),
        "gpu": None,
    },
    "h264_high422": {
        "cpu": ("libx264", ["-profile:v", "high422", "-pix_fmt", "yuv422p", "-preset", "slow", "-crf", "18"]),
        "gpu": None,
    },
    "h265_main": {
        "cpu": ("libx265", ["-profile:v", "main", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20"]),
        "gpu": ("hevc_nvenc", ["-profile:v", "main", "-pix_fmt", "yuv420p", "-cq", "19"]),
    },
    "h265_main10": {
        "cpu": ("libx265", ["-profile:v", "main10", "-pix_fmt", "yuv420p10le", "-preset", "slow", "-crf", "21"]),
        "gpu": ("hevc_nvenc", ["-profile:v", "main10", "-pix_fmt", "p010le", "-cq", "20"]),
    },
    "h265_main12": {
        "cpu": ("libx265", ["-profile:v", "main12", "-pix_fmt", "yuv420p12le", "-preset", "slow", "-crf", "22"]),
        "gpu": None,
    },
    "h265_main422_10": {
        "cpu": ("libx265", ["-profile:v", "main422-10", "-pix_fmt", "yuv422p10le", "-preset", "slow", "-crf", "21"]),
        "gpu": None,
    },
    "av1_main": {
        "cpu": ("libsvtav1", ["-preset", "6", "-crf", "32"]),
        "gpu": ("av1_nvenc", ["-pix_fmt", "yuv420p", "-cq", "28"]),
    },
    "prores_422_lt": {"cpu": ("prores_ks", ["-profile:v", "1", "-pix_fmt", "yuv422p10le"]), "gpu": None},
    "prores_422": {"cpu": ("prores_ks", ["-profile:v", "2", "-pix_fmt", "yuv422p10le"]), "gpu": None},
    "prores_422_hq": {"cpu": ("prores_ks", ["-profile:v", "3", "-pix_fmt", "yuv422p10le"]), "gpu": None},
    "prores_4444": {"cpu": ("prores_ks", ["-profile:v", "4", "-pix_fmt", "yuva444p10le"]), "gpu": None},
    "prores_4444_xq": {"cpu": ("prores_ks", ["-profile:v", "5", "-pix_fmt", "yuva444p10le"]), "gpu": None},
}
_DEFAULT_VIDEO = VIDEO_PARAMS["h264_high"]

AUDIO_ARGS: dict[str, list[str]] = {
    "copy": ["-c:a", "copy"],
    "pcm16": ["-c:a", "pcm_s16le", "-ar", "48000"],
    "pcm24": ["-c:a", "pcm_s24le", "-ar", "48000"],
    "flac": ["-c:a", "flac", "-compression_level", "5"],
    "alac": ["-c:a", "alac"],
    "opus": ["-c:a", "libopus", "-b:a", "160k", "-vbr", "on", "-application", "audio"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    "none": ["-an"],
    "aac_320": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"],
    "aac": ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"],
}


def gpu_available_for(codec: str) -> bool:
    return VIDEO_PARAMS.get(normalize_codec(codec), _DEFAULT_VIDEO)["gpu"] is not None


def video_encode_args(codec: str, use_gpu: bool) -> list[str]:
    params = VIDEO_PARAMS.get(normalize_codec(codec), _DEFAULT_VIDEO)
    entry = params["gpu"] if (use_gpu and params["gpu"] is not None) else params["cpu"]
    encoder, extra = entry  # type: ignore[misc]
    return ["-c:v", encoder, *extra]


def audio_args(mode: str) -> list[str]:
    return list(AUDIO_ARGS.get(mode, AUDIO_ARGS["aac"]))


# -- compatibility ---------------------------------------------------------

def codec_family(codec: str) -> str:
    c = normalize_codec(codec)
    if c.startswith("h264_"):
        return "h264"
    if c.startswith("h265_"):
        return "h265"
    if c == "av1_main":
        return "av1"
    if c.startswith("prores_"):
        return "prores"
    return "h264"


def codec_container_compatible(codec: str, container: str) -> bool:
    fam = codec_family(codec)
    return {
        "mp4": fam in ("h264", "h265", "av1"),
        "mov": fam in ("h264", "h265", "av1", "prores"),
        "mkv": True,
        "mxf": fam == "prores",
    }.get(container, True)


def recommended_container(codec: str) -> str:
    return "mov" if codec_family(codec) == "prores" else "mp4"


def audio_copy_safe(audio_codec: str | None, container: str) -> bool:
    codec = (audio_codec or "").strip().lower()
    if not codec:
        return True
    if container in ("mp4", "m4v", "m4a"):
        return codec in ("aac", "ac3", "eac3", "mp3", "alac", "opus")
    if container == "mov":
        return codec in (
            "aac", "ac3", "eac3", "mp3", "alac",
            "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be", "pcm_s32le", "pcm_f32le", "qdm2",
        )
    if container in ("mkv", "webm"):
        return True
    return True


def fallback_audio_mode(container: str) -> str:
    if container in ("mp4", "mov", "m4v", "m4a"):
        return "aac"
    if container in ("mkv", "webm"):
        return "copy"
    return "aac"


def container_codec_tag_args(codec: str, ext: str) -> list[str]:
    c = normalize_codec(codec)
    e = ext.lower()
    if e in ("mp4", "mov") and c.startswith("h265_"):
        return ["-tag:v", "hvc1"]
    if e == "mov" and c == "av1_main":
        return ["-tag:v", "av01"]
    return []


def stream_copy_bsf(source_video_codec: str | None, target_container: str) -> str | None:
    """Annex-B bitstream filter needed when stream-copying MP4-style H.264/HEVC
    into MKV/WebM/TS (else Windows Media Foundation decodes green/blue snow)."""
    if target_container in ("mp4", "mov", "m4v", "m4a", "3gp", "3g2"):
        return None
    codec = (source_video_codec or "").strip().lower()
    if codec in ("h264", "avc", "avc1"):
        return "h264_mp4toannexb"
    if codec in ("hevc", "h265", "hvc1", "hev1"):
        return "hevc_mp4toannexb"
    return None


# -- arg builders ----------------------------------------------------------

def _time_str(ms: int) -> str:
    value = f"{ms / 1000.0:.6f}"
    return value.rstrip("0").rstrip(".") or "0"


def _hoisted(audio: str, audio_track: int | None, audio_count: int | None) -> bool:
    """Whether the preview-language track should be moved to first."""
    return (
        audio != "none"
        and audio_track is not None and audio_track > 0
        and audio_count is not None and audio_count > 1
        and audio_track < audio_count
    )


def audio_map_args(audio: str, audio_track: int | None, audio_count: int | None,
                   single: bool = False) -> list[str]:
    """``-map`` args for audio. ``audio_track`` is the user's selected preview
    language as a 0-based audio index; when > 0 with multiple tracks, hoist it to
    first so editors (After Effects) default to it. Otherwise map all in order.
    ``single`` keeps only that one track, for merges whose clips disagree on
    layout - concat needs every segment to carry identical streams."""
    if audio == "none":
        return []
    if single:
        return ["-map", f"0:a:{audio_track or 0}?"]
    if not _hoisted(audio, audio_track, audio_count):
        return ["-map", "0:a?"]
    maps = ["-map", f"0:a:{audio_track}"]
    for i in range(audio_count):  # type: ignore[arg-type]
        if i != audio_track:
            maps += ["-map", f"0:a:{i}"]
    return maps


def disposition_args(audio: str, audio_track: int | None, audio_count: int | None,
                     single: bool = False) -> list[str]:
    """When hoisting, mark the first (hoisted) audio output as the default track
    and clear default on the rest, so editors pick the preview language."""
    if single:
        return []
    if not _hoisted(audio, audio_track, audio_count):
        return []
    args = ["-disposition:a:0", "default"]
    for i in range(1, audio_count):  # type: ignore[arg-type]
        args += [f"-disposition:a:{i}", "0"]
    return args


def build_copy_args(
    inp: str, out: str, audio: str,
    seek_ms: int | None = None, dur_ms: int | None = None, bsf: str | None = None,
    audio_track: int | None = None, audio_count: int | None = None,
    audio_single: bool = False,
) -> list[str]:
    """Stream-copy the whole input (pre-cut clip) or a [seek, seek+dur] range
    (cut from a source episode). Input-side ``-ss`` = fast keyframe seek."""
    ext = Path(out).suffix.lstrip(".").lower()
    args = ["-y"]
    if seek_ms and seek_ms > 0:
        args += ["-ss", _time_str(seek_ms)]
    args += ["-i", inp, "-map", "0:v:0"]
    args += audio_map_args(audio, audio_track, audio_count, audio_single)
    if dur_ms and dur_ms > 0:
        args += ["-t", _time_str(dur_ms)]
    args += ["-c:v", "copy"]
    if audio == "none":
        args += ["-an"]
    elif audio == "copy":
        args += ["-c:a", "copy"]
    else:
        args += audio_args(audio)
    args += disposition_args(audio, audio_track, audio_count, audio_single)
    if bsf:
        args += ["-bsf:v", bsf]
    if ext in ("mp4", "mov"):
        args += ["-movflags", "+faststart"]
    args.append(out)
    return args


def build_reencode_args(
    inp: str, out: str, codec: str, audio: str, use_gpu: bool,
    seek_ms: int | None = None, dur_ms: int | None = None,
    audio_track: int | None = None, audio_count: int | None = None,
    audio_single: bool = False,
) -> list[str]:
    """Re-encode the whole input (pre-cut clip) or a [seek, seek+dur] range."""
    ext = Path(out).suffix.lstrip(".").lower()
    args = ["-y"]
    if seek_ms and seek_ms > 0:
        args += ["-ss", _time_str(seek_ms)]
    args += ["-i", inp, "-map", "0:v:0"]
    args += audio_map_args(audio, audio_track, audio_count, audio_single)
    if dur_ms and dur_ms > 0:
        args += ["-t", _time_str(dur_ms)]
    args += ["-vf", "setpts=PTS-STARTPTS"]
    if audio not in ("none", "copy"):
        # aresample pins the first sample to 0 and fills or trims to keep audio
        # against video. Without it a segment whose audio starts a fraction late
        # keeps that offset, and concatenating segments accumulates the drift.
        args += ["-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0"]
    args += video_encode_args(codec, use_gpu)
    args += audio_args(audio)
    args += disposition_args(audio, audio_track, audio_count, audio_single)
    args += ["-fps_mode:v:0", "cfr"]
    if ext in ("mp4", "mov"):
        args += ["-movflags", "+faststart"]
    args += container_codec_tag_args(codec, ext)
    args += ["-max_muxing_queue_size", "1024", out]
    return args
