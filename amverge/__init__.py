"""AMVerge CLI - scene detection and clip management library.

Usage::

    import amverge
    result = amverge.detect_scenes("episode.mp4")
    for s in result.scenes:
        print(s.index, s.start, s.end, s.path)

    from amverge import make_thumbnail, get_ffmpeg
    make_thumbnail("clip.mp4", "thumb.jpg")
"""

from .__version__ import __version__

# -- High-level pipeline ------------------------------------------------
from .pipeline import detect_scenes, DetectResult, Scene, DetectionMethod

# -- Video object -------------------------------------------------------
from .core.wrappers.amverge_video import AmvergeVideo

# -- Scene detector -----------------------------------------------------
from .core.wrappers.scene_detector import SceneDetector

# -- Scene exporter -----------------------------------------------------
from .core.wrappers.scene_exporter import SceneExporter

# -- Scene cache --------------------------------------------------------
from .core.wrappers.scene_cache import SceneCache

# -- Thumbnail generator ------------------------------------------------
from .core.wrappers.thumbnail_generator import ThumbnailGenerator

# -- Similarity checker -------------------------------------------------
from .core.wrappers.similarity_checker import SimilarityChecker

# -- Image crop ----------------------------------------------------------
from .core.wrappers.image_crop import ImageCrop

# -- TransNetV2 config ---------------------------------------------------
from .core.wrappers.transnet_config import TransNetConfig

# -- Binaries -----------------------------------------------------------
from .core.infra.binaries import get_binary, get_ffmpeg, get_ffprobe

# -- Video metadata -----------------------------------------------------
from .core.video import get_video_duration, get_video_info, merge_short_scenes
from .core.video.probe_utils import (
    probe_video_fps,
    probe_video_dimensions,
    probe_video_duration,
    probe_video_total_frames,
)

# -- Keyframe extraction ------------------------------------------------
from .core.keyframes import generate_keyframes
from .core.keyframes.keyframe_align import (
    get_keyframe_timestamps_pyav,
    classify_scenes_by_keyframe_alignment,
)

# -- Scene detection (V1) -----------------------------------------------
from .core.detection.keyframe import detect_cuts_by_keyframe
from .core.detection.edge import detect_cuts_by_edge

# -- Scene detection (V2 TransNetV2) ------------------------------------
from .core.detection.scene_detection import (
    TRANSNET_AVAILABLE,
    decode_and_detect_scenes,
    decode_video_frames_nelux,
    run_model_one_pass,
)

# -- Scene cutting ------------------------------------------------------
from .core.cutting.smart_cut import cut_scene, cut_all_scenes
from .core.cutting.segmenter import run_ffmpeg_segment, collect_scenes

# -- Scene utilities ----------------------------------------------------
from .core.video.scene_utils import (
    scenes_frames_to_seconds,
    convert_scenes_to_timestamps,
    scenes_to_objects,
)

# -- Thumbnails ---------------------------------------------------------
from .core.thumbnails import make_thumbnail, generate_thumbnails

# -- Similarity ---------------------------------------------------------
from .core.similarity import check_pair_similar, find_similar_pairs

# -- Codec detection ----------------------------------------------------
from .core.codec.codec_utils import (
    check_if_hevc,
    is_hevc,
    VALID_CODECS, VALID_AUDIO, VALID_CONTAINERS, VALID_HARDWARE,
    CODEC_ALIASES, CODEC_PROFILES, PRORES_CODECS, AUDIO_FFMPEG,
    resolve_gpu,
)

# -- Image --------------------------------------------------------------
from .core.image import CropData, crop_image

# -- Diagnostics -------------------------------------------------------
from .core.infra.diagnostics import get_gpu_info, get_versions, check_environment, EnvironmentCheck, CheckResult

# -- IPC ----------------------------------------------------------------
from .core.infra.ipc import (
    emit_progress,
    emit_event,
    log,
    check_if_path_exists,
    build_video_cache_prefix,
)

# -- Discord RPC --------------------------------------------------------
from .core.discord.discord_rpc import RPC_AVAILABLE, DiscordRPC

# -- TransNetV2 constants -----------------------------------------------
from .core.transnet.transnet_constants import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FRAME_CHANNELS,
    FRAME_BYTES,
    WINDOW_SIZE,
    STRIDE,
)

# -- Upscaling -----------------------------------------------------------
try:
    from .core.upscaling import (
        UPSCALE_AVAILABLE,
        UPSCALE_MODEL_KEYS,
        MODEL_FILES,
        upscale_video,
        download_weights,
        is_weight_downloaded,
        get_weight_path,
        verify_weight_hash,
        load_weights_if_available,
    )
except ImportError:
    UPSCALE_AVAILABLE = False
    UPSCALE_MODEL_KEYS = []
    MODEL_FILES = {}
    upscale_video = None
    download_weights = None
    is_weight_downloaded = None
    get_weight_path = None
    verify_weight_hash = None
    load_weights_if_available = None

from .core.upscaling.anime4k import (
    upscale_video_anime4k,
    ANIME4K_MODE_PRESETS,
    ANIME4K_SHADER_FILES,
)

from .core.upscaling.artcnn import (
    upscale_video_artcnn,
    ARTCNN_MODELS,
)

from .core.upscaling.registry import (
    UPSCALE_REGISTRY,
    get_model,
    get_models_by_method,
    get_ml_models,
    get_shader_models,
    get_onnx_models,
    get_all_model_keys,
    get_model_scales,
    get_model_credit,
)

__all__ = [
    "__version__",
    # Pipeline
    "detect_scenes", "DetectResult", "Scene", "DetectionMethod",
    # Video object
    "AmvergeVideo",
    # Scene detector
    "SceneDetector",
    # Scene exporter
    "SceneExporter",
    # Scene cache
    "SceneCache",
    # Thumbnail generator
    "ThumbnailGenerator",
    # Similarity checker
    "SimilarityChecker",
    # Image crop
    "ImageCrop",
    # TransNetV2 config
    "TransNetConfig",
    # Binaries
    "get_binary", "get_ffmpeg", "get_ffprobe",
    # Video
    "get_video_duration", "get_video_info", "merge_short_scenes",
    "probe_video_fps", "probe_video_dimensions",
    "probe_video_duration", "probe_video_total_frames",
    # Keyframes
    "generate_keyframes",
    "get_keyframe_timestamps_pyav", "classify_scenes_by_keyframe_alignment",
    # Scene detection V1
    "detect_cuts_by_keyframe", "detect_cuts_by_edge",
    # Scene detection V2
    "TRANSNET_AVAILABLE", "decode_and_detect_scenes",
    "decode_video_frames_nelux", "run_model_one_pass",
    # Scene cutting
    "cut_scene", "cut_all_scenes",
    "run_ffmpeg_segment", "collect_scenes",
    # Scene utils
    "scenes_frames_to_seconds", "convert_scenes_to_timestamps",
    "scenes_to_objects",
    # Thumbnails
    "make_thumbnail", "generate_thumbnails",
    # Similarity
    "check_pair_similar", "find_similar_pairs",
    # Codec
    "check_if_hevc", "is_hevc",
    "VALID_CODECS", "VALID_AUDIO", "VALID_CONTAINERS", "VALID_HARDWARE",
    "CODEC_ALIASES", "CODEC_PROFILES", "PRORES_CODECS", "AUDIO_FFMPEG",
    "resolve_gpu",
    # Image
    "CropData", "crop_image",
    # Diagnostics
    "get_gpu_info", "get_versions",
    "check_environment", "EnvironmentCheck", "CheckResult",
    # IPC
    "emit_progress", "emit_event", "log",
    "check_if_path_exists", "build_video_cache_prefix",
    # Discord RPC
    "RPC_AVAILABLE", "DiscordRPC",
    # TransNetV2 constants
    "FRAME_WIDTH", "FRAME_HEIGHT", "FRAME_CHANNELS",
    "FRAME_BYTES", "WINDOW_SIZE", "STRIDE",
    # Upscaling
    "UPSCALE_AVAILABLE", "QUALITY_PRESETS", "UPSCALE_MODEL_KEYS",
    "MODEL_FILES", "upscale_video",
    "download_weights", "is_weight_downloaded", "get_weight_path",
    "verify_weight_hash", "load_weights_if_available",
    "upscale_video_anime4k", "ANIME4K_MODE_PRESETS", "ANIME4K_SHADER_FILES",
    "upscale_video_artcnn", "ARTCNN_MODELS",
    "UPSCALE_REGISTRY", "get_model", "get_models_by_method",
    "get_ml_models", "get_shader_models", "get_onnx_models",
    "get_all_model_keys", "get_model_scales", "get_model_credit",
]
