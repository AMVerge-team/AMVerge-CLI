"""AMVerge CLI - scene detection and clip management library.

Usage::

    import amverge
    result = amverge.detect_scenes("episode.mp4")
    for s in result.scenes:
        print(s.index, s.start, s.end, s.path)

    from amverge import make_thumbnail, get_ffmpeg
    make_thumbnail("clip.mp4", "thumb.jpg")
"""

from __future__ import annotations

import importlib
from typing import Any

from .__version__ import __version__

# Attribute -> (submodule, name in that submodule). Resolved on first access
# so that `import amverge` does not pull torch and the ML stack; see __getattr__.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # .__version__
    '__version__': ('.__version__', '__version__'),
    # .pipeline
    'DecodeMethod': ('.pipeline', 'DecodeMethod'),
    'DetectResult': ('.pipeline', 'DetectResult'),
    'DetectionMethod': ('.pipeline', 'DetectionMethod'),
    'Scene': ('.pipeline', 'Scene'),
    'detect_scenes': ('.pipeline', 'detect_scenes'),
    # .core.wrappers.amverge_video
    'AmvergeVideo': ('.core.wrappers.amverge_video', 'AmvergeVideo'),
    # .core.wrappers.scene_detector
    'SceneDetector': ('.core.wrappers.scene_detector', 'SceneDetector'),
    # .core.wrappers.scene_exporter
    'SceneExporter': ('.core.wrappers.scene_exporter', 'SceneExporter'),
    # .core.wrappers.scene_cache
    'SceneCache': ('.core.wrappers.scene_cache', 'SceneCache'),
    # .core.wrappers.thumbnail_generator
    'ThumbnailGenerator': ('.core.wrappers.thumbnail_generator', 'ThumbnailGenerator'),
    # .core.wrappers.similarity_checker
    'SimilarityChecker': ('.core.wrappers.similarity_checker', 'SimilarityChecker'),
    # .core.wrappers.image_crop
    'ImageCrop': ('.core.wrappers.image_crop', 'ImageCrop'),
    # .core.wrappers.transnet_config
    'TransNetConfig': ('.core.wrappers.transnet_config', 'TransNetConfig'),
    # .core.infra.binaries
    'get_binary': ('.core.infra.binaries', 'get_binary'),
    'get_ffmpeg': ('.core.infra.binaries', 'get_ffmpeg'),
    'get_ffprobe': ('.core.infra.binaries', 'get_ffprobe'),
    # .core.video
    'get_video_duration': ('.core.video', 'get_video_duration'),
    'get_video_info': ('.core.video', 'get_video_info'),
    'merge_short_scenes': ('.core.video', 'merge_short_scenes'),
    # .core.video.probe_utils
    'probe_video_dimensions': ('.core.video.probe_utils', 'probe_video_dimensions'),
    'probe_video_duration': ('.core.video.probe_utils', 'probe_video_duration'),
    'probe_video_fps': ('.core.video.probe_utils', 'probe_video_fps'),
    'probe_video_total_frames': ('.core.video.probe_utils', 'probe_video_total_frames'),
    # .core.keyframes
    'generate_keyframes': ('.core.keyframes', 'generate_keyframes'),
    # .core.keyframes.keyframe_align
    'classify_scenes_by_keyframe_alignment': ('.core.keyframes.keyframe_align', 'classify_scenes_by_keyframe_alignment'),
    'get_keyframe_timestamps_pyav': ('.core.keyframes.keyframe_align', 'get_keyframe_timestamps_pyav'),
    # .core.detection.keyframe
    'detect_cuts_by_keyframe': ('.core.detection.keyframe', 'detect_cuts_by_keyframe'),
    # .core.detection.edge
    'detect_cuts_by_edge': ('.core.detection.edge', 'detect_cuts_by_edge'),
    # .core.detection.ai_scene_detection
    'TRANSNET_AVAILABLE': ('.core.detection.ai_scene_detection', 'TRANSNET_AVAILABLE'),
    'decode_and_detect_scenes': ('.core.detection.ai_scene_detection', 'decode_and_detect_scenes'),
    'decode_video_frames_nelux': ('.core.detection.ai_scene_detection', 'decode_video_frames_nelux'),
    'run_model_one_pass': ('.core.detection.ai_scene_detection', 'run_model_one_pass'),
    # .core.detection.nelux_runtime
    'nelux_available': ('.core.detection.nelux_runtime', 'nelux_available'),
    # .core.cutting.smart_cut
    'cut_all_scenes': ('.core.cutting.smart_cut', 'cut_all_scenes'),
    'cut_scene': ('.core.cutting.smart_cut', 'cut_scene'),
    # .core.cutting.segmenter
    'collect_scenes': ('.core.cutting.segmenter', 'collect_scenes'),
    'run_ffmpeg_segment': ('.core.cutting.segmenter', 'run_ffmpeg_segment'),
    # .core.video.scene_utils
    'convert_scenes_to_timestamps': ('.core.video.scene_utils', 'convert_scenes_to_timestamps'),
    'scenes_frames_to_seconds': ('.core.video.scene_utils', 'scenes_frames_to_seconds'),
    'scenes_to_objects': ('.core.video.scene_utils', 'scenes_to_objects'),
    # .core.thumbnails
    'generate_thumbnails': ('.core.thumbnails', 'generate_thumbnails'),
    'make_thumbnail': ('.core.thumbnails', 'make_thumbnail'),
    # .core.similarity
    'check_pair_similar': ('.core.similarity', 'check_pair_similar'),
    'find_similar_pairs': ('.core.similarity', 'find_similar_pairs'),
    # .core.codec.codec_utils
    'AUDIO_FFMPEG': ('.core.codec.codec_utils', 'AUDIO_FFMPEG'),
    'CODEC_ALIASES': ('.core.codec.codec_utils', 'CODEC_ALIASES'),
    'CODEC_PROFILES': ('.core.codec.codec_utils', 'CODEC_PROFILES'),
    'PRORES_CODECS': ('.core.codec.codec_utils', 'PRORES_CODECS'),
    'VALID_AUDIO': ('.core.codec.codec_utils', 'VALID_AUDIO'),
    'VALID_CODECS': ('.core.codec.codec_utils', 'VALID_CODECS'),
    'VALID_CONTAINERS': ('.core.codec.codec_utils', 'VALID_CONTAINERS'),
    'VALID_HARDWARE': ('.core.codec.codec_utils', 'VALID_HARDWARE'),
    'check_if_hevc': ('.core.codec.codec_utils', 'check_if_hevc'),
    'is_hevc': ('.core.codec.codec_utils', 'is_hevc'),
    'resolve_gpu': ('.core.codec.codec_utils', 'resolve_gpu'),
    # .core.image
    'CropData': ('.core.image', 'CropData'),
    'crop_image': ('.core.image', 'crop_image'),
    # .core.infra.diagnostics
    'CheckResult': ('.core.infra.diagnostics', 'CheckResult'),
    'EnvironmentCheck': ('.core.infra.diagnostics', 'EnvironmentCheck'),
    'check_environment': ('.core.infra.diagnostics', 'check_environment'),
    'get_gpu_info': ('.core.infra.diagnostics', 'get_gpu_info'),
    'get_versions': ('.core.infra.diagnostics', 'get_versions'),
    # .core.infra.ipc
    'build_video_cache_prefix': ('.core.infra.ipc', 'build_video_cache_prefix'),
    'check_if_path_exists': ('.core.infra.ipc', 'check_if_path_exists'),
    'emit_event': ('.core.infra.ipc', 'emit_event'),
    'emit_progress': ('.core.infra.ipc', 'emit_progress'),
    'log': ('.core.infra.ipc', 'log'),
    # .core.discord.discord_rpc
    'DiscordRPC': ('.core.discord.discord_rpc', 'DiscordRPC'),
    'RPC_AVAILABLE': ('.core.discord.discord_rpc', 'RPC_AVAILABLE'),
    # .core.transnet.transnet_constants
    'FRAME_BYTES': ('.core.transnet.transnet_constants', 'FRAME_BYTES'),
    'FRAME_CHANNELS': ('.core.transnet.transnet_constants', 'FRAME_CHANNELS'),
    'FRAME_HEIGHT': ('.core.transnet.transnet_constants', 'FRAME_HEIGHT'),
    'FRAME_WIDTH': ('.core.transnet.transnet_constants', 'FRAME_WIDTH'),
    'STRIDE': ('.core.transnet.transnet_constants', 'STRIDE'),
    'WINDOW_SIZE': ('.core.transnet.transnet_constants', 'WINDOW_SIZE'),
    # .core.upscaling
    'ANIME4K_MODE_PRESETS': ('.core.upscaling', 'ANIME4K_MODE_PRESETS'),
    'MODEL_FILES': ('.core.upscaling', 'MODEL_FILES'),
    'SystemMonitor': ('.core.upscaling', 'SystemMonitor'),
    'UPSCALE_AVAILABLE': ('.core.upscaling', 'UPSCALE_AVAILABLE'),
    'UPSCALE_MODEL_KEYS': ('.core.upscaling', 'UPSCALE_MODEL_KEYS'),
    'download_weights': ('.core.upscaling', 'download_weights'),
    'format_eta': ('.core.upscaling', 'format_eta'),
    'get_weight_path': ('.core.upscaling', 'get_weight_path'),
    'is_weight_downloaded': ('.core.upscaling', 'is_weight_downloaded'),
    'load_weights_if_available': ('.core.upscaling', 'load_weights_if_available'),
    'sample_cpu': ('.core.upscaling', 'sample_cpu'),
    'sample_gpu': ('.core.upscaling', 'sample_gpu'),
    'upscale_model': ('.core.upscaling', 'upscale_model'),
    'verify_weight_hash': ('.core.upscaling', 'verify_weight_hash'),
    # .core.upscaling.registry
    'QUALITY_PRESETS': ('.core.upscaling.registry', 'QUALITY_PRESETS'),
    'UPSCALE_REGISTRY': ('.core.upscaling.registry', 'UPSCALE_REGISTRY'),
    'get_all_model_keys': ('.core.upscaling.registry', 'get_all_model_keys'),
    'get_ml_models': ('.core.upscaling.registry', 'get_ml_models'),
    'get_model': ('.core.upscaling.registry', 'get_model'),
    'get_model_credit': ('.core.upscaling.registry', 'get_model_credit'),
    'get_model_scales': ('.core.upscaling.registry', 'get_model_scales'),
    'get_models_by_method': ('.core.upscaling.registry', 'get_models_by_method'),
    'get_onnx_models': ('.core.upscaling.registry', 'get_onnx_models'),
    'get_shader_models': ('.core.upscaling.registry', 'get_shader_models'),
    # .core.upscaling.anime4k
    'download_anime4k_shaders': ('.core.upscaling.anime4k', 'download_anime4k_shaders'),
    'is_anime4k_downloaded': ('.core.upscaling.anime4k', 'is_anime4k_downloaded'),
    'libplacebo_available': ('.core.upscaling.anime4k', 'libplacebo_available'),
    # .core.upscaling.artcnn
    'download_artcnn': ('.core.upscaling.artcnn', 'download_artcnn'),
    'get_artcnn_path': ('.core.upscaling.artcnn', 'get_artcnn_path'),
    'is_artcnn_downloaded': ('.core.upscaling.artcnn', 'is_artcnn_downloaded'),
    # .core.interpolation
    'FLOWFRAMES_VERSION': ('.core.interpolation', 'FLOWFRAMES_VERSION'),
    'INTERPOLATION_REGISTRY': ('.core.interpolation', 'INTERPOLATION_REGISTRY'),
    '_INTERP_AVAILABLE': ('.core.interpolation', 'INTERPOLATION_AVAILABLE'),
    'cancel_flowframes': ('.core.interpolation', 'cancel_flowframes'),
    'download_interp_weights': ('.core.interpolation', 'download_weights'),
    'flowframes_available': ('.core.interpolation', 'flowframes_available'),
    'get_all_interp_model_keys': ('.core.interpolation', 'get_all_model_keys'),
    'get_flowframes_path': ('.core.interpolation', 'get_flowframes_path'),
    'get_interp_model': ('.core.interpolation', 'get_model'),
    'get_interp_model_credit': ('.core.interpolation', 'get_model_credit'),
    'get_interp_weight_path': ('.core.interpolation', 'get_weight_path'),
    'get_pervfi_models': ('.core.interpolation', 'get_pervfi_models'),
    'get_rife_models': ('.core.interpolation', 'get_rife_models'),
    'interpolate_video': ('.core.interpolation', 'interpolate_video'),
    'is_interp_weight_downloaded': ('.core.interpolation', 'is_weight_downloaded'),
    'load_interp_weights_if_available': ('.core.interpolation', 'load_weights_if_available'),
    'run_flowframes': ('.core.interpolation', 'run_flowframes'),
    'set_flowframes_path': ('.core.interpolation', 'set_flowframes_path'),
    'verify_interp_weight_hash': ('.core.interpolation', 'verify_weight_hash'),
    # .core.depth
    'COLMAPS': ('.core.depth', 'COLMAPS'),
    'DEPTH_AVAILABLE': ('.core.depth', 'DEPTH_AVAILABLE'),
    'MODEL_CONFIGS': ('.core.depth', 'MODEL_CONFIGS'),
    'download_model': ('.core.depth', 'download_model'),
    'generate_depth_map': ('.core.depth', 'generate_depth_map'),
    'is_model_downloaded': ('.core.depth', 'is_model_downloaded'),
}

# Names whose import is optional: on ImportError they degrade to these
# values instead of propagating, matching the old try/except blocks.
_OPTIONAL_FALLBACKS: dict[str, Any] = {
    'ANIME4K_MODE_PRESETS': {},
    'COLMAPS': {},
    'DEPTH_AVAILABLE': False,
    'MODEL_CONFIGS': {},
    'MODEL_FILES': {},
    'SystemMonitor': None,
    'UPSCALE_AVAILABLE': False,
    'UPSCALE_MODEL_KEYS': [],
    'download_model': None,
    'download_weights': None,
    'format_eta': lambda s: "--:--",
    'generate_depth_map': None,
    'get_weight_path': None,
    'is_model_downloaded': None,
    'is_weight_downloaded': None,
    'load_weights_if_available': None,
    'sample_cpu': None,
    'sample_gpu': None,
    'upscale_model': None,
    'verify_weight_hash': None,
}


# Subpackages that used to become attributes of `amverge` as a side effect of
# the eager `from .core... import ...` lines. `import amverge; amverge.core...`
# is a normal way to use the library, so resolve them on demand.
_LAZY_SUBMODULES: frozenset[str] = frozenset({
    'core',
})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        value = importlib.import_module(f'.{name}', __name__)
        globals()[name] = value
        return value
    try:
        module_name, orig_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    try:
        value = getattr(importlib.import_module(module_name, __name__), orig_name)
    except (ImportError, AttributeError):
        if name not in _OPTIONAL_FALLBACKS:
            raise
        value = _OPTIONAL_FALLBACKS[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "__version__",
    # Pipeline
    "detect_scenes", "DetectResult", "Scene", "DetectionMethod", "DecodeMethod",
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
    "decode_video_frames_nelux", "run_model_one_pass", "nelux_available",
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
    "MODEL_FILES", "upscale_model",
    "download_weights", "is_weight_downloaded", "get_weight_path",
    "verify_weight_hash", "load_weights_if_available",
    "ANIME4K_MODE_PRESETS",
    "SystemMonitor", "sample_gpu", "sample_cpu", "format_eta",
    "UPSCALE_REGISTRY", "get_model", "get_models_by_method",
    "get_ml_models", "get_shader_models", "get_onnx_models",
    "get_all_model_keys", "get_model_scales", "get_model_credit",
    "download_anime4k_shaders", "is_anime4k_downloaded", "libplacebo_available",
    "download_artcnn", "is_artcnn_downloaded", "get_artcnn_path",
    # Interpolation
    "flowframes_available", "run_flowframes", "cancel_flowframes",
    "set_flowframes_path", "get_flowframes_path", "FLOWFRAMES_VERSION",
    "INTERPOLATION_REGISTRY", "interpolate_video",
    "get_interp_model", "get_rife_models", "get_all_interp_model_keys",
    "get_interp_model_credit", "download_interp_weights",
    "is_interp_weight_downloaded", "get_interp_weight_path",
    "verify_interp_weight_hash", "load_interp_weights_if_available",
    # Depth
    "DEPTH_AVAILABLE", "generate_depth_map",
    "download_model", "is_model_downloaded",
    "MODEL_CONFIGS", "COLMAPS",
]
