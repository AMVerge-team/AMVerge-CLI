"""Core modules - pure logic, no Rich/Typer dependencies.

All public functions and classes are re-exported here for convenience::

    from amverge.core import AmvergeVideo, get_keyframe_timestamps_pyav, make_thumbnail
"""

from __future__ import annotations

import importlib
from typing import Any

# Attribute -> (submodule, name in that submodule). Resolved on first access so
# that importing any amverge.core.* leaf does not pull torch via this package.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # .wrappers.amverge_video
    'AmvergeVideo': ('.wrappers.amverge_video', 'AmvergeVideo'),
    # .infra.binaries
    'get_binary': ('.infra.binaries', 'get_binary'),
    'get_ffmpeg': ('.infra.binaries', 'get_ffmpeg'),
    'get_ffprobe': ('.infra.binaries', 'get_ffprobe'),
    # .codec.codec_utils
    'check_if_hevc': ('.codec.codec_utils', 'check_if_hevc'),
    'is_hevc': ('.codec.codec_utils', 'is_hevc'),
    # .wrappers.scene_detector
    'SceneDetector': ('.wrappers.scene_detector', 'SceneDetector'),
    # .wrappers.scene_exporter
    'SceneExporter': ('.wrappers.scene_exporter', 'SceneExporter'),
    # .wrappers.scene_cache
    'SceneCache': ('.wrappers.scene_cache', 'SceneCache'),
    # .wrappers.thumbnail_generator
    'ThumbnailGenerator': ('.wrappers.thumbnail_generator', 'ThumbnailGenerator'),
    # .wrappers.similarity_checker
    'SimilarityChecker': ('.wrappers.similarity_checker', 'SimilarityChecker'),
    # .wrappers.image_crop
    'ImageCrop': ('.wrappers.image_crop', 'ImageCrop'),
    # .wrappers.transnet_config
    'TransNetConfig': ('.wrappers.transnet_config', 'TransNetConfig'),
    # .detection.keyframe
    'detect_cuts_by_keyframe': ('.detection.keyframe', 'detect_cuts_by_keyframe'),
    # .detection.edge
    'detect_cuts_by_edge': ('.detection.edge', 'detect_cuts_by_edge'),
    # .infra.diagnostics
    'CheckResult': ('.infra.diagnostics', 'CheckResult'),
    'EnvironmentCheck': ('.infra.diagnostics', 'EnvironmentCheck'),
    'check_environment': ('.infra.diagnostics', 'check_environment'),
    'get_gpu_info': ('.infra.diagnostics', 'get_gpu_info'),
    'get_versions': ('.infra.diagnostics', 'get_versions'),
    # .discord.discord_rpc
    'DiscordRPC': ('.discord.discord_rpc', 'DiscordRPC'),
    'RPC_AVAILABLE': ('.discord.discord_rpc', 'RPC_AVAILABLE'),
    # .image
    'CropData': ('.image', 'CropData'),
    'crop_image': ('.image', 'crop_image'),
    # .infra.ipc
    'build_video_cache_prefix': ('.infra.ipc', 'build_video_cache_prefix'),
    'check_if_path_exists': ('.infra.ipc', 'check_if_path_exists'),
    'emit_event': ('.infra.ipc', 'emit_event'),
    'emit_progress': ('.infra.ipc', 'emit_progress'),
    'log': ('.infra.ipc', 'log'),
    # .keyframes.keyframe_align
    'classify_scenes_by_keyframe_alignment': ('.keyframes.keyframe_align', 'classify_scenes_by_keyframe_alignment'),
    'get_keyframe_timestamps_pyav': ('.keyframes.keyframe_align', 'get_keyframe_timestamps_pyav'),
    # .keyframes
    'generate_keyframes': ('.keyframes', 'generate_keyframes'),
    # .video.probe_utils
    'probe_video_dimensions': ('.video.probe_utils', 'probe_video_dimensions'),
    'probe_video_duration': ('.video.probe_utils', 'probe_video_duration'),
    'probe_video_fps': ('.video.probe_utils', 'probe_video_fps'),
    'probe_video_total_frames': ('.video.probe_utils', 'probe_video_total_frames'),
    # .detection.ai_scene_detection
    'TRANSNET_AVAILABLE': ('.detection.ai_scene_detection', 'TRANSNET_AVAILABLE'),
    'decode_and_detect_scenes': ('.detection.ai_scene_detection', 'decode_and_detect_scenes'),
    'decode_video_frames_nelux': ('.detection.ai_scene_detection', 'decode_video_frames_nelux'),
    'run_model_one_pass': ('.detection.ai_scene_detection', 'run_model_one_pass'),
    # .video.scene_utils
    'convert_scenes_to_timestamps': ('.video.scene_utils', 'convert_scenes_to_timestamps'),
    'scenes_frames_to_seconds': ('.video.scene_utils', 'scenes_frames_to_seconds'),
    'scenes_to_objects': ('.video.scene_utils', 'scenes_to_objects'),
    # .cutting.segmenter
    'collect_scenes': ('.cutting.segmenter', 'collect_scenes'),
    'run_ffmpeg_segment': ('.cutting.segmenter', 'run_ffmpeg_segment'),
    # .similarity
    'check_pair_similar': ('.similarity', 'check_pair_similar'),
    'find_similar_pairs': ('.similarity', 'find_similar_pairs'),
    # .cutting.smart_cut
    'cut_all_scenes': ('.cutting.smart_cut', 'cut_all_scenes'),
    'cut_scene': ('.cutting.smart_cut', 'cut_scene'),
    # .thumbnails
    'generate_thumbnails': ('.thumbnails', 'generate_thumbnails'),
    'make_thumbnail': ('.thumbnails', 'make_thumbnail'),
    # .thumbnails.thumbnails_streaming
    'generate_thumbnails_streaming': ('.thumbnails.thumbnails_streaming', 'generate_thumbnails_streaming'),
    # .transnet.transnet_constants
    'FRAME_BYTES': ('.transnet.transnet_constants', 'FRAME_BYTES'),
    'FRAME_CHANNELS': ('.transnet.transnet_constants', 'FRAME_CHANNELS'),
    'FRAME_HEIGHT': ('.transnet.transnet_constants', 'FRAME_HEIGHT'),
    'FRAME_WIDTH': ('.transnet.transnet_constants', 'FRAME_WIDTH'),
    'STRIDE': ('.transnet.transnet_constants', 'STRIDE'),
    'WINDOW_SIZE': ('.transnet.transnet_constants', 'WINDOW_SIZE'),
    # .video
    'get_video_duration': ('.video', 'get_video_duration'),
    'get_video_info': ('.video', 'get_video_info'),
    'merge_short_scenes': ('.video', 'merge_short_scenes'),
}

# Bound as attributes by the eager imports this module used to perform; kept
# reachable so `amverge.core.video` still works after `import amverge.core`.
_LAZY_SUBMODULES: frozenset[str] = frozenset({
    'codec',
    'cutting',
    # depth/interpolation/upscaling pull an ML stack, but they must still be
    # reachable as attributes (`amverge.core.upscaling`) the way they were when
    # this package imported them eagerly. Listing them here keeps that working
    # while still deferring the import until someone actually asks.
    'depth',
    'detection',
    'discord',
    'image',
    'infra',
    'interpolation',
    'keyframes',
    'similarity',
    'thumbnails',
    'transnet',
    'upscaling',
    'video',
    'wrappers',
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
    value = getattr(importlib.import_module(module_name, __name__), orig_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


# Mirrors the names the previous eager imports left on this module, so that
# `from amverge.core import *` still yields exactly the same surface.
__all__ = [
    'AmvergeVideo',
    'CheckResult',
    'CropData',
    'DiscordRPC',
    'EnvironmentCheck',
    'FRAME_BYTES',
    'FRAME_CHANNELS',
    'FRAME_HEIGHT',
    'FRAME_WIDTH',
    'ImageCrop',
    'RPC_AVAILABLE',
    'STRIDE',
    'SceneCache',
    'SceneDetector',
    'SceneExporter',
    'SimilarityChecker',
    'TRANSNET_AVAILABLE',
    'ThumbnailGenerator',
    'TransNetConfig',
    'WINDOW_SIZE',
    'build_video_cache_prefix',
    'check_environment',
    'check_if_hevc',
    'check_if_path_exists',
    'check_pair_similar',
    'classify_scenes_by_keyframe_alignment',
    'codec',
    'collect_scenes',
    'convert_scenes_to_timestamps',
    'crop_image',
    'cut_all_scenes',
    'cut_scene',
    'cutting',
    'decode_and_detect_scenes',
    'decode_video_frames_nelux',
    'depth',
    'detect_cuts_by_edge',
    'detect_cuts_by_keyframe',
    'detection',
    'discord',
    'emit_event',
    'emit_progress',
    'find_similar_pairs',
    'generate_keyframes',
    'generate_thumbnails',
    'generate_thumbnails_streaming',
    'get_binary',
    'get_ffmpeg',
    'get_ffprobe',
    'get_gpu_info',
    'get_keyframe_timestamps_pyav',
    'get_versions',
    'get_video_duration',
    'get_video_info',
    'image',
    'infra',
    'interpolation',
    'is_hevc',
    'keyframes',
    'log',
    'make_thumbnail',
    'merge_short_scenes',
    'probe_video_dimensions',
    'probe_video_duration',
    'probe_video_fps',
    'probe_video_total_frames',
    'run_ffmpeg_segment',
    'run_model_one_pass',
    'scenes_frames_to_seconds',
    'scenes_to_objects',
    'similarity',
    'thumbnails',
    'transnet',
    'upscaling',
    'video',
    'wrappers',
]
