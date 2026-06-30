"""Core modules - pure logic, no Rich/Typer dependencies.

All public functions and classes are re-exported here for convenience::

    from amverge.core import AmvergeVideo, get_keyframe_timestamps_pyav, make_thumbnail
"""

from .wrappers.amverge_video import AmvergeVideo
from .infra.binaries import get_binary, get_ffmpeg, get_ffprobe
from .codec.codec_utils import check_if_hevc, is_hevc
from .wrappers.scene_detector import SceneDetector
from .wrappers.scene_exporter import SceneExporter
from .wrappers.scene_cache import SceneCache
from .wrappers.thumbnail_generator import ThumbnailGenerator
from .wrappers.similarity_checker import SimilarityChecker
from .wrappers.image_crop import ImageCrop
from .wrappers.transnet_config import TransNetConfig
from .detection.keyframe import detect_cuts_by_keyframe
from .detection.edge import detect_cuts_by_edge
from .infra.diagnostics import get_gpu_info, get_versions, check_environment, EnvironmentCheck, CheckResult
from .discord.discord_rpc import RPC_AVAILABLE, DiscordRPC
from .image import CropData, crop_image
from .infra.ipc import (
    emit_progress, emit_event, log,
    check_if_path_exists, build_video_cache_prefix,
)
from .keyframes.keyframe_align import (
    get_keyframe_timestamps_pyav,
    classify_scenes_by_keyframe_alignment,
)
from .keyframes import generate_keyframes
from .video.probe_utils import (
    probe_video_fps, probe_video_dimensions,
    probe_video_duration, probe_video_total_frames,
)
from .detection.ai_scene_detection import (
    TRANSNET_AVAILABLE,
    decode_and_detect_scenes,
    decode_video_frames_nelux,
    run_model_one_pass,
)
from .video.scene_utils import (
    scenes_frames_to_seconds,
    convert_scenes_to_timestamps,
    scenes_to_objects,
)
from .cutting.segmenter import run_ffmpeg_segment, collect_scenes
from .similarity import check_pair_similar, find_similar_pairs
from .cutting.smart_cut import cut_scene, cut_all_scenes
from .thumbnails import make_thumbnail, generate_thumbnails
from .thumbnails.thumbnails_streaming import generate_thumbnails_streaming
from .transnet.transnet_constants import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS,
    FRAME_BYTES, WINDOW_SIZE, STRIDE,
)
from .video import get_video_duration, get_video_info, merge_short_scenes
