"""Core modules - pure logic, no Rich/Typer dependencies.

All public functions and classes are re-exported here for convenience::

    from amverge.core import AmvergeVideo, get_keyframe_timestamps_pyav, make_thumbnail
"""

from .amverge_video import AmvergeVideo
from .binaries import get_binary, get_ffmpeg, get_ffprobe
from .codec_utils import check_if_hevc
from .scene_detector import SceneDetector
from .detection.keyframe import detect_cuts_by_keyframe
from .detection.edge import detect_cuts_by_edge
from .diagnostics import get_gpu_info, get_versions
from .discord_rpc import RPC_AVAILABLE, DiscordRPC
from .hevc import is_hevc
from .image import CropData, crop_image
from .ipc import (
    emit_progress, emit_event, log,
    check_if_path_exists, build_video_cache_prefix,
)
from .keyframe_align import (
    get_keyframe_timestamps_pyav,
    classify_scenes_by_keyframe_alignment,
)
from .keyframes import generate_keyframes
from .probe_utils import (
    probe_video_fps, probe_video_dimensions,
    probe_video_duration, probe_video_total_frames,
)
from .scene_detection import (
    TRANSNET_AVAILABLE,
    decode_and_detect_scenes,
    decode_video_frames_nelux,
    run_model_one_pass,
)
from .scene_utils import (
    scenes_frames_to_seconds,
    convert_scenes_to_timestamps,
    scenes_to_objects,
)
from .segmenter import run_ffmpeg_segment, collect_scenes
from .similarity import check_pair_similar, find_similar_pairs
from .smart_cut import cut_scene, cut_all_scenes
from .thumbnails import make_thumbnail, generate_thumbnails
from .thumbnails_streaming import generate_thumbnails_streaming
from .transnet_constants import (
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS,
    FRAME_BYTES, WINDOW_SIZE, STRIDE,
)
from .video import get_video_duration, get_video_info, merge_short_scenes
