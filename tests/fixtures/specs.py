from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

Codec = Literal["h264", "h265", "prores"]
KeyframeLayout = Literal["cuts", "gop", "sparse", "intra"]

COLORS = ("blue", "red")


@dataclass(frozen=True)
class AudioTrack:
    codec: str
    channels: int = 2
    sample_rate: int = 48000
    language: str | None = None
    frequency: int = 440


@dataclass(frozen=True)
class MediaSpec:
    name: str
    codec: Codec
    width: int
    height: int
    fps: str
    segments: tuple[int, ...]
    keyframes: KeyframeLayout
    container: str = "mp4"
    gop: int | None = None
    bframes: int = 0
    open_gop: bool = False
    pix_fmt: str = "yuv420p"
    profile: str | None = None
    audio: tuple[AudioTrack, ...] = field(default_factory=tuple)
    track_timescale: int | None = None

    @property
    def rate(self) -> Fraction:
        return Fraction(self.fps)

    @property
    def pts_tolerance(self) -> float:
        """How far a stored timestamp can sit from its ideal frame time:
        float noise normally, half a tick when the track's timescale rounds
        pts (a 1/1000 timescale stores 24000/1001 frames up to 0.5 ms off)."""
        return 0.5 / self.track_timescale + 1e-6 if self.track_timescale else 1e-4

    @property
    def total_frames(self) -> int:
        return sum(self.segments)

    @property
    def duration(self) -> float:
        return float(self.total_frames / self.rate)

    @property
    def cut_frames(self) -> list[int]:
        out, acc = [], 0
        for n in self.segments[:-1]:
            acc += n
            out.append(acc)
        return out

    @property
    def cut_secs(self) -> list[float]:
        return [self.frame_time(f) for f in self.cut_frames]

    @property
    def scene_frames(self) -> list[tuple[int, int]]:
        bounds = [0, *self.cut_frames, self.total_frames]
        return list(zip(bounds[:-1], bounds[1:]))

    @property
    def scene_secs(self) -> list[tuple[float, float]]:
        return [(self.frame_time(a), self.frame_time(b)) for a, b in self.scene_frames]

    @property
    def keyframe_frames(self) -> list[int]:
        if self.keyframes == "cuts":
            return [0, *self.cut_frames]
        if self.keyframes == "gop":
            return list(range(0, self.total_frames, self.gop))
        if self.keyframes == "sparse":
            return [0]
        return list(range(self.total_frames))

    @property
    def keyframe_secs(self) -> list[float]:
        return [self.frame_time(f) for f in self.keyframe_frames]

    def copy_frames(self, a: int, b: int) -> range:
        start = max(k for k in self.keyframe_frames if k <= a)
        after = [k for k in self.keyframe_frames if k >= b]
        if not after:
            return range(start, self.total_frames)
        return range(start, after[0] + (1 if self.open_gop else 0))

    def frame_time(self, frame: int) -> float:
        return float(frame / self.rate)

    def frame_at(self, seconds: float) -> int:
        return round(seconds * self.rate)

    def segment_color(self, segment: int) -> str:
        return COLORS[segment % len(COLORS)]

    def frame_color(self, frame: int) -> str:
        for i, (a, b) in enumerate(self.scene_frames):
            if a <= frame < b:
                return self.segment_color(i)
        raise IndexError(frame)

    @property
    def extension(self) -> str:
        return self.container


STEREO_AAC = AudioTrack("aac")

FIXTURES: tuple[MediaSpec, ...] = (
    MediaSpec(
        name="h264_24_cuts",
        codec="h264", width=640, height=360, fps="24",
        segments=(48, 36, 60, 30, 48),
        keyframes="cuts", bframes=2,
        audio=(STEREO_AAC,),
    ),
    MediaSpec(
        name="h264_2997_gop",
        codec="h264", width=1280, height=720, fps="30000/1001",
        segments=(45, 90, 20, 75, 60),
        keyframes="gop", gop=30, bframes=3,
        audio=(AudioTrack("aac", channels=1, sample_rate=44100),),
    ),
    MediaSpec(
        name="h264_23976_ms_pts",
        codec="h264", width=640, height=360, fps="24000/1001",
        segments=(40, 36, 80, 40),
        keyframes="cuts", bframes=2,
        audio=(STEREO_AAC,),
        track_timescale=1000,
    ),
    MediaSpec(
        name="h264_25_sparse",
        codec="h264", width=480, height=270, fps="25",
        segments=(40, 40, 40),
        keyframes="sparse",
    ),
    MediaSpec(
        name="h264_30_short",
        codec="h264", width=640, height=360, fps="30",
        segments=(60, 3, 57, 60),
        keyframes="cuts", bframes=2,
        audio=(STEREO_AAC,),
    ),
    MediaSpec(
        name="h265_25_cuts",
        codec="h265", width=960, height=540, fps="25",
        segments=(50, 25, 75, 50),
        keyframes="cuts", bframes=4, profile="main",
        audio=(STEREO_AAC,),
    ),
    MediaSpec(
        name="h265_60_main10_gop",
        codec="h265", width=854, height=480, fps="60",
        segments=(90, 150, 45, 105),
        keyframes="gop", gop=48, bframes=3,
        pix_fmt="yuv420p10le", profile="main10",
        audio=(
            AudioTrack("aac", language="eng", frequency=440),
            AudioTrack("aac", language="jpn", frequency=880),
        ),
    ),
    MediaSpec(
        name="h265_24_opengop",
        codec="h265", width=640, height=360, fps="24",
        segments=(40, 56, 30, 66),
        keyframes="gop", gop=24, bframes=4, open_gop=True, profile="main",
        audio=(STEREO_AAC,),
    ),
    MediaSpec(
        name="prores_23976_422hq",
        codec="prores", width=1920, height=1080, fps="24000/1001",
        segments=(24, 48, 36, 24),
        keyframes="intra", container="mov",
        pix_fmt="yuv422p10le", profile="3",
        audio=(AudioTrack("pcm_s16le"),),
    ),
    MediaSpec(
        name="prores_50_4444",
        codec="prores", width=720, height=576, fps="50",
        segments=(50, 25, 75),
        keyframes="intra", container="mov",
        pix_fmt="yuva444p10le", profile="4",
    ),
)

BY_NAME: dict[str, MediaSpec] = {spec.name: spec for spec in FIXTURES}
