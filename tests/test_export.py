from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess

import pytest

from amverge.core.cutting.smart_cut import cut_scene
from amverge.core.keyframes.keyframe_align import get_keyframe_timestamps_pyav
from amverge.core.export import ExportJob, ExportSettings, export_scenes
from amverge.core.export.engine import CopyContainerError, ExportError, check_copy_container
from amverge.core.infra.binaries import get_ffmpeg, get_ffprobe
from amverge.core.export import params
from amverge.core.video.probe_utils import probe_seek_origin
from tests.fixtures import BY_NAME, FIXTURES, media_params
from tests.fixtures.media import audio_streams, packet_durations, read_frames, video_stream

pytestmark = pytest.mark.media


PROFILES = {
    "h264_main": ("h264", "Main", "yuv420p"),
    "h264_high": ("h264", "High", "yuv420p"),
    "h264_high10": ("h264", "High 10", "yuv420p10le"),
    "h264_high422": ("h264", "High 4:2:2", "yuv422p"),
    "h265_main": ("hevc", "Main", "yuv420p"),
    "h265_main10": ("hevc", "Main 10", "yuv420p10le"),
    "h265_main12": ("hevc", "Rext", "yuv420p12le"),
    "h265_main422_10": ("hevc", "Rext", "yuv422p10le"),
    "av1_main": ("av1", "Main", None),
    "prores_422_lt": ("prores", "LT", "yuv422p10le"),
    "prores_422": ("prores", "Standard", "yuv422p10le"),
    "prores_422_hq": ("prores", "HQ", "yuv422p10le"),
    "prores_4444": ("prores", "4444", None),
    "prores_4444_xq": ("prores", "XQ", None),
}

AUDIO_CODECS = {
    "aac": "aac", "aac_320": "aac", "pcm16": "pcm_s16le", "pcm24": "pcm_s24le",
    "flac": "flac", "alac": "alac", "opus": "opus", "mp3": "mp3",
}


def range_jobs(spec, path, indices) -> list[ExportJob]:
    jobs = []
    for i in indices:
        start, end = spec.scene_secs[i]
        jobs.append(ExportJob(
            scene_index=i, input=str(path),
            seek_ms=int(round(start * 1000)), dur_ms=int(round((end - start) * 1000)),
        ))
    return jobs


def run_export(tmp_path, jobs, **settings) -> list[str]:
    events: list[str] = []
    outputs = export_scenes(
        jobs, str(tmp_path / "out"), "clip", ExportSettings(hardware="cpu", **settings),
        on_event=events.append,
    )
    run_export.events = events
    return outputs


def numbers(path) -> list[int]:
    return [f.number for f in read_frames(path)]


def expected_frames(spec, indices) -> list[int]:
    return [n for i in indices for n in range(*spec.scene_frames[i])]


def assert_av_in_sync(path, tolerance: float = 0.05) -> None:
    spans = packet_durations(path)
    (video,) = spans["video"]
    for audio in spans["audio"]:
        assert audio == pytest.approx(video, abs=tolerance)


class TestReencodeCodecs:
    @pytest.mark.parametrize("codec", sorted(PROFILES))
    def test_codec_profile_and_exact_frames(self, media_files, tmp_path, encoders, codec):
        encoder = params.VIDEO_PARAMS[codec]["cpu"][0]
        if encoder not in encoders:
            pytest.skip(f"{encoder} not in this ffmpeg build")
        spec = BY_NAME["h264_24_cuts"]
        container = params.recommended_container(codec)
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [1, 3]),
            codec=codec, container=container, audio="aac",
        )
        assert [Path(p).name for p in outputs] == [f"clip_0001.{container}", f"clip_0003.{container}"]
        name, profile, pix_fmt = PROFILES[codec]
        for out, idx in zip(outputs, [1, 3]):
            v = video_stream(out)
            assert v["codec_name"] == name
            assert v.get("profile") == profile
            if pix_fmt:
                assert v["pix_fmt"] == pix_fmt
            assert Fraction(v["avg_frame_rate"]) == spec.rate
            assert numbers(out) == list(range(*spec.scene_frames[idx]))
            assert_av_in_sync(out)

    @pytest.mark.parametrize("spec", media_params())
    def test_every_source_reencodes_frame_exact(self, media_files, tmp_path, spec):
        indices = list(range(len(spec.segments)))
        outputs = run_export(tmp_path, range_jobs(spec, media_files[spec.name], indices), codec="h264_high")
        for out, idx in zip(outputs, indices):
            assert numbers(out) == list(range(*spec.scene_frames[idx])), f"scene {idx}"
            assert Fraction(video_stream(out)["avg_frame_rate"]) == spec.rate

    def test_parallel_workers_match_serial(self, media_files, tmp_path):
        spec = BY_NAME["h265_25_cuts"]
        jobs = range_jobs(spec, media_files[spec.name], range(4))
        serial = run_export(tmp_path / "a", jobs, codec="h264_main", workers=1)
        parallel = run_export(tmp_path / "b", jobs, codec="h264_main", workers=3)
        assert [Path(p).name for p in serial] == [Path(p).name for p in parallel]
        assert [numbers(p) for p in serial] == [numbers(p) for p in parallel]
        assert sorted(run_export.events) == sorted(
            f"CLIP_READY|{i}|{p}|reencode" for i, p in enumerate(parallel)
        )

    def test_hash_placeholder_in_stem(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        outputs = export_scenes(
            range_jobs(spec, media_files[spec.name], [2]), str(tmp_path), "ep01_####_final",
            ExportSettings(codec="h264_main", hardware="cpu"),
        )
        assert Path(outputs[0]).name == "ep01_0002_final.mp4"


class TestBitDepth:
    """Every 8-bit and 10-bit source re-encodes to the target profile's bit
    depth, frame-exact, whichever depth it started at; av1_main pins no
    pixel format, so it keeps the source's depth."""

    DEPTH_SOURCES = [
        s.name for s in FIXTURES if s.codec != "prores"
    ]

    @pytest.mark.parametrize("spec", media_params(*DEPTH_SOURCES))
    @pytest.mark.parametrize("codec", ["h264_high", "h264_high10", "h265_main", "h265_main10", "av1_main"])
    def test_reencode_to_profile_depth(self, media_files, tmp_path, encoders, spec, codec):
        encoder = params.VIDEO_PARAMS[codec]["cpu"][0]
        if encoder not in encoders:
            pytest.skip(f"{encoder} not in this ffmpeg build")
        idx = 1
        (out,) = run_export(tmp_path, range_jobs(spec, media_files[spec.name], [idx]), codec=codec)
        v = video_stream(out)
        expected = PROFILES[codec][2] or ("yuv420p10le" if spec.bit_depth == 10 else "yuv420p")
        assert v["pix_fmt"] == expected
        assert numbers(out) == list(range(*spec.scene_frames[idx]))


class TestAudioModes:
    @pytest.mark.parametrize("audio,container", [
        ("aac", "mp4"), ("aac_320", "mp4"), ("mp3", "mp4"), ("opus", "mp4"), ("flac", "mp4"),
        ("alac", "mov"), ("pcm16", "mov"), ("pcm24", "mov"),
    ])
    def test_audio_codec_applied(self, media_files, tmp_path, audio, container):
        spec = BY_NAME["h264_24_cuts"]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [2]),
            codec="h264_high", audio=audio, container=container,
        )
        streams = audio_streams(outputs[0])
        assert [s["codec_name"] for s in streams] == [AUDIO_CODECS[audio]]
        assert_av_in_sync(outputs[0])

    def test_audio_none_strips_audio(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        for codec in ("copy", "h264_high"):
            outputs = run_export(
                tmp_path / codec, range_jobs(spec, media_files[spec.name], [1]), codec=codec, audio="none",
            )
            assert audio_streams(outputs[0]) == []

    def test_audio_copy_keeps_source_codec(self, media_files, tmp_path):
        spec = BY_NAME["prores_23976_422hq"]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [1]),
            codec="prores_422_hq", container="mov", audio="copy",
        )
        assert [s["codec_name"] for s in audio_streams(outputs[0])] == ["pcm_s16le"]

    def test_source_without_audio(self, media_files, tmp_path):
        spec = BY_NAME["prores_50_4444"]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [0, 2]),
            codec="h265_main", audio="aac", merge=True,
        )
        assert audio_streams(outputs[0]) == []
        assert numbers(outputs[0]) == expected_frames(spec, [0, 2])

    @pytest.mark.parametrize("codec", ["copy", "h264_high"])
    def test_language_hoisted_to_first_track(self, media_files, tmp_path, codec):
        spec = BY_NAME["h265_60_main10_gop"]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [1]),
            codec=codec, audio="aac" if codec != "copy" else "copy", audio_language="jpn",
        )
        streams = audio_streams(outputs[0])
        assert [s["tags"]["language"] for s in streams] == ["jpn", "eng"]
        assert streams[0]["disposition"]["default"] == 1
        assert streams[1]["disposition"]["default"] == 0

    def test_audio_track_index_hoisted(self, media_files, tmp_path):
        spec = BY_NAME["h265_60_main10_gop"]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [0]),
            codec="h264_main", audio="aac", audio_track=1,
        )
        assert [s["tags"]["language"] for s in audio_streams(outputs[0])] == ["jpn", "eng"]

    def test_merge_keeps_language_order(self, media_files, tmp_path):
        spec = BY_NAME["h265_60_main10_gop"]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], [0, 2]),
            codec="h265_main10", audio="aac", audio_language="jpn", merge=True,
        )
        assert [s["tags"]["language"] for s in audio_streams(outputs[0])] == ["jpn", "eng"]
        assert_av_in_sync(outputs[0])


class TestCopyExport:
    @pytest.mark.parametrize(
        ("video_spec", "audio_codec"),
        [
            (video, audio)
            for video, targets in {
                "h264_24_cuts": {"avi": True, "mp4": True, "mov": True},
                "h265_25_cuts": {"avi": True, "mp4": True, "mov": True},
                "prores_23976_422hq": {"avi": True, "mp4": False, "mov": True},
            }.items()
            for audio in ("aac", "pcm_s16le", "libopus")
        ],
    )
    def test_copy_preflight_checks_ffmpeg_muxer_matrix(
        self, media_files, tmp_path, video_spec, audio_codec,
    ):
        source = media_files[BY_NAME[video_spec].name]
        input_path = tmp_path / f"{video_spec}-{audio_codec}.mkv"
        result = subprocess.run(
            [
                get_ffmpeg(), "-v", "error", "-y", "-i", str(source), "-f", "lavfi", "-i",
                "sine=frequency=500:sample_rate=48000", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-c:v", "copy", "-c:a", audio_codec, str(input_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode:
            pytest.skip(f"{audio_codec} encoder unavailable: {result.stderr}")
        targets = {"avi": True, "mp4": True, "mov": True}
        if video_spec == "prores_23976_422hq":
            targets["mp4"] = False
        if audio_codec == "pcm_s16le":
            targets["mp4"] = False
        if audio_codec == "libopus":
            targets = {container: accepted and container == "mp4" for container, accepted in targets.items()}
        for container, accepted in targets.items():
            settings = ExportSettings(codec="copy", audio="copy", container=container)
            if accepted:
                check_copy_container([ExportJob(scene_index=0, input=str(input_path))], settings,
                                     ffmpeg=get_ffmpeg(), ffprobe=get_ffprobe())
            else:
                with pytest.raises(CopyContainerError) as error:
                    check_copy_container([ExportJob(scene_index=0, input=str(input_path))], settings,
                                         ffmpeg=get_ffmpeg(), ffprobe=get_ffprobe())
                assert error.value.details["code"] == "copy_container_incompatible"

    @pytest.mark.parametrize("spec", media_params())
    def test_individual_copy_is_keyframe_snapped(self, media_files, tmp_path, spec):
        indices = list(range(len(spec.segments)))
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], indices),
            codec="copy", container=spec.export_container,
        )
        source = video_stream(media_files[spec.name])
        for out, idx in zip(outputs, indices):
            assert numbers(out) == list(spec.copy_frames(*spec.scene_frames[idx])), f"scene {idx}"
            v = video_stream(out)
            assert (v["codec_name"], v.get("profile"), v["pix_fmt"]) == (
                source["codec_name"], source.get("profile"), source["pix_fmt"]
            )
        assert all(e.endswith("|copy") for e in run_export.events)


    @pytest.mark.parametrize("merge", [False, True])
    def test_rejects_container_that_cannot_hold_the_video(self, media_files, tmp_path, merge):
        spec = BY_NAME["prores_23976_422hq"]
        with pytest.raises(ValueError, match="Cannot stream-copy the video .* into mp4"):
            run_export(tmp_path, range_jobs(spec, media_files[spec.name], [0, 2]),
                       codec="copy", container="mp4", merge=merge)
        assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())

    def test_rejects_copied_audio_the_container_cannot_hold(self, media_files, tmp_path):
        spec = BY_NAME["h264_2997_pcm_mov"]
        with pytest.raises(ValueError, match="Cannot stream-copy the audio"):
            run_export(tmp_path, range_jobs(spec, media_files[spec.name], [1]), codec="copy", container="mp4")

    def test_pcm_source_copies_video_into_mp4_with_converted_audio(self, media_files, tmp_path):
        spec = BY_NAME["h264_2997_pcm_mov"]
        (out,) = run_export(tmp_path, range_jobs(spec, media_files[spec.name], [1]),
                            codec="copy", container="mp4", audio="aac")
        assert run_export.events == [f"CLIP_READY|1|{out}|copy"]
        assert video_stream(out)["codec_name"] == "h264"
        assert [a["codec_name"] for a in audio_streams(out)] == ["aac"]
        assert numbers(out) == list(spec.copy_frames(*spec.scene_frames[1]))


class TestAviCopyExport:
    TAGS = {"h264": "H264", "hevc": "HEVC"}

    @pytest.mark.parametrize("spec", media_params(
        "h264_2997_gop", "h265_24_opengop", "h264_23976_mkv", "prores_23976_422hq",
        "huffyuv_24_avi", "utvideo_2997_avi", "magicyuv_25_avi",
    ))
    def test_copy_into_avi_is_keyframe_snapped(self, media_files, tmp_path, spec):
        indices = list(range(len(spec.segments)))
        outputs = run_export(tmp_path, range_jobs(spec, media_files[spec.name], indices),
                             codec="copy", container="avi")
        source = video_stream(media_files[spec.name])
        for out, idx in zip(outputs, indices):
            assert out.endswith(".avi")
            assert numbers(out) == list(spec.copy_frames(*spec.scene_frames[idx])), f"scene {idx}"
            v = video_stream(out)
            assert int(v["nb_frames"]) == len(spec.copy_frames(*spec.scene_frames[idx])), f"scene {idx} has empty frames"
            assert v["codec_name"] == source["codec_name"]
            assert float(Fraction(v["avg_frame_rate"])) == pytest.approx(float(spec.rate), rel=1e-3)
            if v["codec_name"] in self.TAGS:
                assert v["codec_tag_string"] == self.TAGS[v["codec_name"]]
            if spec.audio:
                assert_av_in_sync(out, tolerance=0.1)
        assert all(e.endswith("|copy") for e in run_export.events)

    @pytest.mark.parametrize("spec", media_params("huffyuv_24_avi", "h264_2997_gop"))
    def test_copy_merge_into_avi(self, media_files, tmp_path, spec):
        indices = [0, 2]
        (out,) = run_export(tmp_path, range_jobs(spec, media_files[spec.name], indices),
                            codec="copy", container="avi", merge=True)
        assert run_export.events == [f"CLIP_READY|0|{out}|copy"]
        got = numbers(out)
        assert got == sorted(set(got)) and set(expected_frames(spec, indices)) <= set(got)
        v = video_stream(out)
        assert float(Fraction(v["avg_frame_rate"])) == pytest.approx(float(spec.rate), rel=1e-3)
        assert int(v["nb_frames"]) == len(got)
        if spec.audio:
            assert_av_in_sync(out, tolerance=0.1)

    def test_lossless_source_cannot_go_to_mp4(self, media_files, tmp_path):
        spec = BY_NAME["huffyuv_24_avi"]
        with pytest.raises(CopyContainerError) as error:
            run_export(tmp_path, range_jobs(spec, media_files[spec.name], [1]), codec="copy", container="mp4")
        assert error.value.details["stream_type"] == "video"
        assert "avi" in error.value.details["alternatives"]

    def test_opus_audio_cannot_be_copied_into_avi(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        source = tmp_path / "opus.mkv"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(media_files[spec.name]), "-map", "0:v",
                        "-map", "0:a", "-c:v", "copy", "-c:a", "libopus", str(source)], check=True)
        with pytest.raises(CopyContainerError) as error:
            run_export(tmp_path, range_jobs(spec, source, [1]), codec="copy", container="avi")
        assert (error.value.details["stream_type"], error.value.details["codec"]) == ("audio", "opus")
        (out,) = run_export(tmp_path / "pcm", range_jobs(spec, source, [1]),
                            codec="copy", container="avi", audio="pcm16")
        assert [a["codec_name"] for a in audio_streams(out)] == ["pcm_s16le"]


class TestCutAndExportParity:
    """A single-scene cut and a one-scene export of the same range must give
    the same frames. They share `smart_cut.copy_range` (copy) and the same
    frame-cut args (reencode); this catches either path drifting off again."""

    @pytest.mark.parametrize("spec", media_params())
    def test_copy(self, media_files, tmp_path, spec):
        path = media_files[spec.name]
        keyframes = get_keyframe_timestamps_pyav(str(path))
        for i, (start, end) in enumerate(spec.scene_secs):
            clip, _, _ = cut_scene(path, start, end, i, tmp_path, keyframes, "copy")
            (exported,) = run_export(
                tmp_path / f"export_{i}", range_jobs(spec, path, [i]),
                codec="copy", container=spec.export_container,
            )
            assert numbers(exported) == numbers(clip), f"scene {i}"

    @pytest.mark.parametrize("spec", media_params("h264_24_cuts", "h264_2997_gop", "h264_23976_ms_pts"))
    def test_reencode(self, media_files, tmp_path, spec):
        path = media_files[spec.name]
        for i, (start, end) in enumerate(spec.scene_secs):
            clip, _, _ = cut_scene(path, start, end, i, tmp_path, [], "reencode")
            (exported,) = run_export(
                tmp_path / f"export_{i}", range_jobs(spec, path, [i]), codec="h264_high",
            )
            assert numbers(exported) == numbers(clip) == list(range(*spec.scene_frames[i])), f"scene {i}"


class TestMerge:
    @pytest.mark.parametrize("spec", media_params())
    @pytest.mark.parametrize("codec", ["h264_high", "h265_main", "prores_422_hq"])
    def test_reencode_merge_is_exact_and_in_sync(self, media_files, tmp_path, spec, codec):
        indices = [0, 2, len(spec.segments) - 1] if len(spec.segments) > 3 else [0, 2]
        container = params.recommended_container(codec)
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], indices),
            codec=codec, container=container, merge=True,
        )
        assert len(outputs) == 1 and outputs[0].endswith(f"clip.{container}")
        assert numbers(outputs[0]) == expected_frames(spec, indices)
        assert_av_in_sync(outputs[0])
        assert run_export.events == [f"CLIP_READY|0|{outputs[0]}|reencode"]

    @pytest.mark.parametrize("spec", media_params())
    def test_copy_merge_of_contiguous_scenes_has_no_duplicates(self, media_files, tmp_path, spec):
        indices = list(range(1, len(spec.segments)))
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], indices),
            codec="copy", container=spec.export_container, merge=True,
        )
        a = spec.scene_frames[indices[0]][0]
        expected = list(spec.copy_frames(a, spec.total_frames))
        assert numbers(outputs[0]) == expected

    @pytest.mark.parametrize("spec", media_params("h264_2997_gop", "h265_60_main10_gop", "h264_24_cuts"))
    def test_copy_merge_of_gapped_scenes_never_repeats_frames(self, media_files, tmp_path, spec):
        indices = [0, 2, 3] if len(spec.segments) > 4 else [0, 2]
        outputs = run_export(
            tmp_path, range_jobs(spec, media_files[spec.name], indices),
            codec="copy", container=spec.export_container, merge=True,
        )
        got = numbers(outputs[0])
        assert got == sorted(set(got))
        assert set(expected_frames(spec, indices)) <= set(got)
        assert_av_in_sync(outputs[0], tolerance=0.1)

    @pytest.mark.parametrize("audio", ["mp3", "opus", "none"])
    def test_copy_merge_converts_audio(self, media_files, tmp_path, audio):
        spec = BY_NAME["h264_2997_gop"]
        indices = [0, 2, 3]
        (out,) = run_export(tmp_path, range_jobs(spec, media_files[spec.name], indices),
                            codec="copy", audio=audio, merge=True)
        assert run_export.events == [f"CLIP_READY|0|{out}|copy"]
        assert [a["codec_name"] for a in audio_streams(out)] == ([] if audio == "none" else [AUDIO_CODECS[audio]])
        got = numbers(out)
        assert got == sorted(set(got)) and set(expected_frames(spec, indices)) <= set(got)
        if audio != "none":
            assert_av_in_sync(out, tolerance=0.1)

    def test_copy_merge_of_precut_clips_is_lossless(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        clips = []
        (tmp_path / "clips").mkdir()
        for i, (a, b) in enumerate(spec.scene_frames):
            clip, _, _ = cut_scene(
                media_files[spec.name], spec.frame_time(a), spec.frame_time(b), i, tmp_path / "clips", [], "reencode"
            )
            clips.append(clip)
        jobs = [ExportJob(scene_index=i, input=c) for i, c in enumerate(clips)]
        outputs = run_export(tmp_path, jobs, codec="copy", merge=True)
        assert numbers(outputs[0]) == list(range(spec.total_frames))
        assert run_export.events == [f"CLIP_READY|0|{outputs[0]}|copy"]
        assert_av_in_sync(outputs[0])

    def test_copy_merge_of_mixed_precut_codecs_rejects_without_reencoding(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        first = run_export(tmp_path / "a", range_jobs(spec, media_files[spec.name], [0]), codec="h264_high")[0]
        second = run_export(tmp_path / "b", range_jobs(spec, media_files[spec.name], [1]), codec="h265_main")[0]
        jobs = [ExportJob(scene_index=0, input=first), ExportJob(scene_index=1, input=second)]
        with pytest.raises(ExportError, match="Cannot stream-copy this merge"):
            run_export(tmp_path, jobs, codec="copy", merge=True)


class TestFrameGrid:
    def test_millisecond_rounding_recovers_the_frame(self):
        rate = Fraction(30000, 1001)
        start, end = 1.502, 1.502 + 3.003
        assert params.frame_grid_range(start, end, rate) == (45, 135)

    def test_off_grid_times_are_left_alone(self):
        assert params.frame_grid_range(1.52, 3.0, Fraction(24)) is None
        assert params.frame_grid_range(1.0, 2.0, None) is None

    def test_cut_args_seek_a_quarter_frame_early(self):
        cut = params.frame_cut_args(48, 84, Fraction(24))
        assert cut.input_args == ["-ss", "1.989583333"]
        assert cut.output_args == ["-t", "1.520833333", "-frames:v", "36"]
        assert cut.audio_filters("asetpts=PTS-STARTPTS") == (
            "atrim=start=0.010416667,asetpts=PTS-STARTPTS,atrim=duration=1.5"
        )

    def test_seek_shifts_by_the_origin(self):
        cut = params.frame_cut_args(48, 84, Fraction(24), Fraction(21, 1000))
        assert cut.input_args == ["-ss", "2.010583333"]

    def test_mkv_origin_is_the_audio_priming(self, media_files):
        assert probe_seek_origin(str(media_files["h264_23976_mkv"])) == Fraction(21, 1000)
        assert probe_seek_origin(str(media_files["h264_24_cuts"])) == 0

    def test_first_frame_needs_no_seek(self):
        cut = params.frame_cut_args(0, 10, Fraction(25))
        assert cut.input_args == [] and cut.audio_head is None
        assert cut.output_args == ["-t", "0.42", "-frames:v", "10"]


class TestParams:
    def test_container_compatibility(self):
        assert params.codec_container_compatible("prores_422_hq", "mov")
        assert not params.codec_container_compatible("prores_422_hq", "mp4")
        assert params.codec_container_compatible("h265_main10", "mp4")
        assert params.recommended_container("prores_4444") == "mov"
        assert params.recommended_container("h264") == "mp4"

    def test_audio_copy_safety(self):
        assert params.audio_copy_safe("aac", "mp4")
        assert not params.audio_copy_safe("pcm_s16le", "mp4")
        assert params.audio_copy_safe("pcm_s16le", "mov")
        assert params.fallback_audio_mode("mp4") == "aac"

    def test_hevc_gets_hvc1_tag(self):
        assert params.container_codec_tag_args("h265_main", "mp4") == ["-tag:v", "hvc1"]
        assert params.container_codec_tag_args("h264_high", "mp4") == []

    def test_reencode_args_pin_frame_rate(self):
        args = params.build_reencode_args("in.mp4", "out.mp4", "h264_high", "aac", False,
                                          seek_ms=1500, dur_ms=2000, frame_rate="30000/1001")
        assert args[args.index("-ss") + 1] == "1.5"
        assert args[args.index("-t") + 1] == "2"
        assert args[args.index("-r:v:0") + 1] == "30000/1001"
