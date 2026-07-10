import os
import sys
import zipfile

from .config import get_ffmpeg_dir

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

FFMPEG_DOWNLOAD_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl.zip"
)


def _find_ffmpeg_in_dir(base_dir):
    for fname in ("ffmpeg.exe", "ffmpeg"):
        candidate = os.path.join(base_dir, "bin", fname)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(base_dir, fname)
        if os.path.exists(candidate):
            return candidate
    return None


def is_portable_ffmpeg_installed():
    ffmpeg_dir = get_ffmpeg_dir()
    return _find_ffmpeg_in_dir(ffmpeg_dir) is not None


def get_portable_ffmpeg_path():
    return _find_ffmpeg_in_dir(get_ffmpeg_dir())


def get_portable_ffprobe_path():
    ffmpeg_dir = get_ffmpeg_dir()
    for fname in ("ffprobe.exe", "ffprobe"):
        candidate = os.path.join(ffmpeg_dir, "bin", fname)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(ffmpeg_dir, fname)
        if os.path.exists(candidate):
            return candidate
    return None


def ensure_ffmpeg(progress_cb=None):
    if is_portable_ffmpeg_installed():
        return True

    ffmpeg_dir = get_ffmpeg_dir()
    os.makedirs(ffmpeg_dir, exist_ok=True)
    zip_path = os.path.join(ffmpeg_dir, "ffmpeg-temp.zip")

    import urllib.request
    import ssl

    ctx = ssl._create_unverified_context()

    if progress_cb:
        progress_cb(0, "Downloading portable FFmpeg...")

    try:
        req = urllib.request.Request(
            FFMPEG_DOWNLOAD_URL,
            headers={"User-Agent": "amverge/1.0"},
        )
        with urllib.request.urlopen(req, timeout=600, context=ctx) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            with open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        pct = min(99, int(downloaded * 100 / total))
                        progress_cb(pct, f"Downloading FFmpeg... {pct}%")
    except Exception as e:
        if os.path.exists(zip_path):
            os.unlink(zip_path)
        raise RuntimeError(f"Failed to download FFmpeg: {e}")

    if progress_cb:
        progress_cb(100, "Extracting FFmpeg...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ffmpeg_dir)
        os.unlink(zip_path)
    except Exception as e:
        if os.path.exists(zip_path):
            os.unlink(zip_path)
        raise RuntimeError(f"Failed to extract FFmpeg: {e}")

    extracted_dir = None
    for item in os.listdir(ffmpeg_dir):
        item_path = os.path.join(ffmpeg_dir, item)
        if os.path.isdir(item_path) and _find_ffmpeg_in_dir(item_path):
            extracted_dir = item_path
            break

    if extracted_dir:
        bin_src = os.path.join(extracted_dir, "bin")
        target_bin = os.path.join(ffmpeg_dir, "bin")
        if os.path.exists(bin_src):
            if os.path.exists(target_bin):
                import shutil
                shutil.rmtree(target_bin)
            os.rename(bin_src, target_bin)
        for item in os.listdir(extracted_dir):
            item_path = os.path.join(extracted_dir, item)
            if os.path.isfile(item_path):
                import shutil
                dest = os.path.join(ffmpeg_dir, item)
                if not os.path.exists(dest):
                    shutil.move(item_path, dest)
        import shutil
        shutil.rmtree(extracted_dir, ignore_errors=True)

    return is_portable_ffmpeg_installed()
