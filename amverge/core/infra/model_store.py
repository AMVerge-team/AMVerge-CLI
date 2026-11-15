from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Callable, Optional

QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "archival": {"crf": 14, "x264": "slow"},
    "high": {"crf": 17, "x264": "slow"},
    "balanced": {"crf": 20, "x264": "medium"},
    "fast": {"crf": 22, "x264": "veryfast"},
    "draft": {"crf": 26, "x264": "ultrafast"},
}


def load_registry_json(module_file: str | Path) -> dict[str, Any]:
    json_path = Path(module_file).parent / "registry.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_sha256(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest().lower()


def verify_sha256(file_path: str, expected_hash: Optional[str]) -> bool:
    if not expected_hash:
        return True
    if not os.path.exists(file_path):
        return False
    return compute_sha256(file_path) == expected_hash.lower()


def download_with_resume(
    url: str,
    dest: str,
    *,
    temp_path: Optional[str] = None,
    expected_hash: Optional[str] = None,
    retries: int = 3,
    timeout: int = 120,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    progress_formatter: Optional[Callable[[int, int, int], str]] = None,
    done_message: Optional[str] = None,
    cleanup_temp_dir: Optional[str] = None,
    remove_dest_on_hash_mismatch: bool = False,
) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part_path = temp_path or f"{dest}.part"
    os.makedirs(os.path.dirname(part_path), exist_ok=True)
    ctx = ssl._create_unverified_context()

    for attempt in range(retries):
        try:
            existing = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            headers = {"User-Agent": "amverge/1.0"}
            if existing > 0 and attempt > 0:
                headers["Range"] = f"bytes={existing}-"
            elif existing > 0:
                try:
                    os.remove(part_path)
                except OSError:
                    pass
                existing = 0

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
            code = resp.getcode()
            if code not in (200, 206):
                raise urllib.error.HTTPError(url, code, "", None, None)

            total = int(resp.headers.get("Content-Length", 0))
            file_mode = "ab" if code == 206 else "wb"
            downloaded = existing if code == 206 else 0
            if code == 206 and total > 0:
                total = existing + total

            chunk_size = 65536
            last_pct = -1
            with open(part_path, file_mode) as f:
                while chunk := resp.read(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        pct = min(99, int(downloaded * 100 / total))
                        if pct != last_pct:
                            last_pct = pct
                            msg = (
                                progress_formatter(pct, downloaded, total)
                                if progress_formatter
                                else f"Downloading... {pct}%"
                            )
                            progress_cb(pct, msg)

            if total > 0 and downloaded != total:
                raise ConnectionError(f"Incomplete: {downloaded}/{total} bytes")

            os.replace(part_path, dest)
            if expected_hash and not verify_sha256(dest, expected_hash):
                if remove_dest_on_hash_mismatch:
                    try:
                        os.unlink(dest)
                    except OSError:
                        pass
                raise RuntimeError("Hash mismatch")

            if cleanup_temp_dir:
                try:
                    os.rmdir(cleanup_temp_dir)
                except OSError:
                    pass

            if progress_cb and done_message:
                progress_cb(100, done_message)
            return True

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            IncompleteRead,
            ConnectionError,
            TimeoutError,
            OSError,
            RuntimeError,
        ):
            if attempt == retries - 1:
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                raise

    return False
