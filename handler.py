from __future__ import annotations

import base64
import binascii
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod


FACEFUSION_ROOT = Path(os.getenv("FACEFUSION_ROOT", "/facefusion"))
sys.path.insert(0, str(FACEFUSION_ROOT))
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
DOWNLOAD_TIMEOUT = (15, 180)
UPLOAD_TIMEOUT = (15, 300)
EXECUTION_THREAD_COUNT = int(os.getenv("EXECUTION_THREAD_COUNT", "2"))  # FaceFusion default is 8, which OOMs 24 GB GPUs

CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}

FACE_SWAPPER_MODELS = {
    "hyperswap_1a_256",
    "ghost_1_256",
    "ghost_3_256",
}


class InputError(ValueError):
    pass


class ContentSafetyError(RuntimeError):
    pass


def _https_url(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field} is required")
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.hostname:
        raise InputError(f"{field} must be an HTTPS URL")
    return value.strip()


def _extension_from_magic(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".mp4"
    return None


def _download(url: str, directory: Path, stem: str, max_bytes: int) -> Path:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            raise InputError(f"{stem} exceeds the size limit")

        chunks = response.iter_content(chunk_size=1024 * 1024)
        first_chunk = next(chunks, b"")
        extension = CONTENT_TYPE_EXTENSIONS.get(content_type) or _extension_from_magic(first_chunk)
        if not extension:
            raise InputError(f"unsupported media type for {stem}: {content_type or 'unknown'}")

        path = directory / f"{stem}{extension}"
        size = len(first_chunk)
        if size > max_bytes:
            raise InputError(f"{stem} exceeds the size limit")
        with path.open("wb") as output:
            output.write(first_chunk)
            for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise InputError(f"{stem} exceeds the size limit")
                output.write(chunk)
        if size == 0:
            raise InputError(f"{stem} was empty")
        return path


def _write_b64(value: Any, directory: Path, stem: str, max_bytes: int) -> Path:
    if not isinstance(value, str) or not value:
        raise InputError(f"{stem}_b64 is empty")
    try:
        body = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise InputError(f"{stem}_b64 is invalid") from error
    extension = _extension_from_magic(body)
    if not extension or not body or len(body) > max_bytes:
        raise InputError(f"{stem}_b64 is not a supported media file or is too large")
    path = directory / f"{stem}{extension}"
    path.write_bytes(body)
    return path


def _write_data_image(value: Any, directory: Path) -> Path:
    if not isinstance(value, str) or not value.startswith("data:image/"):
        raise InputError("frame_data_url must be an image data URL")
    try:
        header, encoded = value.split(",", 1)
        media_type = header[5:].split(";", 1)[0].lower()
        extension = CONTENT_TYPE_EXTENSIONS.get(media_type)
        if not extension or not extension.startswith("."):
            raise InputError("frame_data_url uses an unsupported image type")
        body = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise InputError("frame_data_url is invalid") from error
    if not body or len(body) > 8 * 1024 * 1024:
        raise InputError("frame_data_url exceeds the size limit")
    path = directory / f"frame{extension}"
    path.write_bytes(body)
    return path


def _upload(path: Path, upload: Any, media_type: str) -> str:
    if not isinstance(upload, dict):
        raise InputError("output_upload is required")
    url = _https_url(upload.get("url"), "output_upload.url")
    key = upload.get("key")
    if not isinstance(key, str) or not key:
        raise InputError("output_upload.key is required")
    raw_headers = upload.get("headers") or {}
    if not isinstance(raw_headers, dict):
        raise InputError("output_upload.headers must be an object")
    headers = {
        str(name): str(value)
        for name, value in raw_headers.items()
        if str(name).lower() in {"content-type", "cache-control", "x-upsert"}
    }
    headers["Content-Type"] = "video/mp4" if media_type == "video" else "image/jpeg"
    with path.open("rb") as body:
        response = requests.put(url, headers=headers, data=body, timeout=UPLOAD_TIMEOUT)
    response.raise_for_status()
    return key


def _validate_target_content(target_path: Path, media_type: str) -> None:
    if media_type != "image":
        return

    _configure_detection()
    from facefusion import content_analyser

    if content_analyser.analyse_image(str(target_path)):
        raise ContentSafetyError(
            "Target media did not pass FaceFusion's content safety check."
        )


def _run_swap(job_input: dict[str, Any]) -> dict[str, Any]:
    media_type = job_input.get("media_type")
    if media_type not in {"image", "video"}:
        raise InputError("media_type must be image or video")

    use_b64 = bool(job_input.get("target_b64"))
    source_url = target_url = None
    if not use_b64:
        source_url = _https_url(job_input.get("source_url"), "source_url")
        target_url = _https_url(job_input.get("target_url"), "target_url")
    raw_face_mappings = job_input.get("face_mappings")
    face_mappings: list[tuple[int, str]] = []
    if raw_face_mappings is not None:
        if not isinstance(raw_face_mappings, list) or not 1 <= len(raw_face_mappings) <= 20:
            raise InputError("face_mappings must contain 1 to 20 assignments")
        seen_face_indexes: set[int] = set()
        for position, mapping in enumerate(raw_face_mappings):
            if not isinstance(mapping, dict):
                raise InputError("each face mapping must be an object")
            mapping_face_index = mapping.get("face_index")
            if (
                not isinstance(mapping_face_index, int)
                or mapping_face_index < 0
                or mapping_face_index > 20
                or mapping_face_index in seen_face_indexes
            ):
                raise InputError("each face mapping needs a unique face_index from 0 to 20")
            seen_face_indexes.add(mapping_face_index)
            mapping_source_url = _https_url(
                mapping.get("source_url"), f"face_mappings[{position}].source_url"
            )
            face_mappings.append((mapping_face_index, mapping_source_url))
    face_mode = job_input.get("face_mode", "one")
    if face_mode not in {"one", "reference", "many"}:
        raise InputError("face_mode must be one, reference, or many")

    face_swapper_model = job_input.get("face_swapper_model", "hyperswap_1a_256")
    if face_swapper_model not in FACE_SWAPPER_MODELS:
        raise InputError("face_swapper_model is not supported")

    face_index = job_input.get("face_index", 0)
    reference_frame_number = job_input.get("reference_frame_number", 0)
    if not isinstance(face_index, int) or face_index < 0 or face_index > 20:
        raise InputError("face_index must be an integer from 0 to 20")
    if (
        not isinstance(reference_frame_number, int)
        or reference_frame_number < 0
        or reference_frame_number > 1_000_000
    ):
        raise InputError("reference_frame_number is invalid")

    with tempfile.TemporaryDirectory(prefix="facefusion-") as temp_name:
        temp_dir = Path(temp_name)
        t0 = time.time()
        max_target = MAX_VIDEO_BYTES if media_type == "video" else MAX_IMAGE_BYTES
        if use_b64:
            target_path = _write_b64(job_input.get("target_b64"), temp_dir, "target", max_target)
        else:
            target_path = _download(target_url, temp_dir, "target", max_target)
        _validate_target_content(target_path, media_type)
        output_path = temp_dir / ("output.mp4" if media_type == "video" else "output.jpg")
        jobs_path = temp_dir / "jobs"
        jobs_path.mkdir()

        passes = face_mappings or [(face_index, source_url)]
        current_target_path = target_path
        for pass_index, (mapped_face_index, mapped_source_url) in enumerate(passes):
            if use_b64:
                source_path = _write_b64(
                    job_input.get("source_b64"), temp_dir, f"source-{pass_index}", MAX_IMAGE_BYTES
                )
            else:
                source_path = _download(
                    mapped_source_url, temp_dir, f"source-{pass_index}", MAX_IMAGE_BYTES
                )
            t_swap = time.time()
            is_last_pass = pass_index == len(passes) - 1
            pass_output_path = (
                output_path
                if is_last_pass
                else temp_dir
                / (f"pass-{pass_index}.mp4" if media_type == "video" else f"pass-{pass_index}.jpg")
            )
            command = [
                "python",
                "facefusion.py",
                "headless-run",
                "--jobs-path",
                str(jobs_path),
                "--processors",
                "face_swapper",
                "--face-swapper-model",
                face_swapper_model,
                "--execution-providers",
                "cuda",
                "--source-paths",
                str(source_path),
                "--target-path",
                str(current_target_path),
                "--output-path",
                str(pass_output_path),
                "--face-selector-mode",
                "reference" if face_mappings else face_mode,
                "--face-selector-order",
                "left-right",
                "--reference-face-position",
                str(mapped_face_index),
                "--reference-frame-number",
                str(reference_frame_number),
                "--execution-thread-count",
                str(EXECUTION_THREAD_COUNT),
                "--log-level",
                "warn",
            ]
            if media_type == "video":
                command.extend(["--output-video-preset", "veryfast"])

            completed = subprocess.run(
                command,
                cwd=FACEFUSION_ROOT,
                capture_output=True,
                text=True,
                timeout=60 * 45,
            )
            if completed.returncode != 0 or not pass_output_path.is_file():
                detail = (completed.stderr or completed.stdout or "unknown error")[-2000:]
                raise RuntimeError(f"FaceFusion failed: {detail}")
            current_target_path = pass_output_path
            swap_seconds = time.time() - t_swap

        if use_b64:
            return {
                "status": "succeeded",
                "media_type": media_type,
                "bytes": output_path.stat().st_size,
                "output_b64": base64.b64encode(output_path.read_bytes()).decode(),
                "timings": {"prep_s": round(t_swap - t0, 2), "swap_s": round(swap_seconds, 2)},
            }
        output_key = _upload(output_path, job_input.get("output_upload"), media_type)
        return {
            "status": "succeeded",
            "output_key": output_key,
            "media_type": media_type,
            "bytes": output_path.stat().st_size,
        }


def _configure_detection() -> None:
    from facefusion import state_manager

    settings = {
        "execution_device_ids": [0],
        "execution_providers": ["cuda"],
        "download_providers": ["github"],
        "face_detector_angles": [0],
        "face_detector_model": "yolo_face",
        "face_detector_size": "640x640",
        "face_detector_margin": (0, 0, 0, 0),
        "face_detector_score": 0.5,
        "face_landmarker_model": "2dfan4",
        "face_landmarker_score": 0.5,
    }
    for key, value in settings.items():
        state_manager.init_item(key, value)


def _detect_faces(job_input: dict[str, Any]) -> dict[str, Any]:
    from facefusion.face_creator import get_many_faces
    from facefusion.face_selector import sort_faces_by_order
    from facefusion.vision import read_static_image

    with tempfile.TemporaryDirectory(prefix="facefusion-detect-") as temp_name:
        temp_dir = Path(temp_name)
        if job_input.get("frame_data_url"):
            target_path = _write_data_image(job_input.get("frame_data_url"), temp_dir)
        else:
            target_url = _https_url(job_input.get("target_url"), "target_url")
            target_path = _download(target_url, temp_dir, "target", MAX_IMAGE_BYTES)

        frame = read_static_image(str(target_path))
        if frame is None:
            raise InputError("target image could not be decoded")
        _configure_detection()
        faces = sort_faces_by_order(get_many_faces([frame]), "left-right")
        boxes = []
        for face in faces[:20]:
            left, top, right, bottom = face.bounding_box.tolist()
            boxes.append(
                {
                    "left": max(0, round(float(left))),
                    "top": max(0, round(float(top))),
                    "right": max(0, round(float(right))),
                    "bottom": max(0, round(float(bottom))),
                }
            )
        return {"status": "succeeded", "faces": boxes, "count": len(boxes)}


def _gpu_info() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as error:  # diagnostics only
        return f"unavailable ({error})"


def handler(event: dict[str, Any]) -> dict[str, Any]:
    job_input = event.get("input")
    if not isinstance(job_input, dict):
        return {"status": "failed", "error": "input must be an object"}
    try:
        operation = job_input.get("operation", "swap")
        if operation == "detect":
            return _detect_faces(job_input)
        if operation == "swap":
            return _run_swap(job_input)
        raise InputError("operation must be detect or swap")
    except InputError as error:
        return {"status": "failed", "error": str(error), "error_type": "input"}
    except requests.RequestException:
        return {
            "status": "failed",
            "error": "media transfer failed",
            "error_type": "transfer",
        }
    except ContentSafetyError as error:
        return {
            "status": "failed",
            "error": str(error),
            "error_type": "content",
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "error": "FaceFusion processing timed out",
            "error_type": "timeout",
        }
    except Exception:
        traceback.print_exc()
        return {
            "status": "failed",
            "error": "FaceFusion processing failed",
            "error_type": "processing",
            "detail": traceback.format_exc()[-3000:] + "\nGPU: " + _gpu_info(),  # MVP debugging; remove before production
        }


runpod.serverless.start({"handler": handler})
