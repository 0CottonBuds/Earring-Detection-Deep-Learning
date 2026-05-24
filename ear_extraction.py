from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


_FRONTAL_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
)
_PROFILE_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"  # type: ignore[attr-defined]
)


def _to_rgb_array(image: object) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)

    array = np.asarray(image)

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = np.clip(array * 255.0, 0, 255)
        else:
            array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)

    return array


def _detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces: list[tuple[int, int, int, int]] = []

    if not _FRONTAL_FACE_CASCADE.empty():
        for face in _FRONTAL_FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        ):
            faces.append(
                (
                    int(face[0]),
                    int(face[1]),
                    int(face[2]),
                    int(face[3]),
                )
            )

    if not _PROFILE_FACE_CASCADE.empty():
        for face in _PROFILE_FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        ):
            faces.append(
                (
                    int(face[0]),
                    int(face[1]),
                    int(face[2]),
                    int(face[3]),
                )
            )

        flipped_gray = cv2.flip(gray, 1)
        profile_faces = _PROFILE_FACE_CASCADE.detectMultiScale(
            flipped_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        width = image.shape[1]
        for x, y, w, h in profile_faces:
            faces.append((width - x - w, y, w, h))

    return faces


def _crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    return image[y1:y2, x1:x2]


def _score_crop(crop: np.ndarray) -> float:
    if crop.size == 0:
        return -1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var() + gray.std())


def _candidate_ear_boxes(face_box: tuple[int, int, int, int]) -> Iterable[tuple[int, int, int, int]]:
    x, y, w, h = face_box
    band_top = int(y + 0.25 * h)
    band_bottom = int(y + 0.88 * h)
    band_width = max(int(0.42 * w), 24)
    overlap = max(int(0.12 * w), 10)

    yield (x - band_width + overlap, band_top, band_width, band_bottom - band_top)
    yield (x + w - overlap, band_top, band_width, band_bottom - band_top)


def extract_ear_region(image: object, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Crop the most likely ear area from an image and resize it.

    The function accepts PIL images or numpy arrays and always returns an RGB
    numpy array. If no face is detected, it falls back to a centered crop so
    training and inference continue to work.
    """

    image_array = _to_rgb_array(image)
    height, width = image_array.shape[:2]

    faces = _detect_faces(image_array)

    if faces:
        face_box = max(faces, key=lambda box: box[2] * box[3])
        candidate_crops = [
            _crop_box(image_array, box)
            for box in _candidate_ear_boxes(face_box)
        ]
        crop = max(candidate_crops, key=_score_crop)
        if crop.size == 0:
            crop = image_array
    else:
        crop_width = max(int(width * 0.6), 1)
        crop_height = max(int(height * 0.6), 1)
        x1 = max((width - crop_width) // 2, 0)
        y1 = max((height - crop_height) // 2, 0)
        crop = image_array[y1 : y1 + crop_height, x1 : x1 + crop_width]

    if target_size is None:
        target_size = (width, height)

    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA).astype(np.float32)