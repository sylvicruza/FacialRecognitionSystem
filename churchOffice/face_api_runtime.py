from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import requests

from django.conf import settings

from .api_backend import get_all_people

_TORCH = None
_DEVICE = None
_MTCNN = None
_RESNET = None
_RUNTIME_ERROR = None

ENCODING_CACHE: dict[str, Any] = {
    "loaded_at": 0.0,
    "encodings": np.empty((0, 512), dtype=np.float32),
    "people": {},
}
ENCODING_CACHE_TTL_SECONDS = 300


def get_face_runtime():
    global _TORCH, _DEVICE, _MTCNN, _RESNET, _RUNTIME_ERROR
    if _RUNTIME_ERROR is not None:
        raise RuntimeError(_RUNTIME_ERROR)
    if _MTCNN is None or _RESNET is None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1, MTCNN

            _TORCH = torch
            _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _MTCNN = MTCNN(keep_all=True, device=_DEVICE)
            _RESNET = InceptionResnetV1(pretrained="vggface2").eval().to(_DEVICE)
        except Exception as exc:
            _RUNTIME_ERROR = str(exc) or "Face recognition runtime is unavailable."
            raise RuntimeError(_RUNTIME_ERROR) from exc
    return _TORCH, _DEVICE, _MTCNN, _RESNET


def detect_and_encode(image_rgb: np.ndarray):
    torch, device, mtcnn, resnet = get_face_runtime()
    faces = []
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image_rgb)
        if boxes is None:
            return faces

        h, w = image_rgb.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = map(int, map(round, box))
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            face = image_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_LINEAR)
            face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
            face_tensor = torch.from_numpy(face).unsqueeze(0).to(device)
            faces.append((resnet(face_tensor).cpu().numpy().flatten(), box))
    return faces


def _download_image(image_url: str):
    response = requests.get(
        image_url,
        timeout=settings.ATTENDANCE_API_TIMEOUT,
        verify=settings.ATTENDANCE_API_VERIFY_SSL,
    )
    response.raise_for_status()
    arr = np.frombuffer(response.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def load_authorized_face_encodings(request=None):
    now = time.time()
    if now - ENCODING_CACHE["loaded_at"] < ENCODING_CACHE_TTL_SECONDS:
        return ENCODING_CACHE["encodings"], ENCODING_CACHE["people"]

    people = [
        person
        for person in get_all_people(request=request)
        if getattr(person, "authorized", False) and getattr(person, "image", None)
    ]
    encodings = []
    person_map = {}

    for person in people:
        try:
            image = _download_image(person.image.url)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for encoding, _ in detect_and_encode(image_rgb):
                encodings.append(encoding)
                person_map[len(encodings) - 1] = person
        except Exception:
            continue

    ENCODING_CACHE["loaded_at"] = now
    ENCODING_CACHE["encodings"] = (
        np.array(encodings, dtype=np.float32) if encodings else np.empty((0, 512), dtype=np.float32)
    )
    ENCODING_CACHE["people"] = person_map
    return ENCODING_CACHE["encodings"], ENCODING_CACHE["people"]


def recognize_faces(known_encodings: np.ndarray, known_people: dict[int, Any], test_encodings, threshold: float = 0.6):
    results = []
    if known_encodings is None or len(known_encodings) == 0:
        for _, box in test_encodings:
            results.append((None, box, None))
        return results

    for test_encoding, box in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        if distances.size == 0:
            results.append((None, box, None))
            continue
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])
        person = known_people[min_idx] if min_dist < float(threshold) else None
        results.append((person, box, min_dist))
    return results
