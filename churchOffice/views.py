# views.py (FULL) — matches your existing urls.py exactly
# ------------------------------------------------------
# URLs you shared map to these view functions:
#   ''                         -> home
#   'search_user/'             -> search_user
#   'register_user/'           -> register_user
#   'success_page/'            -> success_page
#   'person_list/'             -> person_list
#   'persons/<int:pk>/'        -> person_detail
#   'persons/<int:pk>/authorize/' -> person_authorize
#   'persons/<int:pk>/delete/' -> person_delete
#   'capture-and-recognize/'   -> capture_and_recognize
#   'persons/attendance/'      -> person_attendance_list
#   'camera-config/'           -> camera_config_create
#   'camera-config/list/'      -> camera_config_list
#   'camera-config/update/<int:pk>/' -> camera_config_update
#   'camera-config/delete/<int:pk>/' -> camera_config_delete
#   'stream/<int:cam_id>/'     -> camera_stream
#   'video_feed/<int:cam_id>/' -> video_feed
#   'stream/all/'              -> stream_all_cameras
#   'api/nfc/check-in/'        -> nfc_check_in

from __future__ import annotations

import base64
import json
import os
import time
from collections import defaultdict, deque
from smtplib import SMTPException
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import pygame
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

from django.conf import settings
from django.contrib import messages
from django.core.files.base import ContentFile
from django.db import IntegrityError
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from .models import Attendance, CameraConfiguration, Person
from .services import mark_attendance
from .utils import (
    fetch_user_data,
    fetch_user_data_by_id,
    register_user_on_portal,
    upload_person_photo_to_portal,
)
from django.views.decorators.http import require_GET
from django.db import models

from django.core.mail import EmailMessage
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from openpyxl import Workbook
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
import csv
import io
from datetime import datetime
from .models import Attendance, Person



# =========================================================
# Audio
# =========================================================
try:
    pygame.mixer.init()
    _success_sound = None

    def play_success_sound():
        global _success_sound
        try:
            if _success_sound is None:
                path = os.path.join(settings.BASE_DIR, "media", "audio", "suc.wav")
                _success_sound = pygame.mixer.Sound(path)
            _success_sound.play()
        except Exception as e:
            print("[WARN] Sound playback failed:", e)

except Exception as e:
    print("[WARN] pygame mixer not available:", e)

    def play_success_sound():
        return


# =========================================================
# Face Models (loaded once)
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def detect_and_encode(image_rgb: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    faces: List[Tuple[np.ndarray, np.ndarray]] = []
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

            encoding = resnet(face_tensor).cpu().numpy().flatten()
            faces.append((encoding, box))

    return faces


def encode_uploaded_images() -> Tuple[np.ndarray, List[int]]:
    encodings: List[np.ndarray] = []
    person_ids: List[int] = []

    qs = Person.objects.filter(authorized=True).exclude(image="").only("id", "image")
    for person in qs:
        try:
            img_path = os.path.join(settings.MEDIA_ROOT, str(person.image))
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for encoding, _ in detect_and_encode(img_rgb):
                encodings.append(encoding)
                person_ids.append(person.id)

        except Exception as e:
            print(f"[WARN] encode failed for person_id={person.id}: {e}")

    if not encodings:
        return np.empty((0, 512), dtype=np.float32), []

    return np.array(encodings, dtype=np.float32), person_ids


def recognize_faces(
    known_encodings: np.ndarray,
    known_person_ids: List[int],
    test_encodings: List[Tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.6,
) -> List[Tuple[int | None, np.ndarray, float | None]]:
    results: List[Tuple[int | None, np.ndarray, float | None]] = []

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
        person_id = known_person_ids[min_idx] if min_dist < float(threshold) else None
        results.append((person_id, box, min_dist))

    return results


# =========================================================
# Stream + Attendance throttling
# =========================================================
COOLDOWN_SECONDS = 10
STABLE_HITS_REQUIRED = 3
STABLE_WINDOW_SECONDS = 2


def _open_capture(source: int | str) -> cv2.VideoCapture:
    """
    Open a camera source robustly.

    For numeric webcam indexes on Windows:
    1. Try DirectShow first
    2. Fall back to default OpenCV backend if DirectShow fails

    For URL/RTSP sources:
    Use FFmpeg first, then fall back to default backend if needed.
    """
    cap = None

    if isinstance(source, int):
        # Try DirectShow first
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                return cap
        except Exception as e:
            print(f"[WARN] CAP_DSHOW open failed for webcam {source}: {e}")

        # Fallback to default backend
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        cap = cv2.VideoCapture(source)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            return cap

        return cap

    else:
        # Try FFmpeg for RTSP/HTTP streams
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                return cap
        except Exception as e:
            print(f"[WARN] CAP_FFMPEG open failed for source '{source}': {e}")

        # Fallback to default backend
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        cap = cv2.VideoCapture(source)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            return cap

        return cap


def gen_frames(source, cam_config, max_fps=10, draw_boxes=True, draw_names=True, play_sound=True):
    cap = _open_capture(source)

    if not cap.isOpened():
        err = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(err, f"Camera not opened: {source}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ok, buffer = cv2.imencode(".jpg", err)
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        cap.release()
        return

    known_encodings, known_person_ids = encode_uploaded_images()
    person_by_id = Person.objects.in_bulk(known_person_ids)

    last_marked: Dict[int, timezone.datetime] = {}
    hits: Dict[int, Deque[timezone.datetime]] = defaultdict(lambda: deque(maxlen=30))

    delay = 1.0 / float(max_fps)
    prev_time = 0.0
    last_frame_ok = time.time()

    try:
        while True:
            now_t = time.time()
            if now_t - prev_time < delay:
                time.sleep(0.001)
                continue
            prev_time = now_t

            ret, frame = cap.read()
            if not ret or frame is None:
                if time.time() - last_frame_ok > 2:
                    cap.release()
                    time.sleep(0.5)
                    cap = _open_capture(source)
                    last_frame_ok = time.time()
                continue

            last_frame_ok = time.time()

            frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            test_encodings = detect_and_encode(frame_rgb)

            if test_encodings:
                recognized = recognize_faces(
                    known_encodings,
                    known_person_ids,
                    test_encodings,
                    threshold=float(cam_config.threshold),
                )

                for person_id, box, dist in recognized:
                    if box is None:
                        continue

                    x1, y1, x2, y2 = map(int, map(round, box))
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    x2 = max(0, min(x2, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    y2 = max(0, min(y2, frame.shape[0] - 1))

                    if draw_boxes:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = "Not Recognized"

                    if person_id is not None and person_id in person_by_id:
                        person = person_by_id[person_id]
                        now = timezone.now()

                        hits[person_id].append(now)
                        recent_hits = [
                            t for t in hits[person_id]
                            if (now - t).total_seconds() <= STABLE_WINDOW_SECONDS
                        ]

                        if len(recent_hits) >= STABLE_HITS_REQUIRED:
                            last = last_marked.get(person_id)
                            if not last or (now - last).total_seconds() >= COOLDOWN_SECONDS:
                                outcome = mark_attendance(person, min_checkout_seconds=60, camera=cam_config)
                                last_marked[person_id] = now
                                if play_sound:
                                    play_success_sound()
                                label = f"{person.name} ({outcome.status})"
                            else:
                                label = f"{person.name} (cooldown)"
                        else:
                            label = f"{person.name} (detecting...)"

                    if draw_names:
                        cv2.putText(
                            frame,
                            str(label),
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


# =========================================================
# Streaming views (MATCH your urls.py)
# =========================================================
def _get_bool_qs(request, key: str, default: bool = True) -> bool:
    v = request.GET.get(key, None)
    if v is None:
        return default
    return v in ("1", "true", "True", "yes", "on")


def video_feed(request, cam_id):
    cam_config = get_object_or_404(CameraConfiguration, id=cam_id)
    src = cam_config.camera_source.strip()
    source = int(src) if src.isdigit() else src

    draw_boxes = _get_bool_qs(request, "boxes", True)
    draw_names = _get_bool_qs(request, "names", True)
    play_sound = _get_bool_qs(request, "sound", True)

    return StreamingHttpResponse(
        gen_frames(
            source,
            cam_config,
            max_fps=10,
            draw_boxes=draw_boxes,
            draw_names=draw_names,
            play_sound=play_sound,
        ),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def camera_stream(request, cam_id: int):
    config = get_object_or_404(CameraConfiguration, id=cam_id)
    return render(request, "camera_stream.html", {"config": config})


def stream_all_cameras(request):
    configs = CameraConfiguration.objects.all()
    return render(request, "stream_all_cameras.html", {"configs": configs})

def gen_preview_frames(source, max_fps=10):
    cap = _open_capture(source)

    if not cap.isOpened():
        err = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(err, f"Preview not opened: {source}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ok, buffer = cv2.imencode(".jpg", err)
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        cap.release()
        return

    delay = 1.0 / float(max_fps)
    prev_time = 0.0
    last_frame_ok = time.time()

    try:
        while True:
            now_t = time.time()
            if now_t - prev_time < delay:
                time.sleep(0.001)
                continue
            prev_time = now_t

            ret, frame = cap.read()
            if not ret or frame is None:
                if time.time() - last_frame_ok > 2:
                    cap.release()
                    time.sleep(0.5)
                    cap = _open_capture(source)
                    last_frame_ok = time.time()
                continue

            last_frame_ok = time.time()
            frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


def camera_preview_feed(request, cam_id):
    cam_config = get_object_or_404(CameraConfiguration, id=cam_id)
    src = cam_config.camera_source.strip()
    source = int(src) if src.isdigit() else src

    return StreamingHttpResponse(
        gen_preview_frames(source, max_fps=10),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


# =========================================================
# NFC API (MATCH your urls.py)
# =========================================================
@csrf_exempt
def nfc_check_in(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body or b"{}")
        uid = data.get("uid")
        camera_id = data.get("camera_id")  # optional
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not uid:
        return JsonResponse({"error": "UID is required"}, status=400)

    person = Person.objects.filter(nfc_uid=uid).first()
    if not person:
        return JsonResponse({"error": "UID not registered"}, status=404)

    cam = None
    if camera_id:
        cam = CameraConfiguration.objects.filter(id=camera_id).first()

    outcome = mark_attendance(person, min_checkout_seconds=60, camera=cam)
    play_success_sound()

    return JsonResponse({
        "status": outcome.status,
        "name": person.name,
        "camera": cam.name if cam else None
    })


# =========================================================
# UI / Pages (MATCH your urls.py)
# =========================================================
def home(request):
    total_persons = Person.objects.count()
    total_attendance = Attendance.objects.count()
    total_check_ins = Attendance.objects.filter(check_in_time__isnull=False).count()
    total_check_outs = Attendance.objects.filter(check_out_time__isnull=False).count()
    total_cameras = CameraConfiguration.objects.count()

    context = {
        "total_persons": total_persons,
        "total_attendance": total_attendance,
        "total_check_ins": total_check_ins,
        "total_check_outs": total_check_outs,
        "total_cameras": total_cameras,
    }
    return render(request, "home.html", context)


def search_user(request):
    if request.method == "POST":
        action = request.POST.get("action", "search").strip()

        if action == "search":
            query = request.POST.get("last_name", "").strip()

            if not query:
                return render(
                    request,
                    "search_user.html",
                    {"error": "Enter a last name to search."}
                )

            user_data = fetch_user_data(query)

            return render(
                request,
                "search_user.html",
                {
                    "user_data": user_data or [],
                    "searched_last_name": query,
                    "show_create_option": True,   # always show option after any search
                    "search_performed": True,
                    "prefill_last_name": query,
                }
            )

        elif action == "show_portal_register":
            searched_last_name = request.POST.get("searched_last_name", "").strip()
            return render(
                request,
                "search_user.html",
                {
                    "show_portal_register": True,
                    "searched_last_name": searched_last_name,
                    "prefill_last_name": searched_last_name,
                    "show_create_option": True,
                    "search_performed": True,
                }
            )

        elif action == "portal_register":
            first_name = request.POST.get("first_name", "").strip()
            last_name = request.POST.get("last_name", "").strip()
            gender = request.POST.get("gender", "").strip()
            email = request.POST.get("email", "").strip()
            cell_phone = request.POST.get("cellPhone", "").strip()
            address1 = request.POST.get("address1", "").strip()

            if not first_name or not last_name or gender not in ("1", "2"):
                return render(
                    request,
                    "search_user.html",
                    {
                        "error": "First name, last name, and gender are required.",
                        "show_portal_register": True,
                        "prefill_first_name": first_name,
                        "prefill_last_name": last_name,
                        "prefill_gender": gender,
                        "prefill_email": email,
                        "prefill_cellPhone": cell_phone,
                        "prefill_address1": address1,
                        "show_create_option": True,
                        "search_performed": True,
                        "searched_last_name": last_name,
                    }
                )

            payload = {
                "firstName": first_name,
                "lastName": last_name,
                "gender": gender,
                "email": email or None,
                "cellPhone": cell_phone or None,
                "address1": address1 or None,
            }

            result = register_user_on_portal(payload)

            if not result["ok"]:
                return render(
                    request,
                    "search_user.html",
                    {
                        "error": result["error"] or "Portal registration failed.",
                        "show_portal_register": True,
                        "prefill_first_name": first_name,
                        "prefill_last_name": last_name,
                        "prefill_gender": gender,
                        "prefill_email": email,
                        "prefill_cellPhone": cell_phone,
                        "prefill_address1": address1,
                        "show_create_option": True,
                        "search_performed": True,
                        "searched_last_name": last_name,
                    }
                )

            full_name = f"{first_name} {last_name}".strip()

            user_data = fetch_user_data(last_name) or []
            match = next(
                (u for u in user_data if u.get("text", "").strip().lower() == full_name.lower()),
                user_data[0] if user_data else None
            )

            if not match:
                return render(
                    request,
                    "search_user.html",
                    {
                        "error": "Portal record was created, but we could not locate it automatically. Search again to continue.",
                        "searched_last_name": last_name,
                        "show_create_option": True,
                        "search_performed": True,
                    }
                )

            portal_id = match.get("objid")
            display_name = match.get("text", full_name)

            messages.success(request, f"{display_name} was created successfully.")
            return redirect(f"{reverse('register_user')}?portal_id={portal_id}&name={display_name}")

    return render(request, "search_user.html")


def register_user(request):
    if request.method == "GET":
        portal_id = request.GET.get("portal_id")
        name = request.GET.get("name")
        cams = CameraConfiguration.objects.all().order_by("name")

        return render(
            request,
            "register_user.html",
            {
                "portal_id": portal_id,
                "name": name,
                "camera_configs": cams,
            },
        )

    if request.method == "POST":
        portal_id = request.POST.get("portal_id")
        name = request.POST.get("name")
        image_data = request.POST.get("image_data")

        cams = CameraConfiguration.objects.all().order_by("name")

        if not image_data:
            return render(
                request,
                "register_user.html",
                {
                    "error": "Image is required. Please capture a photo.",
                    "portal_id": portal_id,
                    "name": name,
                    "camera_configs": cams,
                },
            )

        try:
            header, encoded = image_data.split(",", 1)
        except ValueError:
            return render(
                request,
                "register_user.html",
                {
                    "error": "Invalid image data.",
                    "portal_id": portal_id,
                    "name": name,
                    "camera_configs": cams,
                },
            )

        image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

        person = Person(
            name=name,
            portal_id=portal_id,
            image=image_file,
            authorized=True,
        )
        person.save()

        # Upload same image to ChurchCRM behind the scenes
        if portal_id:
            upload_result = upload_person_photo_to_portal(portal_id, image_data)
            print("PORTAL PHOTO RESULT:", upload_result)

            if upload_result.get("ok"):
                messages.success(
                    request,
                    f"{name} registered successfully. Portal photo updated successfully."
                )
            else:
                messages.warning(
                    request,
                    f"{name} registered locally, but portal photo upload failed: {upload_result.get('error', 'Unknown error')}"
                )
        else:
            messages.success(request, f"{name} registered successfully.")

        return redirect("person_list")

    cams = CameraConfiguration.objects.all().order_by("name")
    return render(request, "register_user.html", {"camera_configs": cams})


def success_page(request):
    return render(request, "selfie_success.html")


# =========================================================
# People management (MATCH your urls.py)
# =========================================================
def person_list(request):
    persons_qs = Person.objects.all().order_by("-id")   # latest first

    paginator = Paginator(persons_qs, 10)  # 10 per page
    page_number = request.GET.get("page")
    persons = paginator.get_page(page_number)

    return render(request, "user_list.html", {"persons": persons})


def person_detail(request, pk: int):
    person = get_object_or_404(Person, pk=pk)
    user_data = fetch_user_data_by_id(person.portal_id)
    return render(request, "user_detail.html", {"person": person, "user_data": user_data})


def person_authorize(request, pk: int):
    person = get_object_or_404(Person, pk=pk)

    if request.method == "POST":
        authorized = request.POST.get("authorized", False)
        person.authorized = bool(authorized)
        person.save()
        return redirect("person_detail", pk=pk)

    return render(request, "user_authorize.html", {"person": person})


def person_delete(request, pk: int):
    person = get_object_or_404(Person, pk=pk)

    if request.method == "POST":
        person.delete()
        messages.success(request, "Member deleted successfully.")
        return redirect("person_list")

    return render(request, "user_delete_confirm.html", {"person": person})


# =========================================================
# Attendance views (MATCH your urls.py)
# =========================================================

def _filtered_attendance_qs(search_query: str, date_filter: str):
    """
    Returns a flat Attendance queryset, filtered + ordered newest-first.
    """
    qs = Attendance.objects.select_related("person")

    if search_query:
        qs = qs.filter(person__name__icontains=search_query)

    if date_filter:
        # date_filter comes as 'YYYY-MM-DD'
        qs = qs.filter(date=date_filter)

    # newest first
    qs = qs.order_by("-date", "-check_in_time", "-id")
    return qs


def person_attendance_list(request):
    search_query = request.GET.get("search", "").strip()
    date_filter = request.GET.get("attendance_date", "").strip()

    qs = _filtered_attendance_qs(search_query, date_filter)

    page_size = int(request.GET.get("page_size", 25) or 25)
    page_size = max(10, min(page_size, 200))  # clamp
    paginator = Paginator(qs, page_size)

    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "user_attendance_list.html",
        {
            "attendance_page": page_obj,   # <--- use this in template
            "search_query": search_query,
            "date_filter": date_filter,
            "page_size": page_size,
        },
    )


def capture_and_recognize(request):
    # Your system uses the stream endpoints now; keep this route but redirect somewhere useful.
    return redirect("camera_config_list")


# =========================================================
# Camera configuration UI (MATCH your urls.py)
# =========================================================
def camera_config_create(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        camera_source = request.POST.get("camera_source", "").strip()
        threshold = request.POST.get("threshold", "").strip()

        if not name or not camera_source or not threshold:
            messages.error(request, "All fields are required.")
            return render(request, "camera_config_form.html")

        CameraConfiguration.objects.create(
            name=name,
            camera_source=camera_source,
            threshold=threshold
        )

        messages.success(request, "Camera configuration saved successfully.")
        return redirect("camera_config_list")

    return render(request, "camera_config_form.html")


def camera_config_list(request):
    configs = CameraConfiguration.objects.all()
    return render(request, "camera_config_list.html", {"configs": configs})


def camera_config_update(request, pk: int):
    config = get_object_or_404(CameraConfiguration, pk=pk)

    if request.method == "POST":
        config.name = request.POST.get("name")
        config.camera_source = request.POST.get("camera_source", "")
        config.threshold = request.POST.get("threshold", "0.6")
        config.save()
        return redirect("camera_config_list")

    return render(request, "camera_config_form.html", {"config": config})


def camera_config_delete(request, pk: int):
    config = get_object_or_404(CameraConfiguration, pk=pk)
    if request.method == "POST":
        config.delete()
        messages.success(request, "Camera configuration deleted.")
    return redirect("camera_config_list")

def api_attendance_monitor(request):
    today = timezone.localdate()

    total_checked_in = Attendance.objects.filter(
        date=today,
        check_in_time__isnull=False
    ).count()

    qs = (
        Attendance.objects
        .select_related("person", "camera")
        .filter(date=today)
        .order_by("-check_in_time")[:200]
    )

    events = []
    for a in qs:
        if a.check_in_time:
            events.append({
                "type": "checked_in",
                "time": a.check_in_time.isoformat(),
                "name": a.person.name,
                "person_id": a.person_id,
                "camera": a.camera.name if a.camera else None,
            })
        if a.check_out_time:
            events.append({
                "type": "checked_out",
                "time": a.check_out_time.isoformat(),
                "name": a.person.name,
                "person_id": a.person_id,
                "camera": a.camera.name if a.camera else None,
            })

    events.sort(key=lambda x: x["time"], reverse=True)
    events = events[:50]

    return JsonResponse({
        "date": str(today),
        "total_checked_in": total_checked_in,
        "events": events,
    })



@require_GET
def attendance_today_api(request):
    """
    Returns today's attendance events (latest first).
    Supports:
      - ?since=<epoch_ms>  (optional) for incremental updates
      - ?limit=60
    """
    limit = int(request.GET.get("limit", 60))
    since = request.GET.get("since")

    today = timezone.localdate()

    qs = Attendance.objects.filter(date=today).select_related("person").order_by("-check_in_time", "-check_out_time", "-id")

    if since:
        try:
            if since.isdigit():
                dt = timezone.datetime.fromtimestamp(int(since) / 1000, tz=timezone.get_current_timezone())
                # "changed since" = check_in or check_out happened after dt
                qs = qs.filter(
                    models.Q(check_in_time__gt=dt) |
                    models.Q(check_out_time__gt=dt)
                )
        except Exception:
            pass

    qs = qs[:limit]

    def status_for(a: Attendance):
        if a.check_out_time:
            return "CHECKED OUT"
        if a.check_in_time:
            return "CHECKED IN"
        return "UPDATED"

    def event_time(a: Attendance):
        # pick the most recent meaningful event time
        return a.check_out_time or a.check_in_time

    items = []
    latest_ts = None

    for a in qs:
        t = event_time(a)
        if t and (latest_ts is None or t > latest_ts):
            latest_ts = t

        items.append({
            "id": a.id,
            "person_id": a.person_id,
            "name": a.person.name if a.person else "Unknown",
            "status": status_for(a),
            "time": t.isoformat() if t else None,
            "camera": a.camera.name if a.camera else None,
        })

    return JsonResponse({
        "date": str(today),
        "count": Attendance.objects.filter(date=today).count(),
        "items": items,
        "latest_epoch_ms": int(latest_ts.timestamp() * 1000) if latest_ts else None,
        "latest": latest_ts.isoformat() if latest_ts else None,
    })


def attendance_delete(request, pk: int):
    """
    Deletes an attendance record. POST-only + CSRF protected.
    Redirects back to the referring page (or a safe fallback).
    """
    attendance = get_object_or_404(Attendance, pk=pk)

    # OPTIONAL: if you want staff-only deletes, uncomment:
    # if not request.user.is_staff:
    #     messages.error(request, "You are not allowed to delete attendance records.")
    #     return redirect(request.POST.get("next") or "person_attendance_list")

    display_name = getattr(attendance.person, "name", str(attendance.person))
    date_str = str(attendance.date)

    attendance.delete()

    messages.success(request, f"Deleted attendance for {display_name} on {date_str}.")

    next_url = request.POST.get("next") or request.META.get("HTTP_REFERER") or "/"
    return redirect(next_url)

def _duration_text(a) -> str:
    """
    Safely return the duration string for an Attendance record.
    """
    if a.check_in_time and a.check_out_time:
        # If calculate_duration is a method:
        if callable(getattr(a, "calculate_duration", None)):
            return str(a.calculate_duration())
        # If calculate_duration is a @property:
        return str(getattr(a, "calculate_duration", ""))
    return "Not Checked Out"


def attendance_export_download(request):
    search_query = request.GET.get("search", "").strip()
    date_filter = request.GET.get("attendance_date", "").strip()
    fmt = (request.GET.get("format") or "csv").lower().strip()

    qs = _filtered_attendance_qs(search_query, date_filter)

    # filename
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "attendance"
    if date_filter:
        label += f"_{date_filter}"
    filename_base = f"{label}_{stamp}"

    if fmt == "csv":
        resp = HttpResponse(content_type="text/csv; charset=utf-8")
        resp["Content-Disposition"] = f'attachment; filename="{filename_base}.csv"'
        w = csv.writer(resp)
        w.writerow(["Name", "Portal ID", "Date", "Check-In", "Check-Out", "Duration"])

        for a in qs:
            w.writerow([
                a.person.name if a.person else "",
                getattr(a.person, "portal_id", "") if a.person else "",
                str(a.date) if a.date else "",
                a.check_in_time.strftime("%H:%M:%S") if a.check_in_time else "",
                a.check_out_time.strftime("%H:%M:%S") if a.check_out_time else "",
                _duration_text(a),
            ])
        return resp

    if fmt == "xlsx":
        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance"
        ws.append(["Name", "Portal ID", "Date", "Check-In", "Check-Out", "Duration"])

        for a in qs:
            ws.append([
                a.person.name if a.person else "",
                getattr(a.person, "portal_id", "") if a.person else "",
                str(a.date) if a.date else "",
                a.check_in_time.strftime("%H:%M:%S") if a.check_in_time else "",
                a.check_out_time.strftime("%H:%M:%S") if a.check_out_time else "",
                _duration_text(a),
            ])

        out = io.BytesIO()
        wb.save(out)
        out.seek(0)

        resp = HttpResponse(
            out.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        resp["Content-Disposition"] = f'attachment; filename="{filename_base}.xlsx"'
        return resp

    if fmt == "pdf":
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=landscape(A4))
        width, height = landscape(A4)

        y = height - 40
        c.setFont("Helvetica-Bold", 14)
        title = "Attendance Report"
        if date_filter:
            title += f" ({date_filter})"
        c.drawString(40, y, title)
        y -= 24

        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Filter: search='{search_query or '-'}'")
        y -= 18

        c.setFont("Helvetica-Bold", 10)
        headers = ["Name", "Portal ID", "Date", "In", "Out", "Duration"]
        x_positions = [40, 260, 360, 450, 520, 610]
        for x, h in zip(x_positions, headers):
            c.drawString(x, y, h)
        y -= 14
        c.setFont("Helvetica", 9)

        for a in qs[:2000]:  # safety cap
            if y < 40:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 9)

            name = a.person.name if a.person else ""
            portal = getattr(a.person, "portal_id", "") if a.person else ""
            dt = str(a.date) if a.date else ""
            cin = a.check_in_time.strftime("%H:%M:%S") if a.check_in_time else ""
            cout = a.check_out_time.strftime("%H:%M:%S") if a.check_out_time else ""
            dur = _duration_text(a)

            row = [name, str(portal), dt, cin, cout, str(dur)]
            for x, val in zip(x_positions, row):
                c.drawString(x, y, (val or "")[:28])
            y -= 12

        c.save()
        buf.seek(0)

        resp = HttpResponse(buf.getvalue(), content_type="application/pdf")
        resp["Content-Disposition"] = f'attachment; filename="{filename_base}.pdf"'
        return resp

    return HttpResponse("Invalid format", status=400)


def attendance_email_export(request):
    if request.method != "POST":
        return redirect("person_attendance_list")

    to_email = (request.POST.get("to_email") or "").strip()
    fmt = (request.POST.get("format") or "csv").lower().strip()
    search_query = (request.POST.get("search") or "").strip()
    date_filter = (request.POST.get("attendance_date") or "").strip()

    if not to_email:
        messages.error(request, "Recipient email is required.")
        return redirect("person_attendance_list")

    qs = _filtered_attendance_qs(search_query, date_filter)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "attendance"
    if date_filter:
        label += f"_{date_filter}"
    filename_base = f"{label}_{stamp}"

    attachment_name = ""
    attachment_bytes = b""
    mime = "text/plain"

    if fmt == "csv":
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["Name", "Portal ID", "Date", "Check-In", "Check-Out", "Duration"])

        for a in qs:
            w.writerow([
                a.person.name if a.person else "",
                getattr(a.person, "portal_id", "") if a.person else "",
                str(a.date) if a.date else "",
                a.check_in_time.strftime("%H:%M:%S") if a.check_in_time else "",
                a.check_out_time.strftime("%H:%M:%S") if a.check_out_time else "",
                _duration_text(a),
            ])

        attachment_name = f"{filename_base}.csv"
        attachment_bytes = sio.getvalue().encode("utf-8")
        mime = "text/csv"

    elif fmt == "xlsx":
        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance"
        ws.append(["Name", "Portal ID", "Date", "Check-In", "Check-Out", "Duration"])

        for a in qs:
            ws.append([
                a.person.name if a.person else "",
                getattr(a.person, "portal_id", "") if a.person else "",
                str(a.date) if a.date else "",
                a.check_in_time.strftime("%H:%M:%S") if a.check_in_time else "",
                a.check_out_time.strftime("%H:%M:%S") if a.check_out_time else "",
                _duration_text(a),
            ])

        out = io.BytesIO()
        wb.save(out)
        out.seek(0)

        attachment_name = f"{filename_base}.xlsx"
        attachment_bytes = out.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    elif fmt == "pdf":
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=landscape(A4))
        width, height = landscape(A4)

        y = height - 40
        c.setFont("Helvetica-Bold", 14)
        title = "Attendance Report"
        if date_filter:
            title += f" ({date_filter})"
        c.drawString(40, y, title)
        y -= 24

        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Filter: search='{search_query or '-'}'")
        y -= 18

        c.setFont("Helvetica-Bold", 10)
        headers = ["Name", "Portal ID", "Date", "In", "Out", "Duration"]
        x_positions = [40, 260, 360, 450, 520, 610]
        for x, h in zip(x_positions, headers):
            c.drawString(x, y, h)
        y -= 14
        c.setFont("Helvetica", 9)

        for a in qs[:2000]:
            if y < 40:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 9)

            name = a.person.name if a.person else ""
            portal = getattr(a.person, "portal_id", "") if a.person else ""
            dt = str(a.date) if a.date else ""
            cin = a.check_in_time.strftime("%H:%M:%S") if a.check_in_time else ""
            cout = a.check_out_time.strftime("%H:%M:%S") if a.check_out_time else ""
            dur = _duration_text(a)

            row = [name, str(portal), dt, cin, cout, str(dur)]
            for x, val in zip(x_positions, row):
                c.drawString(x, y, (val or "")[:28])
            y -= 12

        c.save()
        buf.seek(0)

        attachment_name = f"{filename_base}.pdf"
        attachment_bytes = buf.getvalue()
        mime = "application/pdf"

    else:
        messages.error(request, "Invalid export format.")
        return redirect("person_attendance_list")

    subject = "OHC Attendance Report"
    body = (
        "Hello,\n\n"
        "Please find attached the attendance report.\n\n"
        f"Filters:\n- Search: {search_query or '-'}\n- Date: {date_filter or '-'}\n\n"
        "Regards,\nOpened Heavens Chapel"
    )

    try:
        email = EmailMessage(subject=subject, body=body, to=[to_email])
        email.attach(attachment_name, attachment_bytes, mime)
        email.send(fail_silently=False)
        messages.success(request, f"Report sent to {to_email}.")
    except (ConnectionRefusedError, SMTPException, OSError):
        messages.error(
            request,
            "Email could not be sent. Please check SMTP settings (host/port/TLS/SSL) and credentials, then try again."
        )

    return redirect(request.POST.get("next") or reverse("person_attendance_list"))