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

import json
import os
import time
from collections import defaultdict, deque
from functools import wraps
from smtplib import SMTPException
from typing import Deque, Dict, List, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import pygame
from requests import RequestException

from django.conf import settings
from django.contrib import messages
from django.contrib.messages import get_messages
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.csrf import csrf_exempt

from .api_backend import (
    ACCESS_TOKEN_SESSION_KEY,
    REFRESH_TOKEN_SESSION_KEY,
    fetch_all_attendance_logs,
    fetch_attendance_page,
    get_all_cameras,
    get_all_people,
    get_camera,
    get_client,
    normalize_person,
    to_namespace,
    wrap_api_page,
)
from .api_client import AttendanceApiError
from .face_api_runtime import detect_and_encode, load_authorized_face_encodings, recognize_faces
from .utils import (
    fetch_user_data,
    fetch_user_data_by_id,
    register_user_on_portal,
    upload_person_photo_to_portal,
)
from django.views.decorators.http import require_GET

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


DESKTOP_USER_SESSION_KEY = "attendance_desktop_user"
DESKTOP_ROLE_SESSION_KEY = "attendance_desktop_role"
DESKTOP_NAME_SESSION_KEY = "attendance_desktop_name"


def _store_desktop_auth(request, payload: dict):
    request.session[ACCESS_TOKEN_SESSION_KEY] = payload.get("access", "")
    request.session[REFRESH_TOKEN_SESSION_KEY] = payload.get("refresh", "")
    request.session[DESKTOP_USER_SESSION_KEY] = {
        "user_id": payload.get("user_id"),
        "username": payload.get("username"),
        "email": payload.get("email"),
        "user_type": payload.get("user_type"),
        "role": payload.get("role"),
        "full_name": payload.get("full_name"),
        "is_superuser": payload.get("is_superuser", False),
    }
    request.session[DESKTOP_ROLE_SESSION_KEY] = payload.get("role") or payload.get("user_type") or ""
    request.session[DESKTOP_NAME_SESSION_KEY] = payload.get("full_name") or payload.get("username") or "Desktop User"
    request.session.modified = True


def _clear_desktop_auth(request):
    for key in (
        ACCESS_TOKEN_SESSION_KEY,
        REFRESH_TOKEN_SESSION_KEY,
        DESKTOP_USER_SESSION_KEY,
        DESKTOP_ROLE_SESSION_KEY,
        DESKTOP_NAME_SESSION_KEY,
    ):
        request.session.pop(key, None)
    request.session.modified = True


def _update_desktop_tokens(request, access_token: str | None, refresh_token: str | None):
    if access_token:
        request.session[ACCESS_TOKEN_SESSION_KEY] = access_token
    if refresh_token:
        request.session[REFRESH_TOKEN_SESSION_KEY] = refresh_token
    request.session.modified = True


def _is_desktop_authenticated(request) -> bool:
    return bool(request.session.get(ACCESS_TOKEN_SESSION_KEY))


def _is_staff_desktop_user(payload: dict) -> bool:
    role = str(payload.get("role") or payload.get("user_type") or "").lower()
    return role in {"superuser", "admin", "finance", "viewer"}


def _safe_next_url(request, candidate: str | None) -> str:
    if candidate and url_has_allowed_host_and_scheme(candidate, allowed_hosts={request.get_host()}):
        return candidate
    return reverse("home")


def desktop_login_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if _is_desktop_authenticated(request):
            return view_func(request, *args, **kwargs)
        if request.path.startswith("/api/"):
            return JsonResponse({"error": "Desktop login required."}, status=401)
        login_url = f"{reverse('desktop_login')}?next={quote(request.get_full_path())}"
        return redirect(login_url)

    return _wrapped


def _api_client(request=None):
    token_updater = None
    if request is not None:
        token_updater = lambda access_token, refresh_token=None: _update_desktop_tokens(
            request, access_token, refresh_token
        )
    return get_client(request=request, token_updater=token_updater)


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


def gen_frames(source, cam_config, api_client, known_encodings, person_by_index, max_fps=10, draw_boxes=True, draw_names=True, play_sound=True):
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
                    person_by_index,
                    test_encodings,
                    threshold=float(cam_config.threshold),
                )

                for person, box, dist in recognized:
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

                    if person is not None:
                        now = timezone.now()

                        hits[person.id].append(now)
                        recent_hits = [
                            t for t in hits[person.id]
                            if (now - t).total_seconds() <= STABLE_WINDOW_SECONDS
                        ]

                        if len(recent_hits) >= STABLE_HITS_REQUIRED:
                            last = last_marked.get(person.id)
                            if not last or (now - last).total_seconds() >= COOLDOWN_SECONDS:
                                try:
                                    outcome = api_client.face_check_in(
                                        {
                                            "person_id": person.id,
                                            "camera_id": cam_config.id,
                                            "min_checkout_seconds": 60,
                                        }
                                    )
                                    last_marked[person.id] = now
                                    if play_sound:
                                        play_success_sound()
                                    label = f"{person.name} ({outcome.get('status', 'updated')})"
                                except AttendanceApiError:
                                    label = f"{person.name} (api error)"
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


@desktop_login_required
def video_feed(request, cam_id):
    try:
        cam_config = get_camera(cam_id, request=request)
    except AttendanceApiError:
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame")
    src = cam_config.camera_source.strip()
    source = int(src) if src.isdigit() else src
    api_client = _api_client(request)
    known_encodings, person_by_index = load_authorized_face_encodings(request=request)

    draw_boxes = _get_bool_qs(request, "boxes", True)
    draw_names = _get_bool_qs(request, "names", True)
    play_sound = _get_bool_qs(request, "sound", True)

    return StreamingHttpResponse(
        gen_frames(
            source,
            cam_config,
            api_client,
            known_encodings,
            person_by_index,
            max_fps=10,
            draw_boxes=draw_boxes,
            draw_names=draw_names,
            play_sound=play_sound,
        ),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


@desktop_login_required
def camera_stream(request, cam_id: int):
    try:
        config = get_camera(cam_id, request=request)
    except AttendanceApiError as exc:
        messages.error(request, f"Could not load camera: {exc}")
        return redirect("camera_config_list")
    return render(request, "camera_stream.html", {"config": config})


@desktop_login_required
def stream_all_cameras(request):
    try:
        configs = get_all_cameras(request=request)
    except AttendanceApiError as exc:
        configs = []
        messages.error(request, f"Could not load cameras: {exc}")
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


@desktop_login_required
def camera_preview_feed(request, cam_id):
    try:
        cam_config = get_camera(cam_id, request=request)
    except AttendanceApiError:
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame")
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
@desktop_login_required
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

    try:
        outcome = _api_client(request).nfc_check_in(
            {"uid": uid, "camera_id": camera_id, "min_checkout_seconds": 60}
        )
        play_success_sound()
        return JsonResponse(outcome)
    except AttendanceApiError as exc:
        return JsonResponse({"error": str(exc)}, status=400)


# =========================================================
# UI / Pages (MATCH your urls.py)
# =========================================================
def desktop_login(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        next_url = _safe_next_url(request, request.POST.get("next"))

        if not username or not password:
            return render(
                request,
                "desktop_login.html",
                {"next_url": next_url, "error": "Username and password are required."},
            )

        try:
            payload = get_client().authenticate(username=username, password=password)
        except AttendanceApiError as exc:
            return render(
                request,
                "desktop_login.html",
                {"next_url": next_url, "error": f"Login failed: {exc}"},
            )

        if not _is_staff_desktop_user(payload):
            _clear_desktop_auth(request)
            return render(
                request,
                "desktop_login.html",
                {
                    "next_url": next_url,
                    "error": "This desktop app is for admin and staff accounts only.",
                },
            )

        _store_desktop_auth(request, payload)
        messages.success(request, f"Signed in as {request.session.get(DESKTOP_NAME_SESSION_KEY)}.")
        return redirect(next_url)

    if _is_desktop_authenticated(request):
        return redirect(_safe_next_url(request, request.GET.get("next")))

    _clear_desktop_auth(request)
    list(get_messages(request))

    return render(request, "desktop_login.html", {"next_url": _safe_next_url(request, request.GET.get("next"))})


def desktop_logout(request):
    _clear_desktop_auth(request)
    messages.success(request, "Desktop session signed out.")
    return redirect("desktop_login")


@desktop_login_required
def home(request):
    try:
        overview = _api_client(request).overview()
        logs = fetch_all_attendance_logs(request=request)
        context = {
            "total_persons": overview.get("total_persons", 0),
            "total_attendance": overview.get("total_attendance", 0),
            "total_check_ins": sum(1 for item in logs if getattr(item, "check_in_time", None)),
            "total_check_outs": sum(1 for item in logs if getattr(item, "check_out_time", None)),
            "total_cameras": overview.get("total_cameras", 0),
            "desktop_user_name": request.session.get(DESKTOP_NAME_SESSION_KEY, "Desktop User"),
            "desktop_user_role": request.session.get(DESKTOP_ROLE_SESSION_KEY, ""),
        }
    except AttendanceApiError as exc:
        messages.error(request, f"Attendance API unavailable: {exc}")
        context = {
            "total_persons": 0,
            "total_attendance": 0,
            "total_check_ins": 0,
            "total_check_outs": 0,
            "total_cameras": 0,
            "desktop_user_name": request.session.get(DESKTOP_NAME_SESSION_KEY, "Desktop User"),
            "desktop_user_role": request.session.get(DESKTOP_ROLE_SESSION_KEY, ""),
        }
    return render(request, "home.html", context)


@desktop_login_required
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


@desktop_login_required
def register_user(request):
    try:
        cams = get_all_cameras(request=request)
    except AttendanceApiError as exc:
        cams = []
        messages.error(request, f"Could not load camera settings: {exc}")

    if request.method == "GET":
        portal_id = request.GET.get("portal_id")
        name = request.GET.get("name")

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
            person = _api_client(request).create_person(
                {
                    "name": name,
                    "portal_id": portal_id,
                    "image_data": image_data,
                    "authorized": True,
                }
            )
        except AttendanceApiError as exc:
            return render(
                request,
                "register_user.html",
                {
                    "error": f"Could not save member to API: {exc}",
                    "portal_id": portal_id,
                    "name": name,
                    "camera_configs": cams,
                },
            )

        # Upload same image to ChurchCRM behind the scenes
        if portal_id:
            upload_result = upload_person_photo_to_portal(portal_id, image_data)
            print("PORTAL PHOTO RESULT:", upload_result)

            if upload_result.get("ok"):
                messages.success(
                    request,
                    f"{person.get('name', name)} registered successfully. Portal photo updated successfully."
                )
            else:
                messages.warning(
                    request,
                    f"{person.get('name', name)} registered in attendance, but portal photo upload failed: {upload_result.get('error', 'Unknown error')}"
                )
        else:
            messages.success(request, f"{person.get('name', name)} registered successfully.")

        return redirect("person_list")

    return render(request, "register_user.html", {"camera_configs": cams})


def success_page(request):
    return render(request, "selfie_success.html")


# =========================================================
# People management (MATCH your urls.py)
# =========================================================
@desktop_login_required
def person_list(request):
    search = request.GET.get("search", "").strip()
    try:
        persons_qs = get_all_people(search=search, request=request)
    except AttendanceApiError as exc:
        persons_qs = []
        messages.error(request, f"Could not load members: {exc}")

    paginator = Paginator(persons_qs, 10)  # 10 per page
    page_number = request.GET.get("page")
    persons = paginator.get_page(page_number)

    return render(request, "user_list.html", {"persons": persons})


@desktop_login_required
def person_detail(request, pk: int):
    try:
        person = normalize_person(_api_client(request).get_person(pk))
    except AttendanceApiError as exc:
        messages.error(request, f"Could not load member: {exc}")
        return redirect("person_list")
    user_data = fetch_user_data_by_id(person.portal_id)
    return render(request, "user_detail.html", {"person": person, "user_data": user_data})


@desktop_login_required
def person_authorize(request, pk: int):
    if request.method == "POST":
        authorized = request.POST.get("authorized") in ("1", "true", "True", "on", "yes")
        try:
            _api_client(request).authorize_person(pk, authorized)
            messages.success(request, "Member authorization updated.")
        except AttendanceApiError as exc:
            messages.error(request, f"Could not update authorization: {exc}")
        return redirect("person_detail", pk=pk)

    return redirect("person_detail", pk=pk)


@desktop_login_required
def person_delete(request, pk: int):
    try:
        person = normalize_person(_api_client(request).get_person(pk))
    except AttendanceApiError as exc:
        messages.error(request, f"Could not load member: {exc}")
        return redirect("person_list")

    if request.method == "POST":
        try:
            _api_client(request).delete_person(pk)
            messages.success(request, "Member deleted successfully.")
            return redirect("person_list")
        except AttendanceApiError as exc:
            messages.error(request, f"Could not delete member: {exc}")
            return redirect("person_detail", pk=pk)

    return render(request, "user_delete_confirm.html", {"person": person})


# =========================================================
# Attendance views (MATCH your urls.py)
# =========================================================

@desktop_login_required
def person_attendance_list(request):
    search_query = request.GET.get("search", "").strip()
    date_filter = request.GET.get("attendance_date", "").strip()
    page_size = int(request.GET.get("page_size", 25) or 25)
    page_size = max(10, min(page_size, 200))  # clamp
    page_number = int(request.GET.get("page", 1) or 1)
    try:
        page_obj = fetch_attendance_page(search_query, date_filter, page_number, page_size, request=request)
    except AttendanceApiError as exc:
        messages.error(request, f"Could not load attendance logs: {exc}")
        page_obj = wrap_api_page({"count": 0}, [], 1, page_size)

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


@desktop_login_required
def capture_and_recognize(request):
    # Your system uses the stream endpoints now; keep this route but redirect somewhere useful.
    return redirect("camera_config_list")


# =========================================================
# Camera configuration UI (MATCH your urls.py)
# =========================================================
@desktop_login_required
def camera_config_create(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        camera_source = request.POST.get("camera_source", "").strip()
        threshold = request.POST.get("threshold", "").strip()

        if not name or not camera_source or not threshold:
            messages.error(request, "All fields are required.")
            return render(request, "camera_config_form.html")

        try:
            _api_client(request).create_camera(
                {"name": name, "camera_source": camera_source, "threshold": threshold}
            )
            messages.success(request, "Camera configuration saved successfully.")
            return redirect("camera_config_list")
        except AttendanceApiError as exc:
            messages.error(request, f"Could not save camera configuration: {exc}")
            return render(request, "camera_config_form.html", {"config": to_namespace({"name": name, "camera_source": camera_source, "threshold": threshold})})

    return render(request, "camera_config_form.html")


@desktop_login_required
def camera_config_list(request):
    try:
        configs = get_all_cameras(request=request)
    except AttendanceApiError as exc:
        configs = []
        messages.error(request, f"Could not load cameras: {exc}")
    return render(request, "camera_config_list.html", {"configs": configs})


@desktop_login_required
def camera_config_update(request, pk: int):
    if request.method == "POST":
        try:
            _api_client(request).update_camera(
                pk,
                {
                    "name": request.POST.get("name"),
                    "camera_source": request.POST.get("camera_source", ""),
                    "threshold": request.POST.get("threshold", "0.6"),
                },
            )
            messages.success(request, "Camera configuration updated.")
            return redirect("camera_config_list")
        except AttendanceApiError as exc:
            messages.error(request, f"Could not update camera configuration: {exc}")
            return render(
                request,
                "camera_config_form.html",
                {"config": to_namespace({"id": pk, "name": request.POST.get("name"), "camera_source": request.POST.get("camera_source", ""), "threshold": request.POST.get("threshold", "0.6")})},
            )

    try:
        config = get_camera(pk, request=request)
    except AttendanceApiError as exc:
        messages.error(request, f"Could not load camera configuration: {exc}")
        return redirect("camera_config_list")
    return render(request, "camera_config_form.html", {"config": config})


@desktop_login_required
def camera_config_delete(request, pk: int):
    if request.method == "POST":
        try:
            _api_client(request).delete_camera(pk)
            messages.success(request, "Camera configuration deleted.")
        except AttendanceApiError as exc:
            messages.error(request, f"Could not delete camera configuration: {exc}")
    return redirect("camera_config_list")

@desktop_login_required
def api_attendance_monitor(request):
    try:
        return JsonResponse(_api_client(request).monitor())
    except AttendanceApiError as exc:
        return JsonResponse({"error": str(exc)}, status=503)



@require_GET
@desktop_login_required
def attendance_today_api(request):
    """
    Returns today's attendance events (latest first).
    Supports:
      - ?since=<epoch_ms>  (optional) for incremental updates
      - ?limit=60
    """
    try:
        params = {"limit": int(request.GET.get("limit", 60) or 60)}
        since = request.GET.get("since")
        if since:
            params["since"] = since
        return JsonResponse(_api_client(request).today(**params))
    except AttendanceApiError as exc:
        return JsonResponse({"error": str(exc)}, status=503)


@desktop_login_required
def attendance_delete(request, pk: int):
    """
    Deletes an attendance record. POST-only + CSRF protected.
    Redirects back to the referring page (or a safe fallback).
    """
    try:
        _api_client(request).delete_attendance_log(pk)
        messages.success(request, "Deleted attendance record.")
    except AttendanceApiError as exc:
        messages.error(request, f"Could not delete attendance record: {exc}")
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


@desktop_login_required
def attendance_export_download(request):
    search_query = request.GET.get("search", "").strip()
    date_filter = request.GET.get("attendance_date", "").strip()
    fmt = (request.GET.get("format") or "csv").lower().strip()
    qs = fetch_all_attendance_logs(search_query, date_filter, request=request)

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
                str(a.date) if getattr(a, "date", None) else "",
                str(a.check_in_time) if getattr(a, "check_in_time", None) else "",
                str(a.check_out_time) if getattr(a, "check_out_time", None) else "",
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
                str(a.date) if getattr(a, "date", None) else "",
                str(a.check_in_time) if getattr(a, "check_in_time", None) else "",
                str(a.check_out_time) if getattr(a, "check_out_time", None) else "",
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
            dt = str(a.date) if getattr(a, "date", None) else ""
            cin = str(a.check_in_time) if getattr(a, "check_in_time", None) else ""
            cout = str(a.check_out_time) if getattr(a, "check_out_time", None) else ""
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


@desktop_login_required
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

    qs = fetch_all_attendance_logs(search_query, date_filter, request=request)

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
                str(a.date) if getattr(a, "date", None) else "",
                str(a.check_in_time) if getattr(a, "check_in_time", None) else "",
                str(a.check_out_time) if getattr(a, "check_out_time", None) else "",
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
                str(a.date) if getattr(a, "date", None) else "",
                str(a.check_in_time) if getattr(a, "check_in_time", None) else "",
                str(a.check_out_time) if getattr(a, "check_out_time", None) else "",
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
            dt = str(a.date) if getattr(a, "date", None) else ""
            cin = str(a.check_in_time) if getattr(a, "check_in_time", None) else ""
            cout = str(a.check_out_time) if getattr(a, "check_out_time", None) else ""
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
