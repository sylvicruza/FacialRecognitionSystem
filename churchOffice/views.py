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
import subprocess
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from functools import wraps
from smtplib import SMTPException
from typing import Deque, Dict, List, Tuple
from urllib.parse import quote, urlencode

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
from django.views.decorators.http import require_POST

from .api_backend import (
    ACCESS_TOKEN_SESSION_KEY,
    API_BASE_URL_SESSION_KEY,
    REFRESH_TOKEN_SESSION_KEY,
    extract_results,
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
from .api_client import AttendanceApiError, AttendanceAuthError
from .face_api_runtime import detect_and_encode, load_authorized_face_encodings, recognize_faces
from .utils import (
    fetch_user_data,
    fetch_user_data_by_id,
    register_user_on_portal,
    upload_person_photo_to_portal,
)
from django.views.decorators.http import require_GET

from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.core.paginator import Paginator
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse

from openpyxl import Workbook, load_workbook
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
import csv
import io


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
DESKTOP_ORGANIZATION_SESSION_KEY = "attendance_desktop_organization"
DESKTOP_PLAN_SESSION_KEY = "attendance_desktop_plan"
ACTIVE_ATTENDANCE_SESSION_KEY = "active_attendance_session_id"
ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY = "active_attendance_session_cleared"
DESKTOP_SYNC_STATUS_SESSION_KEY = "attendance_desktop_sync_status"
NFC_READER_SETTINGS_SESSION_KEY = "attendance_nfc_reader_settings"
NFC_READER_RUNTIME_SESSION_KEY = "attendance_nfc_reader_runtime"


def _store_desktop_auth(request, payload: dict):
    request.session[ACCESS_TOKEN_SESSION_KEY] = payload.get("access", "")
    request.session[REFRESH_TOKEN_SESSION_KEY] = payload.get("refresh", "")
    user_payload = payload.get("user") or payload
    profile_payload = payload.get("staff_profile") or {}
    organization_payload = payload.get("organization") or {}
    plan_payload = payload.get("plan") or {}
    is_superuser = user_payload.get("is_superuser", payload.get("is_superuser", False))
    if is_superuser and not plan_payload.get("name"):
        plan_payload = {
            "name": "Superuser",
            "tier": "enterprise",
            "allow_manual": True,
            "allow_swipe": True,
            "allow_qr": True,
            "allow_nfc": True,
            "allow_face_recognition": True,
            "allow_geotracking": True,
        }
    request.session[DESKTOP_USER_SESSION_KEY] = {
        "user_id": user_payload.get("id") or payload.get("user_id"),
        "username": user_payload.get("username") or payload.get("username"),
        "email": user_payload.get("email") or payload.get("email"),
        "user_type": payload.get("user_type"),
        "role": profile_payload.get("role") or payload.get("role"),
        "full_name": user_payload.get("name") or payload.get("full_name"),
        "is_superuser": is_superuser,
    }
    request.session[DESKTOP_ORGANIZATION_SESSION_KEY] = organization_payload or {}
    request.session[DESKTOP_PLAN_SESSION_KEY] = plan_payload or {}
    request.session[DESKTOP_ROLE_SESSION_KEY] = profile_payload.get("role") or payload.get("role") or payload.get("user_type") or ""
    request.session[DESKTOP_NAME_SESSION_KEY] = user_payload.get("name") or payload.get("full_name") or user_payload.get("username") or "Desktop User"
    request.session.modified = True
    try:
        _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
    except Exception:
        pass


def _clear_desktop_auth(request):
    for key in (
        ACCESS_TOKEN_SESSION_KEY,
        REFRESH_TOKEN_SESSION_KEY,
        DESKTOP_USER_SESSION_KEY,
        DESKTOP_ROLE_SESSION_KEY,
        DESKTOP_NAME_SESSION_KEY,
        DESKTOP_ORGANIZATION_SESSION_KEY,
        DESKTOP_PLAN_SESSION_KEY,
        ACTIVE_ATTENDANCE_SESSION_KEY,
        ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY,
        DESKTOP_SYNC_STATUS_SESSION_KEY,
        NFC_READER_SETTINGS_SESSION_KEY,
        NFC_READER_RUNTIME_SESSION_KEY,
    ):
        request.session.pop(key, None)
    request.session.modified = True
    try:
        _reader_bridge_config_path().unlink(missing_ok=True)
        _reader_bridge_runtime_path().unlink(missing_ok=True)
    except OSError:
        pass


def _default_nfc_reader_runtime() -> dict:
    return {
        "status": "Waiting",
        "last_uid": "",
        "last_scan_at": "",
        "last_error": "",
        "last_action": "No reader activity yet.",
        "pending_enrollment_uid": "",
        "pending_enrollment_url": "",
    }


def _reader_runtime(request) -> dict:
    file_payload = {}
    try:
        file_payload = json.loads(_reader_bridge_runtime_path().read_text(encoding="utf-8"))
    except Exception:
        file_payload = {}
    return {
        **_default_nfc_reader_runtime(),
        **(request.session.get(NFC_READER_RUNTIME_SESSION_KEY) or {}),
        **(file_payload or {}),
    }


def _store_reader_runtime(request, payload: dict):
    request.session[NFC_READER_RUNTIME_SESSION_KEY] = payload
    request.session.modified = True
    try:
        _reader_bridge_runtime_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _reader_bridge_config_path():
    return settings.RUNTIME_DIR / "nfc_reader_bridge_config.json"


def _reader_bridge_runtime_path():
    return settings.RUNTIME_DIR / "nfc_reader_bridge_runtime.json"


def _reader_bridge_pid_path():
    return settings.RUNTIME_DIR / "nfc_reader_bridge.pid"


def _reader_bridge_running_pid():
    try:
        pid = int(_reader_bridge_pid_path().read_text(encoding="utf-8").strip())
    except Exception:
        return None
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            check=False,
        )
        output = (completed.stdout or "").strip()
        if completed.returncode != 0 or not output or "No tasks are running" in output:
            raise RuntimeError("bridge process is not running")
    except Exception:
        try:
            _reader_bridge_pid_path().unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return pid


def _launch_reader_bridge():
    existing_pid = _reader_bridge_running_pid()
    if existing_pid:
        return existing_pid, False

    bridge_script = settings.BASE_DIR / "run_acr122u_bridge.py"
    venv_python = settings.BASE_DIR / ".venv" / "Scripts" / "python.exe"
    python_executable = venv_python if venv_python.exists() else Path(sys.executable)
    env = os.environ.copy()
    env["DJANGO_SETTINGS_MODULE"] = "ohc_time_attendance.settings"
    env["ATTENDANCE_DESKTOP_MODE"] = "1"
    env["ATTENDANCE_DESKTOP_DATA_DIR"] = str(settings.RUNTIME_DIR)
    creationflags = 0
    for flag_name in ("CREATE_NO_WINDOW", "DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= int(getattr(subprocess, flag_name, 0))
    process = subprocess.Popen(
        [str(python_executable), str(bridge_script)],
        cwd=str(settings.BASE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    return process.pid, True


def _sync_reader_bridge_config(request, *, active_session=None, reader_settings=None, resolve_active=True):
    if active_session is None and resolve_active:
        active_session = _get_active_attendance_session(request)
    reader_settings = reader_settings or request.session.get(NFC_READER_SETTINGS_SESSION_KEY) or {}
    payload = {
        "reader_name": reader_settings.get("reader_name", "Entrance Reader"),
        "reader_mode": reader_settings.get("reader_mode", "keyboard"),
        "reader_suffix": reader_settings.get("reader_suffix", "Enter"),
        "reader_prefix": reader_settings.get("reader_prefix", ""),
        "station_identifier": reader_settings.get("station_identifier", ""),
        "capture_mode": reader_settings.get("capture_mode", "check_in"),
        "backend_api_base_url": request.session.get(API_BASE_URL_SESSION_KEY, settings.ATTENDANCE_API_BASE_URL),
        "access_token": request.session.get(ACCESS_TOKEN_SESSION_KEY, ""),
        "refresh_token": request.session.get(REFRESH_TOKEN_SESSION_KEY, ""),
        "active_session_id": getattr(active_session, "id", None),
        "active_session_label": (
            getattr(active_session, "session_name", None)
            or getattr(active_session, "event_name", None)
            or ""
        ),
        "enrollment_base_url": request.build_absolute_uri(reverse("bulk_nfc_enrollment")),
        "updated_at": timezone.localtime().isoformat(),
    }
    try:
        _reader_bridge_config_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _clear_queued_messages(request):
    list(get_messages(request))
    storage = getattr(request, "_messages", None)
    if storage is not None:
        storage.used = True
        if hasattr(storage, "_queued_messages"):
            storage._queued_messages = []
        try:
            storage._loaded_messages = []
        except AttributeError:
            pass


def _clear_desktop_auth_and_messages(request):
    _clear_desktop_auth(request)
    _clear_queued_messages(request)


def _update_desktop_tokens(request, access_token: str | None, refresh_token: str | None):
    if access_token is None and refresh_token is None:
        request.session.pop(ACCESS_TOKEN_SESSION_KEY, None)
        request.session.pop(REFRESH_TOKEN_SESSION_KEY, None)
    elif access_token:
        request.session[ACCESS_TOKEN_SESSION_KEY] = access_token
    if refresh_token:
        request.session[REFRESH_TOKEN_SESSION_KEY] = refresh_token
    request.session.modified = True


def _is_desktop_authenticated(request) -> bool:
    return bool(request.session.get(ACCESS_TOKEN_SESSION_KEY))


def _is_staff_desktop_user(payload: dict) -> bool:
    if payload.get("user", {}).get("is_superuser") or payload.get("is_superuser"):
        return True
    profile = payload.get("staff_profile") or {}
    role = str(profile.get("role") or payload.get("role") or payload.get("user_type") or "").lower()
    return role in {"owner", "admin", "staff", "superuser", "finance", "viewer"}


def _desktop_role_slug(request) -> str:
    if request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"):
        return "superuser"
    return str(request.session.get(DESKTOP_ROLE_SESSION_KEY, "")).strip().lower()


def _desktop_is_owner_or_admin(request) -> bool:
    return _desktop_role_slug(request) in {"owner", "admin", "superuser"}


def _desktop_can_view_audit(request) -> bool:
    return _desktop_is_owner_or_admin(request)


def _desktop_can_view_platform_oversight(request) -> bool:
    return bool(request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"))


def _desktop_can_manage_authorization(request) -> bool:
    return _desktop_is_owner_or_admin(request)


def _safe_next_url(request, candidate: str | None) -> str:
    if candidate and url_has_allowed_host_and_scheme(candidate, allowed_hosts={request.get_host()}):
        return candidate
    return reverse("home")


def desktop_login_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if _is_desktop_authenticated(request):
            try:
                return view_func(request, *args, **kwargs)
            except AttendanceAuthError:
                return _expire_desktop_session(request)
        if request.path.startswith("/api/"):
            return JsonResponse({"error": "Desktop login required."}, status=401)
        login_url = f"{reverse('desktop_login')}?next={quote(request.get_full_path())}"
        return redirect(login_url)

    return _wrapped


def _expire_desktop_session(request):
    _clear_desktop_auth_and_messages(request)
    if request.path.startswith("/api/"):
        return JsonResponse({"error": "Desktop session expired. Please sign in again."}, status=401)
    login_url = f"{reverse('desktop_login')}?next={quote(request.get_full_path())}"
    return redirect(login_url)


def _is_auth_api_error(exc: AttendanceApiError) -> bool:
    return isinstance(exc, AttendanceAuthError) or getattr(exc, "status_code", None) == 401


def _redirect_if_auth_error(request, exc: AttendanceApiError):
    if _is_auth_api_error(exc):
        return _expire_desktop_session(request)
    return None


def _api_client(request=None):
    token_updater = None
    if request is not None:
        token_updater = lambda access_token, refresh_token=None: _update_desktop_tokens(
            request, access_token, refresh_token
        )
    return get_client(request=request, token_updater=token_updater)


def _normalize_backend_url(url: str | None) -> str:
    value = (url or "").strip().rstrip("/")
    if value in {
        "https://heavensconnect.onrender.com/api/attendance",
    }:
        return settings.ATTENDANCE_API_BASE_URL.rstrip("/")
    return value


def _desktop_backend_url(request) -> str:
    session_url = _normalize_backend_url(request.session.get(API_BASE_URL_SESSION_KEY))
    if session_url:
        if session_url != request.session.get(API_BASE_URL_SESSION_KEY):
            request.session[API_BASE_URL_SESSION_KEY] = session_url
            request.session.modified = True
        return session_url
    return settings.ATTENDANCE_API_BASE_URL.rstrip("/")


def _set_desktop_sync_status(request, *, ok: bool, message: str = ""):
    request.session[DESKTOP_SYNC_STATUS_SESSION_KEY] = {
        "ok": ok,
        "message": message,
        "checked_at": timezone.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    request.session.modified = True


def _friendly_api_error(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, RequestException) or any(
        marker in text.lower()
        for marker in ("connection refused", "failed to establish", "name resolution", "timed out", "max retries")
    ):
        return "The backend is offline or unreachable. Check Settings, internet connection, or the hosted API."
    if isinstance(exc, AttendanceApiError):
        payload = getattr(exc, "payload", None)
        detail = ""
        if isinstance(payload, dict):
            for key in ("detail", "error", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    detail = value.strip()
                    break
                if isinstance(value, list) and value:
                    detail = ", ".join(str(item).strip() for item in value if str(item).strip())
                    if detail:
                        break
        elif isinstance(payload, list) and payload:
            detail = ", ".join(str(item).strip() for item in payload if str(item).strip())
        if detail:
            lower_detail = detail.lower()
            if "uid not registered for this event session" in lower_detail:
                return "This NFC tag was read successfully, but it is not assigned to a member for the active event session yet."
            if "uid is required" in lower_detail:
                return "No NFC tag UID was received. Tap the card again or enter the UID manually."
            return detail
    return text


def _desktop_plan_allows(request, feature: str) -> bool:
    plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}
    if request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"):
        return True
    return bool(plan.get(feature))


def _desktop_plan_allows_or_default(request, feature: str, default: bool = False) -> bool:
    if request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"):
        return True
    plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}
    if feature not in plan:
        return default
    return bool(plan.get(feature))


def _desktop_plan_tier(request) -> str:
    return str((request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}).get("tier") or "").lower()


def _plan_allows_member_import(request) -> bool:
    if request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"):
        return True
    return _desktop_plan_tier(request) in {"standard", "professional", "enterprise"}


def _limit_value(plan: dict, key: str):
    value = (plan or {}).get(key)
    if value in (None, "", "Unlimited"):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _limit_status(current: int, limit: int | None):
    current = int(current or 0)
    remaining = None if limit is None else max(0, limit - current)
    return {
        "current": current,
        "limit": limit,
        "remaining": remaining,
        "is_limited": limit is not None,
        "is_reached": limit is not None and current >= limit,
        "percent": 0 if limit is None else min(100, int((current / limit) * 100)) if limit else 0,
    }


def _member_limit_status(request, overview=None, organization=None):
    plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}
    if request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"):
        return _limit_status(0, None)

    current = 0
    if overview is not None:
        current = overview.get("total_persons", 0) or 0
    elif organization:
        current = organization.get("member_count", 0) or 0
    else:
        try:
            current = _api_client(request).overview().get("total_persons", 0) or 0
        except AttendanceApiError:
            current = (request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}).get("member_count", 0) or 0
    return _limit_status(current, _limit_value(plan, "max_members"))


def _staff_limit_status(request, organization=None):
    plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}
    if request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"):
        return _limit_status(0, None)
    current = (organization or request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}).get("staff_count", 0) or 0
    return _limit_status(current, _limit_value(plan, "max_staff_users"))


def _render_plan_locked(request, title: str, message: str, required_plan: str, back_url: str = "getting_started"):
    return render(
        request,
        "plan_locked.html",
        {
            "lock_title": title,
            "lock_message": message,
            "required_plan": required_plan,
            "current_plan": request.session.get(DESKTOP_PLAN_SESSION_KEY) or {},
            "back_url": back_url,
        },
    )


ATTENDANCE_METHOD_CATALOG = [
    {
        "key": "manual",
        "name": "Manual",
        "feature": "allow_manual",
        "plan": "Starter",
        "icon": "fa-list-check",
        "route": "manual_attendance",
        "description": "Select members and mark present or absent.",
    },
    {
        "key": "swipe",
        "name": "Swipe",
        "feature": "allow_swipe",
        "plan": "Starter",
        "icon": "fa-hand-pointer",
        "route": "",
        "description": "Mobile app swipe-through roll-call.",
    },
    {
        "key": "qr",
        "name": "QR Code",
        "feature": "allow_qr",
        "plan": "Standard",
        "icon": "fa-qrcode",
        "route": "qr_attendance",
        "description": "Members scan an event QR and check themselves in.",
    },
    {
        "key": "nfc",
        "name": "NFC",
        "feature": "allow_nfc",
        "plan": "Professional",
        "icon": "fa-wifi",
        "route": "nfc_attendance",
        "description": "Tap cards or tags to check members in.",
    },
    {
        "key": "face",
        "name": "Facial Recognition",
        "feature": "allow_face_recognition",
        "plan": "Professional",
        "icon": "fa-face-smile",
        "route": "stream_all_cameras",
        "description": "Use local cameras while records sync to the backend.",
    },
    {
        "key": "geo",
        "name": "GeoTracking",
        "feature": "allow_geotracking",
        "plan": "Enterprise",
        "icon": "fa-location-dot",
        "route": "",
        "description": "Mobile location-aware attendance for field teams.",
    },
]


def _attendance_method_cards(request):
    cards = []
    for item in ATTENDANCE_METHOD_CATALOG:
        allowed = _desktop_plan_allows(request, item["feature"])
        route = item.get("route")
        href = reverse(route) if allowed and route else ""
        if allowed and href:
            status = "available"
            status_label = "Included"
            action_label = "Open"
            note = "Available in your current plan."
        elif allowed:
            status = "mobile"
            status_label = "Included"
            action_label = "Mobile app"
            note = "Included in your plan, used from the mobile app."
        else:
            status = "locked"
            status_label = "Not included"
            action_label = f"{item['plan']}+"
            note = f"Upgrade to {item['plan']} or above to use this method."
        cards.append({
            **item,
            "allowed": allowed,
            "href": href,
            "status": status,
            "status_label": status_label,
            "action_label": action_label,
            "note": note,
        })
    return cards


def _preferred_attendance_url(request):
    for method in _attendance_method_cards(request):
        if method["allowed"] and method["href"]:
            return method["href"]
    return reverse("manual_attendance")


def _external_member_integration_enabled(request):
    organization = request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}
    org_enabled = bool(
        organization.get("external_integration_enabled")
        or organization.get("member_integration_enabled")
        or organization.get("portal_integration_enabled")
    )
    return bool(getattr(settings, "EXTERNAL_MEMBER_INTEGRATION_ENABLED", False) or org_enabled)


def _member_registration_context(request, **extra):
    integration_enabled = _external_member_integration_enabled(request)
    context = {
        "integration_enabled": integration_enabled,
        "integration_name": "External member integration",
        "show_direct_register": True,
        "searched_last_name": "",
        "prefill_first_name": "",
        "prefill_last_name": "",
        "prefill_gender": "",
        "prefill_email": "",
        "prefill_cellPhone": "",
        "prefill_address1": "",
        "user_data": [],
        "search_performed": False,
        "show_create_option": False,
        "show_portal_register": False,
        "error": "",
    }
    context.update(extra)
    return context


def _build_setup_context(request, overview=None, active_session=None):
    client = _api_client(request)
    organization = request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}
    plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}

    if overview is None:
        overview = client.overview()
    if active_session is None:
        active_session = _get_active_attendance_session(request)

    branches = [to_namespace(item) for item in extract_results(client.list_branches())]
    events = [to_namespace(item) for item in extract_results(client.list_events())]
    open_sessions = [
        to_namespace(item)
        for item in extract_results(client.list_sessions(status="open"))
    ]

    total_members = int(overview.get("total_persons", 0) or 0)
    total_attendance = int(overview.get("total_attendance", 0) or 0)
    total_cameras = int(overview.get("total_cameras", 0) or 0)
    method_url = _preferred_attendance_url(request)

    checklist = [
        {
            "title": "Confirm organization profile",
            "description": "Add organization contact details and your first branch.",
            "done": bool(organization.get("name")),
            "href": reverse("organization_settings"),
            "cta": "Open organization",
            "icon": "fa-building-user",
        },
        {
            "title": "Add a branch or location",
            "description": "Locations make attendance reports easier to filter later.",
            "done": bool(branches),
            "href": reverse("organization_settings"),
            "cta": "Add branch",
            "icon": "fa-location-dot",
        },
        {
            "title": "Add members",
            "description": "Register members individually or import them from CSV.",
            "done": total_members > 0,
            "href": reverse("register_user"),
            "cta": "Add members",
            "icon": "fa-users",
        },
        {
            "title": "Create an event",
            "description": "Attendance should belong to an event like Sunday Service or Staff Shift.",
            "done": bool(events),
            "href": reverse("attendance_sessions"),
            "cta": "Create event",
            "icon": "fa-calendar-days",
        },
        {
            "title": "Open an attendance session",
            "description": "A session is the actual occurrence where attendance is marked.",
            "done": bool(active_session or open_sessions),
            "href": reverse("attendance_sessions"),
            "cta": "Open session",
            "icon": "fa-calendar-check",
        },
    ]
    completed = sum(1 for item in checklist if item["done"])
    next_step = next((item for item in checklist if not item["done"]), None)
    for item in checklist:
        item["is_next"] = item is next_step and not item["done"]
        if item["done"]:
            item["status"] = "done"
            item["status_label"] = "Complete"
        elif item["is_next"]:
            item["status"] = "next"
            item["status_label"] = "Do this next"
        else:
            item["status"] = "pending"
            item["status_label"] = "Pending"

    return {
        "organization": organization,
        "plan": plan,
        "attendance_methods": _attendance_method_cards(request),
        "setup_checklist": checklist,
        "setup_completed": completed,
        "setup_total": len(checklist),
        "setup_progress_percent": int((completed / len(checklist)) * 100) if checklist else 0,
        "setup_is_complete": bool(checklist) and completed == len(checklist),
        "setup_next_step": next_step,
        "setup_recommended_action": {
            "title": "Mark first attendance",
            "description": "Your setup is ready. Use any included method when the event begins.",
            "done": total_attendance > 0,
            "href": method_url,
            "cta": "Mark attendance",
        },
        "setup_counts": {
            "branches": len(branches),
            "events": len(events),
            "open_sessions": len(open_sessions),
            "members": total_members,
            "attendance": total_attendance,
            "cameras": total_cameras,
        },
    }


def _refresh_desktop_context(request):
    profile_payload = _api_client(request).me()
    user_payload = request.session.get(DESKTOP_USER_SESSION_KEY) or {}
    auth_payload = {
        "access": request.session.get(ACCESS_TOKEN_SESSION_KEY, ""),
        "refresh": request.session.get(REFRESH_TOKEN_SESSION_KEY, ""),
        "user": {**user_payload, **(profile_payload.get("user") or {})},
        "staff_profile": profile_payload.get("staff_profile") or {},
        "organization": profile_payload.get("organization") or {},
        "plan": profile_payload.get("plan") or {},
    }
    _store_desktop_auth(request, auth_payload)
    return profile_payload


def _active_session_id(request):
    session_id = request.session.get(ACTIVE_ATTENDANCE_SESSION_KEY)
    try:
        return int(session_id) if session_id else None
    except (TypeError, ValueError):
        request.session.pop(ACTIVE_ATTENDANCE_SESSION_KEY, None)
        _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
        return None


def _session_ends_at(session):
    raw_value = getattr(session, "ends_at", None)
    if not raw_value:
        return None
    if hasattr(raw_value, "tzinfo"):
        value = raw_value
    else:
        try:
            value = datetime.fromisoformat(str(raw_value))
        except ValueError:
            return None
    if timezone.is_naive(value):
        value = timezone.make_aware(value, timezone.get_current_timezone())
    return timezone.localtime(value)


def _session_has_ended(session) -> bool:
    ends_at = _session_ends_at(session)
    if not ends_at:
        return False
    return timezone.now() >= ends_at


def _session_is_open(session) -> bool:
    return str(getattr(session, "status", "") or "").strip().lower() == "open"


def _close_session_if_expired(request, session):
    if not session or not _session_is_open(session) or not _session_has_ended(session):
        return session
    try:
        updated = _api_client(request).update_session(int(getattr(session, "id")), {"status": "closed"})
        return to_namespace(updated)
    except AttendanceApiError as exc:
        if _is_auth_api_error(exc):
            raise
        return session


def _restore_live_open_session(request):
    if request.session.get(ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY):
        _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
        return None

    open_sessions = [
        to_namespace(item)
        for item in extract_results(_api_client(request).list_sessions(status="open"))
    ]
    live_sessions = []
    for session in open_sessions:
        refreshed = _close_session_if_expired(request, session)
        if _session_is_open(refreshed) and not _session_has_ended(refreshed):
            live_sessions.append(refreshed)

    if not live_sessions:
        request.session.pop(ACTIVE_ATTENDANCE_SESSION_KEY, None)
        _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
        return None

    live_sessions.sort(
        key=lambda item: (
            _session_ends_at(item) is None,
            _session_ends_at(item) or timezone.now(),
            getattr(item, "starts_at", None) or "",
        ),
        reverse=True,
    )
    selected = live_sessions[0]
    request.session[ACTIVE_ATTENDANCE_SESSION_KEY] = getattr(selected, "id", None)
    request.session.pop(ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY, None)
    request.session.modified = True
    _sync_reader_bridge_config(request, active_session=selected)
    return selected


def _get_active_attendance_session(request):
    session_id = _active_session_id(request)
    if not session_id:
        return _restore_live_open_session(request)

    try:
        session = to_namespace(_api_client(request).get_session(session_id))
    except AttendanceApiError as exc:
        if _is_auth_api_error(exc):
            raise
        request.session.pop(ACTIVE_ATTENDANCE_SESSION_KEY, None)
        _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
        return _restore_live_open_session(request)

    session = _close_session_if_expired(request, session)
    if not _session_is_open(session) or _session_has_ended(session):
        request.session.pop(ACTIVE_ATTENDANCE_SESSION_KEY, None)
        request.session.modified = True
        _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
        return _restore_live_open_session(request)
    return session


def _parse_datetime_local(value: str):
    value = (value or "").strip()
    if not value:
        return timezone.now()

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return timezone.now()

    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def _parse_optional_datetime_local(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return _parse_datetime_local(value)


def _datetime_local_value(value):
    if not value:
        return ""
    try:
        if hasattr(value, "strftime"):
            parsed = value
        else:
            raw = str(value).strip()
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                parsed = datetime.strptime(raw, "%d-%m-%Y %H:%M")
    except (TypeError, ValueError):
        return ""
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
    return timezone.localtime(parsed).strftime("%Y-%m-%dT%H:%M")


def _coerce_session_flag(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _session_method_plan_defaults(request):
    return {
        "self": _desktop_plan_allows(request, "allow_qr"),
        "qr": _desktop_plan_allows(request, "allow_qr"),
        "swipe": _desktop_plan_allows_or_default(request, "allow_swipe", default=True),
        "nfc": _desktop_plan_allows(request, "allow_nfc"),
        "face": _desktop_plan_allows(request, "allow_face_recognition"),
        "geo": _desktop_plan_allows(request, "allow_geotracking"),
    }


def _session_method_field_map():
    return {
        "self": "allow_member_self_check_in",
        "qr": "allow_qr_check_in",
        "swipe": "allow_swipe_check_in",
        "nfc": "allow_nfc_check_in",
        "face": "allow_face_check_in",
        "geo": "allow_member_geo_check_in",
    }


def _session_method_enabled(session, request, key: str) -> bool:
    field_name = _session_method_field_map().get(key)
    default = _session_method_plan_defaults(request).get(key, False)
    if not field_name:
        return default
    value = _coerce_session_flag(getattr(session, field_name, None))
    return default if value is None else value


def _session_method_flags(session, request):
    defaults = _session_method_plan_defaults(request)
    flags = {}
    for key, field_name in _session_method_field_map().items():
        value = _coerce_session_flag(getattr(session, field_name, None))
        flags[key] = defaults.get(key, False) if value is None else value
    return flags


def _session_method_guard_response(request, active_session, key: str, label: str):
    if not active_session:
        messages.warning(request, "Select an event session first.")
        return redirect("attendance_sessions")
    if _session_method_enabled(active_session, request, key):
        return None
    messages.warning(
        request,
        f"{label} is turned off for this session. Enable it in Session Settings first.",
    )
    return redirect("attendance_session_detail", pk=active_session.id)


def _session_self_check_payload(request):
    opens_at = _parse_optional_datetime_local(request.POST.get("check_in_opens_at", ""))
    closes_at = _parse_optional_datetime_local(request.POST.get("check_in_closes_at", ""))
    return {
        "allow_member_self_check_in": request.POST.get("allow_member_self_check_in") == "on",
        "allow_qr_check_in": request.POST.get("allow_qr_check_in") == "on",
        "allow_swipe_check_in": request.POST.get("allow_swipe_check_in") == "on",
        "allow_nfc_check_in": request.POST.get("allow_nfc_check_in") == "on",
        "allow_face_check_in": request.POST.get("allow_face_check_in") == "on",
        "allow_member_geo_check_in": request.POST.get("allow_member_geo_check_in") == "on",
        "self_check_in_code": request.POST.get("self_check_in_code", "").strip(),
        "check_in_opens_at": opens_at.isoformat() if opens_at else None,
        "check_in_closes_at": closes_at.isoformat() if closes_at else None,
    }


def _clean_optional_int(value: str, label: str):
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise AttendanceApiError(f"{label} must be a whole number.")


def _session_advanced_payload(request):
    payload = _session_self_check_payload(request)
    payload.update(
        {
            "late_check_in_grace_minutes": _clean_optional_int(
                request.POST.get("late_check_in_grace_minutes", ""),
                "Late check-in grace period",
            ),
            "max_capacity": _clean_optional_int(
                request.POST.get("max_capacity", ""),
                "Max capacity",
            ),
        }
    )
    return payload


def _clean_optional_number(value: str, cast=float):
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise AttendanceApiError("Latitude, longitude, and radius must be valid numbers.")


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


def gen_frames(
    source,
    cam_config,
    api_client,
    known_encodings,
    person_by_index,
    attendance_session_id=None,
    max_fps=10,
    draw_boxes=True,
    draw_names=True,
    play_sound=True,
):
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

            try:
                test_encodings = detect_and_encode(frame_rgb)
            except Exception as exc:
                detail = str(exc).strip()
                if len(detail) > 120:
                    detail = detail[:117] + "..."
                for payload in gen_message_frame(
                    "Face runtime unavailable",
                    detail or "Check the local PyTorch / facenet installation.",
                ):
                    yield payload
                break

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
                                            "attendance_session_id": attendance_session_id,
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


def gen_message_frame(message: str, detail: str = ""):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (28, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    if detail:
        cv2.putText(frame, detail, (28, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 220, 190), 2)
    ok, buffer = cv2.imencode(".jpg", frame)
    if ok:
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


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
    except AttendanceAuthError:
        return _expire_desktop_session(request)
    except AttendanceApiError:
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame")
    src = cam_config.camera_source.strip()
    source = int(src) if src.isdigit() else src
    api_client = _api_client(request)
    try:
        known_encodings, person_by_index = load_authorized_face_encodings(request=request)
    except Exception as exc:
        detail = str(exc).strip()
        if len(detail) > 120:
            detail = detail[:117] + "..."
        return StreamingHttpResponse(
            gen_message_frame(
                "Face runtime unavailable",
                detail or "Check the local PyTorch / facenet installation.",
            ),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )

    draw_boxes = _get_bool_qs(request, "boxes", True)
    draw_names = _get_bool_qs(request, "names", True)
    play_sound = _get_bool_qs(request, "sound", True)
    attendance_session_id = request.GET.get("attendance_session_id") or _active_session_id(request)
    if not attendance_session_id:
        return StreamingHttpResponse(
            gen_message_frame(
                "No active event session",
                "Select Event Sessions before marking attendance.",
            ),
            content_type="multipart/x-mixed-replace; boundary=frame",
        )

    return StreamingHttpResponse(
        gen_frames(
            source,
            cam_config,
            api_client,
            known_encodings,
            person_by_index,
            attendance_session_id=attendance_session_id,
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
        active_session = _get_active_attendance_session(request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load camera: {_friendly_api_error(exc)}")
        return redirect("camera_config_list")
    return render(request, "camera_stream.html", {"config": config, "active_session": active_session})


@desktop_login_required
def stream_all_cameras(request):
    if not _desktop_plan_allows(request, "allow_face_recognition"):
        return _render_plan_locked(
            request,
            "Facial recognition is not included",
            "Upgrade to Professional or above to use local cameras and facial recognition attendance.",
            "Professional",
            back_url="getting_started",
        )
    try:
        configs = get_all_cameras(request=request)
        active_session = _get_active_attendance_session(request)
        guard = _session_method_guard_response(request, active_session, "face", "Face recognition")
        if guard:
            return guard
        method_context = _session_method_context(request, active_session) if active_session else {}
        if active_session:
            face_payload = _api_client(request).list_attendance_logs(
                attendance_session_id=active_session.id,
                page_size=200,
                method="face",
            )
            face_records = [
                record
                for record in extract_results(face_payload)
                if str(record.get("method") or "").strip().lower() in {"face", "facial_recognition"}
            ]
            method_context["method_counts"] = {
                "present": sum(1 for record in face_records if (record.get("status") or "").strip().lower() != "absent"),
                "absent": sum(1 for record in face_records if (record.get("status") or "").strip().lower() == "absent"),
                "pending": 0,
                "marked": len(face_records),
                "members": len(face_records),
            }
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        configs = []
        active_session = None
        method_context = {}
        messages.error(request, f"Could not load cameras: {_friendly_api_error(exc)}")
    return render(
        request,
        "stream_all_cameras.html",
        {
            "configs": configs,
            "active_session": active_session,
            **method_context,
        },
    )

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
    except AttendanceAuthError:
        return _expire_desktop_session(request)
    except AttendanceApiError:
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame")
    src = cam_config.camera_source.strip()
    source = int(src) if src.isdigit() else src

    return StreamingHttpResponse(
        gen_preview_frames(source, max_fps=10),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


@require_GET
@desktop_login_required
def camera_preview_source(request):
    source_value = (request.GET.get("source") or "").strip()
    if not source_value:
        return StreamingHttpResponse(content_type="multipart/x-mixed-replace; boundary=frame")

    if source_value.isdigit():
        source = int(source_value)
    else:
        source = source_value

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
        attendance_session_id = data.get("attendance_session_id") or _active_session_id(request)
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not uid:
        return JsonResponse({"error": "UID is required"}, status=400)
    if not attendance_session_id:
        return JsonResponse({"error": "Select an active attendance session first."}, status=400)

    try:
        outcome = _api_client(request).nfc_check_in(
            {
                "uid": uid,
                "camera_id": camera_id,
                "attendance_session_id": attendance_session_id,
                "min_checkout_seconds": 60,
            }
        )
        play_success_sound()
        return JsonResponse(outcome)
    except AttendanceApiError as exc:
        if _is_auth_api_error(exc):
            return _expire_desktop_session(request)
        return JsonResponse({"error": _friendly_api_error(exc)}, status=400)


# =========================================================
# UI / Pages (MATCH your urls.py)
# =========================================================
def product_about(request):
    return render(
        request,
        "product_about.html",
        {
            "desktop_download_url": settings.TIME_ATTENDANCE_DOWNLOAD_URL,
            "android_download_url": settings.TIME_ATTENDANCE_ANDROID_DOWNLOAD_URL,
            "ios_download_url": settings.TIME_ATTENDANCE_IOS_DOWNLOAD_URL,
        },
    )


@require_POST
def demo_request(request):
    name = request.POST.get("name", "").strip()
    email = request.POST.get("email", "").strip()
    organisation = request.POST.get("organisation", "").strip()
    phone = request.POST.get("phone", "").strip()
    plan = request.POST.get("plan", "").strip()
    message = request.POST.get("message", "").strip()

    if not name or not email:
        messages.error(request, "Please enter your name and email so we can contact you.")
        return redirect("product_about")

    context = {
        "name": name,
        "email": email,
        "organisation": organisation or "-",
        "phone": phone or "-",
        "plan": plan or "-",
        "message": message or "-",
        "submitted_at": timezone.localtime(timezone.now()),
    }
    text_body = render_to_string("emails/demo_request.txt", context)
    html_body = render_to_string("emails/demo_request.html", context)

    try:
        email_message = EmailMultiAlternatives(
            subject=f"KairosTrack Demo Request - {name}",
            body=text_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[settings.DEMO_REQUEST_EMAIL],
            reply_to=[email],
        )
        email_message.attach_alternative(html_body, "text/html")
        email_message.send(fail_silently=False)
        messages.success(request, "Thanks. We received your request and will contact you to schedule a demo.")
    except Exception:
        messages.error(request, "We could not send your request right now. Please try again later.")

    return redirect("product_about")


def desktop_login(request):
    _clear_queued_messages(request)

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
            client = get_client()
            payload = client.authenticate(username=username, password=password)
            try:
                profile_payload = client.me()
                payload = {**payload, **profile_payload}
            except AttendanceApiError:
                pass
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

    return render(request, "desktop_login.html", {"next_url": _safe_next_url(request, request.GET.get("next"))})


def desktop_forgot_password(request):
    if request.method == "POST":
        identifier = request.POST.get("identifier", "").strip()
        if not identifier:
            return render(
                request,
                "desktop_forgot_password.html",
                {"error": "Enter your username or email to reset your password.", "identifier": ""},
            )
        try:
            frontend_base_url = get_client().auth_base_url.rstrip("/")
            get_client().forgot_password(identifier=identifier, frontend_base_url=frontend_base_url)
            messages.success(request, "If the account exists, a password reset link has been sent to the registered email.")
            return redirect("desktop_login")
        except AttendanceApiError as exc:
            return render(
                request,
                "desktop_forgot_password.html",
                {"error": _friendly_api_error(exc), "identifier": identifier},
            )

    return render(request, "desktop_forgot_password.html", {"identifier": ""})


def desktop_reset_password(request):
    uid = (request.POST.get("uid") if request.method == "POST" else request.GET.get("uid") or "").strip()
    token = (request.POST.get("token") if request.method == "POST" else request.GET.get("token") or "").strip()

    if request.method == "POST":
        new_password = request.POST.get("new_password", "")
        confirm_password = request.POST.get("confirm_password", "")
        if not uid or not token:
            return render(
                request,
                "desktop_reset_password.html",
                {"error": "This password reset link is incomplete or invalid.", "uid": uid, "token": token},
            )
        if len(new_password) < 8:
            return render(
                request,
                "desktop_reset_password.html",
                {"error": "Your new password must be at least 8 characters long.", "uid": uid, "token": token},
            )
        if new_password != confirm_password:
            return render(
                request,
                "desktop_reset_password.html",
                {"error": "The new passwords do not match.", "uid": uid, "token": token},
            )
        try:
            get_client().reset_password(uid=uid, token=token, new_password=new_password)
            messages.success(request, "Password reset successful. You can now sign in with your new password.")
            return redirect("desktop_login")
        except AttendanceApiError as exc:
            return render(
                request,
                "desktop_reset_password.html",
                {"error": _friendly_api_error(exc), "uid": uid, "token": token},
            )

    return render(request, "desktop_reset_password.html", {"uid": uid, "token": token})


def desktop_signup(request):
    _clear_queued_messages(request)

    try:
        plans_payload = get_client().list_plans()
        plans = [
            plan for plan in extract_results(plans_payload)
            if plan.get("tier") in {"starter", "standard", "professional", "enterprise"}
        ]
    except AttendanceApiError:
        plans = [
            {"tier": "starter", "name": "Starter", "monthly_price_gbp": "0.00"},
            {"tier": "standard", "name": "Standard", "monthly_price_gbp": "9.99"},
            {"tier": "professional", "name": "Professional", "monthly_price_gbp": "19.99"},
            {"tier": "enterprise", "name": "Enterprise", "monthly_price_gbp": None},
        ]

    if request.method == "POST":
        payload = {
            "organization_name": request.POST.get("organization_name", "").strip(),
            "contact_email": request.POST.get("contact_email", "").strip(),
            "contact_phone": request.POST.get("contact_phone", "").strip(),
            "branch_name": request.POST.get("branch_name", "Main").strip() or "Main",
            "plan_tier": request.POST.get("plan_tier", "starter").strip() or "starter",
            "username": request.POST.get("username", "").strip(),
            "password": request.POST.get("password", ""),
            "first_name": request.POST.get("first_name", "").strip(),
            "last_name": request.POST.get("last_name", "").strip(),
        }
        confirm_password = request.POST.get("confirm_password", "")

        if not payload["organization_name"] or not payload["contact_email"] or not payload["username"] or not payload["password"]:
            return render(
                request,
                "desktop_signup.html",
                {"plans": plans, "form": payload, "error": "Organization, email, username, and password are required."},
            )
        if payload["password"] != confirm_password:
            return render(
                request,
                "desktop_signup.html",
                {"plans": plans, "form": payload, "error": "Passwords do not match."},
            )

        try:
            signup_payload = get_client().signup_organization(payload)
            client = get_client(
                token=signup_payload.get("access"),
                refresh_token=signup_payload.get("refresh"),
            )
            profile_payload = client.me()
            _store_desktop_auth(request, {**signup_payload, **profile_payload})
            messages.success(request, f"{payload['organization_name']} is ready. Welcome to KairosTrack.")
            return redirect("home")
        except AttendanceApiError as exc:
            return render(
                request,
                "desktop_signup.html",
                {"plans": plans, "form": payload, "error": f"Signup failed: {exc}"},
            )

    if _is_desktop_authenticated(request):
        return redirect("home")

    _clear_desktop_auth(request)
    return render(request, "desktop_signup.html", {"plans": plans, "form": {}})


def desktop_logout(request):
    _clear_desktop_auth(request)
    messages.success(request, "Desktop session signed out.")
    return redirect("desktop_login")


@desktop_login_required
def getting_started(request):
    try:
        setup_context = _build_setup_context(request)
        active_session = _get_active_attendance_session(request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load setup guide: {_friendly_api_error(exc)}")
        setup_context = {
            "organization": request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {},
            "plan": request.session.get(DESKTOP_PLAN_SESSION_KEY) or {},
            "attendance_methods": _attendance_method_cards(request),
            "setup_checklist": [],
            "setup_completed": 0,
            "setup_total": 0,
            "setup_progress_percent": 0,
            "setup_is_complete": False,
            "setup_next_step": None,
            "setup_recommended_action": None,
            "setup_counts": {},
        }
        active_session = None

    return render(
        request,
        "getting_started.html",
        {
            **setup_context,
            "active_session": active_session,
        },
    )


@desktop_login_required
def upgrade_request(request):
    requested_plan = request.GET.get("plan", "").strip()
    try:
        context_payload = _refresh_desktop_context(request)
        organization = context_payload.get("organization") or {}
        plan = context_payload.get("plan") or {}
        plans = [
            to_namespace(item)
            for item in extract_results(_api_client(request).list_plans())
            if item.get("tier") in {"standard", "professional", "enterprise"}
        ]
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load upgrade options: {_friendly_api_error(exc)}")
        organization = request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}
        plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}
        plans = [
            to_namespace({"name": "Standard", "tier": "standard"}),
            to_namespace({"name": "Professional", "tier": "professional"}),
            to_namespace({"name": "Enterprise", "tier": "enterprise"}),
        ]

    if request.method == "POST":
        payload = {
            "requested_plan": request.POST.get("requested_plan", "").strip(),
            "contact_email": request.POST.get("contact_email", "").strip(),
            "contact_phone": request.POST.get("contact_phone", "").strip(),
            "message": request.POST.get("message", "").strip(),
        }
        try:
            _api_client(request).request_upgrade(payload)
            messages.success(request, "Upgrade request sent. We will contact you shortly.")
            return redirect("organization_settings")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            return render(
                request,
                "upgrade_request.html",
                {
                    "organization": to_namespace(organization),
                    "plan": to_namespace(plan),
                    "plans": plans,
                    "requested_plan": payload["requested_plan"],
                    "form": payload,
                    "error": f"Could not send upgrade request: {exc}",
                },
            )

    return render(
        request,
        "upgrade_request.html",
        {
            "organization": to_namespace(organization),
            "plan": to_namespace(plan),
            "plans": plans,
            "requested_plan": requested_plan,
            "form": {},
        },
    )


@desktop_login_required
def desktop_settings(request):
    if request.method == "POST":
        action = request.POST.get("action", "").strip()
        if action == "save_backend":
            base_url = request.POST.get("backend_url", "").strip().rstrip("/")
            if not base_url:
                request.session.pop(API_BASE_URL_SESSION_KEY, None)
                messages.success(request, "Backend URL reset to the app default.")
            elif not base_url.startswith(("http://", "https://")) or "/api/attendance" not in base_url:
                messages.error(request, "Enter a full attendance API URL, for example https://example.com/api/attendance.")
            else:
                request.session[API_BASE_URL_SESSION_KEY] = base_url
                request.session.modified = True
                messages.success(request, "Backend URL saved for this desktop session.")
            return redirect("desktop_settings")

        if action == "refresh_context":
            try:
                _refresh_desktop_context(request)
                _set_desktop_sync_status(request, ok=True, message="Account context refreshed.")
                messages.success(request, "Desktop account context refreshed.")
            except AttendanceApiError as exc:
                _set_desktop_sync_status(request, ok=False, message=_friendly_api_error(exc))
                messages.error(request, f"Could not refresh context: {_friendly_api_error(exc)}")
            return redirect("desktop_settings")

    client = _api_client(request)
    backend_status = {"ok": False, "message": "Not checked", "overview": {}}
    try:
        overview = client.overview()
        backend_status = {"ok": True, "message": "Backend reachable", "overview": overview}
        _set_desktop_sync_status(request, ok=True, message="Backend reachable.")
    except AttendanceApiError as exc:
        backend_status = {"ok": False, "message": _friendly_api_error(exc), "overview": {}}
        _set_desktop_sync_status(request, ok=False, message=backend_status["message"])

    organization = request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}
    plan = request.session.get(DESKTOP_PLAN_SESSION_KEY) or {}
    user = request.session.get(DESKTOP_USER_SESSION_KEY) or {}
    active_session = None
    try:
        active_session = _get_active_attendance_session(request)
    except AttendanceApiError:
        active_session = None

    return render(
        request,
        "desktop_settings.html",
        {
            "backend_url": _desktop_backend_url(request),
            "default_backend_url": settings.ATTENDANCE_API_BASE_URL,
            "backend_status": backend_status,
            "organization": organization,
            "plan": plan,
            "desktop_user": user,
            "active_session": active_session,
            "app_version": settings.KAIROSTRACK_DESKTOP_VERSION,
            "download_url": settings.TIME_ATTENDANCE_DOWNLOAD_URL,
            "update_check_url": settings.KAIROSTRACK_UPDATE_CHECK_URL,
            "sync_status": request.session.get(DESKTOP_SYNC_STATUS_SESSION_KEY) or {},
        },
    )


@desktop_login_required
def organization_settings(request):
    client = _api_client(request)

    if request.method == "POST":
        action = request.POST.get("action", "").strip()
        try:
            context_payload = _refresh_desktop_context(request)
            organization = context_payload.get("organization") or {}
            organization_id = organization.get("id")
            if not organization_id:
                messages.error(request, "Your account is not linked to an organization yet.")
                return redirect("organization_settings")

            if action == "update_organization":
                client.update_organization(
                    organization_id,
                    {
                        "name": request.POST.get("name", "").strip(),
                        "contact_email": request.POST.get("contact_email", "").strip(),
                        "contact_phone": request.POST.get("contact_phone", "").strip(),
                    },
                )
                _refresh_desktop_context(request)
                messages.success(request, "Organization details updated.")

            elif action == "add_branch":
                branch_name = request.POST.get("branch_name", "").strip()
                if not branch_name:
                    messages.error(request, "Branch name is required.")
                else:
                    radius = _clean_optional_number(request.POST.get("geofence_radius_meters", ""), int) or 100
                    payload = {
                        "name": branch_name,
                        "address": request.POST.get("address", "").strip(),
                        "latitude": _clean_optional_number(request.POST.get("latitude", "")),
                        "longitude": _clean_optional_number(request.POST.get("longitude", "")),
                        "geofence_radius_meters": radius,
                    }
                    client.create_branch(payload)
                    messages.success(request, "Branch added.")

            elif action == "update_branch":
                branch_id = int(request.POST.get("branch_id", "0") or "0")
                radius = _clean_optional_number(request.POST.get("geofence_radius_meters", ""), int) or 100
                client.update_branch(
                    branch_id,
                    {
                        "name": request.POST.get("branch_name", "").strip(),
                        "address": request.POST.get("address", "").strip(),
                        "latitude": _clean_optional_number(request.POST.get("latitude", "")),
                        "longitude": _clean_optional_number(request.POST.get("longitude", "")),
                        "geofence_radius_meters": radius,
                        "is_active": request.POST.get("is_active") == "on",
                    },
                )
                messages.success(request, "Branch GeoTracking settings updated.")

            elif action == "assign_event_branch":
                event_id = int(request.POST.get("event_id", "0") or "0")
                branch_id = int(request.POST.get("branch_id") or 0) or None
                client.update_event(event_id, {"branch": branch_id})
                messages.success(request, "Event branch updated.")

            elif action == "add_staff":
                password = request.POST.get("password", "")
                confirm_password = request.POST.get("confirm_password", "")
                staff_limit = _staff_limit_status(request, organization)
                if password != confirm_password:
                    messages.error(request, "Staff passwords do not match.")
                elif staff_limit["is_reached"]:
                    messages.warning(request, "Staff user limit reached for the current plan. Upgrade before adding more staff.")
                else:
                    client.create_staff(
                        {
                            "username": request.POST.get("username", "").strip(),
                            "email": request.POST.get("email", "").strip(),
                            "password": password,
                            "first_name": request.POST.get("first_name", "").strip(),
                            "last_name": request.POST.get("last_name", "").strip(),
                            "role": request.POST.get("role", "staff").strip(),
                            "branch": int(request.POST.get("branch") or 0) or None,
                        }
                    )
                    messages.success(request, "Staff user added.")

            elif action == "update_staff":
                staff_id = int(request.POST.get("staff_id", "0") or "0")
                payload = {
                    "first_name": request.POST.get("first_name", "").strip(),
                    "last_name": request.POST.get("last_name", "").strip(),
                    "email": request.POST.get("email", "").strip(),
                    "role": request.POST.get("role", "staff").strip(),
                    "branch": int(request.POST.get("branch") or 0) or None,
                    "is_active": request.POST.get("is_active") == "on",
                }
                client.update_staff(staff_id, payload)
                messages.success(request, "Staff profile updated.")

            elif action == "reset_staff_password":
                staff_id = int(request.POST.get("staff_id", "0") or "0")
                password = request.POST.get("password", "")
                confirm_password = request.POST.get("confirm_password", "")
                if password != confirm_password:
                    messages.error(request, "New passwords do not match.")
                elif len(password) < 8:
                    messages.error(request, "Password must be at least 8 characters.")
                else:
                    client.update_staff(staff_id, {"password": password})
                    messages.success(request, "Staff password updated.")

            elif action == "deactivate_staff":
                staff_id = int(request.POST.get("staff_id", "0") or "0")
                client.delete_staff(staff_id)
                messages.success(request, "Staff user deactivated.")

            else:
                messages.error(request, "Unknown settings action.")

        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not update organization settings: {exc}")

        return redirect("organization_settings")

    try:
        context_payload = _refresh_desktop_context(request)
        organization = context_payload.get("organization") or {}
        plan = context_payload.get("plan") or {}
        staff_profile = context_payload.get("staff_profile") or {}
        branches = [to_namespace(item) for item in extract_results(client.list_branches())]
        events = [to_namespace(item) for item in extract_results(client.list_events(active="true"))]
        staff = [to_namespace(item) for item in extract_results(client.list_staff())]
        overview = client.overview()
        registration_url = ""
        if organization.get("slug"):
            registration_url = f"{client.auth_base_url}/attendance/self-register/{organization['slug']}/"
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load organization settings: {_friendly_api_error(exc)}")
        organization = {}
        plan = {}
        staff_profile = {}
        branches = []
        events = []
        staff = []
        overview = {}
        registration_url = ""

    return render(
        request,
        "organization_settings.html",
        {
            "organization": to_namespace(organization),
            "plan": to_namespace(plan),
            "staff_profile": to_namespace(staff_profile),
            "branches": branches,
            "events": events,
            "staff": staff,
            "overview": overview,
            "registration_url": registration_url,
            "member_limit": _member_limit_status(request, overview=overview, organization=organization),
            "staff_limit": _staff_limit_status(request, organization),
            "attendance_methods": _attendance_method_cards(request),
            "can_manage_staff": staff_profile.get("role") in {"owner", "admin"} or request.session.get(DESKTOP_USER_SESSION_KEY, {}).get("is_superuser"),
        },
    )


@desktop_login_required
def attendance_reports(request):
    today = timezone.localdate()
    date_from = request.GET.get("date_from", "").strip() or str(today.replace(day=1))
    date_to = request.GET.get("date_to", "").strip() or str(today)
    event_id = request.GET.get("event_id", "").strip()
    branch_id = request.GET.get("branch_id", "").strip()
    method = request.GET.get("method", "").strip()

    def _safe_int(value):
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _percent(part, whole):
        if not whole:
            return 0
        return round((part / whole) * 100, 1)

    def _method_label(value):
        labels = {
            "qr": "QR Code",
            "nfc": "NFC",
            "swipe": "Swipe",
            "manual": "Manual",
            "member_self": "Self Check-In",
            "mobile_self": "Self Check-In",
            "face": "Face Recognition",
            "face_recognition": "Face Recognition",
            "geotracking": "GeoTracking",
        }
        cleaned = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return labels.get(cleaned, (value or "Unknown").replace("_", " ").title())

    try:
        client = _api_client(request)
        params = {"date_from": date_from, "date_to": date_to}
        if event_id:
            params["event_id"] = event_id
        if branch_id:
            params["branch_id"] = branch_id
        if method:
            params["method"] = method
        report = to_namespace(client.report_summary(**params))
        events = [to_namespace(item) for item in extract_results(client.list_events())]
        branches = [to_namespace(item) for item in extract_results(client.list_branches())]
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load reports: {_friendly_api_error(exc)}")
        report = to_namespace({"counts": {}, "daily": [], "methods": [], "statuses": [], "top_events": [], "branches": []})
        events = []
        branches = []

    total_records = _safe_int(getattr(report.counts, "total", 0))
    present_count = _safe_int(getattr(report.counts, "present", 0))
    absent_count = _safe_int(getattr(report.counts, "absent", 0))
    pending_count = _safe_int(getattr(report.counts, "pending", 0))
    checked_out_count = _safe_int(getattr(report.counts, "checked_out", 0))

    method_rows = []
    for item in getattr(report, "methods", []) or []:
        count = _safe_int(getattr(item, "count", 0))
        method_rows.append(
            {
                "label": _method_label(getattr(item, "key", "")),
                "count": count,
                "percent": _percent(count, total_records),
            }
        )

    top_events = []
    for item in getattr(report, "top_events", []) or []:
        count = _safe_int(getattr(item, "count", 0))
        top_events.append(
            {
                "label": getattr(item, "event_name", "Unknown event"),
                "count": count,
                "percent": _percent(count, total_records),
            }
        )

    branch_rows = []
    for item in getattr(report, "branches", []) or []:
        count = _safe_int(getattr(item, "count", 0))
        branch_rows.append(
            {
                "label": getattr(item, "branch_name", "No branch"),
                "count": count,
                "percent": _percent(count, total_records),
            }
        )

    daily_rows = []
    max_daily = 0
    for item in getattr(report, "daily", []) or []:
        count = _safe_int(getattr(item, "count", 0))
        max_daily = max(max_daily, count)
        daily_rows.append({"date": getattr(item, "date", ""), "count": count})
    for row in daily_rows:
        row["height"] = 14 if not max_daily else max(14, round((row["count"] / max_daily) * 88))

    donut_segments = [
        {"label": "Present", "count": present_count, "percent": _percent(present_count, total_records), "color": "#22c55e"},
        {"label": "Absent", "count": absent_count, "percent": _percent(absent_count, total_records), "color": "#ff5c4d"},
        {"label": "Pending", "count": pending_count, "percent": _percent(pending_count, total_records), "color": "#f7b53b"},
        {"label": "Checked Out", "count": checked_out_count, "percent": _percent(checked_out_count, total_records), "color": "#4c8bf5"},
    ]
    donut_parts = []
    current_angle = 0
    for segment in donut_segments:
        next_angle = current_angle + (segment["percent"] * 3.6)
        donut_parts.append(f"{segment['color']} {current_angle:.1f}deg {next_angle:.1f}deg")
        current_angle = next_angle
    donut_style = "conic-gradient(" + ", ".join(donut_parts or ["#e5e7eb 0deg 360deg"]) + ")"

    export_query = urlencode(
        {
            key: value
            for key, value in {
                "date_from": date_from,
                "date_to": date_to,
                "event_id": event_id,
                "branch_id": branch_id,
                "method": method,
            }.items()
            if value
        }
    )
    return render(
        request,
        "attendance_reports.html",
        {
            "report": report,
            "events": events,
            "branches": branches,
            "date_from": date_from,
            "date_to": date_to,
            "event_id": event_id,
            "branch_id": branch_id,
            "method": method,
            "export_query": export_query,
            "total_records": total_records,
            "present_count": present_count,
            "absent_count": absent_count,
            "pending_count": pending_count,
            "checked_out_count": checked_out_count,
            "present_percent": _percent(present_count, total_records),
            "absent_percent": _percent(absent_count, total_records),
            "pending_percent": _percent(pending_count, total_records),
            "checked_out_percent": _percent(checked_out_count, total_records),
            "method_rows": method_rows,
            "top_event_rows": top_events,
            "branch_rows": branch_rows,
            "daily_rows": daily_rows,
            "max_daily": max_daily,
            "donut_segments": donut_segments,
            "donut_style": donut_style,
        },
    )


@desktop_login_required
def attendance_sessions(request):
    client = _api_client(request)

    if request.method == "POST":
        action = request.POST.get("action", "").strip()
        try:
            if action == "select":
                session_id = request.POST.get("session_id", "").strip()
                if not session_id:
                    messages.error(request, "Choose an attendance session first.")
                else:
                    session_payload = client.get_session(int(session_id))
                    if (session_payload.get("status") or "").lower() == "scheduled":
                        session_payload = client.update_session(int(session_id), {"status": "open"})
                    request.session[ACTIVE_ATTENDANCE_SESSION_KEY] = session_payload["id"]
                    request.session.pop(ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY, None)
                    request.session.modified = True
                    _sync_reader_bridge_config(request, active_session=to_namespace(session_payload))
                    messages.success(request, f"Active session set to {session_payload.get('session_name') or session_payload.get('event_name')}.")
                return redirect("attendance_sessions")

            if action == "clear":
                request.session.pop(ACTIVE_ATTENDANCE_SESSION_KEY, None)
                request.session[ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY] = True
                request.session.modified = True
                _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
                messages.success(request, "Active attendance session cleared.")
                return redirect("attendance_sessions")

            if action == "delete":
                session_id = request.POST.get("session_id", "").strip()
                if not session_id:
                    messages.error(request, "Choose a session to delete.")
                    return redirect("attendance_sessions")
                client.delete_session(int(session_id))
                if str(_active_session_id(request) or "") == session_id:
                    request.session.pop(ACTIVE_ATTENDANCE_SESSION_KEY, None)
                    request.session[ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY] = True
                    request.session.modified = True
                    _sync_reader_bridge_config(request, active_session=None, resolve_active=False)
                messages.success(request, "Session deleted.")
                return redirect("attendance_sessions")

            messages.error(request, "Unknown session action.")
            return redirect("attendance_sessions")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not update attendance session: {exc}")
            return redirect("attendance_sessions")

    try:
        events = [to_namespace(item) for item in extract_results(client.list_events(active="true"))]
        session_map = {}
        for status_value in ("open", "scheduled", "closed", "cancelled"):
            for item in extract_results(client.list_sessions(status=status_value)):
                session_map[item["id"]] = to_namespace(item)
        sessions = list(session_map.values())
        active_session = _get_active_attendance_session(request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load attendance sessions: {_friendly_api_error(exc)}")
        events = []
        sessions = []
        active_session = None

    def _parse_session_dt(value):
        if not value:
            return None
        if isinstance(value, str):
            raw = value.strip()
            for pattern in (None, "%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")) if pattern is None else datetime.strptime(raw, pattern)
                    if timezone.is_naive(parsed):
                        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
                    return timezone.localtime(parsed)
                except ValueError:
                    continue
            return None
        try:
            return timezone.localtime(value)
        except Exception:
            return None

    method_labels = {
        "self": "Word Check-In",
        "qr": "QR Code",
        "swipe": "Swipe",
        "nfc": "NFC",
        "face": "Face Recognition",
        "geo": "GeoTracking",
    }
    today = timezone.localdate()
    open_count = 0
    scheduled_count = 0
    today_count = 0
    prepared_sessions = []
    for session in sessions:
        starts_dt = _parse_session_dt(getattr(session, "starts_at", None))
        status_key = (getattr(session, "status", "") or "open").strip().lower()
        flags = _session_method_flags(session, request)
        enabled_methods = [label for key, label in method_labels.items() if flags.get(key)]
        if status_key == "open":
            open_count += 1
        if status_key == "scheduled":
            scheduled_count += 1
        if starts_dt and starts_dt.date() == today:
            today_count += 1
        setattr(session, "starts_at_local", starts_dt)
        setattr(session, "date_label", starts_dt.strftime("%d-%m-%Y") if starts_dt else "Not scheduled")
        setattr(session, "time_label", starts_dt.strftime("%H:%M") if starts_dt else "—")
        setattr(session, "status_key", status_key)
        setattr(session, "station_active", bool(active_session and active_session.id == session.id))
        setattr(session, "method_flags", flags)
        setattr(session, "method_summary", " + ".join(enabled_methods[:3]) if enabled_methods else "No check-in methods enabled")
        prepared_sessions.append(session)
    sessions = prepared_sessions

    active_method_summary = "No check-in methods enabled"
    if active_session:
        active_flags = _session_method_flags(active_session, request)
        active_enabled = [label for key, label in method_labels.items() if active_flags.get(key)]
        active_method_summary = " + ".join(active_enabled[:4]) if active_enabled else active_method_summary
        active_starts_dt = _parse_session_dt(getattr(active_session, "starts_at", None))
        setattr(active_session, "starts_at_local", active_starts_dt)
        setattr(active_session, "date_label", active_starts_dt.strftime("%d-%m-%Y") if active_starts_dt else "Not scheduled")
        setattr(active_session, "time_label", active_starts_dt.strftime("%H:%M") if active_starts_dt else "—")
        setattr(active_session, "status_key", ((getattr(active_session, "status", "") or "open").strip().lower()))

    return render(
        request,
        "attendance_sessions.html",
        {
            "events": events,
            "sessions": sessions,
            "active_session": active_session,
            "total_sessions": len(sessions),
            "today_sessions": today_count,
            "open_sessions": open_count,
            "scheduled_sessions": scheduled_count,
            "active_station_count": 1 if active_session else 0,
            "active_method_summary": active_method_summary,
        },
    )


@desktop_login_required
def attendance_session_create(request):
    client = _api_client(request)
    if request.method == "POST":
        action = request.POST.get("action", "").strip()
        if action != "save_session":
            messages.error(request, "Unknown session action.")
            return redirect("attendance_session_create")
        try:
            event_mode = request.POST.get("event_mode", "new").strip() or "new"
            event_id = request.POST.get("event_id", "").strip()
            event_name = request.POST.get("event_name", "").strip()
            session_name = request.POST.get("session_name", "").strip()
            starts_at = _parse_datetime_local(request.POST.get("starts_at", ""))
            ends_at = _parse_optional_datetime_local(request.POST.get("ends_at", ""))
            status_value = request.POST.get("status", "open").strip() or "open"

            if event_mode == "existing":
                if not event_id:
                    messages.error(request, "Choose an existing event first.")
                    return redirect("attendance_session_create")
                selected_event_id = int(event_id)
                success_label = "Session created from existing event."
            else:
                if not event_name:
                    messages.error(request, "Event name is required.")
                    return redirect("attendance_session_create")
                event_payload = client.create_event({"name": event_name, "is_active": True})
                selected_event_id = event_payload["id"]
                success_label = f"Created {event_name}."

            session_payload = client.create_session(
                {
                    "event": selected_event_id,
                    "name": session_name,
                    "starts_at": starts_at.isoformat(),
                    "ends_at": ends_at.isoformat() if ends_at else None,
                    "status": status_value,
                    **_session_advanced_payload(request),
                }
            )
            if status_value == "open":
                request.session[ACTIVE_ATTENDANCE_SESSION_KEY] = session_payload["id"]
                request.session.pop(ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY, None)
                request.session.modified = True
                _sync_reader_bridge_config(request, active_session=to_namespace(session_payload))
                success_label = f"{success_label} Attendance is now active."
            else:
                success_label = f"{success_label} Scheduled for later."
            messages.success(request, success_label)
            return redirect("attendance_sessions")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not create attendance session: {exc}")
            return redirect("attendance_session_create")

    try:
        events = [to_namespace(item) for item in extract_results(client.list_events(active="true"))]
        active_session = _get_active_attendance_session(request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load create session form: {_friendly_api_error(exc)}")
        events = []
        active_session = None

    return render(
        request,
        "attendance_session_create.html",
        {
            "events": events,
            "active_session": active_session,
            "now_local": timezone.localtime().strftime("%Y-%m-%dT%H:%M"),
            "method_defaults": _session_method_plan_defaults(request),
        },
    )


def _attendance_record_status(record) -> str:
    if not record:
        return "pending"
    status_value = (getattr(record, "status", "") or "").lower()
    if status_value == "absent":
        return "absent"
    if status_value == "present" or getattr(record, "check_in_time", None):
        return "present"
    return status_value or "present"


def _attendance_status_label(status_value: str) -> str:
    return {
        "present": "Present",
        "absent": "Absent",
        "pending": "Pending",
    }.get(status_value, (status_value or "Pending").replace("_", " ").title())


def _attendance_method_label(method: str) -> str:
    return {
        "manual": "Manual",
        "swipe": "Swipe",
        "qr": "QR code",
        "member_self": "Self check-in",
        "nfc": "NFC",
        "face": "Facial recognition",
        "facial_recognition": "Facial recognition",
        "geotracking": "GeoTracking",
    }.get((method or "manual").lower(), (method or "Manual").replace("_", " ").title())


def _session_report_context(request, session_id: int):
    client = _api_client(request)
    report = to_namespace(client.session_report(session_id))
    session = report.session
    summary = report.summary
    records = []
    rows = []
    for row in report.rows:
        record = getattr(row, "attendance", None)
        if record:
            records.append(record)
        status_value = _attendance_record_status(record)
        rows.append(
            {
                "person": row.person,
                "record": record,
                "status": status_value,
                "status_label": _attendance_status_label(status_value),
                "method_label": _attendance_method_label(getattr(record, "method", "") if record else ""),
            }
        )

    method_counts = {
        getattr(item, "key", "unknown"): getattr(item, "count", 0)
        for item in getattr(report, "methods", [])
    }
    face_count = method_counts.get("face", 0) + method_counts.get("facial_recognition", 0)
    method_cards = [
        {"key": "manual", "label": "Manual", "count": method_counts.get("manual", 0)},
        {"key": "swipe", "label": "Swipe", "count": method_counts.get("swipe", 0)},
        {"key": "qr", "label": "QR code", "count": method_counts.get("qr", 0)},
        {"key": "member_self", "label": "Self check-in", "count": method_counts.get("member_self", 0)},
        {"key": "nfc", "label": "NFC", "count": method_counts.get("nfc", 0)},
        {"key": "face", "label": "Face", "count": face_count},
        {"key": "geotracking", "label": "GeoTracking", "count": method_counts.get("geotracking", 0)},
    ]
    method_cards = [method for method in method_cards if method["count"]]

    export_query = f"attendance_session_id={session_id}"
    return {
        "session": session,
        "rows": rows,
        "records": records,
        "total_members": getattr(summary, "total_members", 0),
        "present_count": getattr(summary, "present", 0),
        "absent_count": getattr(summary, "absent", 0),
        "pending_count": getattr(summary, "pending", 0),
        "marked_count": getattr(summary, "marked", 0),
        "method_cards": method_cards,
        "export_query": export_query,
        "is_active_session": str(_active_session_id(request) or "") == str(session_id),
        "can_use_swipe": _desktop_plan_allows_or_default(request, "allow_swipe", default=True),
        "can_use_qr": _desktop_plan_allows(request, "allow_qr"),
        "can_use_nfc": _desktop_plan_allows(request, "allow_nfc"),
        "can_use_face": _desktop_plan_allows(request, "allow_face_recognition"),
        "can_use_geo": _desktop_plan_allows(request, "allow_geotracking"),
        "session_methods": _session_method_flags(session, request),
    }


def _session_method_context(request, active_session):
    if not active_session:
        return {
            "session_report_url": reverse("attendance_sessions"),
            "method_counts": {"present": 0, "absent": 0, "pending": 0, "marked": 0, "members": 0},
        }

    people = get_all_people(request=request, authorized=True)
    attendance_by_person = _session_attendance_map(request, active_session.id)
    present = 0
    absent = 0
    for record in attendance_by_person.values():
        if record.get("status") == "absent":
            absent += 1
        else:
            present += 1
    members = len(people)
    return {
        "session_report_url": reverse("attendance_session_detail", args=[active_session.id]),
        "method_counts": {
            "present": present,
            "absent": absent,
            "pending": max(0, members - len(attendance_by_person)),
            "marked": len(attendance_by_person),
            "members": members,
        },
    }


@desktop_login_required
def attendance_session_detail(request, pk: int):
    client = _api_client(request)
    if request.method == "POST":
        action = request.POST.get("action", "").strip()
        try:
            if action == "select":
                session_payload = client.get_session(pk)
                if (session_payload.get("status") or "").lower() == "scheduled":
                    session_payload = client.update_session(pk, {"status": "open"})
                request.session[ACTIVE_ATTENDANCE_SESSION_KEY] = session_payload["id"]
                request.session.pop(ACTIVE_ATTENDANCE_SESSION_CLEARED_KEY, None)
                request.session.modified = True
                _sync_reader_bridge_config(request, active_session=to_namespace(session_payload))
                messages.success(request, "Active attendance session updated.")
                return redirect("attendance_session_detail", pk=pk)

            if action == "status":
                status_value = request.POST.get("status", "open").strip() or "open"
                if status_value not in {"scheduled", "open", "closed", "cancelled"}:
                    status_value = "open"
                client.update_session(pk, {"status": status_value})
                messages.success(request, f"Session marked as {status_value}.")
                return redirect("attendance_session_detail", pk=pk)

            if action == "edit":
                current_session = to_namespace(client.get_session(pk))
                starts_at_value = request.POST.get("starts_at", "").strip()
                starts_at = _parse_optional_datetime_local(starts_at_value)
                payload = {
                    "name": request.POST.get("session_name", "").strip(),
                    "status": request.POST.get("status", "open").strip() or "open",
                    "starts_at": starts_at.isoformat() if starts_at else getattr(current_session, "starts_at", None),
                    **_session_advanced_payload(request),
                }
                ends_at_value = request.POST.get("ends_at", "").strip()
                ends_at = _parse_optional_datetime_local(ends_at_value)
                if ends_at_value:
                    payload["ends_at"] = ends_at.isoformat() if ends_at else None
                else:
                    payload["ends_at"] = getattr(current_session, "ends_at", None)
                if payload["status"] not in {"scheduled", "open", "closed", "cancelled"}:
                    payload["status"] = "open"
                client.update_session(pk, payload)
                messages.success(request, "Session details updated.")
                return redirect("attendance_session_detail", pk=pk)

            if action == "mark":
                person_id = int(request.POST.get("person_id", "0") or "0")
                status_value = request.POST.get("status", "present").strip()
                if status_value not in {"present", "absent"}:
                    status_value = "present"
                _mark_attendance_for_session(request, pk, person_id, status_value, method="manual")
                messages.success(request, "Attendance updated for this session.")
                return redirect("attendance_session_detail", pk=pk)

            messages.error(request, "Unknown session action.")
            return redirect("attendance_session_detail", pk=pk)
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not update session: {exc}")
            return redirect("attendance_session_detail", pk=pk)

    try:
        context = _session_report_context(request, pk)
        session_detail = to_namespace(client.get_session(pk))
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load session report: {_friendly_api_error(exc)}")
        return redirect("attendance_sessions")

    context["session_edit"] = {
        "session_name": getattr(session_detail, "name", "") or "",
        "starts_at": _datetime_local_value(getattr(session_detail, "starts_at", None)),
        "ends_at": _datetime_local_value(getattr(session_detail, "ends_at", None)),
        "status": getattr(session_detail, "status", "open") or "open",
        "allow_member_self_check_in": _session_method_enabled(session_detail, request, "self"),
        "allow_qr_check_in": _session_method_enabled(session_detail, request, "qr"),
        "allow_swipe_check_in": _session_method_enabled(session_detail, request, "swipe"),
        "allow_nfc_check_in": _session_method_enabled(session_detail, request, "nfc"),
        "allow_face_check_in": _session_method_enabled(session_detail, request, "face"),
        "allow_member_geo_check_in": _session_method_enabled(session_detail, request, "geo"),
        "self_check_in_code": getattr(session_detail, "self_check_in_code", "") or "",
        "check_in_opens_at": _datetime_local_value(getattr(session_detail, "check_in_opens_at", None)),
        "check_in_closes_at": _datetime_local_value(getattr(session_detail, "check_in_closes_at", None)),
        "late_check_in_grace_minutes": getattr(session_detail, "late_check_in_grace_minutes", "") or "",
        "max_capacity": getattr(session_detail, "max_capacity", "") or "",
    }

    return render(request, "attendance_session_detail.html", context)


@desktop_login_required
def qr_attendance(request):
    try:
        if not _desktop_plan_allows(request, "allow_qr"):
            return _render_plan_locked(
                request,
                "QR attendance is not included",
                "Upgrade to Standard or above to use event QR codes and member self check-in.",
                "Standard",
                back_url="attendance_sessions",
            )
        active_session = _get_active_attendance_session(request)
        guard = _session_method_guard_response(request, active_session, "qr", "QR check-in")
        if guard:
            return guard

        client = _api_client(request)
        token = getattr(active_session, "qr_code_token", "") or uuid.uuid4().hex
        if not getattr(active_session, "qr_code_token", ""):
            active_session = to_namespace(client.update_session(active_session.id, {"qr_code_token": token}))

        method_context = _session_method_context(request, active_session)
        qr_url = f"{client.auth_base_url}/attendance/qr/{token}/"
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not prepare QR attendance: {exc}")
        return redirect("attendance_sessions")

    return render(
        request,
        "qr_attendance.html",
        {
            "active_session": active_session,
            "qr_url": qr_url,
            **method_context,
        },
    )


@desktop_login_required
def nfc_attendance(request):
    try:
        if not _desktop_plan_allows(request, "allow_nfc"):
            return _render_plan_locked(
                request,
                "NFC attendance is not included",
                "Upgrade to Professional or above to use NFC cards and tag-based check-in.",
                "Professional",
                back_url="attendance_sessions",
            )

        active_session = _get_active_attendance_session(request)
        guard = _session_method_guard_response(request, active_session, "nfc", "NFC check-in")
        if guard:
            return guard

        method_context = _session_method_context(request, active_session)
        recent_logs = _nfc_recent_logs(request, active_session.id)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load NFC attendance: {_friendly_api_error(exc)}")
        return redirect("attendance_sessions")

    return render(
        request,
        "nfc_attendance.html",
        {
            "active_session": active_session,
            "recent_logs": recent_logs,
            "reader_runtime": _reader_runtime(request),
            "reader_settings": request.session.get(NFC_READER_SETTINGS_SESSION_KEY) or {},
            **method_context,
        },
    )


def _nfc_recent_logs(request, session_id: int, limit: int = 8):
    logs_payload = _api_client(request).list_attendance_logs(
        attendance_session_id=session_id,
        page_size=max(limit, 12),
    )
    return [
        to_namespace(record)
        for record in extract_results(logs_payload)
        if record.get("method") == "nfc"
    ][:limit]


def _serialize_nfc_recent_logs(records):
    serialized = []
    for record in records:
        check_in_time = getattr(record, "check_in_time", "") or ""
        if check_in_time:
            check_in_time = str(check_in_time)
        serialized.append(
            {
                "id": getattr(record, "id", None),
                "person_name": getattr(record, "person_name", "") or "Member",
                "status": getattr(record, "status", "") or "present",
                "check_in_time": check_in_time,
            }
        )
    return serialized


@desktop_login_required
@require_GET
def nfc_attendance_status(request):
    if not _desktop_plan_allows(request, "allow_nfc"):
        return JsonResponse({"error": "NFC attendance is not included in this plan."}, status=403)

    try:
        active_session = _get_active_attendance_session(request)
        if not active_session:
            return JsonResponse({"error": "Select an active attendance session first."}, status=400)
        if not _session_method_enabled(active_session, request, "nfc"):
            return JsonResponse({"error": "NFC check-in is turned off for this session."}, status=400)

        method_context = _session_method_context(request, active_session)
        recent_logs = _nfc_recent_logs(request, active_session.id)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return JsonResponse({"error": "Desktop session expired. Please sign in again."}, status=401)
        return JsonResponse({"error": _friendly_api_error(exc)}, status=400)

    return JsonResponse(
        {
            "runtime": _reader_runtime(request),
            "settings": request.session.get(NFC_READER_SETTINGS_SESSION_KEY) or {},
            "active_session": {
                "id": getattr(active_session, "id", None),
                "label": getattr(active_session, "session_name", None) or getattr(active_session, "event_name", None) or "",
                "event_name": getattr(active_session, "event_name", None) or "",
            },
            "method_counts": method_context.get("method_counts", {}),
            "recent_logs": _serialize_nfc_recent_logs(recent_logs),
        }
    )


def _session_attendance_map(request, session_id: int):
    payload = _api_client(request).list_attendance_logs(attendance_session_id=session_id, page_size=200)
    records = extract_results(payload)
    return {record.get("person"): record for record in records}


def _mark_attendance_for_session(request, session_id: int, person_id: int, status_value: str, method: str = "manual"):
    existing = _session_attendance_map(request, session_id).get(person_id)
    now = timezone.now().isoformat()
    payload = {
        "person": person_id,
        "attendance_session": session_id,
        "status": status_value,
        "method": method,
    }

    if status_value == "present":
        payload["check_in_time"] = now
    else:
        payload["check_in_time"] = None
        payload["check_out_time"] = None

    if existing:
        return _api_client(request).update_attendance_log(existing["id"], payload)
    return _api_client(request).create_attendance_log(payload)


def _mark_session_attendance(request, person_id: int, status_value: str, method: str = "manual"):
    active_session = _get_active_attendance_session(request)
    if not active_session:
        raise AttendanceApiError("Select an active attendance session before marking attendance.")
    return _mark_attendance_for_session(request, active_session.id, person_id, status_value, method=method)


@desktop_login_required
def manual_attendance(request):
    try:
        def _manual_redirect_url(source):
            query = {}
            search_value = (source.get("search") or "").strip()
            status_value = (source.get("filter_status") or source.get("status_filter") or request.GET.get("status") or "all").strip().lower() or "all"
            method_value = (source.get("method") or "all").strip().lower() or "all"
            if search_value:
                query["search"] = search_value
            if status_value != "all":
                query["status"] = status_value
            if method_value != "all":
                query["method"] = method_value
            if not query:
                return reverse("manual_attendance")
            return f"{reverse('manual_attendance')}?{urlencode(query)}"

        def _display_dt(value):
            if not value:
                return ""
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if timezone.is_naive(parsed):
                        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
                    value = timezone.localtime(parsed)
                except ValueError:
                    return value
            try:
                localized = timezone.localtime(value)
            except Exception:
                localized = value
            return localized.strftime("%d %b %Y %I:%M %p")

        active_session = _get_active_attendance_session(request)
        if not active_session:
            messages.warning(request, "Select an event session before marking manual attendance.")
            return redirect("attendance_sessions")

        if request.method == "POST":
            action = request.POST.get("action", "mark").strip()
            status_value = request.POST.get("status", "present").strip()
            if status_value not in {"present", "absent"}:
                status_value = "present"
            redirect_url = _manual_redirect_url(request.POST)
            if action in {"mark_all", "mark_pending"}:
                search = request.POST.get("search", request.GET.get("search", "")).strip()
                persons = get_all_people(search=search, request=request, authorized=True)
                attendance_by_person = _session_attendance_map(request, active_session.id)
                updated = 0
                for person in persons:
                    if action == "mark_pending" and attendance_by_person.get(person.id):
                        continue
                    _mark_attendance_for_session(request, active_session.id, person.id, status_value, method="manual")
                    updated += 1
                messages.success(request, f"{updated} member{'s' if updated != 1 else ''} marked {status_value}.")
            elif action == "mark_selected":
                selected_ids = [int(value) for value in request.POST.getlist("person_ids") if str(value).isdigit()]
                if not selected_ids:
                    messages.warning(request, "Select at least one member first.")
                else:
                    updated = 0
                    for person_id in selected_ids:
                        _mark_attendance_for_session(request, active_session.id, person_id, status_value, method="manual")
                        updated += 1
                    messages.success(request, f"{updated} selected member{'s' if updated != 1 else ''} marked {status_value}.")
            else:
                person_id = int(request.POST.get("person_id", "0") or "0")
                _mark_session_attendance(request, person_id, status_value, method="manual")
                messages.success(request, "Attendance updated.")
            return redirect(redirect_url)

        search = request.GET.get("search", "").strip()
        status_filter = (request.GET.get("status") or "all").strip().lower() or "all"
        method_filter = (request.GET.get("method") or "all").strip().lower() or "all"
        persons = get_all_people(search=search, request=request, authorized=True)
        attendance_by_person = _session_attendance_map(request, active_session.id)
        method_context = _session_method_context(request, active_session)
        rows = []
        for person in persons:
            record = attendance_by_person.get(person.id)
            status_key = "pending"
            status_label = "Pending"
            method_key = ""
            method_label = "-"
            attendance_time = ""
            if record:
                status_key = "absent" if record.get("status") == "absent" else "present"
                status_label = "Absent" if status_key == "absent" else "Present"
                method_key = (record.get("method") or "").strip().lower()
                method_label = (record.get("method") or "-").replace("_", " ").title()
                attendance_time = _display_dt(record.get("check_in_time") or "")

            if status_filter != "all" and status_key != status_filter:
                continue
            if method_filter != "all" and method_key != method_filter:
                continue

            rows.append(
                {
                    "person": person,
                    "record": record,
                    "status_key": status_key,
                    "status_label": status_label,
                    "method_key": method_key,
                    "method_label": method_label,
                    "attendance_time": attendance_time,
                }
            )
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load manual attendance: {_friendly_api_error(exc)}")
        return redirect("attendance_sessions")

    return render(
        request,
        "manual_attendance.html",
        {
            "active_session": active_session,
            "rows": rows,
            "search_query": request.GET.get("search", "").strip(),
            "status_filter": status_filter,
            "method_filter": method_filter,
            "row_count": len(rows),
            "session_time_label": _display_dt(getattr(active_session, "attendance_opens_at", None) or getattr(active_session, "starts_at", None)),
            "session_end_label": _display_dt(getattr(active_session, "attendance_closes_at", None) or getattr(active_session, "ends_at", None)),
            **method_context,
        },
    )


@desktop_login_required
def swipe_attendance(request):
    try:
        active_session = _get_active_attendance_session(request)
        guard = _session_method_guard_response(request, active_session, "swipe", "Swipe check-in")
        if guard:
            return guard

        if request.method == "POST":
            person_id = int(request.POST.get("person_id", "0") or "0")
            status_value = request.POST.get("status", "present").strip()
            if status_value not in {"present", "absent"}:
                status_value = "present"
            _mark_session_attendance(request, person_id, status_value, method="swipe")
            return redirect("swipe_attendance")

        persons = get_all_people(request=request, authorized=True)
        attendance_by_person = _session_attendance_map(request, active_session.id)
        method_context = _session_method_context(request, active_session)
        pending = [person for person in persons if person.id not in attendance_by_person]
        current_person = pending[0] if pending else None
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load swipe attendance: {_friendly_api_error(exc)}")
        return redirect("attendance_sessions")

    return render(
        request,
        "swipe_attendance.html",
        {
            "active_session": active_session,
            "current_person": current_person,
            "pending_preview": pending[:3],
            "pending_count": len(pending),
            "completed_count": len(persons) - len(pending),
            "total_count": len(persons),
            **method_context,
        },
    )


@desktop_login_required
def geotracking_attendance(request):
    try:
        if not _desktop_plan_allows(request, "allow_geotracking"):
            return _render_plan_locked(
                request,
                "GeoTracking attendance is not included",
                "Upgrade to Enterprise to use location-aware attendance workflows.",
                "Enterprise",
                back_url="attendance_sessions",
            )

        active_session = _get_active_attendance_session(request)
        guard = _session_method_guard_response(request, active_session, "geo", "GeoTracking")
        if guard:
            return guard

        if request.method == "POST":
            person_id = int(request.POST.get("person_id", "0") or "0")
            latitude = request.POST.get("latitude", "").strip()
            longitude = request.POST.get("longitude", "").strip()
            accuracy = request.POST.get("accuracy", "").strip()
            if not latitude or not longitude:
                messages.error(request, "Location was not captured. Allow location access and try again.")
                return redirect("geotracking_attendance")
            payload = {
                "person_id": person_id,
                "attendance_session_id": active_session.id,
                "latitude": latitude,
                "longitude": longitude,
            }
            if accuracy:
                payload["accuracy_meters"] = accuracy
            outcome = _api_client(request).geotracking_check_in(payload)
            distance = outcome.get("distance_meters")
            radius = outcome.get("radius_meters")
            detail = f" Distance: {distance}m of {radius}m allowed." if distance is not None and radius else ""
            messages.success(request, f"GeoTracking attendance saved for this session.{detail}")
            return redirect("geotracking_attendance")

        search = request.GET.get("search", "").strip()
        persons = get_all_people(search=search, request=request, authorized=True)
        method_context = _session_method_context(request, active_session)
        logs_payload = _api_client(request).list_attendance_logs(
            attendance_session_id=active_session.id,
            page_size=20,
        )
        recent_logs = [
            to_namespace(record)
            for record in extract_results(logs_payload)
            if record.get("method") == "geotracking"
        ][:10]
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load GeoTracking attendance: {_friendly_api_error(exc)}")
        return redirect("attendance_sessions")

    return render(
        request,
        "geotracking_attendance.html",
        {
            "active_session": active_session,
            "persons": persons,
            "recent_logs": recent_logs,
            "search_query": request.GET.get("search", "").strip(),
            **method_context,
        },
    )


def geolocation_attendance(request):
    return geotracking_attendance(request)


@desktop_login_required
def home(request):
    try:
        overview = _api_client(request).overview()
        logs = fetch_all_attendance_logs(request=request)
        active_session = _get_active_attendance_session(request)
        setup_context = _build_setup_context(request, overview=overview, active_session=active_session)
        active_method_flags = _session_method_flags(active_session, request) if active_session else {}
        active_session_logs = (
            fetch_all_attendance_logs(attendance_session_id=str(active_session.id), request=request)
            if active_session else []
        )
        active_present = sum(
            1 for item in active_session_logs
            if getattr(item, "status", "") != "absent" and (
                getattr(item, "status", "") == "present" or getattr(item, "check_in_time", None)
            )
        )
        active_absent = sum(1 for item in active_session_logs if getattr(item, "status", "") == "absent")
        active_marked_people = {
            getattr(getattr(item, "person", None), "id", None)
            for item in active_session_logs
            if getattr(getattr(item, "person", None), "id", None)
        }
        total_persons = overview.get("total_persons", 0)
        context = {
            "total_persons": total_persons,
            "total_attendance": overview.get("total_attendance", 0),
            "total_check_ins": sum(1 for item in logs if getattr(item, "check_in_time", None)),
            "total_check_outs": sum(1 for item in logs if getattr(item, "check_out_time", None)),
            "total_cameras": overview.get("total_cameras", 0),
            "desktop_user_name": request.session.get(DESKTOP_NAME_SESSION_KEY, "Desktop User"),
            "desktop_user_role": request.session.get(DESKTOP_ROLE_SESSION_KEY, ""),
            "active_session": active_session,
            "active_session_methods": [
                {"label": "Mobile Self Check-in", "icon": "fa-mobile-screen", "enabled": active_method_flags.get("self", False)},
                {"label": "QR Code", "icon": "fa-qrcode", "enabled": active_method_flags.get("qr", False)},
                {"label": "NFC Check-in", "icon": "fa-id-card-clip", "enabled": active_method_flags.get("nfc", False)},
                {"label": "Face Recognition", "icon": "fa-face-smile", "enabled": active_method_flags.get("face", False)},
                {"label": "GeoTracking", "icon": "fa-location-dot", "enabled": active_method_flags.get("geo", False)},
            ],
            "active_session_present": active_present,
            "active_session_absent": active_absent,
            "active_session_pending": max(0, int(total_persons or 0) - len(active_marked_people)),
            **setup_context,
        }
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Attendance API unavailable: {_friendly_api_error(exc)}")
        context = {
            "total_persons": 0,
            "total_attendance": 0,
            "total_check_ins": 0,
            "total_check_outs": 0,
            "total_cameras": 0,
            "desktop_user_name": request.session.get(DESKTOP_NAME_SESSION_KEY, "Desktop User"),
                "desktop_user_role": request.session.get(DESKTOP_ROLE_SESSION_KEY, ""),
                "active_session": None,
                "active_session_methods": [],
                "active_session_present": 0,
                "active_session_absent": 0,
                "active_session_pending": 0,
                "organization": request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {},
              "plan": request.session.get(DESKTOP_PLAN_SESSION_KEY) or {},
              "attendance_methods": _attendance_method_cards(request),
              "setup_checklist": [],
              "setup_completed": 0,
              "setup_total": 0,
              "setup_progress_percent": 0,
              "setup_is_complete": False,
              "setup_next_step": None,
              "setup_recommended_action": None,
              "setup_counts": {},
          }
    return render(request, "home.html", context)


@desktop_login_required
def search_user(request):
    messages.info(request, "External member lookup now starts inside Register Member.")
    return redirect(f"{reverse('register_user')}?mode=external")


@desktop_login_required
def register_user(request):
    integration_enabled = _external_member_integration_enabled(request)

    def _split_name_parts(full_name: str | None):
        text = (full_name or "").strip()
        if not text:
            return "", ""
        bits = text.split(None, 1)
        if len(bits) == 1:
            return bits[0], ""
        return bits[0], bits[1]

    def _register_context(**extra):
        base = {
            "registration_mode": "single",
            "portal_id": "",
            "name": "",
            "first_name": "",
            "last_name": "",
            "email": "",
            "phone": "",
            "nfc_uid": "",
            "nfc_handoff_ready": False,
            "camera_configs": cams,
            "member_limit": member_limit,
            "integration_enabled": integration_enabled,
            "user_data": [],
            "search_performed": False,
            "searched_last_name": "",
            "lookup_error": "",
            "created_member_id": "",
            "created_member_name": "",
            "success_mode": False,
        }
        base.update(extra)
        return base

    try:
        cams = get_all_cameras(request=request)
        member_limit = _member_limit_status(request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        cams = []
        member_limit = _member_limit_status(request)
        messages.error(request, f"Could not load camera settings: {_friendly_api_error(exc)}")

    if member_limit["is_reached"]:
        return _render_plan_locked(
            request,
            "Member limit reached",
            "Your current plan has reached its member capacity. Upgrade the plan before adding more members.",
            "Higher member limit",
            back_url="person_list",
        )

    if request.method == "GET":
        requested_mode = (request.GET.get("mode") or "single").strip().lower()
        registration_mode = "external" if requested_mode == "external" else "single"
        portal_id = request.GET.get("portal_id")
        if portal_id and not integration_enabled:
            messages.warning(request, "External IDs are ignored because this workspace has no member integration connected.")
            portal_id = ""
        if portal_id:
            registration_mode = "external"
        name = request.GET.get("name")
        first_name, last_name = _split_name_parts(name)
        email = request.GET.get("email", "")
        phone = request.GET.get("phone", "")
        nfc_uid = request.GET.get("nfc_uid", "")
        reader_runtime = _reader_runtime(request)
        if not nfc_uid:
            nfc_uid = str(reader_runtime.get("pending_enrollment_uid") or "").strip()
        created_member_id = request.GET.get("created_member_id", "").strip()
        created_member_name = request.GET.get("created_member_name", "").strip()
        success_mode = request.GET.get("created") == "1"

        return render(
            request,
            "register_user.html",
            _register_context(
                registration_mode=registration_mode,
                portal_id=portal_id,
                name=name,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                nfc_uid=nfc_uid,
                nfc_handoff_ready=bool(nfc_uid),
                created_member_id=created_member_id,
                created_member_name=created_member_name or name or "",
                success_mode=success_mode,
            ),
        )

    if request.method == "POST":
        action = (request.POST.get("action") or "register").strip().lower()
        registration_mode = (request.POST.get("registration_mode") or "single").strip().lower()
        if registration_mode not in {"single", "external"}:
            registration_mode = "single"

        if action == "external_search":
            searched_last_name = request.POST.get("searched_last_name", "").strip()
            first_name = request.POST.get("first_name", "").strip()
            last_name = request.POST.get("last_name", "").strip()
            portal_id = request.POST.get("portal_id", "").strip()
            email = request.POST.get("email", "").strip()
            phone = request.POST.get("phone", "").strip()
            nfc_uid = request.POST.get("nfc_uid", "").strip()

            if not integration_enabled:
                return render(
                    request,
                    "register_user.html",
                    _register_context(
                        registration_mode="external",
                        portal_id=portal_id,
                        first_name=first_name,
                        last_name=last_name,
                        email=email,
                        phone=phone,
                        nfc_uid=nfc_uid,
                        lookup_error="External member lookup is not connected for this workspace.",
                    ),
                )

            if not searched_last_name:
                return render(
                    request,
                    "register_user.html",
                    _register_context(
                        registration_mode="external",
                        portal_id=portal_id,
                        first_name=first_name,
                        last_name=last_name,
                        email=email,
                        phone=phone,
                        nfc_uid=nfc_uid,
                        lookup_error="Enter a last name to search.",
                    ),
                )

            user_data = fetch_user_data(searched_last_name) or []
            return render(
                request,
                "register_user.html",
                _register_context(
                    registration_mode="external",
                    portal_id=portal_id,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    nfc_uid=nfc_uid,
                    user_data=user_data,
                    searched_last_name=searched_last_name,
                    search_performed=True,
                ),
            )

        portal_id = request.POST.get("portal_id")
        if portal_id and not integration_enabled:
            portal_id = ""
        first_name = request.POST.get("first_name", "").strip()
        last_name = request.POST.get("last_name", "").strip()
        name = request.POST.get("name", "").strip()
        if first_name or last_name:
            name = f"{first_name} {last_name}".strip()
        email = request.POST.get("email", "").strip()
        phone = request.POST.get("phone", "").strip()
        nfc_uid = request.POST.get("nfc_uid", "").strip()
        image_data = request.POST.get("image_data")

        if not name:
            return render(
                request,
                "register_user.html",
                _register_context(
                    error="Name is required.",
                    registration_mode=registration_mode,
                    portal_id=portal_id,
                    name=name,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    nfc_uid=nfc_uid,
                ),
                )

        if not image_data:
            return render(
                request,
                "register_user.html",
                _register_context(
                    error="Image is required. Please capture a photo.",
                    registration_mode=registration_mode,
                    portal_id=portal_id,
                    name=name,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    nfc_uid=nfc_uid,
                ),
                )

        try:
            person = _api_client(request).create_person(
                {
                    "name": name,
                    "portal_id": portal_id or "",
                    "external_id": portal_id or "",
                    "email": email,
                    "phone": phone,
                    "nfc_uid": nfc_uid,
                    "source": "integration" if portal_id else "direct",
                    "image_data": image_data,
                    "authorized": True,
                }
            )
            runtime_payload = _reader_runtime(request)
            if nfc_uid and str(runtime_payload.get("pending_enrollment_uid") or "").strip().upper() == nfc_uid.strip().upper():
                runtime_payload.update(
                    {
                        "pending_enrollment_uid": "",
                        "pending_enrollment_url": "",
                        "last_action": f"NFC tag {nfc_uid.strip().upper()} assigned during member registration.",
                        "status": "Connected",
                        "last_error": "",
                    }
                )
                _store_reader_runtime(request, runtime_payload)
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            return render(
                request,
                "register_user.html",
                _register_context(
                    error=f"Could not save member to API: {exc}",
                    registration_mode=registration_mode,
                    portal_id=portal_id,
                    name=name,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    nfc_uid=nfc_uid,
                ),
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

        success_query = urlencode(
            {
                "created": "1",
                "created_member_id": person.get("id", ""),
                "created_member_name": person.get("name", name),
            }
        )
        return redirect(f"{reverse('register_user')}?{success_query}")

    return render(request, "register_user.html", _register_context())


def success_page(request):
    return render(request, "selfie_success.html")


# =========================================================
# People management (MATCH your urls.py)
# =========================================================
@desktop_login_required
def person_list(request):
    def _member_registered_label(person):
        for attr in ("created_at", "created_on", "created", "registered_on", "date_joined"):
            value = getattr(person, attr, None)
            if value:
                return value
        return "—"

    search = request.GET.get("search", "").strip()
    filter_value = request.GET.get("filter", "all").strip().lower() or "all"
    authorized_filter = None
    source_filter = ""
    if filter_value == "pending":
        authorized_filter = False
    elif filter_value == "authorized":
        authorized_filter = True
    elif filter_value in {"self_registered", "imported", "direct", "integration"}:
        source_filter = filter_value

    try:
        persons_qs = get_all_people(
            search=search,
            request=request,
            authorized=authorized_filter,
            source=source_filter,
        )
        all_people_for_counts = get_all_people(request=request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        persons_qs = []
        all_people_for_counts = []
        messages.error(request, f"Could not load members: {_friendly_api_error(exc)}")

    member_counts = {
        "all": len(all_people_for_counts),
        "pending": len([person for person in all_people_for_counts if not getattr(person, "authorized", False)]),
        "authorized": len([person for person in all_people_for_counts if getattr(person, "authorized", False)]),
        "self_registered": len([person for person in all_people_for_counts if getattr(person, "source", "") == "self_registered"]),
        "imported": len([person for person in all_people_for_counts if getattr(person, "source", "") == "imported"]),
    }
    organization = request.session.get(DESKTOP_ORGANIZATION_SESSION_KEY) or {}
    registration_url = ""
    if organization.get("slug"):
        registration_url = f"{_api_client(request).auth_base_url}/attendance/self-register/{organization['slug']}/"

    paginator = Paginator(persons_qs, 10)  # 10 per page
    page_number = request.GET.get("page")
    persons = paginator.get_page(page_number)
    for person in persons:
        setattr(person, "registered_label", _member_registered_label(person))

    return render(
        request,
        "user_list.html",
        {
            "persons": persons,
            "can_import_members": _plan_allows_member_import(request),
            "member_limit": _member_limit_status(request),
            "filter_value": filter_value,
            "member_counts": member_counts,
            "registration_url": registration_url,
            "search": search,
        },
    )


@desktop_login_required
def bulk_nfc_enrollment(request):
    if not _desktop_plan_allows(request, "allow_nfc"):
        return _render_plan_locked(
            request,
            "NFC tools are not included",
            "Upgrade to Professional or above to manage NFC readers and member tags.",
            "Professional",
            back_url="person_list",
        )
    search = request.GET.get("search", "").strip()
    include_assigned = request.GET.get("include_assigned") == "1"
    current_id = (request.GET.get("current") or "").strip()
    prefill_uid = (request.GET.get("prefill_uid") or "").strip()

    try:
        people = get_all_people(search=search, request=request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load members for NFC enrollment: {_friendly_api_error(exc)}")
        people = []

    queue = [
        person for person in people
        if include_assigned or not getattr(person, "nfc_uid", "")
    ]
    current_person = None
    if current_id:
        try:
            current_person = next((person for person in queue if str(person.id) == current_id), None)
        except StopIteration:
            current_person = None
    if current_person is None and queue:
        current_person = queue[0]

    if request.method == "POST":
        person_id = (request.POST.get("person_id") or "").strip()
        nfc_uid = (request.POST.get("nfc_uid") or "").strip()
        search = (request.POST.get("search") or "").strip()
        include_assigned = request.POST.get("include_assigned") == "1"
        if not person_id:
            messages.error(request, "Choose a member before saving an NFC tag.")
            return redirect("bulk_nfc_enrollment")
        try:
            _api_client(request).update_person(int(person_id), {"nfc_uid": nfc_uid})
            if nfc_uid:
                messages.success(request, "NFC tag saved and the queue moved to the next member.")
            else:
                messages.success(request, "NFC tag cleared for this member.")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not save the NFC tag: {_friendly_api_error(exc)}")
            params = urlencode(
                {
                    "search": search,
                    "include_assigned": "1" if include_assigned else "0",
                    "current": person_id,
                }
            )
            return redirect(f"{reverse('bulk_nfc_enrollment')}?{params}")

        next_people = get_all_people(search=search, request=request)
        next_queue = [person for person in next_people if include_assigned or not getattr(person, "nfc_uid", "")]
        next_person = next((person for person in next_queue if str(person.id) != person_id), None)
        params = {
            "search": search,
            "include_assigned": "1" if include_assigned else "0",
        }
        if next_person:
            params["current"] = next_person.id
        return redirect(f"{reverse('bulk_nfc_enrollment')}?{urlencode(params)}")

    return render(
        request,
        "bulk_nfc_enrollment.html",
        {
            "people": queue,
            "current_person": current_person,
            "search": search,
            "include_assigned": include_assigned,
            "queue_total": len(queue),
            "prefill_uid": prefill_uid,
        },
    )


@desktop_login_required
def nfc_reader_setup(request):
    if not _desktop_plan_allows(request, "allow_nfc"):
        return _render_plan_locked(
            request,
            "NFC tools are not included",
            "Upgrade to Professional or above to manage NFC readers and member tags.",
            "Professional",
            back_url="person_list",
        )
    defaults = {
        "reader_name": "Entrance Reader",
        "reader_mode": "keyboard",
        "reader_suffix": "Enter",
        "reader_prefix": "",
        "station_identifier": "",
        "capture_mode": "check_in",
    }
    settings_payload = {**defaults, **(request.session.get(NFC_READER_SETTINGS_SESSION_KEY) or {})}
    if request.method == "POST":
        settings_payload = {
            "reader_name": (request.POST.get("reader_name") or "").strip() or defaults["reader_name"],
            "reader_mode": (request.POST.get("reader_mode") or "").strip() or defaults["reader_mode"],
            "reader_suffix": (request.POST.get("reader_suffix") or "").strip() or defaults["reader_suffix"],
            "reader_prefix": (request.POST.get("reader_prefix") or "").strip(),
            "station_identifier": (request.POST.get("station_identifier") or "").strip(),
            "capture_mode": (request.POST.get("capture_mode") or "").strip() or defaults["capture_mode"],
        }
        request.session[NFC_READER_SETTINGS_SESSION_KEY] = settings_payload
        request.session.modified = True
        _sync_reader_bridge_config(request, reader_settings=settings_payload)
        messages.success(request, "Reader setup saved for this desktop session.")
        return redirect("nfc_reader_setup")
    active_session = _get_active_attendance_session(request)
    _sync_reader_bridge_config(request, active_session=active_session, reader_settings=settings_payload)
    return render(
        request,
        "nfc_reader_setup.html",
        {
            "reader_settings": settings_payload,
            "desktop_can_view_audit": _desktop_can_view_audit(request),
            "reader_runtime": _reader_runtime(request),
            "bridge_running": bool(_reader_bridge_running_pid()),
            "active_session": active_session,
        },
    )


@desktop_login_required
@require_GET
def nfc_reader_status(request):
    if not _desktop_plan_allows(request, "allow_nfc"):
        return JsonResponse({"error": "NFC tools are not included in this plan."}, status=403)
    active_session = _get_active_attendance_session(request)
    return JsonResponse(
        {
            "runtime": _reader_runtime(request),
            "settings": request.session.get(NFC_READER_SETTINGS_SESSION_KEY) or {},
            "active_session": {
                "id": getattr(active_session, "id", None),
                "label": getattr(active_session, "session_name", None) or getattr(active_session, "event_name", None) or "",
                "branch_name": getattr(active_session, "branch_name", None) or "",
            }
            if active_session
            else None,
            "bridge_running": bool(_reader_bridge_running_pid()),
        }
    )


@desktop_login_required
@require_POST
def nfc_reader_bridge_start(request):
    if not _desktop_plan_allows(request, "allow_nfc"):
        return JsonResponse({"error": "NFC tools are not included in this plan."}, status=403)
    try:
        pid, started = _launch_reader_bridge()
        runtime_payload = _reader_runtime(request)
        runtime_payload.update(
            {
                "status": "Connected" if started else runtime_payload.get("status") or "Waiting",
                "last_error": "",
                "last_action": "Desktop bridge started from KairosTrack." if started else "Desktop bridge is already running.",
            }
        )
        _store_reader_runtime(request, runtime_payload)
        return JsonResponse(
            {
                "ok": True,
                "started": started,
                "pid": pid,
                "message": "Desktop bridge started from KairosTrack." if started else "Desktop bridge is already running.",
            }
        )
    except Exception as exc:
        return JsonResponse({"error": f"Could not start the reader bridge: {exc}"}, status=500)


@desktop_login_required
@require_POST
def nfc_reader_capture(request):
    if not _desktop_plan_allows(request, "allow_nfc"):
        return JsonResponse({"error": "NFC tools are not included in this plan."}, status=403)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    settings_payload = {
        "reader_name": "Entrance Reader",
        "reader_mode": "keyboard",
        "reader_suffix": "Enter",
        "reader_prefix": "",
        "station_identifier": "",
        "capture_mode": "check_in",
        **(request.session.get(NFC_READER_SETTINGS_SESSION_KEY) or {}),
    }
    raw_uid = str(payload.get("uid") or "").strip()
    prefix = str(settings_payload.get("reader_prefix") or "").strip()
    if prefix and raw_uid.startswith(prefix):
        raw_uid = raw_uid[len(prefix):]
    uid = "".join(raw_uid.split()).upper()
    runtime_payload = _reader_runtime(request)
    if not uid:
        runtime_payload.update(
            {
                "status": "Error",
                "last_error": "Empty UID received",
                "last_action": "The workstation received an empty UID.",
            }
        )
        _store_reader_runtime(request, runtime_payload)
        return JsonResponse({"error": "UID is required."}, status=400)

    runtime_payload.update(
        {
            "status": "Connected",
            "last_uid": uid,
            "last_scan_at": timezone.localtime().isoformat(),
            "last_error": "",
        }
    )
    capture_mode = str(payload.get("capture_mode") or settings_payload.get("capture_mode") or "check_in").strip().lower()
    active_session = _get_active_attendance_session(request)
    response_payload = {
        "uid": uid,
        "capture_mode": capture_mode,
        "active_session_id": getattr(active_session, "id", None),
    }

    if capture_mode == "enroll":
        enroll_url = f"{reverse('register_user')}?{urlencode({'nfc_uid': uid})}"
        runtime_payload.update(
            {
                "last_action": "UID captured and ready for member registration.",
                "pending_enrollment_uid": uid,
                "pending_enrollment_url": enroll_url,
            }
        )
        _store_reader_runtime(request, runtime_payload)
        response_payload.update({"message": runtime_payload["last_action"], "enrollment_url": enroll_url})
        return JsonResponse(response_payload)

    if capture_mode == "check_in":
        if not active_session:
            runtime_payload.update(
                {
                    "status": "Error",
                    "last_error": "No active session available",
                    "last_action": "Set an active attendance session before using live NFC check-in.",
                }
            )
            _store_reader_runtime(request, runtime_payload)
            return JsonResponse({"error": "Select an active attendance session first.", "uid": uid}, status=400)
        try:
            outcome = _api_client(request).nfc_station_check_in(
                {
                    "uid": uid,
                    "station_identifier": settings_payload.get("station_identifier") or "",
                    "attendance_session_id": int(getattr(active_session, "id")),
                    "device_label": settings_payload.get("reader_name") or "Desktop Reader Station",
                }
            )
            attendance = outcome.get("attendance") if isinstance(outcome, dict) else {}
            person_name = ""
            if isinstance(attendance, dict):
                person_name = str(attendance.get("person_name") or attendance.get("name") or "").strip()
            runtime_payload.update(
                {
                    "last_action": f"Checked in {person_name or 'member'} with NFC.",
                    "pending_enrollment_uid": "",
                    "pending_enrollment_url": "",
                }
            )
            _store_reader_runtime(request, runtime_payload)
            response_payload.update({"message": runtime_payload["last_action"], "outcome": outcome})
            return JsonResponse(response_payload)
        except AttendanceApiError as exc:
            runtime_payload.update(
                {
                    "status": "Error",
                    "last_error": _friendly_api_error(exc),
                    "last_action": "The reader station could not complete the check-in.",
                }
            )
            _store_reader_runtime(request, runtime_payload)
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return JsonResponse({"error": "Desktop session expired. Please sign in again."}, status=401)
            return JsonResponse({"error": _friendly_api_error(exc), "uid": uid}, status=400)

    runtime_payload.update(
        {
            "last_action": "UID captured in monitor mode.",
            "pending_enrollment_uid": "",
            "pending_enrollment_url": "",
        }
    )
    _store_reader_runtime(request, runtime_payload)
    response_payload["message"] = runtime_payload["last_action"]
    return JsonResponse(response_payload)


def _csv_value(row: dict, *keys: str) -> str:
    normalized = {str(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value:
            return value
    return ""


def _csv_bool(value: str, default: bool = True) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "present", "active"}


def _member_import_template_response():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Members"
    headers = [
        "name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "external_id",
        "branch",
        "authorized",
    ]
    sheet.append(headers)
    sheet.append(["Ada Johnson", "", "", "ada@example.com", "07123456789", "STU-001", "Main", "yes"])
    sheet.append(["", "Tomi", "Cole", "tomi@example.com", "", "STU-002", "Main", "no"])
    for column in sheet.columns:
        letter = column[0].column_letter
        sheet.column_dimensions[letter].width = max(14, max(len(str(cell.value or "")) for cell in column) + 3)

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    response = HttpResponse(
        output.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = 'attachment; filename="kairostrack-member-import-template.xlsx"'
    return response


def _read_member_import_rows(upload):
    filename = upload.name.lower()
    if filename.endswith(".csv"):
        decoded = upload.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded))
        if not reader.fieldnames:
            raise ValueError("The CSV file is empty or missing headers.")
        return list(reader)

    if filename.endswith(".xlsx"):
        workbook = load_workbook(upload, read_only=True, data_only=True)
        sheet = workbook.active
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            raise ValueError("The Excel file is empty.")
        headers = [str(value or "").strip() for value in rows[0]]
        if not any(headers):
            raise ValueError("The Excel file is missing headers.")
        parsed_rows = []
        for row in rows[1:]:
            parsed_rows.append({headers[index]: value for index, value in enumerate(row) if index < len(headers)})
        return parsed_rows

    raise ValueError("Please upload a .csv or .xlsx file.")


def _prepare_member_import_preview(raw_rows, branches, member_limit, default_branch="", authorize_imported=True, existing_people=None):
    branch_by_name = {str(getattr(branch, "name", "")).strip().lower(): getattr(branch, "id", None) for branch in branches}
    branch_name_by_id = {str(getattr(branch, "id", "")): getattr(branch, "name", "") for branch in branches}
    existing_people = existing_people or []
    existing_by_external_id = {
        str(getattr(person, "external_id", "") or "").strip().lower(): person
        for person in existing_people
        if str(getattr(person, "external_id", "") or "").strip()
    }
    existing_by_email = {
        str(getattr(person, "email", "") or "").strip().lower(): person
        for person in existing_people
        if str(getattr(person, "email", "") or "").strip()
    }
    existing_by_phone = {
        str(getattr(person, "phone", "") or "").strip().lower(): person
        for person in existing_people
        if str(getattr(person, "phone", "") or "").strip()
    }

    preview_rows = []
    valid_count = 0

    for row_number, row in enumerate(raw_rows, start=2):
        first_name = _csv_value(row, "first_name", "firstname", "first name")
        last_name = _csv_value(row, "last_name", "lastname", "last name", "surname")
        name = _csv_value(row, "name", "full_name", "full name") or f"{first_name} {last_name}".strip()
        email = _csv_value(row, "email", "email_address", "email address")
        phone = _csv_value(row, "phone", "mobile", "mobile_phone", "cellphone", "cell phone")
        external_id = _csv_value(row, "external_id", "external id", "member_id", "member id", "portal_id", "portal id")
        branch_name = _csv_value(row, "branch", "location", "department")
        authorized_value = _csv_value(row, "authorized", "active")
        branch_id = ""
        resolved_branch_name = ""
        errors = []

        if not name:
            errors.append("Missing member name.")

        if branch_name:
            branch_id = branch_by_name.get(branch_name.lower())
            if branch_id is None:
                errors.append(f"Branch '{branch_name}' was not found.")
            else:
                resolved_branch_name = branch_name
        elif default_branch:
            branch_id = default_branch
            resolved_branch_name = branch_name_by_id.get(str(default_branch), "")

        matched_person = None
        if external_id:
            matched_person = existing_by_external_id.get(external_id.lower())
        if matched_person is None and email:
            matched_person = existing_by_email.get(email.lower())
        if matched_person is None and phone:
            matched_person = existing_by_phone.get(phone.lower())

        action = "update" if matched_person else "create"
        if action == "create" and member_limit["is_limited"] and valid_count >= member_limit["remaining"]:
            errors.append("Member limit reached for the current plan.")

        status_value = "invalid" if errors else action
        if not errors:
            valid_count += 1

        preview_rows.append(
            {
                "row_number": row_number,
                "status": status_value,
                "errors": errors,
                "existing_id": getattr(matched_person, "id", None) if matched_person else None,
                "name": name,
                "email": email,
                "phone": phone,
                "external_id": external_id,
                "branch": branch_id or "",
                "branch_name": resolved_branch_name,
                "source": "imported",
                "authorized": _csv_bool(authorized_value, authorize_imported),
            }
        )

    return preview_rows


@desktop_login_required
def member_import_template(request):
    return _member_import_template_response()


@desktop_login_required
def member_import(request):
    if not _plan_allows_member_import(request):
        return _render_plan_locked(
            request,
            "CSV import is not included",
            "Upgrade to Standard or above to import members from CSV files.",
            "Standard",
            back_url="person_list",
        )

    member_limit = _member_limit_status(request)
    if member_limit["is_reached"]:
        return _render_plan_locked(
            request,
            "Member limit reached",
            "Your current plan has no remaining member capacity for imports.",
            "Higher member limit",
            back_url="person_list",
        )

    client = _api_client(request)
    try:
        branches = [to_namespace(item) for item in extract_results(client.list_branches())]
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        branches = []
        messages.error(request, f"Could not load branches: {_friendly_api_error(exc)}")

    if request.method == "POST":
        action = request.POST.get("action", "preview").strip()
        default_branch = request.POST.get("default_branch", "").strip()
        authorize_imported = request.POST.get("authorize_imported") == "on"

        if action == "import":
            preview_rows = request.session.get("member_import_preview_rows") or []
            if not preview_rows:
                messages.error(request, "Preview expired. Upload the file again before importing.")
                return redirect("member_import")

            created_count = 0
            updated_count = 0
            skipped_count = 0
            errors = []
            for item in preview_rows:
                if item.get("status") == "invalid":
                    skipped_count += 1
                    continue

                payload = {
                    "name": item.get("name", ""),
                    "email": item.get("email", ""),
                    "phone": item.get("phone", ""),
                    "external_id": item.get("external_id", ""),
                    "portal_id": item.get("external_id", ""),
                    "source": "imported",
                    "authorized": bool(item.get("authorized")),
                }
                if item.get("branch"):
                    payload["branch"] = item["branch"]

                try:
                    if item.get("existing_id"):
                        client.update_person(int(item["existing_id"]), payload)
                        updated_count += 1
                    else:
                        client.create_person(payload)
                        created_count += 1
                except AttendanceApiError as exc:
                    skipped_count += 1
                    if len(errors) < 12:
                        errors.append(f"Row {item.get('row_number')}: {exc}")

            request.session.pop("member_import_preview_rows", None)
            if created_count:
                messages.success(request, f"Created {created_count} new member{'s' if created_count != 1 else ''}.")
            if updated_count:
                messages.success(request, f"Updated {updated_count} existing member{'s' if updated_count != 1 else ''}.")
            if skipped_count:
                messages.warning(request, f"Skipped {skipped_count} row{'s' if skipped_count != 1 else ''}.")

            return render(
                request,
                "member_import.html",
                {
                    "branches": branches,
                    "created_count": created_count,
                    "updated_count": updated_count,
                    "skipped_count": skipped_count,
                    "errors": errors,
                    "member_limit": _member_limit_status(request),
                },
            )

        upload = request.FILES.get("import_file")
        if not upload:
            messages.error(request, "Choose a CSV or Excel file to preview.")
            return render(request, "member_import.html", {"branches": branches, "member_limit": member_limit})

        try:
            raw_rows = _read_member_import_rows(upload)
            existing_people = get_all_people(request=request)
            preview_rows = _prepare_member_import_preview(
                raw_rows,
                branches,
                member_limit,
                default_branch=default_branch,
                authorize_imported=authorize_imported,
                existing_people=existing_people,
            )
        except UnicodeDecodeError:
            messages.error(request, "Could not read the CSV. Please save it as UTF-8 and try again.")
            return render(request, "member_import.html", {"branches": branches, "member_limit": member_limit})
        except ValueError as exc:
            messages.error(request, str(exc))
            return render(request, "member_import.html", {"branches": branches, "member_limit": member_limit})

        request.session["member_import_preview_rows"] = preview_rows
        request.session.modified = True
        created_count = len([row for row in preview_rows if row["status"] == "create"])
        updated_count = len([row for row in preview_rows if row["status"] == "update"])
        skipped_count = len([row for row in preview_rows if row["status"] == "invalid"])

        return render(
            request,
            "member_import.html",
            {
                "branches": branches,
                "preview_rows": preview_rows,
                "created_count": created_count,
                "updated_count": updated_count,
                "skipped_count": skipped_count,
                "member_limit": member_limit,
                "default_branch": default_branch,
                "authorize_imported": authorize_imported,
            },
        )

    return render(request, "member_import.html", {"branches": branches, "member_limit": member_limit})


@desktop_login_required
def person_detail(request, pk: int):
    def _member_date_label(member, *attrs):
        for attr in attrs:
            value = getattr(member, attr, None)
            if value:
                return value
        return "—"

    try:
        person = normalize_person(_api_client(request).get_person(pk))
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load member: {_friendly_api_error(exc)}")
        return redirect("person_list")
    if getattr(person, "portal_id", ""):
        user_data = fetch_user_data_by_id(person.portal_id) or {}
    else:
        user_data = {
            "Email": getattr(person, "email", ""),
            "CellPhone": getattr(person, "phone", ""),
            "Address": "",
        }
    return render(
        request,
        "user_detail.html",
        {
            "person": person,
            "user_data": user_data,
            "created_label": _member_date_label(person, "created_at", "created_on", "created", "date_joined"),
            "updated_label": _member_date_label(person, "updated_at", "modified_at", "updated", "created_at", "created"),
        },
    )


@desktop_login_required
def audit_history(request):
    if not _desktop_can_view_audit(request):
        messages.error(request, "Only owners and admins can view audit history.")
        return redirect("home")
    action = request.GET.get("action", "").strip()
    actor = request.GET.get("actor", "").strip()
    branch_id = request.GET.get("branch_id", "").strip()
    person_id = request.GET.get("person_id", "").strip()
    session_id = request.GET.get("session_id", "").strip()
    target_id = request.GET.get("target_id", "").strip()
    date_from = request.GET.get("date_from", "").strip()
    date_to = request.GET.get("date_to", "").strip()
    station_identifier = request.GET.get("station_identifier", "").strip()
    page_number = int(request.GET.get("page", "1") or "1")
    page_size = 30
    params = {"page": page_number, "page_size": page_size}
    if action:
        params["action"] = action
    for key, value in {
        "actor": actor,
        "branch_id": branch_id,
        "person_id": person_id,
        "session_id": session_id,
        "target_id": target_id,
        "date_from": date_from,
        "date_to": date_to,
        "station_identifier": station_identifier,
    }.items():
        if value:
            params[key] = value
    try:
        payload = _api_client(request).list_audit_logs(**params)
        items = [to_namespace(item) for item in extract_results(payload)]
        for item in items:
            metadata = getattr(item, "metadata", {}) or {}
            formatted = []
            changes = metadata.get("changes") if isinstance(metadata, dict) else None
            if isinstance(changes, dict):
                for field, values in changes.items():
                    if isinstance(values, dict):
                        formatted.append(f"{field}: {values.get('before', '—')} -> {values.get('after', '—')}")
            elif isinstance(metadata, dict):
                for key, value in metadata.items():
                    formatted.append(f"{key}: {value}")
            setattr(item, "formatted_metadata", formatted)
            link = ""
            if getattr(item, "target_type", "") == "person" and str(getattr(item, "target_id", "")).isdigit():
                link = reverse("person_detail", args=[item.target_id])
            elif getattr(item, "target_type", "") == "attendance_session" and str(getattr(item, "target_id", "")).isdigit():
                link = reverse("attendance_session_detail", args=[item.target_id])
            setattr(item, "target_link", link)
        page = wrap_api_page(payload, items, page_number, page_size)
        summary = _api_client(request).audit_summary()
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load audit history: {_friendly_api_error(exc)}")
        page = wrap_api_page({"count": 0}, [], 1, page_size)
        summary = {"today": {}, "anomalies": {}}

    return render(
        request,
        "audit_history.html",
        {
            "audit_page": page,
            "action": action,
            "actor": actor,
            "branch_id": branch_id,
            "person_id": person_id,
            "session_id": session_id,
            "target_id": target_id,
            "date_from": date_from,
            "date_to": date_to,
            "station_identifier": station_identifier,
            "summary": summary,
            "action_options": [
                ("", "All activity"),
                ("person_created", "Member created"),
                ("person_updated", "Member updated"),
                ("person_authorized", "Authorization changes"),
                ("person_nfc_updated", "NFC changes"),
                ("session_updated", "Session updates"),
                ("session_state_changed", "Session state changes"),
                ("public_self_register_checkin", "Public self-registration"),
                ("member_logout", "Member logouts"),
                ("nfc_station_check_in", "NFC station check-ins"),
                ("anomaly", "Operational anomalies"),
            ],
        },
    )


@desktop_login_required
def anomaly_summary(request):
    if not _desktop_can_view_audit(request):
        messages.error(request, "Only owners and admins can view monitoring summaries.")
        return redirect("home")
    try:
        summary = _api_client(request).audit_summary()
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load monitoring summary: {_friendly_api_error(exc)}")
        summary = {"today": {}, "anomalies": {}, "throttles": {}}
    return render(request, "anomaly_summary.html", {"summary": summary})


@desktop_login_required
@require_POST
def person_update(request, pk: int):
    payload = {
        "name": request.POST.get("name", "").strip(),
        "email": request.POST.get("email", "").strip(),
        "phone": request.POST.get("phone", "").strip(),
    }

    if not payload["name"]:
        messages.error(request, "Member name is required.")
        return redirect("person_detail", pk=pk)

    try:
        _api_client(request).update_person(pk, payload)
        messages.success(request, "Member details updated.")
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not update member details: {_friendly_api_error(exc)}")
    return redirect("person_detail", pk=pk)


@desktop_login_required
@require_POST
def person_nfc_update(request, pk: int):
    if not _desktop_plan_allows(request, "allow_nfc"):
        messages.warning(request, "NFC member tags are available on the Professional plan and above.")
        return redirect("person_detail", pk=pk)

    nfc_uid = "" if request.POST.get("clear_nfc") else request.POST.get("nfc_uid", "").strip()
    try:
        _api_client(request).update_person(pk, {"nfc_uid": nfc_uid})
        if nfc_uid:
            messages.success(request, "NFC tag saved for this member.")
        else:
            messages.success(request, "NFC tag removed from this member.")
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not update NFC tag: {exc}")
    return redirect("person_detail", pk=pk)


@desktop_login_required
def person_authorize(request, pk: int):
    if not _desktop_can_manage_authorization(request):
        messages.error(request, "Only owners and admins can change member authorization.")
        return redirect("person_detail", pk=pk)
    if request.method == "POST":
        authorized = request.POST.get("authorized") in ("1", "true", "True", "on", "yes")
        next_url = request.POST.get("next") or request.GET.get("next")
        try:
            _api_client(request).authorize_person(pk, authorized)
            messages.success(request, "Member authorization updated.")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not update authorization: {exc}")
        if next_url:
            return redirect(next_url)
        return redirect("person_detail", pk=pk)

    return redirect("person_detail", pk=pk)


@desktop_login_required
def person_delete(request, pk: int):
    try:
        person = normalize_person(_api_client(request).get_person(pk))
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load member: {_friendly_api_error(exc)}")
        return redirect("person_list")

    if request.method == "POST":
        try:
            _api_client(request).delete_person(pk)
            messages.success(request, "Member deleted successfully.")
            return redirect("person_list")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
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
    event_filter = request.GET.get("event_id", "").strip()
    session_filter = request.GET.get("attendance_session_id", "").strip()
    status_filter = request.GET.get("status", "").strip().lower() or "all"
    method_filter = request.GET.get("method", "").strip().lower()
    page_size = int(request.GET.get("page_size", 25) or 25)
    page_size = max(10, min(page_size, 200))  # clamp
    page_number = int(request.GET.get("page", 1) or 1)

    def _normalize_method_label(value):
        labels = {
            "qr": "QR",
            "nfc": "NFC",
            "swipe": "Swipe",
            "manual": "Manual",
            "mobile_self": "Self Check-in",
            "mobile_self_check_in": "Self Check-in",
            "self_check_in": "Self Check-in",
            "geotracking": "GeoTracking",
            "face": "Face Recognition",
            "face_recognition": "Face Recognition",
        }
        cleaned = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return labels.get(cleaned, (value or "—").replace("_", " ").title())

    def _parse_dt(value):
        if not value:
            return None
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if timezone.is_naive(parsed):
                    parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
                return timezone.localtime(parsed)
            except ValueError:
                return None
        try:
            return timezone.localtime(value)
        except Exception:
            return None

    def _status_matches(record, selected_status):
        if not selected_status or selected_status == "all":
            return True
        return (getattr(record, "status", "") or "").strip().lower() == selected_status

    try:
        client = _api_client(request)
        events = [to_namespace(item) for item in extract_results(client.list_events())]
        session_params = {"event_id": event_filter} if event_filter else {}
        sessions = [to_namespace(item) for item in extract_results(client.list_sessions(**session_params))]
        active_session = _get_active_attendance_session(request)
        all_records = fetch_all_attendance_logs(
            search_query,
            date_filter,
            event_id=event_filter,
            attendance_session_id=session_filter,
            method=method_filter,
            request=request,
        )
        filtered_records = [record for record in all_records if _status_matches(record, status_filter)]
        for record in filtered_records:
            record.method_label = _normalize_method_label(getattr(record, "method", ""))
            current_status = (getattr(record, "status", "") or "present").strip().lower()
            record.status_css = current_status if current_status in {"present", "absent", "pending"} else "present"
        paginator = Paginator(filtered_records, page_size)
        page_obj = paginator.get_page(page_number)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load attendance logs: {_friendly_api_error(exc)}")
        all_records = []
        filtered_records = []
        paginator = Paginator([], page_size)
        page_obj = paginator.get_page(1)
        events = []
        sessions = []
        active_session = None

    present_count = sum(1 for record in filtered_records if (getattr(record, "status", "") or "").lower() == "present")
    absent_count = sum(1 for record in filtered_records if (getattr(record, "status", "") or "").lower() == "absent")
    pending_count = sum(1 for record in filtered_records if (getattr(record, "status", "") or "").lower() == "pending")
    total_count = len(filtered_records)

    activity_source = []
    for record in filtered_records:
        person = getattr(record, "person", None)
        timestamp = _parse_dt(getattr(record, "check_in_time", None)) or _parse_dt(getattr(record, "check_out_time", None))
        activity_source.append(
            {
                "record": record,
                "person_name": getattr(person, "name", "Unknown Member") if person else "Unknown Member",
                "status": (getattr(record, "status", "") or "pending").replace("_", " ").title(),
                "method_label": _normalize_method_label(getattr(record, "method", "")),
                "timestamp": timestamp,
                "timestamp_label": timestamp.strftime("%I:%M %p") if timestamp else "—",
            }
        )
    recent_activity = sorted(activity_source, key=lambda item: item["timestamp"] or datetime.min.replace(tzinfo=timezone.get_current_timezone()), reverse=True)[:6]

    completed_methods = {
        _normalize_method_label(getattr(record, "method", ""))
        for record in filtered_records
        if getattr(record, "method", None)
    }

    checkin_minutes = []
    for record in filtered_records:
        check_in_dt = _parse_dt(getattr(record, "check_in_time", None))
        if check_in_dt:
            checkin_minutes.append(check_in_dt.hour * 60 + check_in_dt.minute)

    if checkin_minutes:
        average_minutes = round(sum(checkin_minutes) / len(checkin_minutes))
        avg_hour = average_minutes // 60
        avg_minute = average_minutes % 60
        avg_meridiem = "AM" if avg_hour < 12 else "PM"
        avg_hour_display = avg_hour % 12 or 12
        average_checkin_label = f"{avg_hour_display:02d}:{avg_minute:02d} {avg_meridiem}"
    else:
        average_checkin_label = "—"

    status_tabs = [
        {"key": "all", "label": "All", "count": total_count},
        {"key": "present", "label": "Present", "count": present_count},
        {"key": "absent", "label": "Absent", "count": absent_count},
        {"key": "pending", "label": "Pending", "count": pending_count},
    ]

    method_options = [
        ("", "All methods"),
        ("manual", "Manual"),
        ("qr", "QR Code"),
        ("nfc", "NFC"),
        ("swipe", "Swipe"),
        ("face_recognition", "Face Recognition"),
        ("geotracking", "GeoTracking"),
        ("mobile_self", "Mobile Self Check-in"),
    ]

    return render(
        request,
        "user_attendance_list.html",
        {
            "attendance_page": page_obj,
            "search_query": search_query,
            "date_filter": date_filter,
            "event_filter": event_filter,
            "session_filter": session_filter,
            "status_filter": status_filter,
            "method_filter": method_filter,
            "page_size": page_size,
            "events": events,
            "sessions": sessions,
            "active_session": active_session,
            "present_count": present_count,
            "absent_count": absent_count,
            "pending_count": pending_count,
            "total_count": total_count,
            "recent_activity": recent_activity,
            "methods_used_count": len(completed_methods),
            "average_checkin_label": average_checkin_label,
            "status_tabs": status_tabs,
            "method_options": method_options,
        },
    )


@desktop_login_required
def capture_and_recognize(request):
    # Your system uses the stream endpoints now; keep this route but redirect somewhere useful.
    if not _desktop_plan_allows(request, "allow_face_recognition"):
        return _render_plan_locked(
            request,
            "Facial recognition is not included",
            "Upgrade to Professional or above to use camera-based attendance.",
            "Professional",
            back_url="getting_started",
        )
    return redirect("camera_config_list")


# =========================================================
# Camera configuration UI (MATCH your urls.py)
# =========================================================
@desktop_login_required
def camera_config_create(request):
    if not _desktop_plan_allows(request, "allow_face_recognition"):
        return _render_plan_locked(
            request,
            "Camera setup is not included",
            "Camera setup is used for facial recognition and is available on Professional plans and above.",
            "Professional",
            back_url="getting_started",
        )
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
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not save camera configuration: {exc}")
            return render(request, "camera_config_form.html", {"config": to_namespace({"name": name, "camera_source": camera_source, "threshold": threshold})})

    return render(request, "camera_config_form.html")


@desktop_login_required
def camera_config_list(request):
    if not _desktop_plan_allows(request, "allow_face_recognition"):
        return _render_plan_locked(
            request,
            "Camera setup is not included",
            "Upgrade to Professional or above to manage local camera sources for facial recognition.",
            "Professional",
            back_url="getting_started",
        )
    try:
        configs = get_all_cameras(request=request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        configs = []
        messages.error(request, f"Could not load cameras: {_friendly_api_error(exc)}")
    active_count = 0
    inactive_count = 0
    normalized_configs = []
    for config in configs:
        source_value = str(getattr(config, "camera_source", "") or "").strip()
        if hasattr(config, "is_active"):
            is_active = bool(getattr(config, "is_active"))
        else:
            is_active = bool(source_value)
        if is_active:
            active_count += 1
        else:
            inactive_count += 1

        if source_value.isdigit():
            source_kind = "Local Webcam"
        elif source_value.lower().startswith("rtsp://"):
            source_kind = "IP Camera (RTSP)"
        elif source_value.lower().startswith("http://") or source_value.lower().startswith("https://"):
            source_kind = "IP Camera (HTTP)"
        else:
            source_kind = "Custom Source"

        setattr(config, "source_kind", source_kind)
        normalized_configs.append(config)

    return render(
        request,
        "camera_config_list.html",
        {
            "configs": normalized_configs,
            "camera_stats": {
                "active": active_count,
                "inactive": inactive_count,
                "total": len(normalized_configs),
            },
        },
    )


@desktop_login_required
def camera_config_update(request, pk: int):
    if not _desktop_plan_allows(request, "allow_face_recognition"):
        return _render_plan_locked(
            request,
            "Camera setup is not included",
            "Upgrade to Professional or above to manage camera settings.",
            "Professional",
            back_url="getting_started",
        )
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
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not update camera configuration: {exc}")
            return render(
                request,
                "camera_config_form.html",
                {"config": to_namespace({"id": pk, "name": request.POST.get("name"), "camera_source": request.POST.get("camera_source", ""), "threshold": request.POST.get("threshold", "0.6")})},
            )

    try:
        config = get_camera(pk, request=request)
    except AttendanceApiError as exc:
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
        messages.error(request, f"Could not load camera configuration: {_friendly_api_error(exc)}")
        return redirect("camera_config_list")
    return render(request, "camera_config_form.html", {"config": config})


@desktop_login_required
def camera_config_delete(request, pk: int):
    if not _desktop_plan_allows(request, "allow_face_recognition"):
        return _render_plan_locked(
            request,
            "Camera setup is not included",
            "Upgrade to Professional or above to manage camera settings.",
            "Professional",
            back_url="getting_started",
        )
    if request.method == "POST":
        try:
            _api_client(request).delete_camera(pk)
            messages.success(request, "Camera configuration deleted.")
        except AttendanceApiError as exc:
            auth_response = _redirect_if_auth_error(request, exc)
            if auth_response:
                return auth_response
            messages.error(request, f"Could not delete camera configuration: {exc}")
    return redirect("camera_config_list")

@desktop_login_required
def api_attendance_monitor(request):
    try:
        params = {}
        attendance_session_id = request.GET.get("attendance_session_id") or _active_session_id(request)
        if attendance_session_id:
            params["attendance_session_id"] = attendance_session_id
        return JsonResponse(_api_client(request).monitor(**params))
    except AttendanceApiError as exc:
        if _is_auth_api_error(exc):
            return _expire_desktop_session(request)
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
        method_filter = (request.GET.get("method") or "").strip().lower()
        face_only = (request.GET.get("face_only") or "").strip().lower() in {"1", "true", "yes"}
        if since:
            params["since"] = since
        attendance_session_id = request.GET.get("attendance_session_id") or _active_session_id(request)
        if attendance_session_id:
            params["attendance_session_id"] = attendance_session_id
        if method_filter:
            params["method"] = method_filter
        payload = _api_client(request).today(**params)
        if face_only and isinstance(payload, dict):
            allowed_methods = {"face", "facial_recognition"}
            items = payload.get("items", []) or []
            items = [
                item
                for item in items
                if str(item.get("method") or item.get("source") or "").strip().lower() in allowed_methods
            ]
            payload["items"] = items
            payload["count"] = len(items)
            if items:
                latest_item = max(
                    items,
                    key=lambda item: item.get("epoch_ms") or item.get("id") or 0,
                )
                payload["latest"] = latest_item.get("time") or payload.get("latest")
                payload["latest_epoch_ms"] = latest_item.get("epoch_ms") or payload.get("latest_epoch_ms")
            else:
                payload["latest"] = None
                payload["latest_epoch_ms"] = None
        return JsonResponse(payload)
    except AttendanceApiError as exc:
        if _is_auth_api_error(exc):
            return _expire_desktop_session(request)
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
        auth_response = _redirect_if_auth_error(request, exc)
        if auth_response:
            return auth_response
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


def _attendance_status_matches(record, selected_status: str = "") -> bool:
    selected = (selected_status or "").strip().lower()
    if not selected or selected == "all":
        return True
    return (getattr(record, "status", "") or "").strip().lower() == selected


ATTENDANCE_EXPORT_HEADERS = [
    "Name",
    "Member Code",
    "Event",
    "Session",
    "Status",
    "Method",
    "Date",
    "Check-In",
    "Check-Out",
    "Duration",
    "Geo Branch",
    "Geo Latitude",
    "Geo Longitude",
    "Geo Accuracy (m)",
    "Geo Distance (m)",
    "Geo Radius (m)",
]


def _attendance_export_row(a):
    person = getattr(a, "person", None)
    return [
        getattr(person, "name", "") if person else "",
        (getattr(person, "member_code", "") or getattr(person, "portal_id", "")) if person else "",
        getattr(a, "event_name", "") or "",
        getattr(a, "session_name", "") or "",
        (getattr(a, "status", "") or "").replace("_", " ").title(),
        (getattr(a, "method", "") or "").replace("_", " ").title(),
        str(getattr(a, "date", "") or ""),
        str(getattr(a, "check_in_time", "") or ""),
        str(getattr(a, "check_out_time", "") or ""),
        _duration_text(a),
        getattr(getattr(a, "geotracking_branch", None), "name", "") or getattr(a, "geotracking_branch_name", "") or "",
        str(getattr(a, "geotracking_latitude", "") or ""),
        str(getattr(a, "geotracking_longitude", "") or ""),
        str(getattr(a, "geotracking_accuracy_meters", "") or ""),
        str(getattr(a, "geotracking_distance_meters", "") or ""),
        str(getattr(a, "geotracking_radius_meters", "") or ""),
    ]


def _report_filter_summary(
    search_query="",
    date_filter="",
    event_filter="",
    session_filter="",
    date_from="",
    date_to="",
    branch_filter="",
    method_filter="",
):
    return (
        f"Search: {search_query or '-'} | "
        f"Date: {date_filter or '-'} | "
        f"Range: {date_from or '-'} to {date_to or '-'} | "
        f"Event ID: {event_filter or '-'} | "
        f"Session ID: {session_filter or '-'} | "
        f"Branch ID: {branch_filter or '-'} | "
        f"Method: {method_filter or '-'}"
    )


def _write_pdf_report(qs, title, filter_summary):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    width, height = landscape(A4)
    headers = ["Name", "Event", "Session", "Status", "Method", "Date", "In", "Out", "Duration"]
    x_positions = [32, 160, 260, 365, 425, 490, 555, 625, 695]

    def draw_header(y_pos):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(32, y_pos, title)
        y_pos -= 22
        c.setFont("Helvetica", 9)
        c.drawString(32, y_pos, filter_summary[:145])
        y_pos -= 18
        c.setFont("Helvetica-Bold", 8)
        for x, h in zip(x_positions, headers):
            c.drawString(x, y_pos, h)
        y_pos -= 12
        c.setFont("Helvetica", 8)
        return y_pos

    y = draw_header(height - 36)

    for a in qs[:2000]:  # safety cap
        if y < 36:
            c.showPage()
            y = draw_header(height - 36)

        row = _attendance_export_row(a)
        pdf_row = [row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]]
        for x, val in zip(x_positions, pdf_row):
            c.drawString(x, y, (str(val or ""))[:18])
        y -= 11

    c.save()
    buf.seek(0)
    return buf.getvalue()


@desktop_login_required
def attendance_export_download(request):
    search_query = request.GET.get("search", "").strip()
    date_filter = request.GET.get("attendance_date", "").strip()
    date_from = request.GET.get("date_from", "").strip()
    date_to = request.GET.get("date_to", "").strip()
    event_filter = request.GET.get("event_id", "").strip()
    session_filter = request.GET.get("attendance_session_id", "").strip()
    branch_filter = request.GET.get("branch_id", "").strip()
    method_filter = request.GET.get("method", "").strip()
    status_filter = request.GET.get("status", "").strip()
    fmt = (request.GET.get("format") or "csv").lower().strip()
    qs = fetch_all_attendance_logs(
        search_query,
        date_filter,
        event_id=event_filter,
        attendance_session_id=session_filter,
        date_from=date_from,
        date_to=date_to,
        branch_id=branch_filter,
        method=method_filter,
        request=request,
    )
    if status_filter:
        qs = [record for record in qs if _attendance_status_matches(record, status_filter)]

    # filename
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "attendance"
    if date_filter:
        label += f"_{date_filter}"
    elif date_from or date_to:
        label += f"_{date_from or 'start'}_to_{date_to or 'today'}"
    filename_base = f"{label}_{stamp}"

    if fmt == "csv":
        resp = HttpResponse(content_type="text/csv; charset=utf-8")
        resp["Content-Disposition"] = f'attachment; filename="{filename_base}.csv"'
        w = csv.writer(resp)
        w.writerow(ATTENDANCE_EXPORT_HEADERS)

        for a in qs:
            w.writerow(_attendance_export_row(a))
        return resp

    if fmt == "xlsx":
        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance"
        ws.append(ATTENDANCE_EXPORT_HEADERS)

        for a in qs:
            ws.append(_attendance_export_row(a))

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
        title = "Attendance Report"
        if date_filter:
            title += f" ({date_filter})"
        elif date_from or date_to:
            title += f" ({date_from or 'start'} to {date_to or 'today'})"
        resp = HttpResponse(
            _write_pdf_report(
                qs,
                title,
                _report_filter_summary(
                    search_query,
                    date_filter,
                    event_filter,
                    session_filter,
                    date_from,
                    date_to,
                    branch_filter,
                    method_filter,
                ),
            ),
            content_type="application/pdf",
        )
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
    date_from = (request.POST.get("date_from") or "").strip()
    date_to = (request.POST.get("date_to") or "").strip()
    event_filter = (request.POST.get("event_id") or "").strip()
    session_filter = (request.POST.get("attendance_session_id") or "").strip()
    branch_filter = (request.POST.get("branch_id") or "").strip()
    method_filter = (request.POST.get("method") or "").strip()
    status_filter = (request.POST.get("status") or "").strip()

    if not to_email:
        messages.error(request, "Recipient email is required.")
        return redirect("person_attendance_list")

    qs = fetch_all_attendance_logs(
        search_query,
        date_filter,
        event_id=event_filter,
        attendance_session_id=session_filter,
        date_from=date_from,
        date_to=date_to,
        branch_id=branch_filter,
        method=method_filter,
        request=request,
    )
    if status_filter:
        qs = [record for record in qs if _attendance_status_matches(record, status_filter)]

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
        w.writerow(ATTENDANCE_EXPORT_HEADERS)

        for a in qs:
            w.writerow(_attendance_export_row(a))

        attachment_name = f"{filename_base}.csv"
        attachment_bytes = sio.getvalue().encode("utf-8")
        mime = "text/csv"

    elif fmt == "xlsx":
        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance"
        ws.append(ATTENDANCE_EXPORT_HEADERS)

        for a in qs:
            ws.append(_attendance_export_row(a))

        out = io.BytesIO()
        wb.save(out)
        out.seek(0)

        attachment_name = f"{filename_base}.xlsx"
        attachment_bytes = out.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    elif fmt == "pdf":
        title = "Attendance Report"
        if date_filter:
            title += f" ({date_filter})"

        attachment_name = f"{filename_base}.pdf"
        attachment_bytes = _write_pdf_report(
            qs,
            title,
            _report_filter_summary(search_query, date_filter, event_filter, session_filter),
        )
        mime = "application/pdf"

    else:
        messages.error(request, "Invalid export format.")
        return redirect("person_attendance_list")

    subject = "KairosTrack Attendance Report"
    body = (
        "Hello,\n\n"
        "Please find attached the attendance report.\n\n"
        f"Filters:\n"
        f"- Search: {search_query or '-'}\n"
        f"- Date: {date_filter or '-'}\n"
        f"- Range: {(date_from or '-') + ' to ' + (date_to or '-') if (date_from or date_to) else '-'}\n"
        f"- Event ID: {event_filter or '-'}\n"
        f"- Session ID: {session_filter or '-'}\n\n"
        "Regards,\nKairosTrack"
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
