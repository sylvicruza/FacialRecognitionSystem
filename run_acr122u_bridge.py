import json
import os
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlencode


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ohc_time_attendance.settings")

import django

django.setup()

from django.conf import settings

from churchOffice.api_client import AttendanceApiClient, AttendanceApiError

try:
    from smartcard.Exceptions import CardConnectionException, NoCardException
    from smartcard.System import readers
except Exception as exc:  # pragma: no cover - dependency bootstrap path
    print("The ACR122U bridge requires pyscard. Install it with:")
    print(r"  .\.venv\Scripts\python.exe -m pip install pyscard")
    print(f"Import error: {exc}")
    sys.exit(1)


UID_APDU = [0xFF, 0xCA, 0x00, 0x00, 0x00]
READ_INTERVAL_SECONDS = 0.35
UID_COOLDOWN_SECONDS = 2.0


def friendly_api_error(exc: Exception) -> str:
    text = str(exc)
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
                return "No NFC tag UID was received. Tap the card again."
            return detail
    return text


def runtime_dir_candidates():
    candidates = []
    seen = set()
    for path in [
        settings.RUNTIME_DIR,
        getattr(settings, "BASE_DIR", None),
        Path(os.getenv("LOCALAPPDATA", "")) / "KairosTrack" if os.getenv("LOCALAPPDATA") else None,
    ]:
        if not path:
            continue
        resolved = Path(path)
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(resolved)
    return candidates


def resolve_runtime_paths():
    for directory in runtime_dir_candidates():
        config = directory / "nfc_reader_bridge_config.json"
        runtime = directory / "nfc_reader_bridge_runtime.json"
        if config.exists():
            return directory, config, runtime
    directory = runtime_dir_candidates()[0]
    return directory, directory / "nfc_reader_bridge_config.json", directory / "nfc_reader_bridge_runtime.json"


RUNTIME_DIR, CONFIG_PATH, RUNTIME_PATH = resolve_runtime_paths()
PID_PATH = RUNTIME_DIR / "nfc_reader_bridge.pid"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config() -> dict:
    return _load_json(CONFIG_PATH)


def write_runtime(**updates):
    payload = {
        "status": "Waiting",
        "last_uid": "",
        "last_scan_at": "",
        "last_error": "",
        "last_action": "Bridge started. Waiting for reader activity.",
        "pending_enrollment_uid": "",
        "pending_enrollment_url": "",
        **_load_json(RUNTIME_PATH),
        **updates,
    }
    RUNTIME_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_pid():
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")


def clear_pid():
    try:
        PID_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def persist_tokens(access: str | None, refresh: str | None):
    config = load_config()
    if access is not None:
        config["access_token"] = access
    if refresh is not None:
        config["refresh_token"] = refresh
    config["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_client(config: dict) -> AttendanceApiClient:
    return AttendanceApiClient(
        base_url=(config.get("backend_api_base_url") or settings.ATTENDANCE_API_BASE_URL).rstrip("/"),
        token=config.get("access_token") or "",
        refresh_token=config.get("refresh_token") or "",
        token_updater=persist_tokens,
    )


def reader_candidates():
    configured = (load_config().get("reader_name") or "").strip().lower()
    result = []
    for reader in readers():
        name = str(reader)
        if configured and configured in name.lower():
            result.insert(0, reader)
        else:
            result.append(reader)
    return result


def read_uid_once():
    available = reader_candidates()
    if not available:
        write_runtime(
            status="Error",
            last_error="No smart card reader found.",
            last_action="Connect the ACR122U reader and keep this bridge running.",
        )
        return None
    for reader in available:
        try:
            connection = reader.createConnection()
            connection.connect()
            data, sw1, sw2 = connection.transmit(UID_APDU)
            if (sw1, sw2) != (0x90, 0x00) or not data:
                continue
            return "".join(f"{byte:02X}" for byte in data)
        except (NoCardException, CardConnectionException):
            continue
        except Exception as exc:
            write_runtime(
                status="Error",
                last_error=str(exc),
                last_action=f"Reader communication failed on {reader}.",
            )
    return None


def handle_uid(uid: str, last_opened_uid: str):
    config = load_config()
    capture_mode = str(config.get("capture_mode") or "check_in").strip().lower()
    station_identifier = (config.get("station_identifier") or "").strip()
    reader_name = (config.get("reader_name") or "ACR122U Reader").strip()
    active_session_id = config.get("active_session_id")
    enrollment_base_url = config.get("enrollment_base_url") or ""
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    if capture_mode == "monitor":
        write_runtime(
            status="Connected",
            last_uid=uid,
            last_scan_at=now_iso,
            last_error="",
            last_action="UID captured in monitor mode.",
            pending_enrollment_uid="",
            pending_enrollment_url="",
        )
        return last_opened_uid

    if capture_mode == "enroll":
        enrollment_url = f"{enrollment_base_url}?{urlencode({'prefill_uid': uid})}" if enrollment_base_url else ""
        write_runtime(
            status="Connected",
            last_uid=uid,
            last_scan_at=now_iso,
            last_error="",
            last_action="UID captured and enrollment handoff is ready.",
            pending_enrollment_uid=uid,
            pending_enrollment_url=enrollment_url,
        )
        if enrollment_url and uid != last_opened_uid:
            webbrowser.open(enrollment_url)
            return uid
        return last_opened_uid

    if not active_session_id:
        write_runtime(
            status="Error",
            last_uid=uid,
            last_scan_at=now_iso,
            last_error="No active session selected.",
            last_action="Open KairosTrack and choose an active attendance session first.",
        )
        return last_opened_uid

    client = build_client(config)
    try:
        outcome = client.nfc_station_check_in(
            {
                "uid": uid,
                "attendance_session_id": int(active_session_id),
                "station_identifier": station_identifier,
                "device_label": reader_name,
            }
        )
        attendance = outcome.get("attendance") if isinstance(outcome, dict) else {}
        person_name = ""
        if isinstance(attendance, dict):
            person_name = str(attendance.get("person_name") or attendance.get("name") or "").strip()
        write_runtime(
            status="Connected",
            last_uid=uid,
            last_scan_at=now_iso,
            last_error="",
            last_action=f"Checked in {person_name or 'member'} with NFC.",
            pending_enrollment_uid="",
            pending_enrollment_url="",
        )
    except AttendanceApiError as exc:
        write_runtime(
            status="Error",
            last_uid=uid,
            last_scan_at=now_iso,
            last_error=friendly_api_error(exc),
            last_action="The bridge could not complete the NFC check-in.",
        )
    return last_opened_uid


def main():
    if not CONFIG_PATH.exists():
        print("No reader bridge config found yet.")
        print("Open KairosTrack desktop, visit /members/nfc-reader/, save the reader setup once, then start this bridge again.")
        print("Checked these locations:")
        for directory in runtime_dir_candidates():
            print(f"  - {directory / 'nfc_reader_bridge_config.json'}")
        return

    print("KairosTrack ACR122U bridge starting...")
    print(f"Runtime directory: {RUNTIME_DIR}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Runtime: {RUNTIME_PATH}")
    write_pid()
    write_runtime(
        status="Waiting",
        last_error="",
        last_action="Bridge started. Waiting for reader activity.",
    )

    last_uid = ""
    last_uid_at = 0.0
    last_opened_uid = ""

    while True:
        uid = read_uid_once()
        if uid:
            now = time.time()
            if uid != last_uid or (now - last_uid_at) >= UID_COOLDOWN_SECONDS:
                last_opened_uid = handle_uid(uid, last_opened_uid)
                last_uid = uid
                last_uid_at = now
        time.sleep(READ_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        write_runtime(
            status="Waiting",
            last_action="Bridge stopped.",
        )
        print("Bridge stopped.")
    finally:
        clear_pid()
