from __future__ import annotations

from typing import Any

import requests
from django.conf import settings
from requests import RequestException


class AttendanceApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class AttendanceAuthError(AttendanceApiError):
    pass


class AttendanceApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        refresh_token: str | None = None,
        timeout: int | None = None,
        verify_ssl: bool | None = None,
        auth_base_url: str | None = None,
        token_updater=None,
    ):
        self.base_url = (base_url or settings.ATTENDANCE_API_BASE_URL).rstrip("/")
        self.token = token if token is not None else settings.ATTENDANCE_API_TOKEN
        self.refresh_token = refresh_token
        self.timeout = timeout if timeout is not None else settings.ATTENDANCE_API_TIMEOUT
        self.verify_ssl = verify_ssl if verify_ssl is not None else settings.ATTENDANCE_API_VERIFY_SSL
        self.auth_base_url = (auth_base_url or self._derive_auth_base_url()).rstrip("/")
        self.token_updater = token_updater

    def _derive_auth_base_url(self) -> str:
        marker = "/api/attendance"
        if marker in self.base_url:
            return self.base_url.split(marker, 1)[0]
        return self.base_url

    def _auth_url(self, path: str) -> str:
        return f"{self.auth_base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _update_tokens(self, payload: dict[str, Any]):
        access = payload.get("access")
        refresh = payload.get("refresh")
        if access:
            self.token = access
        if refresh:
            self.refresh_token = refresh
        if self.token_updater and (access or refresh):
            self.token_updater(self.token, self.refresh_token)

    def _network_error(self, exc: RequestException, url: str) -> AttendanceApiError:
        return AttendanceApiError(
            f"Could not reach the attendance backend at {url}. Check your internet connection or backend URL.",
            payload=str(exc),
        )

    def authenticate(self, username: str, password: str):
        url = self._auth_url("api/token/")
        try:
            response = requests.post(
                url,
                json={"username": username, "password": password},
                headers={"Accept": "application/json"},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except RequestException as exc:
            raise self._network_error(exc, url) from exc
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            raise AttendanceApiError(
                f"{response.status_code} calling {response.url}: {payload}",
                status_code=response.status_code,
                payload=payload,
            )
        payload = response.json()
        self._update_tokens(payload)
        return payload

    def refresh_access_token(self):
        if not self.refresh_token:
            raise AttendanceApiError("No refresh token available for the desktop session.")
        url = self._auth_url("api/token/refresh/")
        try:
            response = requests.post(
                url,
                json={"refresh": self.refresh_token},
                headers={"Accept": "application/json"},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except RequestException as exc:
            raise self._network_error(exc, url) from exc
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            error_class = AttendanceAuthError if response.status_code == 401 else AttendanceApiError
            raise error_class(
                f"{response.status_code} calling {response.url}: {payload}",
                status_code=response.status_code,
                payload=payload,
            )
        payload = response.json()
        if "refresh" not in payload:
            payload["refresh"] = self.refresh_token
        self._update_tokens(payload)
        return payload

    def _request(self, method: str, path: str, _retry_on_auth_failure: bool = True, **kwargs):
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        merged_headers = self._headers()
        merged_headers.update(headers)

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=merged_headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                **kwargs,
            )
        except RequestException as exc:
            raise self._network_error(exc, url) from exc

        if response.status_code == 401 and _retry_on_auth_failure and self.refresh_token:
            try:
                self.refresh_access_token()
            except AttendanceApiError:
                if self.token_updater:
                    self.token_updater(None, None)
                pass
            else:
                return self._request(method, path, _retry_on_auth_failure=False, headers=headers, **kwargs)

        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            error_class = AttendanceAuthError if response.status_code == 401 else AttendanceApiError
            raise error_class(
                f"{response.status_code} calling {url}: {payload}",
                status_code=response.status_code,
                payload=payload,
            )

        if response.status_code == 204 or not response.content:
            return None

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()
        return response.content

    def overview(self):
        return self._request("GET", "")

    def me(self):
        return self._request("GET", "me/")

    def signup_organization(self, payload: dict[str, Any]):
        return self._request("POST", "onboarding/signup/", json=payload)

    def request_upgrade(self, payload: dict[str, Any]):
        return self._request("POST", "billing/upgrade-request/", json=payload)

    def list_plans(self):
        return self._request("GET", "plans/")

    def list_organizations(self, **params):
        return self._request("GET", "organizations/", params=params)

    def create_organization(self, payload: dict[str, Any]):
        return self._request("POST", "organizations/", json=payload)

    def get_organization(self, organization_id: int):
        return self._request("GET", f"organizations/{organization_id}/")

    def update_organization(self, organization_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"organizations/{organization_id}/", json=payload)

    def delete_organization(self, organization_id: int):
        return self._request("DELETE", f"organizations/{organization_id}/")

    def list_staff(self, **params):
        return self._request("GET", "staff/", params=params)

    def create_staff(self, payload: dict[str, Any]):
        return self._request("POST", "staff/", json=payload)

    def get_staff(self, staff_id: int):
        return self._request("GET", f"staff/{staff_id}/")

    def update_staff(self, staff_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"staff/{staff_id}/", json=payload)

    def delete_staff(self, staff_id: int):
        return self._request("DELETE", f"staff/{staff_id}/")

    def list_branches(self, **params):
        return self._request("GET", "branches/", params=params)

    def create_branch(self, payload: dict[str, Any]):
        return self._request("POST", "branches/", json=payload)

    def get_branch(self, branch_id: int):
        return self._request("GET", f"branches/{branch_id}/")

    def update_branch(self, branch_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"branches/{branch_id}/", json=payload)

    def delete_branch(self, branch_id: int):
        return self._request("DELETE", f"branches/{branch_id}/")

    def list_events(self, **params):
        return self._request("GET", "events/", params=params)

    def create_event(self, payload: dict[str, Any]):
        return self._request("POST", "events/", json=payload)

    def get_event(self, event_id: int):
        return self._request("GET", f"events/{event_id}/")

    def update_event(self, event_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"events/{event_id}/", json=payload)

    def delete_event(self, event_id: int):
        return self._request("DELETE", f"events/{event_id}/")

    def list_sessions(self, **params):
        return self._request("GET", "sessions/", params=params)

    def create_session(self, payload: dict[str, Any]):
        return self._request("POST", "sessions/", json=payload)

    def get_session(self, session_id: int):
        return self._request("GET", f"sessions/{session_id}/")

    def update_session(self, session_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"sessions/{session_id}/", json=payload)

    def delete_session(self, session_id: int):
        return self._request("DELETE", f"sessions/{session_id}/")

    def list_devices(self, **params):
        return self._request("GET", "devices/", params=params)

    def create_device(self, payload: dict[str, Any]):
        return self._request("POST", "devices/", json=payload)

    def get_device(self, device_id: int):
        return self._request("GET", f"devices/{device_id}/")

    def update_device(self, device_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"devices/{device_id}/", json=payload)

    def delete_device(self, device_id: int):
        return self._request("DELETE", f"devices/{device_id}/")

    def list_persons(self, **params):
        return self._request("GET", "persons/", params=params)

    def create_person(self, payload: dict[str, Any]):
        return self._request("POST", "persons/", json=payload)

    def get_person(self, person_id: int):
        return self._request("GET", f"persons/{person_id}/")

    def update_person(self, person_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"persons/{person_id}/", json=payload)

    def authorize_person(self, person_id: int, authorized: bool):
        return self._request("PATCH", f"persons/{person_id}/authorize/", json={"authorized": authorized})

    def delete_person(self, person_id: int):
        return self._request("DELETE", f"persons/{person_id}/")

    def list_cameras(self):
        return self._request("GET", "cameras/")

    def create_camera(self, payload: dict[str, Any]):
        return self._request("POST", "cameras/", json=payload)

    def get_camera(self, camera_id: int):
        return self._request("GET", f"cameras/{camera_id}/")

    def update_camera(self, camera_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"cameras/{camera_id}/", json=payload)

    def delete_camera(self, camera_id: int):
        return self._request("DELETE", f"cameras/{camera_id}/")

    def list_attendance_logs(self, **params):
        return self._request("GET", "logs/", params=params)

    def create_attendance_log(self, payload: dict[str, Any]):
        return self._request("POST", "logs/", json=payload)

    def get_attendance_log(self, attendance_id: int):
        return self._request("GET", f"logs/{attendance_id}/")

    def update_attendance_log(self, attendance_id: int, payload: dict[str, Any]):
        return self._request("PATCH", f"logs/{attendance_id}/", json=payload)

    def delete_attendance_log(self, attendance_id: int):
        return self._request("DELETE", f"logs/{attendance_id}/delete/")

    def report_summary(self, **params):
        return self._request("GET", "reports/summary/", params=params)

    def session_report(self, session_id: int, **params):
        return self._request("GET", f"reports/sessions/{session_id}/", params=params)

    def member_report(self, person_id: int, **params):
        return self._request("GET", f"reports/members/{person_id}/", params=params)

    def face_check_in(self, payload: dict[str, Any]):
        return self._request("POST", "check-in/face/", json=payload)

    def nfc_check_in(self, payload: dict[str, Any]):
        return self._request("POST", "nfc/check-in/", json=payload)

    def geotracking_check_in(self, payload: dict[str, Any]):
        return self._request("POST", "geotracking/check-in/", json=payload)

    def today(self, **params):
        return self._request("GET", "today/", params=params)

    def monitor(self, **params):
        return self._request("GET", "monitor/", params=params)

    def public_member_check(self, payload: dict[str, Any]):
        return self._request("POST", "public/member-check/", json=payload)

    def public_self_register(self, payload: dict[str, Any]):
        return self._request("POST", "public/self-register/", json=payload)
