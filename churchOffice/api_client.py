from __future__ import annotations

from typing import Any

import requests
from django.conf import settings


class AttendanceApiError(RuntimeError):
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

    def authenticate(self, username: str, password: str):
        response = requests.post(
            self._auth_url("api/token/"),
            json={"username": username, "password": password},
            headers={"Accept": "application/json"},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            raise AttendanceApiError(f"{response.status_code} calling {response.url}: {payload}")
        payload = response.json()
        self._update_tokens(payload)
        return payload

    def refresh_access_token(self):
        if not self.refresh_token:
            raise AttendanceApiError("No refresh token available for the desktop session.")
        response = requests.post(
            self._auth_url("api/token/refresh/"),
            json={"refresh": self.refresh_token},
            headers={"Accept": "application/json"},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            raise AttendanceApiError(f"{response.status_code} calling {response.url}: {payload}")
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

        response = requests.request(
            method=method,
            url=url,
            headers=merged_headers,
            timeout=self.timeout,
            verify=self.verify_ssl,
            **kwargs,
        )

        if response.status_code == 401 and _retry_on_auth_failure and self.refresh_token:
            try:
                self.refresh_access_token()
            except AttendanceApiError:
                pass
            else:
                return self._request(method, path, _retry_on_auth_failure=False, headers=headers, **kwargs)

        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = response.text
            raise AttendanceApiError(f"{response.status_code} calling {url}: {payload}")

        if response.status_code == 204 or not response.content:
            return None

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()
        return response.content

    def overview(self):
        return self._request("GET", "")

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

    def get_attendance_log(self, attendance_id: int):
        return self._request("GET", f"logs/{attendance_id}/")

    def delete_attendance_log(self, attendance_id: int):
        return self._request("DELETE", f"logs/{attendance_id}/delete/")

    def face_check_in(self, payload: dict[str, Any]):
        return self._request("POST", "check-in/face/", json=payload)

    def nfc_check_in(self, payload: dict[str, Any]):
        return self._request("POST", "nfc/check-in/", json=payload)

    def today(self, **params):
        return self._request("GET", "today/", params=params)

    def monitor(self):
        return self._request("GET", "monitor/")

    def public_member_check(self, payload: dict[str, Any]):
        return self._request("POST", "public/member-check/", json=payload)

    def public_self_register(self, payload: dict[str, Any]):
        return self._request("POST", "public/self-register/", json=payload)
