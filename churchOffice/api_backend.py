from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from django.conf import settings

from .api_client import AttendanceApiClient

ACCESS_TOKEN_SESSION_KEY = "attendance_api_access_token"
REFRESH_TOKEN_SESSION_KEY = "attendance_api_refresh_token"


def get_client(request=None, token_updater=None) -> AttendanceApiClient:
    token = settings.ATTENDANCE_API_TOKEN
    refresh_token = None
    if request is not None:
        token = request.session.get(ACCESS_TOKEN_SESSION_KEY) or token
        refresh_token = request.session.get(REFRESH_TOKEN_SESSION_KEY)
    return AttendanceApiClient(token=token, refresh_token=refresh_token, token_updater=token_updater)


def to_namespace(value: Any):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def extract_results(payload):
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    return payload


def normalize_person(person: dict[str, Any]):
    image_url = person.get("image_url")
    return to_namespace(
        {
            **person,
            "pk": person["id"],
            "image": {"url": image_url} if image_url else None,
        }
    )


def normalize_camera(camera: dict[str, Any]):
    return to_namespace({**camera, "pk": camera["id"]})


def get_all_people(search: str = "", request=None):
    client = get_client(request=request)
    payload = client.list_persons(search=search) if search else client.list_persons()
    return [normalize_person(person) for person in extract_results(payload)]


def get_people_map(request=None):
    return {person.id: person for person in get_all_people(request=request)}


def get_all_cameras(request=None):
    payload = get_client(request=request).list_cameras()
    return [normalize_camera(camera) for camera in extract_results(payload)]


def get_camera(camera_id: int, request=None):
    return normalize_camera(get_client(request=request).get_camera(camera_id))


def prepare_attendance_records(records: list[dict[str, Any]], request=None):
    people_map = get_people_map(request=request)
    items = []
    for record in records:
        person = people_map.get(record.get("person"))
        camera_name = record.get("camera_name")
        camera_obj = {"id": record.get("camera"), "name": camera_name} if record.get("camera") else None
        items.append(
            to_namespace(
                {
                    **record,
                    "person": person,
                    "camera": camera_obj,
                    "calculate_duration": record.get("duration"),
                }
            )
        )
    return items


def wrap_api_page(response: dict[str, Any], object_list: list[Any], page_number: int, page_size: int):
    count = int(response.get("count", len(object_list)))
    num_pages = max(1, (count + page_size - 1) // page_size)
    paginator = SimpleNamespace(count=count, num_pages=num_pages)

    class ApiPage:
        def __init__(self):
            self.object_list = object_list
            self.number = page_number
            self.paginator = paginator

        def has_previous(self):
            return self.number > 1

        def has_next(self):
            return self.number < paginator.num_pages

        def previous_page_number(self):
            return max(1, self.number - 1)

        def next_page_number(self):
            return min(paginator.num_pages, self.number + 1)

        def start_index(self):
            if count == 0:
                return 0
            return (self.number - 1) * page_size + 1

        def end_index(self):
            if count == 0:
                return 0
            return min(count, self.number * page_size)

    return ApiPage()


def fetch_attendance_page(search_query: str, date_filter: str, page_number: int, page_size: int, request=None):
    response = get_client(request=request).list_attendance_logs(
        search=search_query,
        attendance_date=date_filter,
        page=page_number,
        page_size=page_size,
    )
    records = prepare_attendance_records(response.get("results", []), request=request)
    return wrap_api_page(response, records, page_number, page_size)


def fetch_all_attendance_logs(search_query: str = "", date_filter: str = "", request=None):
    page = 1
    page_size = 200
    records: list[Any] = []
    client = get_client(request=request)

    while True:
        response = client.list_attendance_logs(
            search=search_query,
            attendance_date=date_filter,
            page=page,
            page_size=page_size,
        )
        records.extend(prepare_attendance_records(response.get("results", []), request=request))
        if not response.get("next"):
            break
        page += 1

    return records
