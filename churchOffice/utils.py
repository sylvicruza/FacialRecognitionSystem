import base64
import requests

API_BASE = "https://openedheavenschapel.co.uk/churchcrm/api"
API_KEY = "EtqByhhjSGRS8cv6jxO4rMHvJOadtMlVrDnNDXSv3OfY7eH1m3"


def _headers():
    return {
        "x-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def fetch_user_data(last_name):
    url = f"{API_BASE}/persons/search/{last_name}"
    response = requests.get(url, headers=_headers(), timeout=20)
    if response.status_code == 200:
        return response.json()
    return None


def fetch_user_data_by_id(portal_id):
    url = f"{API_BASE}/person/{portal_id}"
    response = requests.get(url, headers=_headers(), timeout=20)
    if response.status_code == 200:
        return response.json()
    return None


def register_user_on_portal(data: dict):
    url = f"{API_BASE}/public/register/person"

    try:
        response = requests.post(url, headers=_headers(), json=data, timeout=30)

        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}

        if response.status_code in (200, 201):
            return {
                "ok": True,
                "data": payload,
                "status_code": response.status_code,
                "error": None,
            }

        return {
            "ok": False,
            "data": payload,
            "status_code": response.status_code,
            "error": payload.get("message") if isinstance(payload, dict) else str(payload),
        }

    except requests.RequestException as ex:
        return {
            "ok": False,
            "data": None,
            "status_code": 0,
            "error": f"Could not connect to Church Portal: {ex}",
        }


def upload_person_photo_to_portal(person_id, image_data):
    """
    Uploads a person's photo to ChurchCRM private API.

    API expects:
    POST /person/:personId/photo
    payload = {
        "imgBase64": "data:image/jpeg;base64,..."
    }
    """
    if not person_id or not image_data:
        return {
            "ok": False,
            "data": None,
            "status_code": 0,
            "error": "person_id and image_data are required",
        }

    try:
        # Ensure image_data is a full data URL as required by the API
        if "," in image_data:
            header, encoded = image_data.split(",", 1)
            # validate base64 part only
            base64.b64decode(encoded, validate=True)
            full_data_url = image_data
        else:
            # raw base64 was supplied, convert it into the expected data URL format
            base64.b64decode(image_data, validate=True)
            full_data_url = f"data:image/jpeg;base64,{image_data}"

        url = f"{API_BASE}/person/{person_id}/photo"

        payload = {
            "imgBase64": full_data_url
        }

        response = requests.post(
            url,
            headers=_headers(),
            json=payload,
            timeout=30
        )

        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text}

        if response.status_code == 200:
            return {
                "ok": True,
                "data": data,
                "status_code": response.status_code,
                "error": None,
            }

        return {
            "ok": False,
            "data": data,
            "status_code": response.status_code,
            "error": data.get("message") if isinstance(data, dict) else str(data),
        }

    except Exception as ex:
        return {
            "ok": False,
            "data": None,
            "status_code": 0,
            "error": str(ex),
        }