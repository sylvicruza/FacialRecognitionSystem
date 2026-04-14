# Time and Attendance Desktop Client

This project should now be treated as the local desktop client for the Time and Attendance system.

## Why the split is required

The hosted Render app at `https://heavensconnect.onrender.com` cannot open private LAN camera addresses such as:

- `rtsp://admin:Ofueu1000@192.168.1.119:554/h264Preview_01_sub`
- `http://192.168.1.20:4747/video`

Those camera sources only work from a machine on the same local network. That means camera access, face recognition, and the attendance operator UI must run locally on the church PC or laptop.

## New responsibility of this repo

`DjangoProject1` should own:

- local camera access
- face recognition
- operator-facing pages
- desktop packaging and distribution
- API calls to `WelfareAPI` for CRUD and attendance log sync

## API configuration

Set these environment variables on the desktop machine:

- `ATTENDANCE_API_BASE_URL=https://heavensconnect.onrender.com/api/attendance`
- `ATTENDANCE_API_TOKEN=<jwt access token if the API is protected>`
- `ATTENDANCE_API_TIMEOUT=30`
- `ATTENDANCE_API_VERIFY_SSL=True`

The reusable client lives in `churchOffice/api_client.py`.

## Desktop entry points

- `python run_attendance_client.py`
- `start_attendance_client.bat`
- `pyinstaller TimeAndAttendance.spec`

## Recommended next refactor

The next code step is to update the existing `churchOffice/views.py` flows so member, camera, and attendance CRUD pages use `AttendanceApiClient` instead of reading the local SQLite database directly. Camera streaming should remain local in this repo.

That refactor is now started:

- member CRUD pages call the hosted attendance API
- camera CRUD pages call the hosted attendance API
- attendance logs and deletes call the hosted attendance API
- face recognition and NFC check-ins write back through the hosted attendance API
- live camera streaming still runs locally on the desktop machine
