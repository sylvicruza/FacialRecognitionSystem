# KairosTrack Testing Wiki

## Purpose

This guide explains how to test KairosTrack across the hosted backend, desktop app, and Flutter mobile app.

KairosTrack is an event-based attendance platform. Every attendance record should belong to an event session, for example:

`Sylvester checked in for Sunday Service on April 21, 2026.`

## Project Locations

| App | Local Path |
| --- | --- |
| Backend API | `C:\Users\cents\PycharmProjects\WelfareAPI` |
| Desktop App | `C:\Users\cents\PycharmProjects\DjangoProject1` |
| Mobile App | `C:\Users\cents\AndroidStudioProjects\kairostrack_mobile` |
| Hosted Backend | `https://heavensconnect.onrender.com/api/attendance` |

## Required Test Data

Before testing, make sure the backend has:

- One organization.
- One branch/location.
- One active staff/admin user.
- At least five members.
- At least one event, for example `Sunday Service`.
- At least one attendance session for that event.
- At least one camera setup if testing face recognition.
- NFC UID saved against at least one member if testing NFC.

## Start Backend Locally

```powershell
cd C:\Users\cents\PycharmProjects\WelfareAPI
.\.venv\Scripts\python.exe manage.py runserver 0.0.0.0:8085
```

Open:

```text
http://127.0.0.1:8085/swagger/
```

Expected result:

- Swagger loads.
- `/api/attendance/` returns a response.
- Protected endpoints return `401` unless logged in.

## Start Desktop Locally

```powershell
cd C:\Users\cents\PycharmProjects\DjangoProject1
.\.venv\Scripts\python.exe manage.py runserver 8000
```

Open:

```text
http://127.0.0.1:8000/login/
```

Expected result:

- KairosTrack login page loads.
- Staff can sign in with backend username and password.
- After login, dashboard loads.

## Connect Desktop To Backend

1. Open `Desktop Settings`.
2. Confirm the backend URL.
3. For hosted backend, use:

```text
https://heavensconnect.onrender.com/api/attendance
```

4. For local backend, use:

```text
http://127.0.0.1:8085/api/attendance
```

5. Click `Refresh Context`.

Expected result:

- Backend status shows reachable.
- Organization, plan, and signed-in staff details load.

## Start Mobile Locally

```powershell
cd C:\Users\cents\AndroidStudioProjects\kairostrack_mobile
flutter pub get
flutter run
```

If Gradle fails with `Unsupported class file major version`, confirm Java and Gradle compatibility:

```powershell
flutter doctor --verbose
```

Expected result:

- Mobile app builds.
- KairosTrack login screen loads.
- Staff can sign in.
- Member self check-in mode is available.

## Event Session Test

1. Sign in to desktop.
2. Open `Event Sessions`.
3. Create or select an event session.
4. Set it as the active session.
5. Open the session detail/report page.

Expected result:

- The active session is visible.
- Attendance methods point to the selected session.
- Report counts show total members, present, absent, and pending.

## Manual Attendance Test

1. Open `Manual Attendance`.
2. Confirm the selected event session appears.
3. Search for a member.
4. Mark the member `Present`.
5. Mark another member `Absent`.
6. Use quick action `Mark Pending Present`.
7. Open the session report.

Expected result:

- Attendance records are saved with method `manual`.
- Counts update correctly.
- No record is saved without an event session.

## Swipe Attendance Test

Swipe attendance is intended for mobile staff mode.

1. Open the mobile app.
2. Sign in as staff.
3. Select an event session.
4. Open swipe attendance.
5. Swipe right for present.
6. Swipe left for absent.
7. Open desktop session report.

Expected result:

- Records sync to the backend.
- Method shows as `swipe`.
- Desktop report reflects mobile actions.

## QR Attendance Test

### Staff QR Check-In

1. Open mobile staff mode.
2. Select an event session.
3. Open QR check-in.
4. Scan or paste a member QR code.

Expected result:

- Member is checked in for the selected event session.
- Method shows as `qr`.
- Duplicate check-ins are handled gracefully.

### Member QR Self Check-In

1. Open mobile member self check-in.
2. Search by organization and member identifier.
3. Select an available event session.
4. Scan or enter event QR.

Expected result:

- Member only sees their own check-in flow.
- Member cannot view other member records.
- Wrong organization, expired QR, or already checked-in states show clear messages.

## NFC Attendance Test

1. Assign an NFC UID to a member.
2. Open desktop `NFC Check-In`.
3. Tap the card or type the UID.
4. Press Enter or click `Check In`.
5. Open session report.

Expected result:

- Record is saved with method `nfc`.
- Recent scans update.
- Unknown UID shows a clear error.

## Face Recognition Test

Face recognition is a desktop/local feature because private IP cameras only work on the local network.

1. Confirm a member has an authorized face image.
2. Confirm camera settings exist.
3. Select an active event session.
4. Open `Mark Attendance` or camera stream.
5. Stand in front of the camera.
6. Open session report.

Expected result:

- Face check-in writes to the active event session.
- Method shows as `face`.
- If no active session exists, the app asks the user to select one first.

## GeoTracking Test

GeoTracking is intended for the mobile app and Enterprise tier.

1. Create a branch with latitude, longitude, and allowed radius.
2. Assign an event/session to the branch.
3. Open mobile member or staff GeoTracking check-in.
4. Allow location permission.
5. Attempt check-in inside the allowed radius.
6. Attempt check-in outside the allowed radius.

Expected result:

- Inside radius: attendance is accepted.
- Outside radius: attendance is rejected with distance/radius details.
- Location permission errors are clear and recoverable.

## Reports Test

1. Open desktop `Reports`.
2. Filter by date range.
3. Filter by event.
4. Filter by branch.
5. Filter by method.
6. Download CSV, Excel, and PDF.

Expected result:

- Counts match attendance records.
- Method breakdown is correct.
- Exports respect selected filters.
- PDF is readable.

## Session Expiry Test

1. Sign in to desktop.
2. Wait until access token expires or invalidate the token.
3. Open All Members, Attendance Logs, or Reports.

Expected result:

- App refreshes token if refresh token is valid.
- If refresh fails, user is redirected to login.
- The app should not repeat raw `401 token expired` messages across pages.

## Offline Backend Test

1. Sign in to desktop using local backend.
2. Stop the backend server.
3. Open members, reports, or settings.

Expected result:

- App shows a friendly backend unavailable message.
- Desktop does not crash.
- User can update backend URL in Desktop Settings.

## Installer Test

1. Build the desktop executable.
2. Build installer with Inno Setup.
3. Install on a Windows PC.
4. Confirm shortcut icon and app name show `KairosTrack`.
5. Launch app.
6. Login and run a basic manual attendance test.
7. Uninstall app.

Expected result:

- App installs under `KairosTrack`.
- Shortcut icon is correct.
- Uninstall removes KairosTrack app folders.
- No stale `TimeAndAttendance` branding remains.

## Final Acceptance Checklist

- Backend can create organizations, branches, staff, members, events, and sessions.
- Desktop can sign in using backend credentials.
- Desktop can select an active event session.
- Manual attendance works.
- QR attendance works.
- NFC attendance works.
- Face recognition writes to selected event session.
- Mobile staff mode can see sessions and mark attendance.
- Mobile member mode can self check in without seeing other members.
- GeoTracking validates allowed radius.
- Reports and exports work.
- Expired sessions redirect to login cleanly.
- Offline backend state is handled gracefully.
- Installer can be built, installed, launched, and uninstalled.

