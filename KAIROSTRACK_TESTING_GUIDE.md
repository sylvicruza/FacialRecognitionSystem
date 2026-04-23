# KairosTrack Attendance Flow Testing Guide

Use this guide to test the full event-based attendance flow after starting the desktop app.

## 1. Start the desktop app

```powershell
cd C:\Users\cents\PycharmProjects\DjangoProject1
.\.venv\Scripts\python.exe manage.py runserver 8000
```

Open `http://127.0.0.1:8000/` and sign in with a staff account.

## 2. Confirm the required setup

1. Open `Event Sessions`.
2. Create or select an event session, for example `Sunday Service - 10:00 AM`.
3. Open the session report from the same page.
4. Confirm the report page shows members, present, absent, pending, methods, exports, and action buttons.

Attendance should always be recorded against this selected session.

## 3. Test Manual Attendance

1. Open `Manual Attendance`.
2. Confirm the selected event session appears at the top.
3. Mark one member `Present`.
4. Mark one member `Absent`.
5. Open `Report` and confirm the counts changed.

Expected result: records are saved with method `manual`.

## 4. Test Swipe Attendance

Swipe is intended for the mobile app experience. The desktop app keeps the operational attendance flow on Manual, QR/NFC, and Face Recognition.

Mobile app test flow:

1. Start the Flutter app from `C:\Users\cents\AndroidStudioProjects\kairostrack_mobile`.
2. Sign in with a KairosTrack staff username and password.
3. Select an open or scheduled event session.
4. Swipe right to mark a member present.
5. Swipe left to mark a member absent.
6. Continue until the queue is empty, then open the desktop session report.

Expected result: records are saved with method `swipe`, and the pending count reduces.

## 5. Test QR Attendance

Desktop/admin flow:

1. Open `QR Check-In`.
2. Confirm the QR page shows the selected event session and current counts.
3. Click `Test Check-In`.
4. Complete the public check-in flow from the hosted backend page.
5. Return to the session report.

Mobile staff flow:

1. Start the Flutter app and sign in as staff.
2. Open an event session.
3. Tap `QR Check-In`.
4. Paste or scan a member QR payload from the member's phone.
5. Confirm the session count updates.

Mobile member flow:

1. Open `Member Self Check-In`.
2. Find the member using organization plus member code, email, or phone.
3. Confirm the member QR appears.
4. Tap `Use Event QR`.
5. Paste or scan the event QR token for an open self-check-in session.

Expected result: records are saved with method `qr`.

## 6. Test NFC Attendance

1. Assign an NFC UID to a member from the member profile.
2. Open `NFC Check-In`.
3. Tap a card/tag or type the UID into the reader field.
4. Press Enter or click `Check In`.
5. Check the recent scan panel and session report.

Expected result: records are saved with method `nfc`.

## 7. Test Face Recognition

1. Confirm camera settings exist.
2. Open `Face Recognition`.
3. Confirm the active event session appears at the top.
4. Resume a camera stream.
5. Stand in front of the camera with an authorized member face.
6. Open the session report.

Expected result: records are saved with method `face`.

## 8. Test GeoTracking Attendance

Enterprise plan is required.

GeoTracking is intended for the mobile app. The desktop app is now mainly used to configure branch/location rules. The hidden desktop route can still be used as a temporary validation screen.

1. Open `Organization`.
2. Add or edit a branch with latitude, longitude, and allowed radius.
3. Assign the event to that branch in `Event Branch Assignment`.
4. Open `/attendance/geotracking/` only as a temporary desktop test screen.
5. Click `Capture Location`.
6. Allow browser location permission.
7. Select a member and click `Mark Present`.
8. Open the session report.
9. Export CSV/Excel and confirm GeoTracking branch, latitude, longitude, accuracy, distance, and radius are included.

Expected result: records are saved with method `geotracking`.

Expected rejection: if the captured location is outside the branch radius, the backend rejects the check-in and returns the distance/radius details.

## 9. Test Mobile API Endpoints

The mobile app uses these hosted backend endpoints:

- `GET /api/attendance/mobile/sessions/`
- `GET /api/attendance/mobile/sessions/<id>/summary/`
- `GET /api/attendance/mobile/sessions/<id>/queue/`
- `POST /api/attendance/mobile/swipe/`
- `POST /api/attendance/mobile/qr/staff-check-in/`
- `POST /api/attendance/mobile/geotracking/check-in/`
- `POST /api/attendance/mobile/member/lookup/`
- `GET /api/attendance/mobile/member/sessions/`
- `POST /api/attendance/mobile/member/check-in/`
- `POST /api/attendance/mobile/member/event-qr-check-in/`
- `POST /api/attendance/mobile/member/geotracking/check-in/`

Run the mobile app:

```powershell
cd C:\Users\cents\AndroidStudioProjects\kairostrack_mobile
flutter run
```

Expected result: mobile login loads event sessions from the backend, swipe actions update the session report, and GeoTracking validates against the branch radius.

## 10. Test Member Self Check-In

1. Open the mobile app.
2. Tap `Member Self Check-In`.
3. Enter the organization name or slug.
4. Enter the member code, email, or phone for an authorized member.
5. Select an open event session.
6. Tap `Self Check-In` or `GeoTracking Check-In`.
7. Open the desktop session report.

Expected result: the member can only see their own check-in screen, and attendance is saved against the selected event session.

## 11. Test Exports

From the session report:

1. Click `CSV`.
2. Click `Excel`.
3. Click `PDF`.

Expected result: each export should contain only records for the selected event session.

## 12. Session expiry check

If the backend token expires:

1. Open any attendance page.
2. The desktop app should redirect to the login screen.
3. Sign in again.
4. Continue testing.

The app should not keep repeating raw `401 token expired` messages on every page.
