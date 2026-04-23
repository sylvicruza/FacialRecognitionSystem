# KairosTrack Desktop Testing And Installation

## Local End-To-End Test

1. Start the backend:
   ```powershell
   cd C:\Users\cents\PycharmProjects\WelfareAPI
   .\.venv\Scripts\python.exe manage.py runserver 0.0.0.0:8085
   ```

2. Start the desktop app:
   ```powershell
   cd C:\Users\cents\PycharmProjects\DjangoProject1
   .\.venv\Scripts\python.exe manage.py runserver 8000
   ```

3. Open `http://127.0.0.1:8000/login/`.

4. Sign in with a staff/admin account from the backend.

5. Open `Desktop Settings` and confirm:
   - Backend URL is correct.
   - Backend status says reachable.
   - Organization and plan are loaded.

6. Create or select an event session from `Event Sessions`.

7. Test attendance methods:
   - `Manual Attendance`: search, mark one member, mark pending present, mark filtered absent.
   - `QR Check-In`: open the test link and submit a member check-in.
   - `NFC Check-In`: type or scan a UID assigned to a member.
   - `Face Recognition`: open live cameras only after an active event session is selected.

8. Open `Reports` and verify:
   - Date range filters.
   - Event filter.
   - Branch filter.
   - Method breakdown.

9. Open `Attendance Logs` and export CSV, Excel, and PDF.

## Hosted Backend Test

In `Desktop Settings`, set the backend URL to:

```text
https://heavensconnect.onrender.com/api/attendance
```

Refresh context, then repeat the same attendance test flow.

## Installer Build

The installer expects the packaged executable at:

```text
dist\KairosTrack.exe
```

Then build with Inno Setup using:

```text
TimeAndAttendanceInstaller.iss
```

The installer includes:

- KairosTrack app name.
- KairosTrack icon.
- Version metadata.
- Start menu shortcut.
- Optional desktop shortcut.
- Uninstall cleanup for KairosTrack local app data folders.

## Release / Update Flow

The app currently uses a simple update notification strategy:

- Desktop Settings shows the installed app version.
- Desktop Settings links to the latest GitHub release.
- Users can download the latest installer from the same page.

Full automatic background updates are not enabled yet. That should be added later with a signed updater flow.
