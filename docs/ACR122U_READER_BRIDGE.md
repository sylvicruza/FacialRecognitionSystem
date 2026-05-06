# ACR122U Reader Bridge

KairosTrack's browser reader page works directly with keyboard-wedge readers. The ACS ACR122U is usually a PC/SC smart card reader instead, so it needs a small local bridge.

## What the bridge does

- Reads NFC tag UIDs from the ACR122U
- Uses the reader setup you saved in KairosTrack desktop
- Supports:
  - `monitor only`
  - `member enrollment handoff`
  - `live entrance check-in`

## One-time setup

1. Plug in the ACR122U.
2. Open KairosTrack desktop and go to `/members/nfc-reader/`.
3. Save the reader setup once.
4. Install the bridge dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install pyscard
```

## Start the bridge

```powershell
.\start_nfc_reader_bridge.bat
```

The bridge reads config from the local desktop runtime folder and updates the reader runtime file that KairosTrack shows in the reader setup page.

## Notes

- `live entrance check-in` needs an active attendance session selected in KairosTrack.
- `member enrollment handoff` opens bulk enrollment with the UID already filled in.
- If the bridge says no reader is found, confirm Windows still shows the ACR122U under smart card readers.
