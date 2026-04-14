# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()

a = Analysis(
    ["run_attendance_client.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        ("templates", "templates"),
        ("static", "static"),
        ("media", "media"),
    ],
    hiddenimports=[
        "waitress",
        "django.contrib.sessions",
        "django.contrib.messages",
        "django.template.context_processors.request",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TimeAndAttendance",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TimeAndAttendance",
)
