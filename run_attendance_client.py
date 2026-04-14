import os
import socket
import sys
import threading
import time
import webbrowser


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _open_browser_when_ready(host: str, port: int):
    if _wait_for_port(host, port):
        webbrowser.open(f"http://{host}:{port}/login/")


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ohc_time_attendance.settings")
    os.environ.setdefault("ATTENDANCE_DESKTOP_MODE", "1")

    import django

    django.setup()

    from django.core.management import call_command
    from django.core.wsgi import get_wsgi_application
    from django.contrib.staticfiles.handlers import StaticFilesHandler
    from waitress import serve

    host = os.getenv("ATTENDANCE_CLIENT_HOST", "127.0.0.1")
    port = int(os.getenv("ATTENDANCE_CLIENT_PORT", "8000"))
    threads = int(os.getenv("ATTENDANCE_CLIENT_THREADS", "8"))

    # Ensure the packaged desktop app has the local session/runtime tables it needs.
    call_command("migrate", interactive=False, run_syncdb=True, verbosity=0)

    browser_thread = threading.Thread(
        target=_open_browser_when_ready,
        args=(host, port),
        daemon=True,
    )
    browser_thread.start()

    print(f"Time and Attendance desktop starting on http://{host}:{port}/login/")
    application = StaticFilesHandler(get_wsgi_application())
    serve(application, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
