from dataclasses import dataclass
from datetime import timedelta
from django.db import transaction
from django.utils import timezone
from .models import Attendance, Person, CameraConfiguration


@dataclass(frozen=True)
class AttendanceOutcome:
    status: str
    attendance_id: int

@transaction.atomic
def mark_attendance(person: Person, min_checkout_seconds: int = 60, camera: CameraConfiguration | None = None):
    today = timezone.localdate()
    now = timezone.now()

    attendance, created = Attendance.objects.select_for_update().get_or_create(
        person=person,
        date=today
    )

    # Check-in
    if created or not attendance.check_in_time:
        attendance.check_in_time = now
        if hasattr(attendance, "camera"):
            attendance.camera = camera
        attendance.save(update_fields=["check_in_time"] + (["camera"] if hasattr(attendance, "camera") else []))
        return AttendanceOutcome("checked_in", attendance.id)

    # Check-out (optional)
    if attendance.check_in_time and not attendance.check_out_time:
        if now >= attendance.check_in_time + timedelta(seconds=min_checkout_seconds):
            attendance.check_out_time = now
            if hasattr(attendance, "camera"):
                attendance.camera = camera
            attendance.save(update_fields=["check_out_time"] + (["camera"] if hasattr(attendance, "camera") else []))
            return AttendanceOutcome("checked_out", attendance.id)
        return AttendanceOutcome("already_checked_in", attendance.id)

    return AttendanceOutcome("already_checked_out", attendance.id)