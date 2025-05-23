from django.db import models
from django.utils import timezone

# Create your models here.
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=255)
    portal_id = models.CharField(max_length=100)
    image = models.ImageField(upload_to='person_images/', null=True, blank=True)
    authorized = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class Attendance(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    date = models.DateField()
    check_in_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.person.name} - {self.date}"

    def mark_checked_in(self):
        today = timezone.now().date()

        # Check if there's already an attendance record for today
        attendance_record = Attendance.objects.filter(person=self.person, date=today).first()

        if attendance_record:
            # If the record exists, update it
            attendance_record.check_in_time = timezone.now()
            attendance_record.save()
            print(f"[ATTENDANCE] Checked in: {self.person.name} at {attendance_record.check_in_time}")
        else:
            # If no record exists, create a new one
            new_attendance = Attendance(person=self.person, date=today, check_in_time=timezone.now())
            new_attendance.save()
            print(f"[ATTENDANCE] New record created. Checked in: {self.person.name} at {new_attendance.check_in_time}")

    def mark_checked_out(self):
        if self.check_in_time:
            self.check_out_time = timezone.now()
            self.save()
        else:
            raise ValueError("Cannot mark check-out without check-in.")

    def calculate_duration(self):
        if self.check_in_time and self.check_out_time:
            duration = self.check_out_time - self.check_in_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        return None

    def save(self, *args, **kwargs):
        if not self.pk:
            self.date = timezone.now().date()
        super().save(*args, **kwargs)


class CameraConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Give a name to this camera configuration")
    camera_source = models.CharField(max_length=255, help_text="Camera index (0 for default webcam or RTSP/HTTP URL for IP camera)")
    threshold = models.FloatField(default=0.6, help_text="Face recognition confidence threshold")

    def __str__(self):
        return self.name



