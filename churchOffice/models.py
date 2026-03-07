from django.db import models
from django.utils import timezone


class Person(models.Model):
    name = models.CharField(max_length=255)
    portal_id = models.CharField(max_length=100)
    image = models.ImageField(upload_to='person_images/', null=True, blank=True)
    authorized = models.BooleanField(default=False)
    nfc_uid = models.CharField(max_length=100, unique=True, null=True, blank=True)

    def __str__(self):
        return self.name


class Attendance(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    date = models.DateField()
    check_in_time = models.DateTimeField(null=True, blank=True)
    check_out_time = models.DateTimeField(null=True, blank=True)

    camera = models.ForeignKey(
        "CameraConfiguration",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="attendance_events",
    )



    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["person", "date"], name="uniq_attendance_person_date")
        ]

    def __str__(self):
        return f"{self.person.name} - {self.date}"

    def calculate_duration(self):
        if self.check_in_time and self.check_out_time:
            duration = self.check_out_time - self.check_in_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        return None

    def save(self, *args, **kwargs):
        if not self.pk and not self.date:
            self.date = timezone.localdate()
        super().save(*args, **kwargs)


class CameraConfiguration(models.Model):
    name = models.CharField(max_length=100, unique=True, help_text="Give a name to this camera configuration")
    camera_source = models.CharField(
        max_length=255,
        help_text="Camera index (0 for default webcam or RTSP/HTTP URL for IP camera)"
    )
    threshold = models.FloatField(default=0.6, help_text="Face recognition confidence threshold")

    def __str__(self):
        return self.name
