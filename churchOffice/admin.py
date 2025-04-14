from django.contrib import admin

from churchOffice.models import Person, Attendance, CameraConfiguration

# Register your models here.
admin.site.register(Person)
admin.site.register(Attendance)
admin.site.register(CameraConfiguration)