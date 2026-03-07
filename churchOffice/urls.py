from django.urls import path
from . import views

urlpatterns = [
    # UI / Pages
    path('', views.home, name='home'),
    path('search_user/', views.search_user, name='search_user'),
    path('register_user/', views.register_user, name='register_user'),
    path('success_page/', views.success_page, name='success_page'),

    # People management
    path('person_list/', views.person_list, name='person_list'),
    path('persons/<int:pk>/', views.person_detail, name='person_detail'),
    path('persons/<int:pk>/authorize/', views.person_authorize, name='person_authorize'),
    path('persons/<int:pk>/delete/', views.person_delete, name='person_delete'),

    # Attendance views
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('persons/attendance/', views.person_attendance_list, name='person_attendance_list'),
    path("attendance/<int:pk>/delete/", views.attendance_delete, name="attendance_delete"),
    path("attendance/export/download/", views.attendance_export_download, name="attendance_export_download"),
    path("attendance/export/email/", views.attendance_email_export, name="attendance_email_export"),

    # Camera configuration (UI)
    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),

    # Streaming (MJPEG)
    path('stream/<int:cam_id>/', views.camera_stream, name='camera_stream'),
    path('video_feed/<int:cam_id>/', views.video_feed, name='video_feed'),
    path('camera_preview/<int:cam_id>/', views.camera_preview_feed, name='camera_preview_feed'),
    path('stream/all/', views.stream_all_cameras, name='stream_all_cameras'),

    # NFC API
    path('api/nfc/check-in/', views.nfc_check_in, name='nfc_check_in'),

    path('api/attendance/monitor/', views.api_attendance_monitor, name='api_attendance_monitor'),
    path("api/attendance/today/", views.attendance_today_api, name="attendance_today_api"),
]