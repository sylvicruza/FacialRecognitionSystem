from django.urls import path
from . import views

urlpatterns = [
    # Public product pages
    path('about/', views.product_about, name='product_about'),
    path('pricing/', views.product_about, name='product_pricing'),
    path('demo-request/', views.demo_request, name='demo_request'),

    # UI / Pages
    path('login/', views.desktop_login, name='desktop_login'),
    path('signup/', views.desktop_signup, name='desktop_signup'),
    path('logout/', views.desktop_logout, name='desktop_logout'),
    path('getting-started/', views.getting_started, name='getting_started'),
    path('upgrade/', views.upgrade_request, name='upgrade_request'),
    path('settings/', views.desktop_settings, name='desktop_settings'),
    path('organization/settings/', views.organization_settings, name='organization_settings'),
    path('', views.home, name='home'),
    path('search_user/', views.search_user, name='search_user'),
    path('register_user/', views.register_user, name='register_user'),
    path('success_page/', views.success_page, name='success_page'),

    # People management
    path('person_list/', views.person_list, name='person_list'),
    path('members/import/', views.member_import, name='member_import'),
    path('members/import/template/', views.member_import_template, name='member_import_template'),
    path('persons/<int:pk>/', views.person_detail, name='person_detail'),
    path('persons/<int:pk>/authorize/', views.person_authorize, name='person_authorize'),
    path('persons/<int:pk>/nfc/', views.person_nfc_update, name='person_nfc_update'),
    path('persons/<int:pk>/delete/', views.person_delete, name='person_delete'),

    # Attendance views
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('persons/attendance/', views.person_attendance_list, name='person_attendance_list'),
    path('reports/', views.attendance_reports, name='attendance_reports'),
    path('attendance/sessions/', views.attendance_sessions, name='attendance_sessions'),
    path('attendance/sessions/<int:pk>/', views.attendance_session_detail, name='attendance_session_detail'),
    path('attendance/qr/', views.qr_attendance, name='qr_attendance'),
    path('attendance/nfc/', views.nfc_attendance, name='nfc_attendance'),
    path('attendance/manual/', views.manual_attendance, name='manual_attendance'),
    path('attendance/swipe/', views.swipe_attendance, name='swipe_attendance'),
    path('attendance/geotracking/', views.geotracking_attendance, name='geotracking_attendance'),
    path('attendance/geolocation/', views.geolocation_attendance, name='geolocation_attendance'),
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
