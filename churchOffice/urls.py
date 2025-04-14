from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('search_user/', views.search_user, name='search_user'),
    path('register_user/', views.register_user, name='register_user'),  # URL for user registration
    path('success_page/', views.success_page, name='success_page'),
    path('person_list/', views.person_list, name='person_list'),
    path('persons/<int:pk>/', views.person_detail, name='person_detail'),
    path('persons/<int:pk>/authorize/', views.person_authorize, name='person_authorize'),
    path('persons/<int:pk>/delete/', views.person_delete, name='person_delete'),
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('persons/attendance/', views.person_attendance_list, name='person_attendance_list'),
    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),
]