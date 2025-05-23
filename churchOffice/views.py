from .utils import fetch_user_data, fetch_user_data_by_id  # Import both functions
from .models import Person, Attendance, CameraConfiguration  # Assuming you have a Person model for storing the user
from django.core.files.base import ContentFile
from django.contrib import messages
import base64
from django.db import IntegrityError
import numpy as np
import os
import cv2
import torch
from django.conf import settings
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
import pygame
from datetime import datetime, timedelta
from django.utils import timezone
import threading
import time
from django.http import StreamingHttpResponse
import pygame

pygame.mixer.init()


# Load models once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Face detection & encoding
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, map(round, box))
                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                encoding = resnet(face_tensor).cpu().numpy().flatten()
                faces.append((encoding, box))
    return faces

# Preload known encodings
def encode_uploaded_images():
    encodings, names = [], []
    for person in Person.objects.filter(authorized=True):
        img_path = os.path.join(settings.MEDIA_ROOT, str(person.image))
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for encoding, _ in detect_and_encode(img_rgb):
            encodings.append(encoding)
            names.append(person.name)
    return np.array(encodings), names

# Face recognition
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized = []
    for test_encoding, box in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        if len(distances) == 0:
            recognized.append(('Not Recognized', box))
            continue
        min_idx = np.argmin(distances)
        name = known_names[min_idx] if distances[min_idx] < threshold else 'Not Recognized'
        recognized.append((name, box))
    return recognized

# MJPEG stream generator with FPS limiting
def gen_frames(source, cam_config, known_encodings, known_names, max_fps=10):
    cap = cv2.VideoCapture(source)
    prev = 0
    delay = 1 / max_fps

    while cap.isOpened():
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - prev < delay:
            continue
        prev = now

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_encodings = detect_and_encode(frame_rgb)

        if test_encodings and len(known_encodings) > 0:
            recognized = recognize_faces(known_encodings, known_names, test_encodings, cam_config.threshold)
            for name, box in recognized:
                if box is None:
                    continue
                x1, y1, x2, y2 = map(int, map(round, box))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                print(f"[FRAME] Processing camera {cam_config.name} at {datetime.now()}")
                print(f"[INFO] Detected faces: {len(test_encodings)}")

                if name != 'Not Recognized':
                    print(f"[MATCH] Recognized {name}")

                    person = Person.objects.filter(name=name).first()
                    if person:
                        today = timezone.now().date()  # always use timezone-aware value
                        now = timezone.now()
                        attendance, created = Attendance.objects.get_or_create(person=person, date=today)

                        if created:
                            attendance.mark_checked_in()
                            play_success_sound()
                        elif attendance.check_in_time and not attendance.check_out_time:
                            if now >= attendance.check_in_time + timedelta(seconds=60):
                                attendance.mark_checked_out()
                                play_success_sound()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def play_success_sound():
    try:
        path = os.path.join(settings.BASE_DIR, 'media/audio/suc.wav')  # or wherever your sound is
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print("[ERROR] Sound playback failed:", e)



# Streaming view
def video_feed(request, cam_id):
    cam_config = get_object_or_404(CameraConfiguration, id=cam_id)
    known_encodings, known_names = encode_uploaded_images()
    source = int(cam_config.camera_source) if cam_config.camera_source.isdigit() else cam_config.camera_source
    return StreamingHttpResponse(
        gen_frames(source, cam_config, known_encodings, known_names),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

# Controller view to load camera page
def camera_stream(request, cam_id):
    config = get_object_or_404(CameraConfiguration, id=cam_id)
    return render(request, 'camera_stream.html', {'config': config})

def stream_all_cameras(request):
    configs = CameraConfiguration.objects.all()
    return render(request, 'stream_all_cameras.html', {'configs': configs})

# this is for showing Attendance list
def person_attendance_list(request):
    # Get the search query and date filter from the request
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')

    # Get all members
    persons = Person.objects.all()

    # Filter persons based on the search query
    if search_query:
        persons = persons.filter(name__icontains=search_query)

    # Prepare the attendance data
    person_attendance_data = []

    for person in persons:
        # Get the attendance records for each student, filtering by attendance date if provided
        attendance_records = Attendance.objects.filter(person=person)

        if date_filter:
            # Assuming date_filter is in the format YYYY-MM-DD
            attendance_records = attendance_records.filter(date=date_filter)

        attendance_records = attendance_records.order_by('date')

        person_attendance_data.append({
            'person': person,
            'attendance_records': attendance_records
        })

    context = {
        'person_attendance_data': person_attendance_data,
        'search_query': search_query,  # Pass the search query to the template
        'date_filter': date_filter  # Pass the date filter to the template
    }
    return render(request, 'user_attendance_list.html', context)


# Create your views here.
def home(request):
    total_persons = Person.objects.count()
    total_attendance = Attendance.objects.count()
    total_check_ins = Attendance.objects.filter(check_in_time__isnull=False).count()
    total_check_outs = Attendance.objects.filter(check_out_time__isnull=False).count()
    total_cameras = CameraConfiguration.objects.count()

    context = {
        'total_persons': total_persons,
        'total_attendance': total_attendance,
        'total_check_ins': total_check_ins,
        'total_check_outs': total_check_outs,
        'total_cameras': total_cameras,
    }
    return render(request, 'home.html', context)

def search_user(request):
    if request.method == 'POST':
        # Get the last name entered by the user in the form
        last_name = request.POST.get('last_name')

        # Call the API to search for the user by last name
        user_data = fetch_user_data(last_name)

        if user_data:
            # If users are found, pass the data to the template to display
            return render(request, 'search_user.html', {'user_data': user_data})
        else:
            # If no users found, show an error message
            return render(request, 'search_user.html', {'error': 'User not found, please register on the Church Portal first.'})

    # If it's a GET request, just render the search form
    return render(request, 'search_user.html')


def register_user(request):
    if request.method == 'GET':
        portal_id = request.GET.get('portal_id')
        name = request.GET.get('name')
        return render(request, 'register_user.html', {
            'portal_id': portal_id,
            'name': name,
        })

    if request.method == 'POST':
        portal_id = request.POST.get('portal_id')
        name = request.POST.get('name')
        image_data = request.POST.get('image_data')
        print(image_data)

        if not image_data:
            return render(request, 'register_user.html', {
                'error': 'Image is required.',
                'portal_id': portal_id,
                'name': name
            })

        if image_data:
            header, encoded = image_data.split(',', 1)
            image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

            person = Person(
                name=name,
                portal_id=portal_id,
                image=image_file,
                authorized=True
            )
            person.save()

            return redirect('success_page')

    return render(request, 'register_user.html')

# Success view after capturing Member information and image
def success_page(request):
    return render(request, 'selfie_success.html')

# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

def person_list(request):
    persons = Person.objects.all()
    return render(request, 'user_list.html', {'persons': persons})

def person_detail(request, pk):
    person = get_object_or_404(Person, pk=pk)
    user_data = fetch_user_data_by_id(person.portal_id)
    context = {'person': person, 'user_data': user_data}
    return render(request, 'user_detail.html', context)

def person_authorize(request, pk):
    person = get_object_or_404(Person, pk=pk)

    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        person.authorized = bool(authorized)
        person.save()
        return redirect('person_detail', pk=pk)

    return render(request, 'user_authorize.html', {'person': person})

def person_delete(request, pk):
    person = get_object_or_404(Person, pk=pk)

    if request.method == 'POST':
        person.delete()
        messages.success(request, 'Member deleted successfully.')
        return redirect('person_list')  # Redirect to the member list after deletion

    return render(request, 'user_delete_confirm.html', {'person': person})


# Function to handle the creation of a new camera configuration
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        # Save the changes to the database
        config.save()

        # Redirect to the list page after successful update
        return redirect('camera_config_list')

        # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config})


# DELETE: Function to delete a camera configuration
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

    # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})


def capture_and_recognize(request):
    return None


