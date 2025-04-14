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

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models once
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, map(round, box))
                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_LINEAR)
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                encoding = resnet(face_tensor).cpu().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Encode uploaded images only once
def encode_uploaded_images():
    from .models import Person  # Ensure models import here to avoid circular deps
    known_face_encodings, known_face_names = [], []
    for person in Person.objects.filter(authorized=True):
        image_path = os.path.join(settings.MEDIA_ROOT, str(person.image))
        known_image = cv2.imread(image_path)
        if known_image is None:
            continue
        known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
        encodings = detect_and_encode(known_image_rgb)
        if encodings:
            known_face_encodings.extend(encodings)
            known_face_names.extend([person.name] * len(encodings))
    return known_face_encodings, known_face_names

# Face recognition
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        if len(distances) == 0:
            recognized_names.append('Not Recognized')
            continue
        min_idx = np.argmin(distances)
        recognized_names.append(known_names[min_idx] if distances[min_idx] < threshold else 'Not Recognized')
    return recognized_names

# View for recognition
def capture_and_recognize(request):
    from .models import CameraConfiguration, Person, Attendance
    stop_events, camera_threads, camera_windows, error_messages = [], [], [], []
    known_face_encodings, known_face_names = encode_uploaded_images()

    def process_frame(cam_config, stop_event):
        cap = None
        try:
            source = int(cam_config.camera_source) if cam_config.camera_source.isdigit() else cam_config.camera_source
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {cam_config.name}")

            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('churchOffice/suc.wav')
            window_name = f"Face Recognition - {cam_config.name}"
            camera_windows.append(window_name)

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_face_encodings = detect_and_encode(frame_rgb)

                if test_face_encodings and known_face_encodings:
                    names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, cam_config.threshold)
                    for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
                        if box is None:
                            continue
                        x1, y1, x2, y2 = map(int, map(round, box))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        if name != 'Not Recognized':
                            person = Person.objects.filter(name=name).first()
                            if not person:
                                continue

                            today = datetime.now().date()
                            attendance, created = Attendance.objects.get_or_create(person=person, date=today)

                            now = timezone.now()
                            if created:
                                attendance.mark_checked_in()
                                success_sound.play()
                                msg = f"{name}, checked in."
                            elif attendance.check_in_time and not attendance.check_out_time:
                                if now >= attendance.check_in_time + timedelta(seconds=60):
                                    attendance.mark_checked_out()
                                    success_sound.play()
                                    msg = f"{name}, checked out."
                                else:
                                    msg = f"{name}, already checked in."
                            else:
                                msg = f"{name}, already checked out."

                            cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

        except Exception as e:
            error_messages.append(str(e))
        finally:
            if cap: cap.release()
            cv2.destroyWindow(window_name)

    try:
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found.")

        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)
            t = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(t)
            t.start()

        while any(t.is_alive() for t in camera_threads):
            time.sleep(1)

    except Exception as e:
        error_messages.append(str(e))
    finally:
        for e in stop_events: e.set()
        for w in camera_windows:
            if cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(w)

    if error_messages:
        return render(request, 'error.html', {'error_message': "\n".join(error_messages)})
    return redirect('person_attendance_list')

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
