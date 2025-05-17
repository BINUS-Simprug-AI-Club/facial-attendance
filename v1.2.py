"""
Facial Recognition Attendance System
-----------------------------------
This program uses facial recognition to track student attendance and 
stores records in Firebase. It displays motivational quotes and provides
visual feedback when students are recognized.
"""

# Standard library imports
import os
import pickle
import random
import time
import tempfile
from datetime import datetime, time as dt_time

# Third-party imports
import cv2
import dlib
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'facial-attendance-binus.firebasestorage.app'})
db = firestore.client()

# Configuration settings
CONFIG = {
    # Recognition settings
    "tolerance": 0.35,              # Face matching threshold (lower = stricter)
    "frame_resize": 0.25,           # Resize factor for processing (smaller = faster)
    "skip_frames": 2,               # Only process every nth frame
    "enhanced_facial_recognition": True,  # Use enhanced recognition features
    
    # Display settings
    "display_fps": True,            # Show FPS counter
    "show_all_faces": True,         # Show boxes for unknown faces too
    "flip_camera": 1,               # -1: 180°, 0: upside down, 1: mirror
    "corner_display": True,         # Show corner screenshot
    "show_landmarks": True,         # Display facial landmarks on detected faces
    
    # Attendance settings
    "latest_login_time": "07:30",   # Latest time to login without being marked late
    
    # File paths
    "facial_landmarks_path": "shape_predictor_68_face_landmarks.dat", # Path to facial landmarks model
}

# Motivational quotes based on BINUS Values and IB Learner Profile
MOTIVATIONAL_QUOTES = [
    "Strive for excellence in all you do today.",
    "Embrace innovation in your learning journey.",
    "Perseverance leads to great accomplishments.",
    "Today is another opportunity for personal growth.",
    "Your integrity defines who you are as a person.",
    "Respect for others builds a stronger community.",
    "Be an inquirer: nurture your curiosity today.",
    "Think deeply as a knowledgeable learner.",
    "Be a creative and critical thinker.",
    "Communicate confidently and creatively today.",
    "Act with integrity and honesty in all you do.",
    "Keep an open mind to different perspectives.",
    "Show caring towards your fellow students.",
    "Take risks and approach challenges positively.",
    "Balance your academic, physical, and emotional well-being.",
    "Reflect thoughtfully on your learning journey."
]

# Global variables
attendance = {}  # Track attendance to prevent duplicate entries
thank_you_message = {
    'active': False,
    'name': '',
    'time': 0,
    'duration': 3.0,
    'quote': '',
    'status': ''
}
facial_landmark_predictor = None


def initialize_landmark_predictor():
    """Initialize the facial landmark predictor if available."""
    global facial_landmark_predictor
    
    print("🔄 Loading facial landmark predictor...")
    try:
        facial_landmark_predictor = dlib.shape_predictor(CONFIG["facial_landmarks_path"])
        print("✅ Facial landmark predictor loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading facial landmark predictor: {str(e)}")
        print("⚠️ Continuing without facial landmarks detection.")
        CONFIG["show_landmarks"] = False
        return False


def log_attendance(name, class_name=None):
    """
    Record attendance for a person in Firebase and show thank you message.
    
    Args:
        name (str): Name of the person to log
        class_name (str, optional): Class name of the person
        
    Returns:
        bool: True if new attendance was logged, False if already logged
    """
    attendance_key = name
    if class_name:
        attendance_key = f"{class_name}/{name}"
        
    if attendance_key in attendance:
        return False  # Already logged
    
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    date_only = current_time.strftime("%Y-%m-%d")

    latest_time = dt_time(
        int(CONFIG["latest_login_time"].split(":")[0]), 
        int(CONFIG["latest_login_time"].split(":")[1])
    )
    
    is_late = current_time.time() > latest_time
    status = "Late" if is_late else "Present"

    attendance_ref = db.collection('attendance').document(date_only)
    
    @firestore.transactional
    def update_in_transaction(transaction, doc_ref):
        doc = doc_ref.get(transaction=transaction)
        
        attendance_data = {
            name: {
                'timestamp': timestamp,
                'status': status,
                'late': is_late
            }
        }
        
        if class_name:
            attendance_data[name]['class'] = class_name
        
        if doc.exists:
            transaction.update(doc_ref, attendance_data)
        else:
            transaction.set(doc_ref, attendance_data)
    
    transaction = db.transaction()
    update_in_transaction(transaction, attendance_ref)
    
    attendance[attendance_key] = True
    
    if class_name:
        print(f"👋 Thank you, {name} from {class_name}! Your attendance has been recorded as {status}.")
    else:
        print(f"👋 Thank you, {name}! Your attendance has been recorded as {status}.")
    
    global thank_you_message
    thank_you_message = {
        'active': True,
        'name': name if not class_name else f"{name} ({class_name})",
        'time': time.time(),
        'duration': 3.0,
        'quote': random.choice(MOTIVATIONAL_QUOTES),
        'status': status
    }
    
    return True


def load_face_encodings():
    """
    Load known face encodings from cache or build from image dataset in Firebase Storage.
    
    Returns:
        tuple: (known_face_encodings, known_face_names, known_face_classes)
    """
    print("🔄 Loading face encodings...")
    known_face_encodings = []
    known_face_names = []
    known_face_classes = []
    
    encodings_file = "encodings.pickle"
    if os.path.exists(encodings_file):
        print("Loading encodings from cache...")
        try:
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data["encodings"]
                known_face_names = data["names"]
                if "classes" in data:
                    known_face_classes = data["classes"]
                else:
                    known_face_classes = ["" for _ in known_face_names]
            print(f"✅ Loaded {len(known_face_encodings)} encodings from cache")
        except (pickle.PickleError, KeyError) as e:
            print(f"⚠️ Error loading cache: {e}. Will rebuild from Firebase Storage.")
            known_face_encodings = []
            known_face_names = []
            known_face_classes = []
    
    if not known_face_encodings:
        dataset_path = "face_dataset"
        print(f"Building encodings from Firebase Storage: {dataset_path}")
        
        bucket = storage.bucket()
        
        blobs = list(bucket.list_blobs(prefix=dataset_path + "/"))
        class_prefixes = set()
        
        for blob in blobs:
            path_parts = blob.name.split('/')
            if len(path_parts) > 2:
                class_prefixes.add(path_parts[1])
        
        for class_name in class_prefixes:
            class_path = f"{dataset_path}/{class_name}/"
            print(f"Processing class: {class_name}")
            
            student_prefixes = set()
            for blob in blobs:
                if blob.name.startswith(class_path):
                    path_parts = blob.name.split('/')
                    if len(path_parts) > 3:
                        student_prefixes.add(path_parts[2])
            
            for student_name in student_prefixes:
                print(f"Processing student: {student_name} in class: {class_name}")
                student_encodings = []
                
                student_path = f"{class_path}{student_name}/"
                student_images = [blob for blob in blobs if blob.name.startswith(student_path) and blob.name.split('/')[-1]]
                
                temp_dir = tempfile.mkdtemp()
                
                for blob in student_images:
                    image_name = blob.name.split("/")[-1]
                    if not image_name:
                        continue
                        
                    temp_path = os.path.join(temp_dir, image_name)
                    blob.download_to_filename(temp_path)
                    print(f"Processing image: {blob.name}")

                    image = cv2.imread(temp_path)
                    if image is None:
                        print(f"⚠️ Warning: Unable to read {image_name}, skipping.")
                        continue

                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb)
                    if len(face_locations) > 0:
                        encodings = face_recognition.face_encodings(rgb, face_locations)
                        student_encodings.extend(encodings)
                        print(f"✓ Found {len(face_locations)} faces in {image_name}")
                    else:
                        print(f"⚠️ No faces detected in {image_name}")
                    
                    os.remove(temp_path)
                
                os.rmdir(temp_dir)
                        
                if student_encodings:
                    for encoding in student_encodings:
                        known_face_encodings.append(encoding)
                        known_face_names.append(student_name)
                        known_face_classes.append(class_name)
                    print(f"✅ Loaded {len(student_encodings)} images for {student_name} in class {class_name}")
                else:
                    print(f"❌ No usable images found for {student_name} in class {class_name}")
        
        print("Saving encodings to cache...")
        data = {
            "encodings": known_face_encodings, 
            "names": known_face_names,
            "classes": known_face_classes
        }
        with open(encodings_file, 'wb') as f:
            pickle.dump(data, f)

    print(f"✅ Face data loaded successfully! {len(known_face_encodings)} faces stored.")
    return known_face_encodings, known_face_names, known_face_classes


def display_thank_you_message(frame):
    """Display thank you message overlay on frame."""
    if not thank_you_message['active']:
        return frame
        
    if time.time() - thank_you_message['time'] < thank_you_message['duration']:
        bg_color = (0, 200, 0) if thank_you_message['status'] == "Present" else (0, 165, 255)
        
        cv2.rectangle(frame, (50, frame.shape[0]//2 - 70), 
                    (frame.shape[1]-50, frame.shape[0]//2 + 70), 
                    bg_color, -1)
        
        cv2.putText(frame, f"Thank you, {thank_you_message['name']}!", 
                (100, frame.shape[0]//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Your attendance has been recorded as {thank_you_message['status']}", 
                (100, frame.shape[0]//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        quote = thank_you_message['quote']
        if len(quote) > 50:
            split_idx = quote.find(' ', 40)
            if split_idx == -1:
                split_idx = 50
            line1 = quote[:split_idx]
            line2 = quote[split_idx:]
            cv2.putText(frame, f'"{line1}', 
                    (100, frame.shape[0]//2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f'{line2}"', 
                    (100, frame.shape[0]//2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f'"{quote}"', 
                    (100, frame.shape[0]//2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        thank_you_message['active'] = False
        
    return frame


def draw_facial_landmarks(frame, face_coords):
    """Draw facial landmarks on the detected face."""
    if not CONFIG["show_landmarks"] or not CONFIG["enhanced_facial_recognition"]:
        return
        
    left, top, right, bottom = face_coords
    try:
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        shape = facial_landmark_predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dlib_rect)
        
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
    except Exception:
        pass


def main():
    """Main function to run the facial recognition system."""
    initialize_landmark_predictor()
    
    known_face_encodings, known_face_names, known_face_classes = load_face_encodings()
    
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    print("🎥 Camera initialized. Press 'q' to quit.")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break
            
        frame = cv2.flip(frame, CONFIG["flip_camera"])
        
        if CONFIG["corner_display"]:
            corner_size = (int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.2))
            corner_frame = cv2.resize(frame, corner_size)
            
            x_offset = frame.shape[1] - corner_size[0] - 10
            y_offset = 10
            
            cv2.rectangle(frame, (x_offset-2, y_offset-2), 
                         (x_offset + corner_size[0]+2, y_offset + corner_size[1]+2), 
                         (255, 255, 255), 2)
            
            frame[y_offset:y_offset + corner_size[1], x_offset:x_offset + corner_size[0]] = corner_frame
        
        process_this_frame = frame_count % CONFIG["skip_frames"] == 0
        frame_count += 1
        
        if time.time() - fps_start_time >= 1.0:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()
        
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=CONFIG["frame_resize"], fy=CONFIG["frame_resize"])
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            face_classes = []
            face_confidences = []
            
            for face_encoding in face_encodings:
                name = "Unknown"
                class_name = ""
                confidence = 0.0
                
                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        best_match_distance = face_distances[best_match_index]
                        
                        best_match_score = 1 - best_match_distance
                        
                        if best_match_score >= CONFIG["tolerance"]:
                            name = known_face_names[best_match_index]
                            class_name = known_face_classes[best_match_index]
                            confidence = best_match_score
                            
                            if log_attendance(name, class_name):
                                print(f"✔️ Attendance marked for {name}" + 
                                      (f" in class {class_name}" if class_name else ""))
                
                face_names.append(name)
                face_classes.append(class_name)
                face_confidences.append(confidence)
            
            for (top, right, bottom, left), name, class_name, confidence in zip(
                    face_locations, face_names, face_classes, face_confidences):
                scale = 1.0 / CONFIG["frame_resize"]
                top = int(top * scale)
                right = int(right * scale)
                bottom = int(bottom * scale)
                left = int(left * scale)
                
                if name == "Unknown" or confidence < CONFIG["tolerance"]:
                    if CONFIG["show_all_faces"]:
                        color = (0, 0, 255) if name == "Unknown" else (0, 165, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        label = f"Unknown ({confidence:.2f})" if name == "Unknown" else f"{name} ({confidence:.2f})"
                        cv2.putText(frame, label, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    display_name = name
                    if class_name:
                        display_name = f"{name} ({class_name})"
                    
                    cv2.putText(frame, f"{display_name} ({confidence:.2f})", (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                draw_facial_landmarks(frame, (left, top, right, bottom))
        
        if CONFIG["display_fps"]:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        frame = display_thank_you_message(frame)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    print("🔴 Face recognition system closed.")


if __name__ == "__main__":
    main()
