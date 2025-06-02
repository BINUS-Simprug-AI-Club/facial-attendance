"""
Facial Recognition Attendance System - High Performance Edition
--------------------------------------------------------------
This program uses optimized facial recognition to track student attendance with
maximum speed while maintaining high accuracy. Features include GPU acceleration,
intelligent frame processing, and advanced caching.
"""

# Standard library imports
import os
import pickle
import random
import time
import tempfile
from datetime import datetime, time as dt_time
import threading
from collections import deque
import queue

# Third-party imports
import cv2 
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage
import concurrent.futures
from dotenv import load_dotenv

# Try to import face_alignment for landmarks
try:
    import face_alignment
    FACE_ALIGNMENT_AVAILABLE = True
except ImportError:
    FACE_ALIGNMENT_AVAILABLE = False

# Try to import dlib for faster face detection
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# Try to import GPU acceleration libraries
try:
    import cupy as cp # type: ignore
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Get secret key from .env file
load_dotenv()
secret_key = os.getenv("MY_SECRET_KEY")
print("My secret key is:", secret_key)

# Initialize Firebase
cred = credentials.Certificate("facial-attendance-binus-firebase-adminsdk-fbsvc-663eb05a63.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'facial-attendance-binus.firebasestorage.app'})
db = firestore.client()

# Configuration settings with performance optimizations
CONFIG = {
    # Recognition settings - optimized for speed/accuracy balance
    "tolerance": 0.45,
    "frame_resize": 0.3,
    "skip_frames": 1,
    "enhanced_facial_recognition": True,
    
    # Display settings
    "display_fps": True,
    "show_all_faces": True,
    "flip_camera": 1,
    "corner_display": False,
    "show_landmarks": False,
    
    # Attendance settings
    "latest_login_time": "07:30",
    
    # Performance settings
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    "first_run_warning": True,
    "use_gpu_acceleration": GPU_AVAILABLE,
    "use_dlib_detector": DLIB_AVAILABLE,
    "detector_scale_factor": 1.2,
    "detector_min_neighbors": 3,
    
    # Advanced Performance settings
    "face_detection_threads": 2,
    "encoding_threads": 4,
    "recognition_cache_size": 100,
    "preload_known_faces": True,
    "dynamic_quality_adjustment": True,
    "performance_monitoring": True,
    "target_fps": 30,
    
    # PIN Authentication settings
    "confidence_threshold_for_pin": 0.45,
    "pin_timeout": 15,
    "pin_allowed_attempts": 3,
    
    # Advanced Recognition settings - optimized
    "min_recognition_threshold": 0.50,
    "confident_recognition_threshold": 0.55,
    "use_gpu_if_available": True,
    "adaptive_processing": True,
    "max_parallel_recognitions": 4,
    "face_tracking_enabled": True,
    "tracking_quality_threshold": 5,
    "max_tracking_age": 20,
    
    # New performance settings
    "batch_processing_size": 8,
    "use_fast_detector": True,
    "roi_optimization": True,
    "memory_optimization": True,
}

# Motivational quotes based on BINUS Values and IB Learner Profile

MOTIVATIONAL_QUOTES = [

"Strive for excellence.",

"Embrace innovation",

"Persevere daily.",

"Grow every day.",

"Be honest.",

"Respect others.",

"Stay curious.",

"Think deeply.",

"Be creative.",

"Communicate well.",

"Act with integrity.",

"Stay open-minded.",

"Care for others.",

"Take risks.",

"Stay balanced.",

"Reflect often",

]

# Global variables
attendance = {}
thank_you_message = {
    'active': False,
    'name': '',
    'time': 0,
    'duration': 3.0,
    'quote': '',
    'status': ''
}
facial_landmark_predictor = None

# PIN authentication globals
pin_verification_mode = {
    'active': False,
    'name': '',
    'class_name': '',
    'pin': '',
    'correct_pin': '',
    'start_time': 0,
    'attempts': 0,
    'input_pin': '',
    'error_message': ''
}

# Dictionary to store user PINs
user_pins = {}

# Performance monitoring globals
performance_metrics = {
    'frame_times': deque(maxlen=30),
    'detection_times': deque(maxlen=30),
    'recognition_times': deque(maxlen=30),
    'total_faces_detected': 0,
    'total_faces_recognized': 0,
    'cache_hits': 0,
    'cache_misses': 0
}

# Recognition cache for fast lookups
recognition_cache = {}
cache_lock = threading.Lock()

# Preloaded face data for faster access
preloaded_face_data = {
    'encodings': None,
    'names': None,
    'classes': None,
    'loaded': False
}

def initialize_landmark_predictor():
    """Initialize the facial landmark predictor using face-alignment with HRNet."""
    global facial_landmark_predictor
    
    if not FACE_ALIGNMENT_AVAILABLE:
        print("⚠️ face-alignment package not available. Landmarks disabled.")
        CONFIG["show_landmarks"] = False
        return False
    
    print("🔄 Loading HRNet facial landmark predictor...")
    try:
        if CONFIG["first_run_warning"]:
            print("⚠️ Note: On first run, face-alignment will download model weights (~100MB)")
            print("⚠️ This requires internet connection and may take a few minutes")
        
        # Initialize face alignment with HRNet backbone
        facial_landmark_predictor = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=CONFIG["device"],
            flip_input=False
        )
        print("✅ HRNet facial landmark predictor loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading HRNet: {str(e)}")
        print("⚠️ Continuing without facial landmarks detection.")
        CONFIG["show_landmarks"] = False
        return False


def load_user_pins():
    """Load user PINs from Firebase Storage."""
    print("🔄 Loading user PINs from Firebase...")
    pins = {}
    
    bucket = storage.bucket()
    dataset_path = "face_dataset"
    
    blobs = list(bucket.list_blobs(prefix=dataset_path + "/"))
    
    for blob in blobs:
        if blob.name.endswith('/pin.txt'):
            path_parts = blob.name.split('/')
            if len(path_parts) >= 4:
                class_name = path_parts[1]
                name = path_parts[2]
                
                try:
                    pin_content = blob.download_as_text()
                    user_key = f"{class_name}/{name}"
                    pins[user_key] = pin_content.strip()
                    print(f"✓ Loaded PIN for {name} in class {class_name}")
                except Exception as e:
                    print(f"⚠️ Error loading PIN for {name}: {str(e)}")
    
    print(f"✅ Loaded {len(pins)} user PINs from Firebase")
    return pins


def display_pin_pad(frame, person_name, class_name):
    """Display a sleek, modern PIN entry UI."""
    global pin_verification_mode
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    
    # Set the opacity of the overlay
    alpha = 0.85
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Main PIN pad container
    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    box_width = 400
    box_height = 500
    
    start_x = center_x - box_width // 2
    start_y = center_y - box_height // 2
    end_x = center_x + box_width // 2
    end_y = center_y + box_height // 2
    
    # Draw main container with rounded corners
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (50, 50, 50), -1)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (100, 100, 100), 2)
    
    # Title area
    cv2.rectangle(frame, (start_x, start_y), (end_x, start_y + 60), (0, 120, 200), -1)
    cv2.putText(frame, "PIN Verification", (start_x + 20, start_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display name
    display_text = f"{person_name}"
    if class_name:
        display_text += f" ({class_name})"
    
    cv2.putText(frame, display_text, (start_x + 20, start_y + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # PIN display area
    pin_display_y = start_y + 130
    cv2.rectangle(frame, (start_x + 50, pin_display_y - 30), 
                 (end_x - 50, pin_display_y + 10), (30, 30, 30), -1)
    cv2.rectangle(frame, (start_x + 50, pin_display_y - 30), 
                 (end_x - 50, pin_display_y + 10), (100, 100, 100), 1)
    
    # Show masked PIN
    masked_pin = "-" * len(pin_verification_mode['input_pin'])
    cv2.putText(frame, masked_pin, (start_x + 60, pin_display_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Error message if any
    if pin_verification_mode['error_message']:
        cv2.putText(frame, pin_verification_mode['error_message'], 
                   (start_x + 50, pin_display_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # PIN buttons
    buttons = [
        '1', '2', '3',
        '4', '5', '6',
        '7', '8', '9',
        'C', '0', 'V'
    ]
    
    button_width = 80
    button_height = 60
    button_margin = 10
    
    button_start_y = pin_display_y + 60
    
    for i, button in enumerate(buttons):
        row = i // 3
        col = i % 3
        
        button_x = start_x + 50 + col * (button_width + button_margin)
        button_y = button_start_y + row * (button_height + button_margin)
        
        # Button colors
        if button == 'C':
            button_color = (0, 80, 160)
        elif button == 'V':
            button_color = (0, 160, 80)
        else:
            button_color = (80, 80, 80)
            
        # Draw button
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     button_color, -1)
        cv2.rectangle(frame, (button_x, button_y), 
                     (button_x + button_width, button_y + button_height), 
                     (150, 150, 150), 1)
        
        # Button text
        text_x = button_x + button_width // 2 - 10
        text_y = button_y + button_height // 2 + 10
        cv2.putText(frame, button, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Instructions and timeout
    time_left = CONFIG["pin_timeout"] - (time.time() - pin_verification_mode['start_time'])
    time_text = f"Time remaining: {int(time_left)}s"
    cv2.putText(frame, time_text, (start_x + 50, end_y - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Attempts: {pin_verification_mode['attempts'] + 1}/{CONFIG['pin_allowed_attempts']}", 
                (start_x + 50, end_y - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame


def handle_pin_input(key):
    """Process keyboard input for PIN verification mode."""
    global pin_verification_mode
    
    # Handle numeric keys (0-9)
    if ord('0') <= key <= ord('9'):
        if len(pin_verification_mode['input_pin']) < 6:  # Limit PIN length to 6 digits
            pin_verification_mode['input_pin'] += chr(key)
    
    # Handle backspace/delete/clear (multiple options for flexibility)
    elif key in [ord('c'), ord('C'), 8, 127]:
        pin_verification_mode['input_pin'] = ""
        pin_verification_mode['error_message'] = ""
    
    # Handle enter/return for verification
    elif key in [13, ord('v'), ord('V')]:
        verify_pin()


def verify_pin():
    """Verify the entered PIN."""
    global pin_verification_mode
    
    if pin_verification_mode['input_pin'] == pin_verification_mode['correct_pin']:
        # PIN is correct, log attendance
        log_attendance(pin_verification_mode['name'], pin_verification_mode['class_name'])
        # Reset PIN mode
        pin_verification_mode['active'] = False
    else:
        # Incorrect PIN
        pin_verification_mode['attempts'] += 1
        pin_verification_mode['input_pin'] = ""
        
        if pin_verification_mode['attempts'] >= CONFIG['pin_allowed_attempts']:
            pin_verification_mode['error_message'] = "Too many incorrect attempts. Try again later."
            pin_verification_mode['active'] = False
        else:
            pin_verification_mode['error_message'] = f"Incorrect PIN. Try again. ({pin_verification_mode['attempts']}/{CONFIG['pin_allowed_attempts']})"


def handle_mouse_click(event, x, y, flags, param):
    """Handle mouse clicks for PIN pad UI."""
    global pin_verification_mode
    
    if not pin_verification_mode['active'] or event != cv2.EVENT_LBUTTONDOWN:
        return
        
    frame_width, frame_height = param
    center_x = frame_width // 2
    center_y = frame_height // 2
    box_width = 400
    box_height = 500
    
    start_x = center_x - box_width // 2
    start_y = center_y - box_height // 2
    
    button_width = 80
    button_height = 60
    button_margin = 10
    pin_display_y = start_y + 130
    button_start_y = pin_display_y + 60
    
    buttons = [
        '1', '2', '3',
        '4', '5', '6',
        '7', '8', '9',
        'C', '0', 'V'
    ]
    
    for i, button in enumerate(buttons):
        row = i // 3
        col = i % 3
        
        button_x = start_x + 50 + col * (button_width + button_margin)
        button_y = button_start_y + row * (button_height + button_margin)
        
        # Check if click is inside this button
        if (button_x <= x <= button_x + button_width and 
            button_y <= y <= button_y + button_height):
            
            # Handle numeric buttons
            if button in '0123456789' and len(pin_verification_mode['input_pin']) < 6:
                pin_verification_mode['input_pin'] += button
                pin_verification_mode['error_message'] = ""
                
            # Handle clear button
            elif button == 'C':
                pin_verification_mode['input_pin'] = ""
                pin_verification_mode['error_message'] = ""
                
            # Handle verify button
            elif button == 'V':
                verify_pin()
                
            break


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
        print(f"👋 Thank you, {name} from {class_name}! ({status}).")
    else:
        print(f"👋 Thank you, {name}! ({status}).")
    
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
                student_images = [blob for blob in blobs if blob.name.startswith(student_path) and blob.name.split('/')[-1] and 
                                  not blob.name.endswith('/pin.txt')]
                
                for blob in student_images:
                    image_name = blob.name.split("/")[-1]
                    if not image_name:
                        continue
                    
                    print(f"Processing image: {blob.name}")
                    
                    try:
                        # Download image directly to memory instead of file
                        image_bytes = blob.download_as_bytes()
                        
                        # Convert bytes to numpy array
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is None:
                            print(f"⚠️ Warning: Unable to decode {image_name}, skipping.")
                            continue

                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        face_locations = face_recognition.face_locations(rgb)
                        if len(face_locations) > 0:
                            encodings = face_recognition.face_encodings(rgb, face_locations)
                            student_encodings.extend(encodings)
                            print(f"✓ Found {len(face_locations)} faces in {image_name}")
                        else:
                            print(f"⚠️ No faces detected in {image_name}")
                            
                    except Exception as e:
                        print(f"❌ Error processing {image_name}: {str(e)}")
                        continue
                        
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
        cv2.putText(frame, f"You are {thank_you_message['status']}", 
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
    """Draw facial landmarks on the detected face using HRNet."""
    if not CONFIG["show_landmarks"] or not CONFIG["enhanced_facial_recognition"] or facial_landmark_predictor is None:
        return
        
    left, top, right, bottom = face_coords
    try:
        # Safety check for region dimensions
        if right <= left or bottom <= top:
            return
            
        # Extract face region with some margin
        face_region = frame[max(0, top-30):min(frame.shape[0], bottom+30), 
                            max(0, left-30):min(frame.shape[1], right+30)]
        
        if face_region.size == 0 or face_region.shape[0] <= 0 or face_region.shape[1] <= 0:
            return
            
        # Predict landmarks using HRNet
        preds = facial_landmark_predictor.get_landmarks_from_image(face_region)
        
        if preds and len(preds) > 0:
            landmarks = preds[0]
            
            # Adjust coordinates to the original frame
            x_offset = max(0, left-30)
            y_offset = max(0, top-30)
            
            # Draw landmarks
            for point in landmarks:
                x, y = int(point[0] + x_offset), int(point[1] + y_offset)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
    except Exception as e:
        # Silent fail for landmark drawing failures
        pass


# Class for face tracking to improve performance
class FaceTracker:
    def __init__(self, max_disappeared=30, min_quality=7):
        """
        Initialize a face tracker
        
        Args:
            max_disappeared: Maximum number of frames a face can disappear before being removed
            min_quality: Minimum quality score to consider a track valid
        """
        self.next_id = 0
        self.tracks = {}  # Dictionary of active face tracks
        self.disappeared = {}  # Count of frames since last detection
        self.max_disappeared = max_disappeared
        self.min_quality = min_quality
        
    def register(self, face_rect, face_encoding, name="Unknown", class_name="", confidence=0.0):
        """
        Register a new face with the tracker
        """
        track_id = self.next_id
        self.tracks[track_id] = {
            "rect": face_rect,
            "encoding": face_encoding,
            "name": name,
            "class_name": class_name,
            "confidence": confidence,
            "quality": 10  # Initial quality score
        }
        self.disappeared[track_id] = 0
        self.next_id += 1
        return track_id
        
    def update(self, face_locations, face_encodings, face_names=None, face_classes=None, face_confidences=None):
        """
        Update tracker with new detections
        """
        if face_names is None:
            face_names = ["Unknown"] * len(face_locations)
        if face_classes is None:
            face_classes = [""] * len(face_locations)
        if face_confidences is None:
            face_confidences = [0.0] * len(face_locations)
            
        # If no faces detected, mark all tracks as disappeared
        if len(face_locations) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
            return self.tracks
        
        # If no existing tracks, register all faces
        if len(self.tracks) == 0:
            for i, (rect, encoding, name, class_name, confidence) in enumerate(
                zip(face_locations, face_encodings, face_names, face_classes, face_confidences)):
                self.register(rect, encoding, name, class_name, confidence)
            return self.tracks
            
        # Match new detections with existing tracks
        track_ids = list(self.tracks.keys())
        track_encodings = [self.tracks[tid]["encoding"] for tid in track_ids]
        
        # Map each face to the best matching track
        matched_tracks = {}
        unmatched_detections = list(range(len(face_locations)))
        
        for i, face_encoding in enumerate(face_encodings):
            if len(track_encodings) == 0:  # Skip if no track encodings available
                continue
                
            # Calculate distances to all tracks
            face_distances = face_recognition.face_distance(track_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Consider it a match if distance is below threshold
            if min_distance < 0.6:  # This threshold determines how strict the tracking matching is
                track_id = track_ids[best_match_index]
                matched_tracks[track_id] = i
                if i in unmatched_detections:
                    unmatched_detections.remove(i)
                    
                # Update track with new information
                self.tracks[track_id]["rect"] = face_locations[i]
                
                # Only update name if the new detection has a known name with good confidence
                if face_names[i] != "Unknown" and face_confidences[i] > self.tracks[track_id]["confidence"]:
                    self.tracks[track_id]["name"] = face_names[i]
                    self.tracks[track_id]["class_name"] = face_classes[i]
                    self.tracks[track_id]["confidence"] = face_confidences[i]
                    
                # Increase quality score for matched tracks (capped at 10)
                self.tracks[track_id]["quality"] = min(10, self.tracks[track_id]["quality"] + 1)
                self.disappeared[track_id] = 0
        
        # Handle tracks that weren't matched
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.disappeared[track_id] += 1
                # Decrease quality score
                self.tracks[track_id]["quality"] = max(0, self.tracks[track_id]["quality"] - 1)
                
                if self.disappeared[track_id] > self.max_disappeared or self.tracks[track_id]["quality"] < self.min_quality:
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
        
        # Register new detections that didn't match any track
        for i in unmatched_detections:
            self.register(face_locations[i], face_encodings[i], 
                         face_names[i], face_classes[i], face_confidences[i])
        
        return self.tracks

def initialize_optimized_detectors():
    """Initialize optimized face detectors based on available hardware."""
    detectors = {}
    
    # Initialize OpenCV cascade (fastest, least accurate)
    opencv_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not opencv_detector.empty():
        detectors['opencv'] = opencv_detector
        print("✅ OpenCV cascade detector loaded")
    
    # Initialize dlib detector (balanced speed/accuracy)
    if DLIB_AVAILABLE and CONFIG["use_dlib_detector"]:
        try:
            dlib_detector = dlib.get_frontal_face_detector()
            detectors['dlib'] = dlib_detector
            print("✅ Dlib frontal face detector loaded")
        except Exception as e:
            print(f"⚠️ Failed to load dlib detector: {e}")
    
    return detectors

def fast_face_detection(frame, detectors):
    """Ultra-fast face detection using multiple optimized detectors."""
    start_time = time.time()
    
    faces = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try dlib first (fastest for face detection)
    if 'dlib' in detectors and CONFIG["use_dlib_detector"]:
        try:
            dlib_faces = detectors['dlib'](gray)
            for face in dlib_faces:
                faces.append((face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()))
        except Exception:
            pass
    
    # Fallback to OpenCV if no faces found or dlib unavailable
    if not faces and 'opencv' in detectors:
        opencv_faces = detectors['opencv'].detectMultiScale(
            gray, 
            scaleFactor=CONFIG["detector_scale_factor"],
            minNeighbors=CONFIG["detector_min_neighbors"],
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces.extend(opencv_faces)
    
    detection_time = time.time() - start_time
    performance_metrics['detection_times'].append(detection_time)
    performance_metrics['total_faces_detected'] += len(faces)
    
    return faces

def preload_face_encodings():
    """Preload all face encodings for faster access."""
    global preloaded_face_data
    
    if preloaded_face_data['loaded']:
        return
    
    print("🚀 Preloading face encodings for maximum performance...")
    
    encodings, names, classes = load_face_encodings()
    
    # Convert to optimized numpy arrays
    if encodings:
        preloaded_face_data['encodings'] = np.array(encodings, dtype=np.float32)
        preloaded_face_data['names'] = names
        preloaded_face_data['classes'] = classes
        preloaded_face_data['loaded'] = True
        
        print(f"✅ Preloaded {len(encodings)} face encodings")
    else:
        print("⚠️ No face encodings to preload")

def optimized_face_recognition(face_encoding, use_cache=True):
    """Ultra-fast face recognition with caching and GPU acceleration."""
    start_time = time.time()
    
    # Check cache first
    encoding_hash = hash(face_encoding.tobytes()) if use_cache else None
    
    if use_cache and encoding_hash in recognition_cache:
        with cache_lock:
            result = recognition_cache[encoding_hash]
            performance_metrics['cache_hits'] += 1
            return result
    
    # Perform recognition
    if not preloaded_face_data['loaded']:
        preload_face_encodings()
    
    name = "Unknown"
    class_name = ""
    confidence = 0.0
    
    if preloaded_face_data['encodings'] is not None and len(preloaded_face_data['encodings']) > 0:
        # Use GPU acceleration if available
        if GPU_AVAILABLE and CONFIG["use_gpu_acceleration"]:
            try:
                # Move data to GPU for faster computation
                gpu_face_encoding = cp.asarray(face_encoding)
                gpu_known_encodings = cp.asarray(preloaded_face_data['encodings'])
                
                # Compute distances on GPU
                distances = cp.linalg.norm(gpu_known_encodings - gpu_face_encoding, axis=1)
                distances = cp.asnumpy(distances)  # Move back to CPU
            except Exception:
                # Fallback to CPU computation
                distances = np.linalg.norm(preloaded_face_data['encodings'] - face_encoding, axis=1)
        else:
            # CPU computation with optimized numpy operations
            distances = np.linalg.norm(preloaded_face_data['encodings'] - face_encoding, axis=1)
        
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            confidence = max(0, 1 - best_distance)
            
            if confidence >= CONFIG["min_recognition_threshold"]:
                name = preloaded_face_data['names'][best_match_index]
                class_name = preloaded_face_data['classes'][best_match_index]
    
    result = (name, class_name, confidence)
    
    # Cache the result
    if use_cache and encoding_hash:
        with cache_lock:
            recognition_cache[encoding_hash] = result
            # Limit cache size
            if len(recognition_cache) > CONFIG["recognition_cache_size"]:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(recognition_cache))
                del recognition_cache[oldest_key]
        performance_metrics['cache_misses'] += 1
    
    recognition_time = time.time() - start_time
    performance_metrics['recognition_times'].append(recognition_time)
    performance_metrics['total_faces_recognized'] += 1
    
    return result

def batch_face_encoding(rgb_frame, face_locations, batch_size=None):
    """Process multiple face encodings in optimized batches."""
    if not batch_size:
        batch_size = CONFIG["batch_processing_size"]
    
    if not face_locations:
        return []
    
    # Process in batches for better memory usage and speed
    all_encodings = []
    
    for i in range(0, len(face_locations), batch_size):
        batch_locations = face_locations[i:i + batch_size]
        
        try:
            batch_encodings = face_recognition.face_encodings(rgb_frame, batch_locations)
            all_encodings.extend(batch_encodings)
        except Exception as e:
            print(f"⚠️ Error in batch encoding: {e}")
            # Fallback to individual processing
            for location in batch_locations:
                try:
                    encoding = face_recognition.face_encodings(rgb_frame, [location])
                    if encoding:
                        all_encodings.append(encoding[0])
                except Exception:
                    continue
    
    return all_encodings

def adaptive_quality_control(fps, target_fps=None):
    """Dynamically adjust processing parameters based on performance."""
    if not target_fps:
        target_fps = CONFIG["target_fps"]
    
    if not CONFIG["dynamic_quality_adjustment"]:
        return
    
    # Adjust frame resize factor based on FPS
    if fps < target_fps * 0.7:  # If FPS is significantly below target
        CONFIG["frame_resize"] = max(0.15, CONFIG["frame_resize"] - 0.05)
        CONFIG["skip_frames"] = min(5, CONFIG["skip_frames"] + 1)
        CONFIG["batch_processing_size"] = max(2, CONFIG["batch_processing_size"] - 1)
    elif fps > target_fps * 1.2:  # If FPS is significantly above target
        CONFIG["frame_resize"] = min(0.5, CONFIG["frame_resize"] + 0.05)
        CONFIG["skip_frames"] = max(1, CONFIG["skip_frames"] - 1)
        CONFIG["batch_processing_size"] = min(12, CONFIG["batch_processing_size"] + 1)

def display_performance_metrics(frame):
    """Display real-time performance metrics on frame."""
    if not CONFIG["performance_monitoring"]:
        return frame
    
    y_offset = 60
    
    # Calculate averages
    avg_frame_time = np.mean(performance_metrics['frame_times']) if performance_metrics['frame_times'] else 0
    avg_detection_time = np.mean(performance_metrics['detection_times']) if performance_metrics['detection_times'] else 0
    avg_recognition_time = np.mean(performance_metrics['recognition_times']) if performance_metrics['recognition_times'] else 0
    
    # Cache hit rate
    total_cache_requests = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
    cache_hit_rate = (performance_metrics['cache_hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
    
    metrics = [
        f"Frame: {avg_frame_time*1000:.1f}ms",
        f"Detection: {avg_detection_time*1000:.1f}ms", 
        f"Recognition: {avg_recognition_time*1000:.1f}ms",
        f"Cache Hit Rate: {cache_hit_rate:.1f}%",
        f"Total Faces: {performance_metrics['total_faces_detected']}",
        f"Recognized: {performance_metrics['total_faces_recognized']}"
    ]
    
    for i, metric in enumerate(metrics):
        cv2.putText(frame, metric, (10, y_offset + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame

# Enhanced Face Tracker with performance optimizations
class OptimizedFaceTracker:
    def __init__(self, max_disappeared=20, min_quality=3):
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.min_quality = min_quality
        self.frame_count = 0
        
    def register(self, face_rect, face_encoding, name="Unknown", class_name="", confidence=0.0):
        track_id = self.next_id
        self.tracks[track_id] = {
            "rect": face_rect,
            "encoding": face_encoding,
            "name": name,
            "class_name": class_name,
            "confidence": confidence,
            "quality": 8,
            "last_update": self.frame_count
        }
        self.disappeared[track_id] = 0
        self.next_id += 1
        return track_id
    
    def update(self, face_locations, face_encodings, face_names=None, face_classes=None, face_confidences=None):
        self.frame_count += 1
        
        # Use optimized defaults
        if face_names is None:
            face_names = ["Unknown"] * len(face_locations)
        if face_classes is None:
            face_classes = [""] * len(face_locations)
        if face_confidences is None:
            face_confidences = [0.0] * len(face_locations)
        
        # Early return for no faces
        if len(face_locations) == 0:
            # Age out disappeared tracks
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
            return self.tracks
        
        # Register all faces if no existing tracks
        if len(self.tracks) == 0:
            for i, (rect, encoding, name, class_name, confidence) in enumerate(
                zip(face_locations, face_encodings, face_names, face_classes, face_confidences)):
                self.register(rect, encoding, name, class_name, confidence)
            return self.tracks
        
        # Fast matching using vectorized operations
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0 and len(face_encodings) > 0:
            track_encodings = np.array([self.tracks[tid]["encoding"] for tid in track_ids])
            face_encodings_array = np.array(face_encodings)
            
            # Compute all pairwise distances at once
            distances_matrix = np.linalg.norm(
                track_encodings[:, np.newaxis] - face_encodings_array[np.newaxis, :], 
                axis=2
            )
            
            # Find best matches
            matched_tracks = {}
            unmatched_detections = set(range(len(face_locations)))
            
            # Use greedy approach for speed
            while distances_matrix.size > 0 and unmatched_detections:
                # Find minimum distance
                min_idx = np.unravel_index(np.argmin(distances_matrix), distances_matrix.shape)
                track_idx, face_idx = min_idx
                min_distance = distances_matrix[min_idx]
                
                if min_distance < 0.5:  # Threshold for matching
                    track_id = track_ids[track_idx]
                    matched_tracks[track_id] = face_idx
                    unmatched_detections.discard(face_idx)
                    
                    # Update track
                    self.tracks[track_id]["rect"] = face_locations[face_idx]
                    if face_names[face_idx] != "Unknown" and face_confidences[face_idx] > self.tracks[track_id]["confidence"]:
                        self.tracks[track_id]["name"] = face_names[face_idx]
                        self.tracks[track_id]["class_name"] = face_classes[face_idx]
                        self.tracks[track_id]["confidence"] = face_confidences[face_idx]
                    
                    self.tracks[track_id]["quality"] = min(10, self.tracks[track_id]["quality"] + 2)
                    self.tracks[track_id]["last_update"] = self.frame_count
                    self.disappeared[track_id] = 0
                    
                    # Remove matched elements from consideration
                    distances_matrix = np.delete(distances_matrix, track_idx, axis=0)
                    distances_matrix = np.delete(distances_matrix, face_idx, axis=1)
                    track_ids.pop(track_idx)
                else:
                    break  # No more good matches
        
        # Handle unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.disappeared[track_id] += 1
                self.tracks[track_id]["quality"] = max(0, self.tracks[track_id]["quality"] - 1)
                
                if (self.disappeared[track_id] > self.max_disappeared or 
                    self.tracks[track_id]["quality"] < self.min_quality):
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
        
        # Register unmatched detections
        for face_idx in unmatched_detections:
            self.register(face_locations[face_idx], face_encodings[face_idx],
                         face_names[face_idx], face_classes[face_idx], face_confidences[face_idx])
        
        return self.tracks

def main():
    """Main function with optimized performance pipeline."""
    global pin_verification_mode
    
    print("🚀 Starting High-Performance Facial Recognition System...")
    
    # Initialize optimized components
    detectors = initialize_optimized_detectors()
    
    if CONFIG["preload_known_faces"]:
        preload_face_encodings()
    
    # Load PINs
    global user_pins
    user_pins = load_user_pins()
    
    # Initialize camera with optimized settings
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    # Optimize camera settings for performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("🎥 High-performance camera initialized.")
    
    # Set up optimized window
    cv2.namedWindow('High-Speed Face Recognition', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('High-Speed Face Recognition', handle_mouse_click,
                        (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # Initialize optimized face tracker
    face_tracker = OptimizedFaceTracker(
        max_disappeared=CONFIG["max_tracking_age"],
        min_quality=CONFIG["tracking_quality_threshold"]
    )
    
    # Performance monitoring
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Thread pool for parallel processing
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONFIG["encoding_threads"]
    )
    
    print("✅ System ready! Ultra-fast facial recognition active.")
    
    try:
        while True:
            frame_start_time = time.time()
            
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, CONFIG["flip_camera"])
            
            # Handle PIN verification mode
            if pin_verification_mode['active']:
                if time.time() - pin_verification_mode['start_time'] > CONFIG["pin_timeout"]:
                    pin_verification_mode['active'] = False
                    print("⏱️ PIN verification timed out")
                
                frame = display_pin_pad(frame, pin_verification_mode['name'], 
                                       pin_verification_mode['class_name'])
            else:
                # High-speed face recognition mode
                process_this_frame = frame_count % CONFIG["skip_frames"] == 0
                frame_count += 1
                
                # Calculate FPS
                if time.time() - fps_start_time >= 1.0:
                    fps = frame_count / (time.time() - fps_start_time)
                    frame_count = 0
                    fps_start_time = time.time()
                    
                    # Adaptive quality control
                    adaptive_quality_control(fps)
                
                # Process frame for face recognition
                active_faces = {}
                
                if process_this_frame:
                    # Resize frame for processing
                    small_frame = cv2.resize(frame, (0, 0), 
                                           fx=CONFIG["frame_resize"], 
                                           fy=CONFIG["frame_resize"])
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Fast face detection
                    face_locations = fast_face_detection(small_frame, detectors)
                    
                    if face_locations:
                        # Convert to face_recognition format
                        face_recognition_locations = []
                        for (x, y, w, h) in face_locations:
                            # Convert from (x,y,w,h) to (top, right, bottom, left)
                            top, right, bottom, left = y, x + w, y + h, x
                            face_recognition_locations.append((top, right, bottom, left))
                        
                        # Batch face encoding
                        face_encodings = batch_face_encoding(rgb_small_frame, face_recognition_locations)
                        
                        if face_encodings:
                            # Parallel face recognition
                            if len(face_encodings) <= CONFIG["max_parallel_recognitions"]:
                                futures = [
                                    executor.submit(optimized_face_recognition, encoding)
                                    for encoding in face_encodings
                                ]
                                
                                recognition_results = []
                                for future in concurrent.futures.as_completed(futures):
                                    recognition_results.append(future.result())
                            else:
                                # Sequential processing for many faces
                                recognition_results = [
                                    optimized_face_recognition(encoding)
                                    for encoding in face_encodings
                                ]
                        
                        # Extract results
                        face_names = [result[0] for result in recognition_results]
                        face_classes = [result[1] for result in recognition_results]
                        face_confidences = [result[2] for result in recognition_results]
                        
                        # Handle attendance logging and PIN verification
                        for i, (name, class_name, confidence) in enumerate(recognition_results):
                            if name != "Unknown" and confidence >= CONFIG["min_recognition_threshold"]:
                                attendance_key = f"{class_name}/{name}" if class_name else name
                                
                                if attendance_key not in attendance:
                                    if confidence >= CONFIG["confident_recognition_threshold"]:
                                        # High confidence - log directly
                                        if log_attendance(name, class_name):
                                            print(f"⚡ Fast attendance logged: {name}")
                                    else:
                                        # Lower confidence - require PIN
                                        user_key = f"{class_name}/{name}"
                                        if user_key in user_pins:
                                            pin_verification_mode = {
                                                'active': True,
                                                'name': name,
                                                'class_name': class_name,
                                                'correct_pin': user_pins[user_key],
                                                'start_time': time.time(),
                                                'attempts': 0,
                                                'input_pin': '',
                                                'error_message': ''
                                            }
                        
                        # Update face tracker
                        active_faces = face_tracker.update(
                            face_recognition_locations, face_encodings, 
                            face_names, face_classes, face_confidences
                        )
                else:
                    # Use existing tracks without new detection
                    active_faces = face_tracker.tracks
                
                # Render faces with optimized drawing
                scale = 1.0 / CONFIG["frame_resize"]
                for face_id, face_data in active_faces.items():
                    top, right, bottom, left = face_data["rect"]
                    
                    # Scale coordinates back to original frame size
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)
                    left = int(left * scale)
                    
                    name = face_data["name"]
                    class_name = face_data.get("class_name", "")
                    confidence = face_data.get("confidence", 0.0)
                    
                    # Choose color based on recognition status
                    if name == "Unknown":
                        color = (0, 0, 255)  # Red
                        label = f"Unknown ({confidence:.2f})"
                    else:
                        attendance_key = f"{class_name}/{name}" if class_name else name
                        if attendance_key in attendance:
                            color = (255, 165, 0)  # Orange - already logged
                            label = f"{name} - Logged"
                        else:
                            color = (0, 255, 0)  # Green - new detection
                            label = f"{name} ({confidence:.2f})"
                        
                        if class_name and name != "Unknown":
                            label = f"{name} ({class_name}) ({confidence:.2f})"
                    
                    # Fast rectangle and text rendering
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display performance metrics
            if CONFIG["display_fps"]:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            frame = display_performance_metrics(frame)
            frame = display_thank_you_message(frame)
            
            # Record frame processing time
            frame_time = time.time() - frame_start_time
            performance_metrics['frame_times'].append(frame_time)
            
            cv2.imshow('High-Speed Face Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif pin_verification_mode['active']:
                handle_pin_input(key)
                
    except KeyboardInterrupt:
        print("\n⚡ System stopped by user")
    finally:
        # Cleanup
        executor.shutdown(wait=False)
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        if CONFIG["performance_monitoring"]:
            print("\n📊 Performance Summary:")
            print(f"Total faces detected: {performance_metrics['total_faces_detected']}")
            print(f"Total faces recognized: {performance_metrics['total_faces_recognized']}")
            print(f"Cache hit rate: {(performance_metrics['cache_hits'] / max(1, performance_metrics['cache_hits'] + performance_metrics['cache_misses']) * 100):.1f}%")
            avg_fps = 1.0 / np.mean(performance_metrics['frame_times']) if performance_metrics['frame_times'] else 0
            print(f"Average FPS: {avg_fps:.1f}")
        
        print("🔴 High-Performance Face Recognition System closed.")

if __name__ == "__main__":
    main()
