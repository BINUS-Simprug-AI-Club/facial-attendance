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
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage
import face_alignment  # Add face-alignment package
import concurrent.futures  # For parallel processing

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'facial-attendance-binus.firebasestorage.app'})
db = firestore.client()

# Configuration settings
CONFIG = {
    # Recognition settings
    "tolerance": 0.65,              # Face matching threshold (lower = stricter)
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
    
    # HRNet settings
    "device": "cpu",                # Device for face alignment model: 'cpu' or 'cuda'
    "first_run_warning": True,      # Show warning about first-run model download
    
    # PIN Authentication settings
    "confidence_threshold_for_pin": 0.55,  # Threshold below which PIN authentication is required
    "pin_timeout": 30,              # Seconds to allow PIN entry before timeout
    "pin_allowed_attempts": 3,      # Number of PIN attempts before timeout
    
    # Advanced Recognition settings
    "min_recognition_threshold": 0.5,     # Minimum threshold to consider a possible match
    "confident_recognition_threshold": 0.65, # Threshold for confident recognition without verification
    "use_gpu_if_available": True,         # Try to use GPU acceleration if available
    "adaptive_processing": True,          # Dynamically adjust processing parameters based on load
    "max_parallel_recognitions": 5,       # Maximum number of parallel face recognitions
    "face_tracking_enabled": True,        # Enable face tracking to improve performance
    "tracking_quality_threshold": 7,      # Quality threshold for the tracker
    "max_tracking_age": 30,               # Maximum number of frames to keep tracking without recognition
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


def initialize_landmark_predictor():
    """Initialize the facial landmark predictor using face-alignment with HRNet."""
    global facial_landmark_predictor
    
    print("🔄 Loading HRNet facial landmark predictor...")
    try:
        # First run warning
        if CONFIG["first_run_warning"]:
            print("⚠️ Note: On first run, face-alignment will download model weights (~100MB)")
            print("⚠️ This requires internet connection and may take a few minutes")
        
        # Initialize face alignment with HRNet backbone
        # The enum is "TWO_D" not "_2D"
        facial_landmark_predictor = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,  # Fixed enum value
            device=CONFIG["device"],
            flip_input=False
        )
        print("✅ HRNet facial landmark predictor loaded successfully!")
        return True
    except ImportError:
        print("❌ Error: face-alignment package not installed.")
        print("⚠️ Please install it using: pip install face-alignment")
        CONFIG["show_landmarks"] = False
        return False
    except AttributeError as e:
        # Handle case where enum value might have different name in different versions
        print(f"❌ Error with landmark type: {e}")
        print("⚠️ Trying alternative enum value...")
        try:
            # Try with the most common alternative name
            facial_landmark_predictor = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D if hasattr(face_alignment.LandmarksType, '_2D') else 
                getattr(face_alignment.LandmarksType, '2D', None) if hasattr(face_alignment.LandmarksType, '2D') else
                getattr(face_alignment.LandmarksType, 'D2', None) if hasattr(face_alignment.LandmarksType, 'D2') else
                0,  # Fallback to first enum value (usually 2D)
                device=CONFIG["device"],
                flip_input=False
            )
            print("✅ HRNet facial landmark predictor loaded with alternative enum!")
            return True
        except Exception as e2:
            print(f"❌ Still error loading HRNet: {e2}")
            CONFIG["show_landmarks"] = False
            return False
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("❌ CUDA error: HRNet couldn't use GPU. Switching to CPU...")
            CONFIG["device"] = "cpu"
            return initialize_landmark_predictor()
        print(f"❌ Error loading HRNet: {str(e)}")
        CONFIG["show_landmarks"] = False
        return False
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
    masked_pin = "•" * len(pin_verification_mode['input_pin'])
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
        'C', '0', '✓'
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
        elif button == '✓':
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
    elif key in [ord('\r'), ord('\n'), 13]:
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
        'C', '0', '✓'
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
            elif button == '✓':
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

def main():
    """Main function to run the facial recognition system."""
    initialize_landmark_predictor()
    
    # Load face encodings and PINs
    known_face_encodings, known_face_names, known_face_classes = load_face_encodings()
    global user_pins
    user_pins = load_user_pins()
    
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    print("🎥 Camera initialized. Press 'q' to quit.")
    
    # Set up mouse callback for PIN pad interaction
    cv2.namedWindow('Face Recognition')
    cv2.setMouseCallback('Face Recognition', handle_mouse_click, 
                        (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                         video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # Initialize face tracker if enabled
    face_tracker = None
    if CONFIG["face_tracking_enabled"]:
        face_tracker = FaceTracker(
            max_disappeared=CONFIG["max_tracking_age"],
            min_quality=CONFIG["tracking_quality_threshold"]
        )
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Dynamic parameters
    current_resize_factor = CONFIG["frame_resize"]
    current_skip_frames = CONFIG["skip_frames"]
    
    # For parallel processing
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["max_parallel_recognitions"])
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break
            
        frame = cv2.flip(frame, CONFIG["flip_camera"])
        
        # Check if PIN verification mode is active
        global pin_verification_mode
        if pin_verification_mode['active']:
            # Check for timeout
            if time.time() - pin_verification_mode['start_time'] > CONFIG["pin_timeout"]:
                pin_verification_mode['active'] = False
                print("⏱️ PIN verification timed out")
                
            # Display PIN pad UI
            frame = display_pin_pad(frame, pin_verification_mode['name'], 
                                   pin_verification_mode['class_name'])
        else:
            # Normal face recognition mode
            if CONFIG["corner_display"]:
                corner_size = (int(frame.shape[1] * 0.2), int(frame.shape[0] * 0.2))
                corner_frame = cv2.resize(frame, corner_size)
                
                x_offset = frame.shape[1] - corner_size[0] - 10
                y_offset = 10
                
                cv2.rectangle(frame, (x_offset-2, y_offset-2), 
                             (x_offset + corner_size[0]+2, y_offset + corner_size[1]+2), 
                             (255, 255, 255), 2)
                
                frame[y_offset:y_offset + corner_size[1], x_offset:x_offset + corner_size[0]] = corner_frame
            
            process_this_frame = frame_count % current_skip_frames == 0
            frame_count += 1
            
            if time.time() - fps_start_time >= 1.0:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()
            
            # Use tracking or perform detection
            active_faces = {}
            
            if CONFIG["face_tracking_enabled"] and face_tracker is not None and not process_this_frame:
                # Use existing tracks without performing new detections
                active_faces = face_tracker.tracks
            elif process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=current_resize_factor, fy=current_resize_factor)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                # Adaptive processing: adjust parameters based on number of detected faces
                if CONFIG["adaptive_processing"]:
                    faces_count = len(face_locations)
                    if faces_count > 10:
                        # Heavy load - reduce processing
                        current_resize_factor = max(0.15, CONFIG["frame_resize"] - 0.1)
                        current_skip_frames = CONFIG["skip_frames"] + 2
                    elif faces_count > 5:
                        # Moderate load
                        current_resize_factor = max(0.2, CONFIG["frame_resize"] - 0.05)
                        current_skip_frames = CONFIG["skip_frames"] + 1
                    else:
                        # Light load - return to default
                        current_resize_factor = CONFIG["frame_resize"]
                        current_skip_frames = CONFIG["skip_frames"]
                
                if face_locations:
                    # Process faces in parallel for better performance
                    face_encodings = []
                    if len(face_locations) <= CONFIG["max_parallel_recognitions"]:
                        # For fewer faces, process in parallel
                        futures = [executor.submit(face_recognition.face_encodings, rgb_small_frame, [loc]) 
                                  for loc in face_locations]
                        for future in concurrent.futures.as_completed(futures):
                            result = future.result()
                            if result:
                                face_encodings.append(result[0])
                    else:
                        # For many faces, use batch processing
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    # Process face recognition with improved accuracy checks
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
                                
                                # Double threshold approach for better accuracy
                                if best_match_score >= CONFIG["min_recognition_threshold"]:
                                    # We have a potential match
                                    if best_match_score >= CONFIG["confident_recognition_threshold"]:
                                        # Confident match - mark attendance directly
                                        name = known_face_names[best_match_index]
                                        class_name = known_face_classes[best_match_index]
                                        confidence = best_match_score
                                        
                                        if log_attendance(name, class_name):
                                            print(f"✔️ Attendance marked for {name}" + 
                                                  (f" in class {class_name}" if class_name else ""))
                                    
                                    # For matches below confident threshold but above minimum threshold
                                    elif CONFIG["min_recognition_threshold"] <= best_match_score < CONFIG["confident_recognition_threshold"]:
                                        # Potential match - require verification
                                        name = known_face_names[best_match_index]
                                        class_name = known_face_classes[best_match_index]
                                        confidence = best_match_score
                                        
                                        user_key = f"{class_name}/{name}"
                                        if user_key in user_pins:
                                            # Activate PIN verification mode
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
                                            print(f"🔑 PIN verification required for {name} (confidence: {confidence:.2f})")
                                        else:
                                            # No PIN but too low confidence - mark as Unknown
                                            name = "Unknown"
                                            class_name = ""
                                            print(f"⚠️ No PIN found for potential match {name}, marking as Unknown")
                        
                        face_names.append(name)
                        face_classes.append(class_name)
                        face_confidences.append(confidence)
                    
                    # Update face tracker with new detections
                    if CONFIG["face_tracking_enabled"] and face_tracker is not None:
                        active_faces = face_tracker.update(
                            face_locations, face_encodings, face_names, face_classes, face_confidences
                        )
                    else:
                        # Without tracking, create a simple dictionary of faces
                        active_faces = {
                            i: {
                                "rect": loc,
                                "name": name,
                                "class_name": cls,
                                "confidence": conf
                            } for i, (loc, name, cls, conf) in enumerate(zip(
                                face_locations, face_names, face_classes, face_confidences
                            ))
                        }
            
            # Display all active faces
            for face_id, face_data in active_faces.items():
                if CONFIG["face_tracking_enabled"] and face_tracker is not None:
                    # Scale coordinates from tracking
                    top, right, bottom, left = face_data["rect"]
                    scale = 1.0 / current_resize_factor
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)
                    left = int(left * scale)
                else:
                    # Direct coordinates without tracking
                    top, right, bottom, left = face_data["rect"]
                    scale = 1.0 / current_resize_factor
                    top = int(top * scale)
                    right = int(right * scale)
                    bottom = int(bottom * scale)
                    left = int(left * scale)
                
                name = face_data["name"]
                class_name = face_data.get("class_name", "")
                confidence = face_data.get("confidence", 0.0)
                
                if name == "Unknown" or confidence < CONFIG["min_recognition_threshold"]:
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
                
                # Draw facial landmarks for this face
                draw_facial_landmarks(frame, (left, top, right, bottom))
                
        if CONFIG["display_fps"]:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        frame = display_thank_you_message(frame)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        # Handle PIN input if in PIN verification mode
        if pin_verification_mode['active']:
            handle_pin_input(key)
    
    # Clean up resources
    executor.shutdown()
    video_capture.release()
    cv2.destroyAllWindows()
    print("🔴 Face recognition system closed.")


if __name__ == "__main__":
    main()
