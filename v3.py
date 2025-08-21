# Standard library imports
import os
import pickle
import random
import time
import tempfile
import hashlib
import logging
from datetime import datetime, time as dt_time
import threading
from collections import deque
import queue
import math

# Third-party imports
import cv2 
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore, storage
import concurrent.futures
from dotenv import load_dotenv
# Add SciPy for mathematical operations
import scipy.spatial as spatial

# Try to import face_alignment for landmarks
try:
    import face_alignment
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# Try to import GPU acceleration libraries
try:
    # this is only if you have NVIDIA gpu, then you can install then use cupy
    import cupy as cp # type: ignore
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available, using CPU only")

# Get secret key from .env file
load_dotenv()
secret_key = os.getenv("MY_SECRET_KEY")
print("My secret key is:", secret_key)

# Initialize Firebase
cred = credentials.Certificate("facial-attendance-binus-firebase-adminsdk-fbsvc-663eb05a63.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'facial-attendance-binus.firebasestorage.app'})
db = firestore.client()

# Configuration settings with performance optimizations and best practices
CONFIG = {
    # Recognition settings - optimized for accuracy and security
    "tolerance": 0.4,               # Stricter tolerance for better security
    "frame_resize": 0.6,           # Higher quality for better accuracy
    "skip_frames": 1,               # Process more frames for better tracking
    "enhanced_facial_recognition": True,
    "min_face_size": (50, 50),     # Minimum face size for detection
    "max_face_size": (400, 400),   # Maximum face size for processing
    "face_quality_threshold": 0.4, # Minimum quality score for face processing
    "brightness_threshold": (50, 200), # Acceptable brightness range
    "blur_threshold": 100,          # Minimum sharpness threshold
    
    # Multi-model ensemble settings
    "use_ensemble_models": True,    # Use multiple models for better accuracy
    "ensemble_threshold": 0.7,      # Confidence threshold for ensemble
    "model_weights": {              # Weights for different models
        "cnn": 1.0
    },
    
    # Display settings
    "display_fps": True,
    "show_all_faces": True,
    "flip_camera": 1,
    "corner_display": False,        # Disabled for performance
    "show_landmarks": False,        # Disabled by default for speed
    
    # Attendance settings
    "latest_login_time": "07:30",
    "duplicate_detection_window": 300,  # 5 minutes window to prevent duplicates
    
    # Performance settings
    "device": "cuda" if GPU_AVAILABLE else "cpu",
    "first_run_warning": True,
    "use_gpu_acceleration": GPU_AVAILABLE,
    "use_dlib_detector": DLIB_AVAILABLE,
    "detector_scale_factor": 1.1,  # More conservative for accuracy
    "detector_min_neighbors": 5,    # Higher for better accuracy
    
    # Advanced Performance settings
    "face_detection_threads": 2,
    "encoding_threads": 4,
    "recognition_cache_size": 500,  # Larger cache for better hit rate
    "preload_known_faces": True,
    "dynamic_quality_adjustment": True,
    "performance_monitoring": True,
    "target_fps": 25,               # More realistic target
    
    # PIN Authentication settings
    "confidence_threshold_for_pin": 0.60,  # Higher threshold for PIN mode
    "pin_timeout": 15,
    "pin_allowed_attempts": 3,
    
    # Advanced Recognition settings - optimized for security
    "min_recognition_threshold": 0.60,     # Higher for better security
    "confident_recognition_threshold": 0.61, # Much higher for confident matches
    "use_gpu_if_available": True,
    "adaptive_processing": True,
    "max_parallel_recognitions": 2,        # Conservative for stability
    "face_tracking_enabled": True,
    "tracking_quality_threshold": 15,      # Higher for better tracking
    "max_tracking_age": 20,                # Conservative tracking age
    "recognition_voting_threshold": 5,     # Require multiple confirmations
    
    # Security and anti-spoofing
    "require_multiple_angles": True,       # Require face from multiple angles
    "liveness_detection": True,            # Enable liveness detection
    "texture_analysis": True,              # Analyze face texture for spoofing
    "motion_detection": True,              # Detect natural head motion
    "eye_movement_detection": True,        # Track eye movements
    "depth_analysis": True,                # Analyze face depth (if available)
    
    # Face quality assessment
    "enable_quality_assessment": True,     # Enable comprehensive quality checks
    "symmetry_threshold": 0.8,            # Face symmetry requirement
    "pose_angle_threshold": 30,           # Maximum pose angle in degrees
    "illumination_consistency": True,      # Check lighting consistency
    "resolution_check": True,             # Ensure adequate resolution
    
    # debugging
    "show_face_boxes": True,               # Enable for debugging
    "show_quality_metrics": True,         # Show quality assessment
    "log_recognition_details": True,      # Detailed logging

    # Enhanced high-performance settings
    "batch_processing_size": 8,            # Optimized batch size
    "use_fast_detector": True,
    "roi_optimization": True,
    "memory_optimization": True,
    "cascade_optimization": True,          # Use optimized cascade detection
    "parallel_face_detection": True,      # Parallel detection
    "smart_frame_skipping": True,         # Intelligent frame skipping
    "adaptive_roi": True,                 # Adaptive region of interest
    "temporal_consistency": True,         # Use temporal information
    
    # Anti-spoofing settings - enhanced
    "enable_blink_detection": True,       # Enable blink detection
    "ear_threshold": 0.25,               # Eye aspect ratio threshold for blink
    "min_blinks_required": 3,            # More blinks required for security
    "blink_detection_time": 8,           # Longer time window
    "show_eye_landmarks": False,         # Show eye landmarks for debugging
    "micro_expression_detection": True,   # Detect micro-expressions
    "pulse_detection": True,             # Detect pulse from face
    "challenge_response": True,          # Random challenges (smile, nod, etc.)
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
attendance = {}  # Track attendance to prevent duplicate entries
attendance_timestamps = {}  # Track when people last logged in
thank_you_message = {
    'active': False,
    'name': '',
    'time': 0,
    'duration': 3.0,
    'quote': '',
    'status': ''
}
facial_landmark_predictor = None

# Security and quality assessment globals
face_quality_assessor = None
liveness_detector = None
challenge_response_active = {
    'active': False,
    'challenge': '',
    'start_time': 0,
    'person_name': '',
    'class_name': '',
    'required_action': '',
    'completed': False
}

# Enhanced face tracking with quality metrics
face_quality_history = {}  # Track quality over time for each person

# Security monitoring and logging
security_monitor = {
    'failed_attempts': deque(maxlen=100),
    'suspicious_activity': deque(maxlen=50),
    'recognition_logs': deque(maxlen=1000),
    'system_alerts': deque(maxlen=100)
}

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_recognition_security.log'),
        logging.StreamHandler()
    ]
)
security_logger = logging.getLogger('FacialRecognitionSecurity')

def log_security_event(event_type, details, severity='INFO'):
    """Log security events with structured data."""
    timestamp = datetime.now().isoformat()
    event_data = {
        'timestamp': timestamp,
        'event_type': event_type,
        'details': details,
        'severity': severity
    }
    
    security_monitor['system_alerts'].append(event_data)
    
    if severity == 'WARNING':
        security_logger.warning(f"{event_type}: {details}")
    elif severity == 'ERROR':
        security_logger.error(f"{event_type}: {details}")
    elif severity == 'CRITICAL':
        security_logger.critical(f"{event_type}: {details}")
    else:
        security_logger.info(f"{event_type}: {details}")

def detect_suspicious_activity(face_locations, face_names, face_confidences):
    """Detect patterns that might indicate spoofing or suspicious behavior."""
    current_time = time.time()
    
    # Multiple unknown faces might indicate spoofing attempts
    unknown_count = sum(1 for name in face_names if name == "Unknown")
    if unknown_count > 3:
        log_security_event(
            "MULTIPLE_UNKNOWN_FACES",
            f"Detected {unknown_count} unknown faces simultaneously",
            "WARNING"
        )
    
    # Very low confidence across multiple faces
    if face_confidences and len(face_confidences) > 1:
        avg_confidence = np.mean([c for c in face_confidences if c > 0])
        if avg_confidence < 0.3:
            log_security_event(
                "LOW_CONFIDENCE_PATTERN",
                f"Average confidence {avg_confidence:.2f} across {len(face_confidences)} faces",
                "WARNING"
            )
    
    # Faces appearing and disappearing rapidly
    if hasattr(detect_suspicious_activity, 'last_face_count'):
        face_count_change = abs(len(face_locations) - detect_suspicious_activity.last_face_count)
        if face_count_change > 2:
            log_security_event(
                "RAPID_FACE_CHANGES",
                f"Face count changed by {face_count_change} in one frame",
                "INFO"
            )
    
    detect_suspicious_activity.last_face_count = len(face_locations)

def comprehensive_system_health_check():
    """Perform comprehensive system health and security checks."""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'components': {},
        'security_status': 'OK',
        'performance_status': 'OK',
        'alerts': []
    }
    
    # Check critical components
    health_status['components']['gpu_available'] = GPU_AVAILABLE
    health_status['components']['dlib_available'] = DLIB_AVAILABLE
    health_status['components']['face_alignment_available'] = FACE_ALIGNMENT_AVAILABLE
    health_status['components']['firebase_connected'] = True  # Assume connected if no exception
    
    # Check performance metrics
    if performance_metrics['frame_times']:
        avg_frame_time = np.mean(performance_metrics['frame_times'])
        if avg_frame_time > 0.1:  # More than 100ms per frame
            health_status['performance_status'] = 'DEGRADED'
            health_status['alerts'].append(f"High frame processing time: {avg_frame_time*1000:.1f}ms")
    
    # Check recognition cache health
    total_cache_requests = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
    if total_cache_requests > 0:
        cache_hit_rate = performance_metrics['cache_hits'] / total_cache_requests
        if cache_hit_rate < 0.5:  # Less than 50% hit rate
            health_status['alerts'].append(f"Low cache hit rate: {cache_hit_rate*100:.1f}%")
    
    # Check recent security events
    recent_alerts = [alert for alert in security_monitor['system_alerts'] 
                    if alert['severity'] in ['WARNING', 'ERROR', 'CRITICAL']]
    if len(recent_alerts) > 10:  # Many recent alerts
        health_status['security_status'] = 'ELEVATED'
        health_status['alerts'].append(f"{len(recent_alerts)} recent security alerts")
    
    return health_status

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
    'frame_times': deque(maxlen=50),
    'detection_times': deque(maxlen=50),
    'recognition_times': deque(maxlen=50),
    'encoding_times': deque(maxlen=50),
    'total_faces_detected': 0,
    'total_faces_recognized': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'avg_fps': 0,
    'peak_fps': 0
}

# Global variables for blink detection
blink_detection = {
    'active': False,
    'start_time': 0,
    'blink_count': 0,
    'last_blink_time': 0,
    'ear_values': deque(maxlen=10),  # Store recent EAR values
    'person_name': '',
    'class_name': '',
    'status_message': '',
    'previous_ear': 1.0,  # Initialize with open eye value
}

def initialize_landmark_predictor():
    """Initialize the facial landmark predictor using face-alignment with HRNet."""
    global facial_landmark_predictor
    
    if not FACE_ALIGNMENT_AVAILABLE:
        print("⚠️ face-alignment package not available. Landmarks disabled.")
        CONFIG["show_landmarks"] = False
        CONFIG["enable_blink_detection"] = False
        return False
    
    print("🔄 Loading HRNet facial landmark predictor...")
    try:
        # First run warning
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
        CONFIG["enable_blink_detection"] = False
        return False

class FaceQualityAssessor:
    """Comprehensive face quality assessment for security and accuracy."""
    
    def __init__(self):
        self.blur_threshold = CONFIG.get("blur_threshold", 100)
        self.brightness_range = CONFIG.get("brightness_threshold", (50, 200))
        self.symmetry_threshold = CONFIG.get("symmetry_threshold", 0.8)
        self.pose_threshold = CONFIG.get("pose_angle_threshold", 30)
        
    def assess_face_quality(self, face_image, landmarks=None):
        """
        Comprehensive face quality assessment.
        
        Returns:
            dict: Quality metrics and overall score
        """
        if face_image is None or face_image.size == 0:
            return {"overall_score": 0.0, "reason": "Invalid image"}
        
        quality_metrics = {
            "sharpness": self._assess_sharpness(face_image),
            "brightness": self._assess_brightness(face_image),
            "contrast": self._assess_contrast(face_image),
            "symmetry": self._assess_symmetry(face_image, landmarks),
            "pose_angle": self._assess_pose_angle(landmarks),
            "resolution": self._assess_resolution(face_image),
            "noise_level": self._assess_noise(face_image),
            "overall_score": 0.0
        }
        
        # Calculate weighted overall score
        weights = {
            "sharpness": 0.25,
            "brightness": 0.15,
            "contrast": 0.15,
            "symmetry": 0.15,
            "pose_angle": 0.15,
            "resolution": 0.10,
            "noise_level": 0.05
        }
        
        overall_score = sum(quality_metrics[metric] * weights[metric] 
                           for metric in weights.keys())
        quality_metrics["overall_score"] = overall_score
        
        return quality_metrics
    
    def _assess_sharpness(self, image):
        """Assess image sharpness using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return min(1.0, laplacian_var / self.blur_threshold)
        except Exception:
            return 0.0
    
    def _assess_brightness(self, image):
        """Assess image brightness."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            
            min_bright, max_bright = self.brightness_range
            if min_bright <= mean_brightness <= max_bright:
                # Optimal range
                return 1.0
            elif mean_brightness < min_bright:
                # Too dark
                return max(0.0, mean_brightness / min_bright)
            else:
                # Too bright
                return max(0.0, (255 - mean_brightness) / (255 - max_bright))
        except Exception:
            return 0.0
    
    def _assess_contrast(self, image):
        """Assess image contrast using standard deviation."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = np.std(gray)
            return min(1.0, contrast / 50.0)  # Normalize to 0-1
        except Exception:
            return 0.0
    
    def _assess_symmetry(self, image, landmarks=None):
        """Assess face symmetry."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default score when landmarks unavailable
        
        try:
            # Calculate symmetry based on landmark positions
            left_points = landmarks[0:17]  # Face outline left
            right_points = landmarks[16:0:-1]  # Face outline right (reversed)
            
            # Calculate center line
            nose_tip = landmarks[30]
            center_x = nose_tip[0]
            
            # Measure symmetry by comparing distances from center
            left_distances = [abs(point[0] - center_x) for point in left_points]
            right_distances = [abs(point[0] - center_x) for point in right_points]
            
            # Calculate symmetry score
            symmetry_diffs = [abs(l - r) for l, r in zip(left_distances, right_distances)]
            avg_diff = np.mean(symmetry_diffs)
            
            # Normalize to 0-1 score
            symmetry_score = max(0.0, 1.0 - (avg_diff / 20.0))
            return symmetry_score
        except Exception:
            return 0.5
    
    def _assess_pose_angle(self, landmarks=None):
        """Assess head pose angle."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5  # Default score
        
        try:
            # Calculate pose using key landmarks
            nose_tip = landmarks[30]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            
            # Calculate eye center line angle
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            angle_degrees = abs(np.degrees(angle))
            
            # Score based on how close to frontal the pose is
            pose_score = max(0.0, 1.0 - (angle_degrees / self.pose_threshold))
            return pose_score
        except Exception:
            return 0.5
    
    def _assess_resolution(self, image):
        """Assess if image resolution is adequate."""
        try:
            height, width = image.shape[:2]
            min_size = min(height, width)
            
            if min_size >= CONFIG["min_face_size"][0]:
                return 1.0
            else:
                return max(0.0, min_size / CONFIG["min_face_size"][0])
        except Exception:
            return 0.0
    
    def _assess_noise(self, image):
        """Assess image noise level."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Use gradient magnitude to estimate noise
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # High gradient variance indicates noise
            noise_estimate = np.var(gradient_magnitude)
            noise_score = max(0.0, 1.0 - (noise_estimate / 10000.0))
            return noise_score
        except Exception:
            return 0.5

class LivenessDetector:
    """Advanced liveness detection to prevent spoofing attacks."""
    
    def __init__(self):
        self.texture_patterns = []
        self.motion_history = deque(maxlen=30)
        self.challenge_responses = {
            'blink': False,
            'smile': False,
            'nod': False,
            'turn_left': False,
            'turn_right': False
        }
    
    def detect_liveness(self, face_image, landmarks=None, frame_history=None):
        """
        Comprehensive liveness detection.
        
        Returns:
            dict: Liveness assessment results
        """
        liveness_score = {
            "texture_analysis": self._analyze_texture(face_image),
            "motion_detection": self._detect_motion(frame_history),
            "depth_estimation": self._estimate_depth(face_image, landmarks),
            "micro_expressions": self._detect_micro_expressions(landmarks),
            "pulse_detection": self._detect_pulse(face_image),
            "overall_liveness": 0.0
        }
        
        # Calculate overall liveness score
        weights = {
            "texture_analysis": 0.3,
            "motion_detection": 0.25,
            "depth_estimation": 0.2,
            "micro_expressions": 0.15,
            "pulse_detection": 0.1
        }
        
        overall_score = sum(liveness_score[metric] * weights[metric] 
                           for metric in weights.keys())
        liveness_score["overall_liveness"] = overall_score
        
        return liveness_score
    
    def _analyze_texture(self, face_image):
        """Analyze face texture for signs of spoofing."""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            # Local Binary Pattern for texture analysis
            # Simplified version - in production, use more sophisticated methods
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            texture_variance = np.var(sobel_x) + np.var(sobel_y)
            
            # Real faces have more texture variance than photos/screens
            texture_score = min(1.0, texture_variance / 50000.0)
            return texture_score
        except Exception:
            return 0.5
    
    def _detect_motion(self, frame_history):
        """Detect natural head motion indicative of a live person."""
        if frame_history is None or len(frame_history) < 3:
            return 0.5
        
        try:
            # Calculate motion between consecutive frames
            motion_scores = []
            for i in range(1, len(frame_history)):
                frame1 = frame_history[i-1]
                frame2 = frame_history[i]
                
                if frame1 is not None and frame2 is not None:
                    # Calculate optical flow
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    
                    flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
                    if flow is not None:
                        motion_magnitude = np.mean(np.abs(flow))
                        motion_scores.append(motion_magnitude)
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                # Natural motion should be present but not excessive
                motion_score = min(1.0, avg_motion / 5.0) if avg_motion < 50 else max(0.0, 1.0 - avg_motion / 100.0)
                return motion_score
            
        except Exception:
            pass
        
        return 0.5
    
    def _estimate_depth(self, face_image, landmarks=None):
        """Estimate face depth to detect flat surfaces."""
        if landmarks is None or len(landmarks) < 68:
            return 0.5
        
        try:
            # Use landmarks to estimate 3D structure
            # Calculate nose bridge depth relative to eye level
            nose_tip = landmarks[30]
            nose_bridge = landmarks[27]
            left_eye = landmarks[36]
            right_eye = landmarks[45]
            
            # Estimate depth based on proportions
            eye_line_y = (left_eye[1] + right_eye[1]) / 2
            nose_depth_ratio = abs(nose_bridge[1] - eye_line_y) / abs(nose_tip[1] - eye_line_y)
            
            # Real faces have certain depth proportions
            depth_score = min(1.0, nose_depth_ratio / 0.5)
            return depth_score
        except Exception:
            return 0.5
    
    def _detect_micro_expressions(self, landmarks):
        """Detect subtle micro-expressions that indicate a live person."""
        if landmarks is None:
            return 0.5
        
        try:
            # Store landmarks for temporal analysis
            if hasattr(self, 'previous_landmarks') and self.previous_landmarks is not None:
                # Calculate subtle movements in facial features
                movement_scores = []
                
                # Check eye movements
                for eye_idx in range(36, 48):  # Eye landmark indices
                    if eye_idx < len(landmarks) and eye_idx < len(self.previous_landmarks):
                        movement = np.linalg.norm(
                            np.array(landmarks[eye_idx]) - np.array(self.previous_landmarks[eye_idx])
                        )
                        movement_scores.append(movement)
                
                if movement_scores:
                    avg_movement = np.mean(movement_scores)
                    # Micro-movements should be present but subtle
                    micro_expr_score = min(1.0, avg_movement / 2.0) if avg_movement < 10 else 0.5
                    self.previous_landmarks = landmarks
                    return micro_expr_score
            
            self.previous_landmarks = landmarks
            return 0.5
        except Exception:
            return 0.5
    
    def _detect_pulse(self, face_image):
        """Detect pulse from facial color variations (simplified)."""
        try:
            # Extract forehead region for pulse detection
            height, width = face_image.shape[:2]
            forehead_region = face_image[int(height*0.1):int(height*0.3), 
                                       int(width*0.3):int(width*0.7)]
            
            if forehead_region.size > 0:
                # Calculate average color intensity
                avg_intensity = np.mean(forehead_region)
                
                # Store for temporal analysis
                if hasattr(self, 'pulse_history'):
                    self.pulse_history.append(avg_intensity)
                    if len(self.pulse_history) > 100:
                        self.pulse_history.pop(0)
                    
                    # Look for periodic variations (pulse)
                    if len(self.pulse_history) > 30:
                        pulse_variance = np.var(self.pulse_history)
                        pulse_score = min(1.0, pulse_variance / 10.0)
                        return pulse_score
                else:
                    self.pulse_history = [avg_intensity]
            
            return 0.5
        except Exception:
            return 0.5

# Initialize quality assessment components
face_quality_assessor = FaceQualityAssessor()
liveness_detector = LivenessDetector()

def validate_face_region(image, face_location):
    """
    Validate that a detected face region meets quality standards.
    
    Args:
        image: Input image
        face_location: Face bounding box (top, right, bottom, left)
    
    Returns:
        tuple: (is_valid, quality_score, reason)
    """
    try:
        top, right, bottom, left = face_location
        
        # Basic boundary checks
        if top < 0 or left < 0 or bottom >= image.shape[0] or right >= image.shape[1]:
            return False, 0.0, "Face region out of bounds"
        
        if right <= left or bottom <= top:
            return False, 0.0, "Invalid face dimensions"
        
        # Extract face region
        face_image = image[top:bottom, left:right]
        
        if face_image.size == 0:
            return False, 0.0, "Empty face region"
        
        # Size validation
        face_width = right - left
        face_height = bottom - top
        
        min_size = CONFIG["min_face_size"]
        max_size = CONFIG["max_face_size"]
        
        if face_width < min_size[0] or face_height < min_size[1]:
            return False, 0.2, f"Face too small: {face_width}x{face_height}"
        
        if face_width > max_size[0] or face_height > max_size[1]:
            return False, 0.3, f"Face too large: {face_width}x{face_height}"
        
        # Quality assessment
        quality_metrics = face_quality_assessor.assess_face_quality(face_image)
        overall_quality = quality_metrics["overall_score"]
        
        threshold = CONFIG.get("face_quality_threshold", 0.3)
        
        if overall_quality < threshold:
            reasons = []
            if quality_metrics["sharpness"] < 0.3:
                reasons.append("blurry")
            if quality_metrics["brightness"] < 0.3:
                reasons.append("poor lighting")
            if quality_metrics["contrast"] < 0.3:
                reasons.append("low contrast")
            
            return False, overall_quality, f"Low quality: {', '.join(reasons)}"
        
        return True, overall_quality, "Valid face"
        
    except Exception as e:
        return False, 0.0, f"Validation error: {str(e)}"

def enhanced_face_preprocessing(face_image):
    """
    Apply advanced preprocessing to improve face recognition accuracy.
    
    Args:
        face_image: Input face image
    
    Returns:
        Preprocessed face image
    """
    try:
        if face_image is None or face_image.size == 0:
            return face_image
        
        # Convert to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            # Assume BGR input from OpenCV
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Histogram equalization for lighting normalization
        if len(face_image.shape) == 3:
            # Convert to LAB color space for better lighting correction
            lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            face_image = cv2.equalizeHist(face_image)
        
        # Noise reduction using bilateral filter
        if len(face_image.shape) == 3:
            face_image = cv2.bilateralFilter(face_image, 9, 75, 75)
        else:
            face_image = cv2.bilateralFilter(face_image, 9, 75, 75)
        
        # Sharpening filter
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        face_image = cv2.filter2D(face_image, -1, kernel)
        
        # Ensure proper data type and range
        face_image = np.clip(face_image, 0, 255).astype(np.uint8)
        
        return face_image
        
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        return face_image

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
    """Display a sleek, modern PIN entry UI for self-identification."""
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
    box_width = 450
    box_height = 550
    
    start_x = center_x - box_width // 2
    start_y = center_y - box_height // 2
    end_x = center_x + box_width // 2
    end_y = center_y + box_height // 2
    
    # Draw main container with rounded corners
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (50, 50, 50), -1)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (100, 100, 100), 2)
    
    # Title area
    cv2.rectangle(frame, (start_x, start_y), (end_x, start_y + 60), (0, 120, 200), -1)
    cv2.putText(frame, "Enter YOUR PIN to Identify", (start_x + 20, start_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Instruction text
    cv2.putText(frame, "Enter YOUR PIN to confirm identity:", (start_x + 20, start_y + 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # PIN display area
    pin_display_y = start_y + 150
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
    
    button_start_y = pin_display_y + 80
    
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
    cv2.putText(frame, time_text, (start_x + 50, end_y - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Attempts: {pin_verification_mode['attempts'] + 1}/{CONFIG['pin_allowed_attempts']}", 
                (start_x + 50, end_y - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "C=Clear, V=Verify", 
                (start_x + 50, end_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
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
    """Verify the entered PIN against all users to identify the correct person."""
    global pin_verification_mode
    
    entered_pin = pin_verification_mode['input_pin']
    
    # Search through all user PINs to find a match
    correct_user = None
    correct_class = None
    
    for user_key, stored_pin in user_pins.items():
        if stored_pin == entered_pin:
            # Found matching PIN
            if '/' in user_key:
                correct_class, correct_user = user_key.split('/', 1)
            else:
                correct_user = user_key
                correct_class = ""
            break
    
    if correct_user:
        # PIN matched a known user - use enhanced attendance logging
        attendance_key = f"{correct_class}/{correct_user}" if correct_class else correct_user
        
        # Get stored recognition metrics if available
        confidence = pin_verification_mode.get('confidence', 0.8)  # Higher confidence due to PIN verification
        quality_score = pin_verification_mode.get('quality_score', 0.0)
        security_score = pin_verification_mode.get('security_score', 1.0)  # High security due to PIN
        
        # Enhanced duplicate check
        current_time = time.time()
        should_log = True
        
        if attendance_key in attendance_timestamps:
            last_time = attendance_timestamps[attendance_key]
            time_diff = current_time - last_time
            if time_diff < CONFIG.get("duplicate_detection_window", 300):
                should_log = False
                minutes_left = int((CONFIG.get("duplicate_detection_window", 300) - time_diff) / 60)
                pin_verification_mode['error_message'] = f"{correct_user} already logged in recently! Wait {minutes_left} min."
                pin_verification_mode['input_pin'] = ""
                pin_verification_mode['attempts'] += 1
                return
        
        if should_log:
            # Log attendance with enhanced data
            success = log_attendance(correct_user, correct_class, confidence, quality_score, security_score)
            if success:
                print(f"✅ PIN verified! Enhanced attendance logged for {correct_user}" + 
                      (f" in class {correct_class}" if correct_class else "") +
                      f" (conf: {confidence:.2f}, PIN security)")
                # Reset PIN mode
                pin_verification_mode['active'] = False
            else:
                pin_verification_mode['error_message'] = "Failed to log attendance. Try again."
                pin_verification_mode['input_pin'] = ""
        else:
            pin_verification_mode['error_message'] = f"{correct_user} attendance already recorded!"
            pin_verification_mode['input_pin'] = ""
    else:
        # No matching PIN found
        pin_verification_mode['attempts'] += 1
        pin_verification_mode['input_pin'] = ""
        
        if pin_verification_mode['attempts'] >= CONFIG['pin_allowed_attempts']:
            pin_verification_mode['error_message'] = "Too many incorrect attempts. Access denied."
            pin_verification_mode['active'] = False
            print(f"🚨 Security alert: {pin_verification_mode['attempts']} failed PIN attempts")
        else:
            pin_verification_mode['error_message'] = f"PIN not found. Try again. ({pin_verification_mode['attempts']}/{CONFIG['pin_allowed_attempts']})"


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


def log_attendance(name, class_name=None, confidence=0.0, quality_score=0.0, security_score=0.0):
    """
    Record attendance for a person in Firebase with enhanced security and duplicate prevention.
    
    Args:
        name (str): Name of the person to log
        class_name (str, optional): Class name of the person
        confidence (float): Recognition confidence score
        quality_score (float): Face quality assessment score
        security_score (float): Security/liveness assessment score
        
    Returns:
        bool: True if new attendance was logged, False if already logged or rejected
    """
    attendance_key = name
    if class_name:
        attendance_key = f"{class_name}/{name}" 
    
    current_time = datetime.now()
    current_timestamp = current_time.timestamp()
    
    # Check for duplicate entries within time window
    if attendance_key in attendance_timestamps:
        last_login_time = attendance_timestamps[attendance_key]
        time_diff = current_timestamp - last_login_time
        
        if time_diff < CONFIG.get("duplicate_detection_window", 300):  # 5 minutes default
            minutes_remaining = int((CONFIG.get("duplicate_detection_window", 300) - time_diff) / 60)
            print(f"⚠️ {name} already logged in recently. Please wait {minutes_remaining} more minutes.")
            return False
    
    # Security checks before logging
    min_confidence = CONFIG.get("min_recognition_threshold", 0.65)
    min_quality = CONFIG.get("face_quality_threshold", 0.3)
    min_security = 0.4  # Minimum liveness score
    
    if confidence < min_confidence:
        print(f"⚠️ Attendance rejected for {name}: Low confidence ({confidence:.2f} < {min_confidence})")
        return False
    
    if quality_score > 0 and quality_score < min_quality:
        print(f"⚠️ Attendance rejected for {name}: Poor image quality ({quality_score:.2f} < {min_quality})")
        return False
    
    if security_score > 0 and security_score < min_security:
        print(f"⚠️ Attendance rejected for {name}: Failed liveness detection ({security_score:.2f} < {min_security})")
        return False
        
    # Check daily attendance limit (basic fraud prevention)
    if attendance_key in attendance:
        return False  # Already logged today
    
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    date_only = current_time.strftime("%Y-%m-%d")

    latest_time = dt_time(
        int(CONFIG["latest_login_time"].split(":")[0]), 
        int(CONFIG["latest_login_time"].split(":")[1])
    )
    
    is_late = current_time.time() > latest_time
    status = "Late" if is_late else "Present"

    # Enhanced attendance data with security metrics
    attendance_data = {
        name: {
            'timestamp': timestamp,
            'status': status,
            'late': is_late,
            'confidence': round(confidence, 3),
            'quality_score': round(quality_score, 3) if quality_score > 0 else None,
            'security_score': round(security_score, 3) if security_score > 0 else None,
            'ip_address': 'localhost',  # Could be expanded for network deployments
            'system_info': {
                'gpu_used': GPU_AVAILABLE and CONFIG["use_gpu_acceleration"],
                'detection_method': 'ensemble' if CONFIG.get("use_ensemble_models", True) else 'standard'
            }
        }
    }
    
    if class_name:
        attendance_data[name]['class'] = class_name

    try:
        attendance_ref = db.collection('attendance').document(date_only)
        
        @firestore.transactional
        def update_in_transaction(transaction, doc_ref):
            doc = doc_ref.get(transaction=transaction)
            
            if doc.exists:
                transaction.update(doc_ref, attendance_data)
            else:
                transaction.set(doc_ref, attendance_data)
        
        transaction = db.transaction()
        update_in_transaction(transaction, attendance_ref)
        
        # Update local tracking
        attendance[attendance_key] = True
        attendance_timestamps[attendance_key] = current_timestamp
        
        # Log successful attendance
        log_message = f"✅ Attendance logged: {name}"
        if class_name:
            log_message += f" ({class_name})"
        log_message += f" - {status} (conf: {confidence:.2f}"
        if quality_score > 0:
            log_message += f", qual: {quality_score:.2f}"
        if security_score > 0:
            log_message += f", sec: {security_score:.2f}"
        log_message += ")"
        print(log_message)
        
        # Show thank you message
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
        
    except Exception as e:
        print(f"❌ Error logging attendance for {name}: {str(e)}")
        return False
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

# Advanced caching system
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.capacity:
                oldest = self.order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.order.append(key)

# High-performance recognition cache
recognition_cache = LRUCache(CONFIG["recognition_cache_size"])

# Preloaded face data for faster access
preloaded_face_data = {
    'encodings': None,
    'names': None,
    'classes': None,
    'loaded': False,
    'gpu_encodings': None  # GPU version of encodings
}

# Face detection optimization
class OptimizedFaceDetector:
    def __init__(self):
        self.detectors = {}
        self.initialize_detectors()
        self.detection_history = deque(maxlen=10)
        
    def initialize_detectors(self):
        """Initialize multiple optimized face detectors."""
        # OpenCV Haar Cascade (fastest)
        try:
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not haar_cascade.empty():
                self.detectors['haar'] = haar_cascade
                print("✅ Haar cascade detector loaded")
        except Exception as e:
            print(f"⚠️ Error loading Haar cascade: {e}")
        
        # dlib detector (accurate)
        if DLIB_AVAILABLE:
            try:
                dlib_detector = dlib.get_frontal_face_detector()
                self.detectors['dlib'] = dlib_detector
                print("✅ dlib detector loaded")
            except Exception as e:
                print(f"⚠️ dlib detector failed: {e}")
    
    def detect_faces_optimized(self, frame):
        """Ultra-fast optimized face detection with multiple methods."""
        start_time = time.time()
        faces = []
        
        # Use smart detection based on recent history
        if len(self.detection_history) > 0:
            avg_faces = sum(self.detection_history) / len(self.detection_history)
            if avg_faces < 2:
                # Few faces detected recently, use faster method
                faces = self._detect_with_haar(frame)
            else:
                # Many faces, use more accurate method
                faces = self._detect_with_dlib(frame) if 'dlib' in self.detectors else self._detect_with_haar(frame)
        else:
            # First detection, use balanced method
            faces = self._detect_with_haar(frame)
        
        # Update detection history
        self.detection_history.append(len(faces))
        
        detection_time = time.time() - start_time
        performance_metrics['detection_times'].append(detection_time)
        performance_metrics['total_faces_detected'] += len(faces)
        
        return faces
    
    def _detect_with_haar(self, frame):
        """Fast Haar cascade detection."""
        if 'haar' not in self.detectors:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detectors['haar'].detectMultiScale(
                gray,
                scaleFactor=CONFIG["detector_scale_factor"],
                minNeighbors=CONFIG["detector_min_neighbors"],
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_DO_CANNY_PRUNING
            )
            return faces.tolist() if len(faces) > 0 else []
        except Exception as e:
            print(f"⚠️ Haar detection error: {e}")
            return []
    
    def _detect_with_dlib(self, frame):
        """Accurate dlib detection."""
        if 'dlib' not in self.detectors:
            return self._detect_with_haar(frame)
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dlib_faces = self.detectors['dlib'](gray, 0)
            
            faces = []
            for face in dlib_faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces.append([x, y, w, h])
            return faces
        except Exception as e:
            print(f"⚠️ dlib detection error: {e}")
            return self._detect_with_haar(frame)

# Initialize optimized detector
face_detector = OptimizedFaceDetector()

def preload_face_encodings():
    """Preload all face encodings for maximum performance."""
    global preloaded_face_data
    
    if preloaded_face_data['loaded']:
        return
    
    print("🚀 Preloading face encodings for maximum performance...")
    
    encodings, names, classes = load_face_encodings()
    
    if encodings:
        # Convert to optimized numpy arrays
        preloaded_face_data['encodings'] = np.array(encodings, dtype=np.float32)
        preloaded_face_data['names'] = names
        preloaded_face_data['classes'] = classes
        
        # Preload to GPU if available
        if GPU_AVAILABLE and CONFIG["use_gpu_acceleration"]:
            try:
                preloaded_face_data['gpu_encodings'] = cp.asarray(preloaded_face_data['encodings'])
                print("✅ Face encodings loaded to GPU")
            except Exception as e:
                print(f"⚠️ GPU loading failed: {e}")
        
        preloaded_face_data['loaded'] = True
        print(f"✅ Preloaded {len(encodings)} face encodings")
    else:
        print("⚠️ No face encodings to preload")

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) based on facial landmarks.
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Args:
        eye_landmarks: List of (x,y) coordinates for eye landmarks
        
    Returns:
        float: Eye Aspect Ratio
    """
    if len(eye_landmarks) != 6:
        return 1.0  # Default open eye value
    
    try:
        # Compute euclidean distances
        # Vertical distances
        A = spatial.distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = spatial.distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        C = spatial.distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 1.0
        return ear
    except Exception as e:
        print(f"⚠️ Error calculating EAR: {str(e)}")
        return 1.0

def get_eye_landmarks(landmarks):
    """
    Extract eye landmarks from the full set of facial landmarks.
    HRNet landmarks have specific indices for left and right eye.
    
    Args:
        landmarks: Array of facial landmarks from HRNet
        
    Returns:
        tuple: (left_eye_landmarks, right_eye_landmarks)
    """
    if landmarks is None or len(landmarks) < 68:
        return None, None
    
    try:
        # HRNet provides 68-point landmarks
        # Left eye indices (36-41)
        left_eye = landmarks[36:42]
        
        # Right eye indices (42-47)
        right_eye = landmarks[42:48]
        
        return left_eye, right_eye
    except Exception as e:
        print(f"⚠️ Error extracting eye landmarks: {str(e)}")
        return None, None

def detect_blink(ear_value):
    """
    Detect a blink based on EAR values.
    A blink is detected when EAR drops below threshold and then rises back.
    
    Args:
        ear_value: Current Eye Aspect Ratio value
        
    Returns:
        bool: True if blink detected, False otherwise
    """
    global blink_detection
    
    # Store the current EAR value
    blink_detection['ear_values'].append(ear_value)
    
    # Blink is detected when EAR drops below threshold and then rises back
    if (blink_detection['previous_ear'] > CONFIG['ear_threshold'] and 
        ear_value <= CONFIG['ear_threshold']):
        # Potential blink start
        blink_start = True
    else:
        blink_start = False
    
    if (blink_detection['previous_ear'] <= CONFIG['ear_threshold'] and 
        ear_value > CONFIG['ear_threshold']):
        # Potential blink end - count as a blink
        if time.time() - blink_detection['last_blink_time'] > 0.2:  # Prevent counting rapid blinks
            blink_detection['blink_count'] += 1
            blink_detection['last_blink_time'] = time.time()
            blink_detection['status_message'] = f"Blink detected! ({blink_detection['blink_count']}/{CONFIG['min_blinks_required']})"
            print(f"👁️ Blink detected! Count: {blink_detection['blink_count']}")
            
            # Check if we've reached required blinks
            if blink_detection['blink_count'] >= CONFIG['min_blinks_required']:
                return True
    
    # Update previous EAR
    blink_detection['previous_ear'] = ear_value
    return False

def draw_eye_landmarks(frame, landmarks, left_eye, right_eye, ear_value):
    """
    Draw eye landmarks and EAR value on the frame for debugging.
    
    Args:
        frame: Video frame
        landmarks: All facial landmarks
        left_eye: Left eye landmarks
        right_eye: Right eye landmarks
        ear_value: Current EAR value
    """
    if not CONFIG["show_eye_landmarks"]:
        return
    
    try:
        # Draw all facial landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1)
            
        # Draw left eye landmarks
        for (x, y) in left_eye:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            
        # Draw right eye landmarks
        for (x, y) in right_eye:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            
        # Display EAR value
        cv2.putText(frame, f"EAR: {ear_value:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    except Exception:
        pass  # Fail silently for drawing

def start_blink_detection(name, class_name):
    """
    Start the blink detection process for anti-spoofing.
    
    Args:
        name: Person's name
        class_name: Class name
    """
    global blink_detection
    
    blink_detection = {
        'active': True,
        'start_time': time.time(),
        'blink_count': 0,
        'last_blink_time': 0,
        'ear_values': deque(maxlen=10),
        'person_name': name,
        'class_name': class_name,
        'status_message': "Please blink naturally...",
        'previous_ear': 1.0
    }
    
    print(f"👁️ Anti-spoofing activated: Waiting for {CONFIG['min_blinks_required']} blinks from {name}")

def display_blink_detection_ui(frame):
    """Display blink detection UI overlay on frame."""
    if not blink_detection['active']:
        return frame
    
    # Calculate time remaining
    elapsed_time = time.time() - blink_detection['start_time']
    time_remaining = max(0, CONFIG['blink_detection_time'] - elapsed_time)
    
    # Create overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    
    # Set opacity
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw instruction box
    cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 100, 200), 3)
    
    # Add title and instructions
    cv2.putText(frame, "ANTI-SPOOFING VERIFICATION", 
                (frame.shape[1]//2 - 200, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    
    cv2.putText(frame, f"Hello, {blink_detection['person_name']}!", 
                (frame.shape[1]//2 - 150, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.putText(frame, "Please blink naturally to verify you are a real person", 
                (frame.shape[1]//2 - 250, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    # Show status message
    cv2.putText(frame, blink_detection['status_message'], 
                (frame.shape[1]//2 - 150, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show time remaining with progress bar
    progress_width = int((time_remaining / CONFIG['blink_detection_time']) * (frame.shape[1] - 200))
    cv2.rectangle(frame, (100, frame.shape[0] - 100), 
                 (frame.shape[1] - 100, frame.shape[0] - 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (100, frame.shape[0] - 100), 
                 (100 + progress_width, frame.shape[0] - 80), (0, 200, 0), -1)
    cv2.putText(frame, f"Time remaining: {int(time_remaining)}s", 
                (100, frame.shape[0] - 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Handle timeout
    if time_remaining <= 0:
        blink_detection['active'] = False
        print(f"⏱️ Blink detection timed out for {blink_detection['person_name']}")
    
    return frame

def enhanced_face_recognition_ensemble(face_encoding, face_image=None, landmarks=None, use_cache=True):
    """
    Enhanced face recognition with ensemble models, security checks, and quality assessment.
    
    Args:
        face_encoding: Face encoding vector
        face_image: Original face image for quality assessment
        landmarks: Facial landmarks for additional analysis
        use_cache: Whether to use recognition cache
    
    Returns:
        tuple: (name, class_name, confidence, quality_metrics, security_score)
    """
    start_time = time.time()
    
    # Check cache first
    if use_cache:
        try:
            encoding_hash = hash(face_encoding.tobytes())
            cached_result = recognition_cache.get(encoding_hash)
            if cached_result is not None:
                performance_metrics['cache_hits'] += 1
                return cached_result
        except Exception:
            # If hashing fails, continue without cache
            pass
    
    # Ensure face data is loaded
    if not preloaded_face_data['loaded']:
        preload_face_encodings()
    
    name = "Unknown"
    class_name = ""
    base_confidence = 0.0
    quality_metrics = {}
    security_score = 0.0
    
    # Quality assessment if image is provided
    if face_image is not None and CONFIG.get("enable_quality_assessment", True):
        quality_metrics = face_quality_assessor.assess_face_quality(face_image, landmarks)
        
        # Reject low-quality faces early
        if quality_metrics["overall_score"] < CONFIG.get("face_quality_threshold", 0.3):
            result = (name, class_name, 0.0, quality_metrics, 0.0)
            if use_cache:
                try:
                    recognition_cache.put(encoding_hash, result)
                except Exception:
                    pass
            return result
    
    # Liveness detection if enabled
    if face_image is not None and CONFIG.get("liveness_detection", True):
        liveness_metrics = liveness_detector.detect_liveness(face_image, landmarks)
        security_score = liveness_metrics["overall_liveness"]
        
        # Reject if liveness score is too low
        if security_score < 0.4:  # Threshold for liveness
            result = (name, class_name, 0.0, quality_metrics, security_score)
            if use_cache:
                try:
                    recognition_cache.put(encoding_hash, result)
                except Exception:
                    pass
            return result
    
    if preloaded_face_data['encodings'] is not None and len(preloaded_face_data['encodings']) > 0:
        try:
            # Ensemble recognition with multiple methods
            recognition_results = []
            
            # Method 1: Standard Euclidean distance
            euclidean_distances = np.linalg.norm(preloaded_face_data['encodings'] - face_encoding, axis=1)
            euclidean_best_idx = np.argmin(euclidean_distances)
            euclidean_confidence = max(0, 1 - euclidean_distances[euclidean_best_idx])
            recognition_results.append({
                'method': 'euclidean',
                'index': euclidean_best_idx,
                'confidence': euclidean_confidence,
                'weight': CONFIG.get("model_weights", {}).get("euclidean", 0.4)
            })
            
            # Method 2: Cosine similarity
            try:
                # Normalize vectors for cosine similarity
                norm_encoding = face_encoding / np.linalg.norm(face_encoding)
                norm_stored = preloaded_face_data['encodings'] / np.linalg.norm(preloaded_face_data['encodings'], axis=1, keepdims=True)
                
                cosine_similarities = np.dot(norm_stored, norm_encoding)
                cosine_best_idx = np.argmax(cosine_similarities)
                cosine_confidence = max(0, cosine_similarities[cosine_best_idx])
                recognition_results.append({
                    'method': 'cosine',
                    'index': cosine_best_idx,
                    'confidence': cosine_confidence,
                    'weight': CONFIG.get("model_weights", {}).get("cosine", 0.3)
                })
            except Exception as e:
                print(f"⚠️ Cosine similarity failed: {e}")
            
            # Method 3: Manhattan distance
            try:
                manhattan_distances = np.sum(np.abs(preloaded_face_data['encodings'] - face_encoding), axis=1)
                manhattan_best_idx = np.argmin(manhattan_distances)
                manhattan_confidence = max(0, 1 - (manhattan_distances[manhattan_best_idx] / 100))  # Normalize
                recognition_results.append({
                    'method': 'manhattan',
                    'index': manhattan_best_idx,
                    'confidence': manhattan_confidence,
                    'weight': CONFIG.get("model_weights", {}).get("manhattan", 0.3)
                })
            except Exception as e:
                print(f"⚠️ Manhattan distance failed: {e}")
            
            # Ensemble decision making
            if recognition_results:
                # Weighted voting
                candidate_scores = {}
                
                for result in recognition_results:
                    idx = result['index']
                    confidence = result['confidence']
                    weight = result['weight']
                    
                    if idx not in candidate_scores:
                        candidate_scores[idx] = {
                            'total_score': 0.0,
                            'vote_count': 0,
                            'confidence_sum': 0.0
                        }
                    
                    candidate_scores[idx]['total_score'] += confidence * weight
                    candidate_scores[idx]['vote_count'] += 1
                    candidate_scores[idx]['confidence_sum'] += confidence
                
                # Find best candidate
                best_candidate = None
                best_score = 0.0
                
                for idx, scores in candidate_scores.items():
                    # Require minimum number of votes for confident recognition
                    if scores['vote_count'] >= CONFIG.get("recognition_voting_threshold", 2):
                        avg_confidence = scores['confidence_sum'] / scores['vote_count']
                        final_score = scores['total_score'] * (scores['vote_count'] / len(recognition_results))
                        
                        if final_score > best_score and avg_confidence >= CONFIG["min_recognition_threshold"]:
                            best_score = final_score
                            best_candidate = idx
                
                if best_candidate is not None:
                    base_confidence = best_score
                    
                    # Additional security checks
                    security_passed = True
                    
                    # Check if confidence is high enough
                    if base_confidence < CONFIG["min_recognition_threshold"]:
                        security_passed = False
                    
                    # Check against confident threshold for automatic acceptance
                    if base_confidence >= CONFIG["confident_recognition_threshold"]:
                        security_passed = True
                    elif base_confidence >= CONFIG["confidence_threshold_for_pin"]:
                        # Medium confidence - might require PIN verification
                        security_passed = True  # Let higher level decide on PIN
                    
                    # Quality-based confidence adjustment
                    if quality_metrics:
                        quality_factor = quality_metrics["overall_score"]
                        base_confidence *= (0.5 + 0.5 * quality_factor)  # Scale by quality
                    
                    # Security-based confidence adjustment
                    if security_score > 0:
                        base_confidence *= (0.6 + 0.4 * security_score)  # Scale by liveness
                    
                    if security_passed and base_confidence >= CONFIG["min_recognition_threshold"]:
                        name = preloaded_face_data['names'][best_candidate]
                        class_name = preloaded_face_data['classes'][best_candidate] if best_candidate < len(preloaded_face_data['classes']) else ""
                        
                        if CONFIG.get("log_recognition_details", False):
                            print(f"✅ Ensemble recognition: {name} (confidence: {base_confidence:.3f}, quality: {quality_metrics.get('overall_score', 0):.3f}, security: {security_score:.3f})")
                
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
    
    result = (name, class_name, base_confidence, quality_metrics, security_score)
    
    # Cache the result
    if use_cache:
        try:
            recognition_cache.put(encoding_hash, result)
            performance_metrics['cache_misses'] += 1
        except Exception:
            pass
    
    recognition_time = time.time() - start_time
    performance_metrics['recognition_times'].append(recognition_time)
    performance_metrics['total_faces_recognized'] += 1
    
    return result

def ultra_fast_face_recognition(face_encoding, use_cache=True):
    """Backward compatibility wrapper for the enhanced recognition function."""
    name, class_name, confidence, _, _ = enhanced_face_recognition_ensemble(
        face_encoding, None, None, use_cache
    )
    return (name, class_name, confidence)

def parallel_face_encoding(rgb_frame, face_locations):
    """Process face encodings in parallel for maximum speed."""
    start_time = time.time()
    
    if not face_locations:
        return []
    
    # If face_alignment is available, use it for more accurate landmark-based encoding
    if 'facial_landmark_predictor' in globals() and facial_landmark_predictor is not None and FACE_ALIGNMENT_AVAILABLE:
        all_encodings = []
        for (top, right, bottom, left) in face_locations:
            # Extract face region
            face_img = rgb_frame[top:bottom, left:right]
            try:
                landmarks = facial_landmark_predictor.get_landmarks_from_image(face_img)
                if landmarks and len(landmarks) > 0:
                    # Use face_recognition.face_encodings without model argument (default is hog)
                    encoding = face_recognition.face_encodings(face_img, [(0, face_img.shape[1], face_img.shape[0], 0)], model="cnn")
                    if encoding:
                        all_encodings.append(encoding[0])
            except Exception as e:
                print(f"⚠️ face_alignment encoding error: {e}")
        encoding_time = time.time() - start_time
        performance_metrics['encoding_times'].append(encoding_time)
        return all_encodings
    # Otherwise, use the original CNN-based encoding
    if len(face_locations) > 1 and CONFIG["max_parallel_recognitions"] > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(face_locations), CONFIG["encoding_threads"])) as executor:
            futures = []
            # Split face locations into batches
            batch_size = max(1, len(face_locations) // CONFIG["encoding_threads"])
            for i in range(0, len(face_locations), batch_size):
                batch = face_locations[i:i + batch_size]
                # Always use model as a keyword argument
                future = executor.submit(lambda *args, **kwargs: face_recognition.face_encodings(*args, **kwargs), rgb_frame, batch, model="cnn")
                futures.append(future)
            all_encodings = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_encodings = future.result()
                    all_encodings.extend(batch_encodings)
                except Exception as e:
                    print(f"⚠️ Batch encoding error: {e}")
            encoding_time = time.time() - start_time
            performance_metrics['encoding_times'].append(encoding_time)
            return all_encodings
    else:
        try:
            encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="cnn")
            encoding_time = time.time() - start_time
            performance_metrics['encoding_times'].append(encoding_time)
            return encodings
        except Exception as e:
            print(f"⚠️ Encoding error: {e}")
            return []

def adaptive_quality_control(fps):
    """Dynamically adjust processing parameters based on performance."""
    if not CONFIG["dynamic_quality_adjustment"]:
        return
    
    target_fps = CONFIG["target_fps"]
    
    # Adjust parameters based on FPS performance
    if fps < target_fps * 0.6:  # Significantly below target
        # Reduce quality for speed
        CONFIG["frame_resize"] = max(0.25, CONFIG["frame_resize"] - 0.05)
        CONFIG["skip_frames"] = min(4, CONFIG["skip_frames"] + 1)
        CONFIG["batch_processing_size"] = max(4, CONFIG["batch_processing_size"] - 2)
        CONFIG["max_parallel_recognitions"] = max(2, CONFIG["max_parallel_recognitions"] - 1)
    elif fps > target_fps * 1.3:  # Significantly above target
        # Increase quality
        CONFIG["frame_resize"] = min(0.5, CONFIG["frame_resize"] + 0.02)
        CONFIG["skip_frames"] = max(1, CONFIG["skip_frames"] - 1)
        CONFIG["batch_processing_size"] = min(16, CONFIG["batch_processing_size"] + 1)
        CONFIG["max_parallel_recognitions"] = min(8, CONFIG["max_parallel_recognitions"] + 1)

def display_enhanced_performance_metrics(frame):
    """Display comprehensive performance and security metrics."""
    if not CONFIG["performance_monitoring"]:
        return frame
    
    # Calculate enhanced metrics
    avg_frame_time = np.mean(performance_metrics['frame_times']) if performance_metrics['frame_times'] else 0
    avg_detection_time = np.mean(performance_metrics['detection_times']) if performance_metrics['detection_times'] else 0
    avg_recognition_time = np.mean(performance_metrics['recognition_times']) if performance_metrics['recognition_times'] else 0
    avg_encoding_time = np.mean(performance_metrics['encoding_times']) if performance_metrics['encoding_times'] else 0
    
    # Cache performance
    total_cache_requests = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
    cache_hit_rate = (performance_metrics['cache_hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
    
    # Security metrics
    recent_alerts = len([alert for alert in security_monitor['system_alerts'] 
                        if time.time() - datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00')).timestamp() < 300])
    
    # System health status
    health = comprehensive_system_health_check()
    
    # Display enhanced metrics with color coding
    y_offset = 50
    
    # Performance metrics
    performance_color = (0, 255, 0) if health['performance_status'] == 'OK' else (0, 255, 255)
    security_color = (0, 255, 0) if health['security_status'] == 'OK' else (0, 165, 255)
    
    metrics = [
        f"Frame: {avg_frame_time*1000:.1f}ms",
        f"Detection: {avg_detection_time*1000:.1f}ms",
        f"Recognition: {avg_recognition_time*1000:.1f}ms",
        f"Cache: {cache_hit_rate:.0f}% hit rate",
        f"Security: {health['security_status']}",
        f"Performance: {health['performance_status']}",
        f"Alerts: {recent_alerts} recent",
        f"Quality: {'ON' if CONFIG.get('enable_quality_assessment', True) else 'OFF'}",
        f"Liveness: {'ON' if CONFIG.get('liveness_detection', True) else 'OFF'}"
    ]
    
    for i, metric in enumerate(metrics):
        if i < 3:
            color = (0, 255, 255)  # Cyan for timing
        elif i == 3:
            color = (255, 255, 0)  # Yellow for cache
        elif i == 4:
            color = security_color
        elif i == 5:
            color = performance_color
        else:
            color = (255, 255, 255)  # White for others
            
        cv2.putText(frame, metric, (10, y_offset + i * 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    # Display alerts if any
    if health['alerts']:
        alert_y = y_offset + len(metrics) * 18 + 20
        cv2.putText(frame, "ALERTS:", (10, alert_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for i, alert in enumerate(health['alerts'][:3]):  # Show only first 3 alerts
            cv2.putText(frame, f"• {alert}", (10, alert_y + (i + 1) * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
    
    return frame

def display_face_quality_info(frame, face_location, quality_metrics, security_score, confidence):
    """Display quality and security information for a detected face."""
    if not CONFIG.get("show_quality_metrics", False):
        return frame
    
    top, right, bottom, left = face_location
    
    # Calculate display position
    info_x = left
    info_y = top - 60
    
    if info_y < 0:
        info_y = bottom + 20
    
    # Quality indicators
    quality_score = quality_metrics.get("overall_score", 0.0) if quality_metrics else 0.0
    
    # Color coding based on quality and security
    if quality_score > 0.7 and security_score > 0.7 and confidence > 0.8:
        status_color = (0, 255, 0)  # Green - excellent
        status_text = "EXCELLENT"
    elif quality_score > 0.5 and security_score > 0.5 and confidence > 0.6:
        status_color = (0, 255, 255)  # Yellow - good
        status_text = "GOOD"
    elif quality_score > 0.3 and confidence > 0.4:
        status_color = (0, 165, 255)  # Orange - fair
        status_text = "FAIR"
    else:
        status_color = (0, 0, 255)  # Red - poor
        status_text = "POOR"
    
    # Draw info box
    box_height = 50
    cv2.rectangle(frame, (info_x, info_y - box_height), 
                 (info_x + 200, info_y), (0, 0, 0), -1)
    cv2.rectangle(frame, (info_x, info_y - box_height), 
                 (info_x + 200, info_y), status_color, 1)
    
    # Display metrics
    cv2.putText(frame, f"Quality: {status_text}", 
               (info_x + 5, info_y - 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
    cv2.putText(frame, f"Q:{quality_score:.2f} S:{security_score:.2f} C:{confidence:.2f}", 
               (info_x + 5, info_y - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    if quality_metrics:
        detail_text = f"Sharp:{quality_metrics.get('sharpness', 0):.1f} Bright:{quality_metrics.get('brightness', 0):.1f}"
        cv2.putText(frame, detail_text, 
                   (info_x + 5, info_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    return frame

# Ultra-fast face tracker with vectorized operations
class UltraFastFaceTracker:
    def __init__(self, max_disappeared=15, min_quality=3):
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
        
        # Use defaults if not provided
        if face_names is None:
            face_names = ["Unknown"] * len(face_locations)
        if face_classes is None:
            face_classes = [""] * len(face_locations)
        if face_confidences is None:
            face_confidences = [0.0] * len(face_locations)
        
        # Handle no faces
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
        
        # Ultra-fast vectorized matching
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0 and len(face_encodings) > 0:
            track_encodings = np.array([self.tracks[tid]["encoding"] for tid in track_ids])
            face_encodings_array = np.array(face_encodings)
            
            # Vectorized distance computation
            distances_matrix = np.linalg.norm(
                track_encodings[:, np.newaxis] - face_encodings_array[np.newaxis, :], 
                axis=2
            )
            
            # Fast Hungarian-like assignment
            matched_tracks = {}
            unmatched_detections = set(range(len(face_locations)))
            
            # Greedy matching for speed
            for _ in range(min(len(track_ids), len(face_locations))):
                if distances_matrix.size == 0:
                    break
                    
                min_idx = np.unravel_index(np.argmin(distances_matrix), distances_matrix.shape)
                track_idx, face_idx = min_idx
                
                if distances_matrix[min_idx] < 0.6:  # Match threshold
                    track_id = track_ids[track_idx]
                    matched_tracks[track_id] = face_idx
                    unmatched_detections.discard(face_idx)
                    
                    # Update track
                    self.tracks[track_id]["rect"] = face_locations[face_idx]
                    if face_names[face_idx] != "Unknown" and face_confidences[face_idx] > self.tracks[track_id]["confidence"]:
                        self.tracks[track_id]["name"] = face_names[face_idx]
                        self.tracks[track_id]["class_name"] = face_classes[face_idx]
                        self.tracks[track_id]["confidence"] = face_confidences[face_idx]
                    
                    self.tracks[track_id]["quality"] = min(10, self.tracks[track_id]["quality"] + 3)
                    self.tracks[track_id]["last_update"] = self.frame_count
                    self.disappeared[track_id] = 0
                    
                    # Remove matched from consideration
                    distances_matrix = np.delete(distances_matrix, track_idx, axis=0)
                    distances_matrix = np.delete(distances_matrix, face_idx, axis=1)
                    track_ids.pop(track_idx)
                else:
                    break
        
        # Handle unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.disappeared[track_id] += 1
                self.tracks[track_id]["quality"] = max(0, self.tracks[track_id]["quality"] - 2)
                
                if (self.disappeared[track_id] > self.max_disappeared or 
                    self.tracks[track_id]["quality"] < self.min_quality):
                    self.tracks.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
        
        # Register unmatched detections
        for face_idx in unmatched_detections:
            self.register(face_locations[face_idx], face_encodings[face_idx],
                         face_names[face_idx], face_classes[face_idx], face_confidences[face_idx])
        
        return self.tracks

def check_lighting_conditions(frame):
    """
    Check if the frame has adequate lighting for face detection.
    
    Args:
        frame: The video frame to analyze
        
    Returns:
        tuple: (is_lighting_good, brightness_value, message)
    """
    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    brightness = np.mean(gray)
    
    # Calculate contrast using standard deviation
    contrast = np.std(gray)
    
    # Get brightness thresholds from config
    min_brightness, max_brightness = CONFIG.get("brightness_threshold", (50, 200))
    min_contrast = 30  # Minimum contrast for good quality
    
    # Check lighting conditions
    is_good_lighting = True
    message = "Lighting is good"
    
    if brightness < min_brightness:
        is_good_lighting = False
        message = f"Too dark ({brightness:.1f}). Increase lighting."
    elif brightness > max_brightness:
        is_good_lighting = False
        message = f"Too bright ({brightness:.1f}). Reduce lighting."
    elif contrast < min_contrast:
        is_good_lighting = False
        message = f"Low contrast ({contrast:.1f}). Improve lighting setup."
    
    return is_good_lighting, brightness, message

def display_lighting_warning(frame, message, brightness):
    """Display lighting warning overlay on frame."""
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 50), -1)
    
    # Blend overlay
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Display warning message
    warning_lines = [
        "⚠️ LIGHTING WARNING ⚠️",
        message,
        "Please adjust lighting for better recognition",
        f"Current brightness: {brightness:.1f}"
    ]
    
    y_start = frame.shape[0] // 2 - 60
    for i, line in enumerate(warning_lines):
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        y = y_start + i * 30
        
        if i == 0:
            color = (0, 255, 255)  # Yellow for title
            thickness = 3
        else:
            color = (255, 255, 255)  # White for message
            thickness = 2
            
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
    
    return frame
    
    # Check for overexposed areas (bright spots)
    overexposed = np.sum(gray > 240) / gray.size
    
    # Check for underexposed areas (dark spots)
    underexposed = np.sum(gray < 30) / gray.size
    
    # Determine if lighting is good
    is_good = True
    message = "Lighting OK"
    
    if brightness < 40:
        is_good = False
        message = "Too dark! Please turn on more lights."
    elif brightness > 210:
        is_good = False
        message = "Too bright! Reduce lighting or adjust camera."
    elif contrast < 15:
        is_good = False
        message = "Low contrast! Improve lighting conditions."
    elif overexposed > 0.15:
        is_good = False
        message = "Overexposed areas detected! Reduce bright light sources."
    elif underexposed > 0.30:
        is_good = False
        message = "Underexposed areas detected! Increase lighting."
    
    return (is_good, brightness, message)

def display_lighting_warning(frame, message, brightness):
    """Display warning message when lighting conditions are poor."""
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    
    # Set the opacity of the overlay
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Show warning message
    cv2.putText(frame, "⚠️ " + message, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show brightness meter
    cv2.putText(frame, f"Brightness: {brightness:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    
    # Draw brightness meter
    cv2.rectangle(frame, (220, 50), (220 + 200, 65), (50, 50, 50), -1)
    
    # Determine color based on brightness
    if brightness < 40:  # Too dark
        bar_color = (0, 0, 150)  # Dark red
    elif brightness > 210:  # Too bright
        bar_color = (0, 150, 255)  # Yellow-orange
    else:
        # Gradient from red to green
        green = min(255, int((brightness / 120) * 255))
        red = min(255, int(((255 - brightness) / 120) * 255))
        bar_color = (0, green, red)
    
    # Draw filled portion of brightness meter
    filled_width = min(200, int((brightness / 255) * 200))
    cv2.rectangle(frame, (220, 50), (220 + filled_width, 65), bar_color, -1)
    
    return frame

# Initialize the lighting check variables
lighting_check_enabled = True
lighting_check_interval = 5  # Check every 5 seconds
last_lighting_check_time = time.time()

def main():
    """Main function with ultra-high performance optimizations."""
    global pin_verification_mode, blink_detection
    
    print("🚀 Starting Ultra-High Performance Facial Recognition System...")
    
    # Initialize landmark predictor if available
    if FACE_ALIGNMENT_AVAILABLE and CONFIG["show_landmarks"]:
        initialize_landmark_predictor()
    
    # Initialize high-performance components
    if CONFIG["preload_known_faces"]:
        preload_face_encodings()
    
    # Load PINs
    global user_pins
    user_pins = load_user_pins()
    
    # Initialize camera with maximum performance settings
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    # Optimize camera for maximum performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video_capture.set(cv2.CAP_PROP_FPS, 30)  # Reduced from 60 for stability
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    except Exception:
        pass  # Some cameras don't support this codec
    
    print("🎥 Ultra-high performance camera initialized.")
    
    # Set up optimized window
    cv2.namedWindow('Ultra-Fast Face Recognition', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Ultra-Fast Face Recognition', handle_mouse_click,
                        (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # Initialize ultra-fast face tracker
    face_tracker = UltraFastFaceTracker(
        max_disappeared=CONFIG["max_tracking_age"],
        min_quality=CONFIG["tracking_quality_threshold"]
    )
    
    # Performance monitoring
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    skip_counter = 0
    
    # High-performance thread pool
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=CONFIG["encoding_threads"]
    )
    
    print("✅ Ultra-fast facial recognition system ready!")
    
    try:
        while True:
            frame_start_time = time.time()
            
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break
            
            frame = cv2.flip(frame, CONFIG["flip_camera"])
            
            # Check lighting conditions before proceeding with face detection
            lighting_good, brightness, lighting_message = check_lighting_conditions(frame)
            
            if not lighting_good:
                # Display warning about poor lighting
                frame = display_lighting_warning(frame, lighting_message, brightness)
                
                # Skip face detection in poor lighting conditions
                if CONFIG["display_fps"]:
                    fps_color = (0, 255, 0) if fps >= CONFIG["target_fps"] * 0.8 else (0, 255, 255)
                    cv2.putText(frame, f"FPS: {fps:.1f} (Peak: {performance_metrics.get('peak_fps', 0):.1f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
                
                frame = display_enhanced_performance_metrics(frame)
                frame = display_thank_you_message(frame)
                
                # Record frame processing time
                frame_time = time.time() - frame_start_time
                performance_metrics['frame_times'].append(frame_time)
                
                cv2.imshow('Ultra-Fast Face Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue  # Skip the rest of the loop and start with a new frame
            
            # Handle PIN verification mode
            if pin_verification_mode['active']:
                if time.time() - pin_verification_mode['start_time'] > CONFIG["pin_timeout"]:
                    pin_verification_mode['active'] = False
                    print("⏱️ PIN verification timed out")
                
                frame = display_pin_pad(frame, pin_verification_mode['name'], 
                                       pin_verification_mode['class_name'])
            elif blink_detection['active']:
                # Handle blink detection mode
                small_frame = cv2.resize(frame, (0, 0), 
                                       fx=CONFIG["frame_resize"], 
                                       fy=CONFIG["frame_resize"],
                                       interpolation=cv2.INTER_LINEAR)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Get face locations for the person being verified
                face_locations_raw = face_detector.detect_faces_optimized(small_frame)
                
                if face_locations_raw and facial_landmark_predictor is not None:
                    # Process the largest face (closest to the camera)
                    largest_face = max(face_locations_raw, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    
                    # Get face region with some margin
                    face_region = rgb_small_frame[max(0, y-30):min(rgb_small_frame.shape[0], y+h+30), 
                                                max(0, x-30):min(rgb_small_frame.shape[1], x+w+30)]
                    
                    if face_region.size > 0:
                        # Get landmarks
                        try:
                            landmarks = facial_landmark_predictor.get_landmarks_from_image(face_region)
                            
                            if landmarks and len(landmarks) > 0:
                                # Adjust coordinates to the original small frame
                                x_offset = max(0, x-30)
                                y_offset = max(0, y-30)
                                
                                landmarks_adjusted = [(p[0] + x_offset, p[1] + y_offset) for p in landmarks[0]]
                                
                                # Extract eye landmarks
                                left_eye, right_eye = get_eye_landmarks(landmarks_adjusted)
                                
                                if left_eye is not None and right_eye is not None:
                                    # Calculate average EAR
                                    left_ear = calculate_ear(left_eye)
                                    right_ear = calculate_ear(right_eye)
                                    avg_ear = (left_ear + right_ear) / 2.0
                                    
                                    # Scale coordinates back to original frame size
                                    scale = 1.0 / CONFIG["frame_resize"]
                                    scaled_landmarks = [(int(p[0] * scale), int(p[1] * scale)) 
                                                       for p in landmarks_adjusted]
                                    scaled_left_eye = [(int(p[0] * scale), int(p[1] * scale)) 
                                                      for p in left_eye]
                                    scaled_right_eye = [(int(p[0] * scale), int(p[1] * scale)) 
                                                       for p in right_eye]
                                    
                                    # Draw eye landmarks
                                    draw_eye_landmarks(frame, scaled_landmarks, 
                                                     scaled_left_eye, scaled_right_eye, avg_ear)
                                    
                                    # Detect blink
                                    if detect_blink(avg_ear):
                                        print(f"✅ Blink verification successful for {blink_detection['person_name']}")
                                       
                                        blink_detection['status_message'] = "Verification successful!"
                                        
                                        # Log attendance after successful blink verification
                                        log_attendance(blink_detection['person_name'], 
                                                      blink_detection['class_name'])
                                        
                                        # Show success message for a moment
                                        time.sleep(1)
                                        blink_detection['active'] = False
                        except Exception as e:
                            print(f"⚠️ Error processing landmarks: {e}")
                
                # Display blink detection UI
                frame = display_blink_detection_ui(frame)
            else:
                # Regular face recognition mode
                skip_counter += 1
                process_this_frame = skip_counter % CONFIG["skip_frames"] == 0
                frame_count += 1
                
                # Calculate FPS with performance tracking
                if time.time() - fps_start_time >= 1.0:
                    fps = frame_count / (time.time() - fps_start_time)
                    performance_metrics['avg_fps'] = fps
                    performance_metrics['peak_fps'] = max(performance_metrics.get('peak_fps', 0), fps)
                    frame_count = 0
                    fps_start_time = time.time()
                    
                    # Dynamic quality control
                    adaptive_quality_control(fps)
                
                # Process frame for face recognition
                active_faces = {}
                
                if process_this_frame:
                    # Resize frame with optimized interpolation
                    small_frame = cv2.resize(frame, (0, 0), 
                                           fx=CONFIG["frame_resize"], 
                                           fy=CONFIG["frame_resize"],
                                           interpolation=cv2.INTER_LINEAR)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Ultra-fast face detection with validation
                    face_locations_raw = face_detector.detect_faces_optimized(small_frame)
                    
                    if face_locations_raw:
                        # Convert to face_recognition format and validate
                        face_locations = []
                        valid_face_regions = []
                        
                        for face_data in face_locations_raw:
                            if len(face_data) >= 4:
                                x, y, w, h = face_data[:4]
                                # Convert from (x,y,w,h) to (top, right, bottom, left)
                                top, right, bottom, left = y, x + w, y + h, x
                                face_location = (top, right, bottom, left)
                                
                                # Validate face region quality
                                is_valid, quality_score, reason = validate_face_region(rgb_small_frame, face_location)
                                
                                if is_valid:
                                    face_locations.append(face_location)
                                    # Extract and preprocess face region
                                    face_region = rgb_small_frame[top:bottom, left:right]
                                    face_region = enhanced_face_preprocessing(face_region)
                                    valid_face_regions.append(face_region)
                                elif CONFIG.get("show_quality_metrics", False):
                                    print(f"⚠️ Face rejected: {reason} (quality: {quality_score:.2f})")
                        
                        # Parallel face encoding for valid faces only
                        if face_locations:
                            face_encodings = parallel_face_encoding(rgb_small_frame, face_locations)
                            
                            if face_encodings:
                                # Enhanced parallel recognition with quality assessment
                                recognition_results = []
                                if len(face_encodings) <= CONFIG["max_parallel_recognitions"]:
                                    # Get landmarks for liveness detection
                                    landmarks_list = []
                                    if facial_landmark_predictor is not None and CONFIG.get("liveness_detection", True):
                                        for i, face_region in enumerate(valid_face_regions):
                                            try:
                                                landmarks = facial_landmark_predictor.get_landmarks_from_image(face_region)
                                                landmarks_list.append(landmarks[0] if landmarks and len(landmarks) > 0 else None)
                                            except Exception:
                                                landmarks_list.append(None)
                                    else:
                                        landmarks_list = [None] * len(face_encodings)
                                    
                                    # Use enhanced recognition with ensemble models
                                    futures = []
                                    for i, encoding in enumerate(face_encodings):
                                        face_image = valid_face_regions[i] if i < len(valid_face_regions) else None
                                        landmarks = landmarks_list[i] if i < len(landmarks_list) else None
                                        
                                        future = executor.submit(
                                            enhanced_face_recognition_ensemble,
                                            encoding, face_image, landmarks, True
                                        )
                                        futures.append(future)
                                    
                                    for future in concurrent.futures.as_completed(futures):
                                        try:
                                            result = future.result()
                                            recognition_results.append(result)
                                        except Exception as e:
                                            print(f"⚠️ Enhanced recognition future error: {e}")
                                            recognition_results.append(("Unknown", "", 0.0, {}, 0.0))
                                else:
                                    # Sequential for many faces
                                    for i, encoding in enumerate(face_encodings):
                                        face_image = valid_face_regions[i] if i < len(valid_face_regions) else None
                                        try:
                                            landmarks = None
                                            if facial_landmark_predictor is not None and face_image is not None:
                                                landmarks_result = facial_landmark_predictor.get_landmarks_from_image(face_image)
                                                landmarks = landmarks_result[0] if landmarks_result and len(landmarks_result) > 0 else None
                                        except Exception:
                                            landmarks = None
                                        
                                        result = enhanced_face_recognition_ensemble(
                                            encoding, face_image, landmarks, True
                                        )
                                        recognition_results.append(result)
                                
                                # Extract enhanced results
                                face_names = [result[0] for result in recognition_results]
                                face_classes = [result[1] for result in recognition_results]
                                face_confidences = [result[2] for result in recognition_results]
                                face_qualities = [result[3] for result in recognition_results]
                                face_securities = [result[4] for result in recognition_results]
                                
                                # Handle attendance logging with enhanced security
                                for i, (name, class_name, confidence, quality_metrics, security_score) in enumerate(recognition_results):
                                    if name != "Unknown" and confidence >= CONFIG["min_recognition_threshold"]:
                                        attendance_key = f"{class_name}/{name}" if class_name else name
                                        
                                        # Enhanced duplicate check with timestamps
                                        should_log = True
                                        if attendance_key in attendance_timestamps:
                                            last_time = attendance_timestamps[attendance_key]
                                            time_diff = time.time() - last_time
                                            if time_diff < CONFIG.get("duplicate_detection_window", 300):
                                                should_log = False
                                        
                                        if should_log:
                                            # Determine logging method based on confidence and security
                                            quality_score = quality_metrics.get("overall_score", 0.0) if quality_metrics else 0.0
                                            
                                            if confidence >= CONFIG["confident_recognition_threshold"]:
                                                # High confidence - log directly with enhanced data
                                                success = log_attendance(name, class_name, confidence, quality_score, security_score)
                                                if success:
                                                    print(f"⚡ High-confidence attendance: {name} (conf: {confidence:.2f})")
                                            elif confidence >= CONFIG["confidence_threshold_for_pin"]:
                                                # Medium confidence - require PIN verification
                                                pin_verification_mode = {
                                                    'active': True,
                                                    'name': name,
                                                    'class_name': class_name,
                                                    'correct_pin': '',
                                                    'start_time': time.time(),
                                                    'attempts': 0,
                                                    'input_pin': '',
                                                    'error_message': '',
                                                    'confidence': confidence,
                                                    'quality_score': quality_score,
                                                    'security_score': security_score
                                                }
                                                print(f"🔐 PIN verification required for {name} (conf: {confidence:.2f})")
                                            else:
                                                # Low confidence - reject but log attempt
                                                if CONFIG.get("log_recognition_details", True):
                                                    print(f"❌ Low confidence rejected: {name} (conf: {confidence:.2f}, qual: {quality_score:.2f}, sec: {security_score:.2f})")
                            
                            # Update ultra-fast tracker with enhanced data
                            active_faces = face_tracker.update(
                                face_locations, face_encodings, 
                                face_names, face_classes, face_confidences
                            )
                else:
                    # Use existing tracks without new detection
                    active_faces = face_tracker.tracks
                
                # Ultra-fast rendering
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
                    
                    # Enhanced rectangle and text rendering with quality info
                    if CONFIG.get("show_face_boxes", True):
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Display quality information if available
                        if hasattr(face_data, 'quality_metrics') or hasattr(face_data, 'security_score'):
                            quality_metrics = getattr(face_data, 'quality_metrics', {})
                            security_score = getattr(face_data, 'security_score', 0.0)
                            frame = display_face_quality_info(
                                frame, (top, right, bottom, left), 
                                quality_metrics, security_score, confidence
                            )
            
            # Display performance metrics
            if CONFIG["display_fps"]:
                fps_color = (0, 255, 0) if fps >= CONFIG["target_fps"] * 0.8 else (0, 255, 255)
                cv2.putText(frame, f"FPS: {fps:.1f} (Peak: {performance_metrics.get('peak_fps', 0):.1f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
            
            frame = display_enhanced_performance_metrics(frame)
            frame = display_thank_you_message(frame)
            
            # Record frame processing time
            frame_time = time.time() - frame_start_time
            performance_metrics['frame_times'].append(frame_time)
            
            cv2.imshow('Ultra-Fast Face Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif pin_verification_mode['active']:
                handle_pin_input(key)
                
    except KeyboardInterrupt:
        print("\n⚡ System stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        executor.shutdown(wait=False)
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Print comprehensive performance summary
        if CONFIG["performance_monitoring"]:
            print("\n📊 Ultra-Performance Summary:")
            print(f"Peak FPS: {performance_metrics.get('peak_fps', 0):.1f}")
            print(f"Average FPS: {performance_metrics.get('avg_fps', 0):.1f}")
            print(f"Total faces detected: {performance_metrics['total_faces_detected']}")
            print(f"Total faces recognized: {performance_metrics['total_faces_recognized']}")
            cache_total = performance_metrics['cache_hits'] + performance_metrics['cache_misses']
            cache_rate = (performance_metrics['cache_hits'] / max(1, cache_total) * 100)
            print(f"Cache hit rate: {cache_rate:.1f}%")
            print(f"GPU acceleration: {'ON' if GPU_AVAILABLE and CONFIG['use_gpu_acceleration'] else 'OFF'}")
        
        print("🔴 Ultra-High Performance Face Recognition System closed.")

if __name__ == "__main__":
    main()

