
# Facial Recognition Attendance System - High Performance Edition

## Overview
This facial recognition system provides automated attendance tracking using computer vision and facial recognition technology. The system captures video from a webcam, identifies individuals by comparing faces against a known database, and logs attendance with timestamps. Data is stored in the cloud using Firebase.

**v2.7.py is the latest High Performance Edition, featuring GPU acceleration, parallel processing, anti-spoofing, and advanced configuration for speed and accuracy.**

## Features
- Real-time face detection and recognition (OpenCV and dlib supported)
- GPU acceleration (CuPy/NVIDIA GPU) and multi-threaded processing
- Class-based student organization for educational settings
- Automatic attendance logging with timestamps
- Cloud-based attendance storage with Firebase
- Interactive attendance viewing
- Late arrival tracking based on configurable time threshold
- Display of motivational quotes upon attendance registration
- Visual confirmation overlay for successful attendance
- Advanced facial landmark detection using HRNet (if available)
- Performance optimization: smart frame skipping, batch processing, dynamic quality, caching, and parallel face detection/encoding
- Anti-spoofing with blink detection (configurable)
- Caching of facial encodings for faster startup
- PIN-based backup authentication when face recognition confidence is low (now supports 4-6 digit PINs)
- Robust error handling and debugging options
- Ensures no one is logged more than once per session

## Files in this Project
- `v2.7.py`: The main facial recognition system (High Performance Edition)
- `make_dataset.py`: Script for creating the facial recognition database and collecting user PINs (now supports 4-6 digit PINs and improved image capture)
- `serviceAccountKey.json`: Firebase credentials file (must be set up separately)

## Dependencies
- OpenCV (cv2)
- face_recognition
- numpy
- pickle
- datetime
- firebase-admin
- face-alignment (for HRNet facial landmark detection, optional)
- dlib (optional, for faster face detection)
- cupy (optional, for GPU acceleration)
- python-dotenv (for .env support)
- scipy (for advanced math/anti-spoofing)

## Configuration Options
The program includes a powerful CONFIG dictionary with many adjustable settings for performance, recognition, and security. Key options include:

- `tolerance`: Face matching threshold (lower = stricter matching)
- `frame_resize`: Frame size reduction for faster processing
- `skip_frames`: Number of frames to skip between processing
- `display_fps`: Toggle FPS counter display
- `show_all_faces`: Show boxes for unknown faces
- `flip_camera`: Camera orientation adjustment
- `corner_display`: Show corner preview
- `latest_login_time`: Time threshold for late arrival marking
- `enhanced_facial_recognition`: Toggle enhanced features
- `show_landmarks`: Display facial landmarks on detected faces
- `device`: Device for HRNet model ('cpu', 'cuda', or auto)
- `first_run_warning`: Toggle first-run model download warning
- `confidence_threshold_for_pin`: Threshold below which PIN authentication is required
- `pin_timeout`: Time allowed for PIN entry in seconds
- `pin_allowed_attempts`: Number of PIN attempts before timeout
- `min_recognition_threshold` / `confident_recognition_threshold`: Fine-tune PIN/face match logic
- `use_gpu_acceleration`, `use_dlib_detector`, `face_detection_threads`, `encoding_threads`, `batch_processing_size`, `parallel_face_detection`, `dynamic_quality_adjustment`, `smart_frame_skipping`, `adaptive_processing`, `performance_monitoring`, and more for advanced users
- `enable_blink_detection`, `ear_threshold`, `min_blinks_required`, `blink_detection_time`: Anti-spoofing (blink detection)

## Creating a Face Dataset
Before using the recognition system, you need to build a dataset of faces:

1. Run `make_dataset.py`
2. Enter your name and class name when prompted
3. Enter a 4-6 digit PIN (used for backup authentication; only numbers allowed)
4. Position yourself in front of the camera with good lighting (the script uses high-res, high-FPS settings for best results)
5. The script will guide you through capturing images of your face:
   - Press 'c' to capture images manually one by one
   - Press 'a' to enable auto-capture mode
   - Press 'q' to quit the capturing process
6. Images and PIN will be stored in Firebase Storage in the `face_dataset/{class_name}/{your_name}` directory
7. Repeat for each person you want to recognize

## How Face Recognition Works
1. **Dataset Loading**: The system loads facial encodings from a cached pickle file or builds encodings from the face image dataset in Firebase Storage.
2. **Face Detection**: Each video frame is processed using OpenCV or dlib (if available) for fast, accurate face location.
3. **Face Recognition**: Detected faces are encoded and compared against the known dataset, using parallel and batch processing for speed.
4. **Facial Landmark Detection**: HRNet is used to identify facial landmarks for enhanced recognition (if installed).
5. **Anti-Spoofing**: Optional blink detection ensures the subject is real and present.
6. **Attendance Logging**: When a match is found with sufficient confidence, attendance is recorded in Firebase and displayed to the user.
7. **PIN Authentication**: When face recognition confidence is low but above a minimum threshold, the system will prompt for PIN verification as a backup (now supports 4-6 digits).
8. **Visual Feedback**: Recognized individuals receive a confirmation message with a motivational quote directly on the screen.

## PIN Authentication System (v2.7.py)
The v2.7.py version includes an advanced, modern PIN authentication system:

- PIN verification is triggered when face recognition confidence falls between configurable thresholds
- Users can enter their PIN via mouse clicks on an on-screen keypad or via keyboard
- PINs can be 4-6 digits (numbers only)
- Limited attempts and timeout mechanism prevent brute force attacks
- Visual feedback shows remaining time and attempts
- PIN data is securely stored in Firebase alongside facial recognition images

## Attendance Storage
Attendance is stored in two ways:
1. **Firebase Database**: Records are stored in Firestore with the date as the document ID
2. **Interactive Viewing**: Press 'v' during operation to view today's attendance records

## Firebase Integration
The system uses Firebase Firestore to store attendance data and Firebase Storage for face images:
- The system connects to the 'facial-attendance-binus.firebasestorage.app' bucket
- Face images are organized by class and student name
- PINs are stored securely in the same structure
- Each day's attendance is stored as a separate document in Firestore
- Each person's record includes timestamp, status (Present/Late), class information, and lateness flag
- Requires a valid `serviceAccountKey.json` file with appropriate permissions

## Usage Instructions
1. Ensure all dependencies are installed (see above; install optional packages for best performance)
2. Set up a Firebase project and download the serviceAccountKey.json file
3. (Optional) Install face-alignment: `pip install face-alignment` for landmark detection
4. (Optional) Install dlib and cupy for faster face detection and GPU acceleration
5. Build your face dataset using make_dataset.py (including PINs)
6. Run v2.7.py (with advanced PIN authentication and anti-spoofing)
7. Position yourself in front of the camera
8. When recognized with high confidence, your attendance will be recorded automatically
9. If recognition confidence is moderate, you'll be prompted to enter your PIN
10. If anti-spoofing is enabled, blink as prompted to verify liveness
11. Press 'q' to quit the application

## Troubleshooting
- **Camera not working**: Ensure your webcam is properly connected and not used by another application
- **Poor recognition**: Improve lighting conditions and ensure face is clearly visible
- **Performance issues**: Adjust CONFIG settings to reduce processing load (increase frame_resize, skip_frames, or disable advanced features)
- **Unknown faces**: Rebuild your dataset with more varied images of each person
- **Firebase errors**: Check your internet connection and verify serviceAccountKey.json is valid
- **Dataset issues**: If your face isn't recognized properly, delete the encodings.pickle file to force rebuilding the dataset
- **HRNet download**: On first run, the face-alignment package will download model weights (~100MB) which requires internet connection
- **CUDA/GPU errors**: If you encounter GPU errors, set device to 'cpu' in the CONFIG settings or uninstall cupy
- **dlib errors**: If dlib is not installed, the system will fall back to OpenCV for face detection
- **Landmark detection issues**: If facial landmarks aren't working, ensure face-alignment is installed correctly
- **PIN verification not working**: Ensure PINs were properly created during dataset creation (now supports 4-6 digits)
- **Blink detection issues**: If anti-spoofing is enabled and not working, check camera quality and lighting
