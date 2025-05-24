# Facial Recognition Attendance System

## Overview
This facial recognition system provides automated attendance tracking using computer vision and facial recognition technology. The system captures video from a webcam, identifies individuals by comparing faces against a known database, and logs attendance with timestamps. Data is stored in the cloud using Firebase.

## Features
- Real-time face detection and recognition
- Class-based student organization for educational settings
- Automatic attendance logging with timestamps
- Cloud-based attendance storage with Firebase
- Interactive attendance viewing
- Late arrival tracking based on configurable time threshold
- Display of motivational quotes upon attendance registration
- Visual confirmation overlay for successful attendance
- Advanced facial landmark detection using HRNet
- Performance optimization settings
- Caching of facial encodings for faster startup
- PIN-based backup authentication when face recognition confidence is low

## Files in this Project
- `v2.py`: The main facial recognition system with HRNet landmark detection and Firebase integration
- `v2.1.py`: Enhanced version with PIN authentication backup when face recognition confidence is low
- `make_dataset.py`: Script for creating the facial recognition database and collecting user PINs
- `serviceAccountKey.json`: Firebase credentials file (must be set up separately)

## Dependencies
- OpenCV (cv2)
- face_recognition
- numpy
- pickle
- datetime
- firebase-admin
- face-alignment (for HRNet facial landmark detection)

## Configuration Options
The program includes a CONFIG dictionary with the following adjustable settings:

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
- `device`: Device for HRNet model ('cpu' or 'cuda' for GPU acceleration)
- `first_run_warning`: Toggle first-run model download warning
- `confidence_threshold_for_pin`: Threshold below which PIN authentication is required
- `pin_timeout`: Time allowed for PIN entry in seconds
- `pin_allowed_attempts`: Number of PIN attempts before timeout

## Creating a Face Dataset
Before using the recognition system, you need to build a dataset of faces:

1. Run `make_dataset.py`
2. Enter your name when prompted
3. Enter your class name when prompted
4. Enter a 4-6 digit PIN when prompted (used for backup authentication)
5. Position yourself in front of the camera with good lighting
6. The script will guide you through capturing 20 images of your face:
   - Press 'c' to capture images manually one by one
   - Press 'a' to enable auto-capture mode
   - Press 'q' to quit the capturing process
7. Images and PIN will be stored in Firebase Storage in the `face_dataset/{class_name}/{your_name}` directory
8. Repeat for each person you want to recognize

## How Face Recognition Works
1. **Dataset Loading**: The system loads facial encodings from either a cached pickle file or builds encodings from the face image dataset in Firebase Storage.
2. **Face Detection**: Each video frame is processed to identify face locations.
3. **Face Recognition**: Detected faces are encoded and compared against the known dataset.
4. **Facial Landmark Detection**: HRNet is used to identify facial landmarks for enhanced recognition.
5. **Attendance Logging**: When a match is found with sufficient confidence, attendance is recorded in Firebase and displayed to the user.
6. **PIN Authentication**: When face recognition confidence is low but above a minimum threshold, the system will prompt for PIN verification as a backup.
7. **Visual Feedback**: Recognized individuals receive a confirmation message with a motivational quote directly on the screen.

## PIN Authentication System (v2.1.py)
The v2.1.py version includes a sleek, modern PIN authentication system:

- PIN verification is triggered when face recognition confidence falls between configurable thresholds
- Users can enter their PIN via mouse clicks on an on-screen keypad or via keyboard
- Limited attempts prevent brute force attacks
- Timeout mechanism ensures security
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
1. Ensure all dependencies are installed
2. Set up a Firebase project and download the serviceAccountKey.json file
3. Install face-alignment package: `pip install face-alignment`
4. Build your face dataset using make_dataset.py (including PINs)
5. Run v2.py (standard version) or v2.1.py (with PIN authentication)
6. Position yourself in front of the camera
7. When recognized with high confidence, your attendance will be recorded automatically
8. If recognition confidence is moderate, you'll be prompted to enter your PIN
9. Press 'q' to quit the application

## Troubleshooting
- **Camera not working**: Ensure your webcam is properly connected and not used by another application
- **Poor recognition**: Improve lighting conditions and ensure face is clearly visible
- **Performance issues**: Adjust CONFIG settings to reduce processing load (increase frame_resize, skip_frames)
- **Unknown faces**: Rebuild your dataset with more varied images of each person
- **Firebase errors**: Check your internet connection and verify serviceAccountKey.json is valid
- **Dataset issues**: If your face isn't recognized properly, delete the encodings.pickle file to force rebuilding the dataset
- **HRNet download**: On first run, the face-alignment package will download model weights (~100MB) which requires internet connection
- **CUDA errors**: If you encounter GPU errors, set device to 'cpu' in the CONFIG settings
- **Landmark detection issues**: If facial landmarks aren't working, ensure face-alignment is installed correctly
- **PIN verification not working**: Ensure PINs were properly created during dataset creation
