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
- Display of motivational quotes (SPIRIT Values and IB Learner Profile) upon attendance registration
- Visual confirmation overlay for successful attendance
- Facial landmark detection for enhanced recognition
- Performance optimization settings
- Caching of facial encodings for faster startup

## Files in this Project
- `v1.2.py`: The main facial recognition and attendance tracking program with Firebase integration
- `make_dataset.py`: Script for creating the facial recognition database
- `shape_predictor_68_face_landmarks.dat`: Required for facial landmark detection (must be downloaded separately at https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
- `serviceAccountKey.json`: Firebase credentials file

## Dependencies
- OpenCV (cv2)
- face_recognition
- dlib
- numpy
- pickle
- datetime
- firebase-admin

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
- `facial_landmarks_path`: Path to the facial landmark predictor file
- `show_landmarks`: Display facial landmarks on detected faces

## Creating a Face Dataset
Before using the recognition system, you need to build a dataset of faces:

1. Run `make_dataset.py`
2. Enter your name when prompted
3. Enter your class name when prompted
4. Position yourself in front of the camera with good lighting
5. The script will guide you through capturing 20 images of your face:
   - Press 'c' to capture images manually one by one
   - Press 'a' to enable auto-capture mode
   - Press 'q' to quit the capturing process
6. Images will be stored in Firebase Storage in the `face_dataset/{class_name}/{your_name}` directory
7. Repeat for each person you want to recognize

## How Face Recognition Works
1. **Dataset Loading**: The system loads facial encodings from either a cached pickle file or builds encodings from the face image dataset in Firebase Storage.
2. **Face Detection**: Each video frame is processed to identify face locations.
3. **Face Recognition**: Detected faces are encoded and compared against the known dataset.
4. **Attendance Logging**: When a match is found with sufficient confidence, attendance is recorded in Firebase and displayed to the user.
5. **Visual Feedback**: Recognized individuals receive a confirmation message with a motivational quote directly on the screen.

## Attendance Storage
**Firebase Database**: Records are stored in Firestore with the date as the document ID

## Firebase Integration
The system uses Firebase Firestore to store attendance data and Firebase Storage for face images:
- The system connects to the 'facial-attendance-binus.firebasestorage.app' bucket
- Face images are organized by class and student name
- Each day's attendance is stored as a separate document in Firestore
- Each person's record includes timestamp, status (Present/Late), class information, and lateness flag
- Requires a valid `serviceAccountKey.json` file with appropriate permissions

## Usage Instructions
1. Ensure all dependencies are installed
2. Download the shape_predictor_68_face_landmarks.dat file (available from dlib)
3. Set up a Firebase project and download the serviceAccountKey.json file
4. Build your face dataset using make_dataset.py
5. Run v1.2.py
6. Position yourself in front of the camera
7. When recognized, your attendance will be recorded and confirmation displayed with your name and class
8. Press 'q' to quit the application

## Troubleshooting
- **Camera not working**: Ensure your webcam is properly connected and not used by another application
- **Poor recognition**: Improve lighting conditions and ensure face is clearly visible
- **Performance issues**: Adjust CONFIG settings to reduce processing load (increase frame_resize, skip_frames)
- **Unknown faces**: Rebuild your dataset with more varied images of each person
- **Firebase errors**: Check your internet connection and verify serviceAccountKey.json is valid
- **Dataset issues**: If your face isn't recognized properly, delete the encodings.pickle file to force rebuilding the dataset
- **Class organization**: Make sure students are consistently categorized in the same class across dataset creation and recognition
