"""
Face Dataset Creation Tool
-------------------------
This tool captures face images for the facial recognition system and
uploads them to Firebase Storage for use in the attendance system.
"""

import cv2
import os
import time
import tempfile
import shutil
import firebase_admin
from firebase_admin import credentials, storage

def main():
    # Initialize Firebase (only if not already initialized)
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        # Use the same bucket name as in the main application
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'facial-attendance-binus.firebasestorage.app'
        })

    bucket = storage.bucket()

    # Create a temporary folder for face images
    temp_dataset_path = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dataset_path}")

    # Keep a reference to the Firebase dataset path
    dataset_path = "face_dataset"

    # Open camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("❌ Error: Failed to load face detection model.")
        camera.release()
        return

    # Ask for the person's name
    name = input("Enter your name: ").strip()
    class_name = input("Enter your class name: ").strip()
    if not name or not class_name:
        print("❌ Error: Name and class cannot be empty.")
        camera.release()
        return
    
    # Create folder for the person's images
    person_folder = os.path.join(temp_dataset_path, class_name, name)
    os.makedirs(person_folder, exist_ok=True)

    # Configuration
    images_to_capture = 20
    countdown_time = 2  # seconds between captures
    count = 0
    
    print(f"\n📸 We'll capture {images_to_capture} images for facial recognition.")
    print("Position yourself in front of the camera with good lighting.")
    print("Try different angles and expressions for better recognition.")
    print("Press 'q' to quit or 'c' to capture an image when ready.")
    
    # Main loop
    capturing = False  # Start with manual capturing
    countdown = 0
    last_capture_time = 0
    
    while count < images_to_capture:
        success, frame = camera.read()
        if not success:
            print("❌ Error: Failed to capture frame.")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangle around faces
        face_detected = False
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_detected = True
        
        # Add progress information
        cv2.putText(display_frame, f"Progress: {count}/{images_to_capture}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        # Show status message
        if not face_detected:
            cv2.putText(display_frame, "No face detected! Position yourself in front of the camera.", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif capturing and countdown > 0:
            # Show countdown
            cv2.putText(display_frame, f"Taking photo in: {countdown}", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Press 'c' to capture or 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Face Capture", display_frame)
        
        # Handle countdown for automatic capture
        if capturing:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0:
                countdown -= 1
                last_capture_time = current_time
                
                if countdown <= 0:
                    capturing = False
                    # Only capture if face is detected
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            # Crop and save face image
                            face = frame[y:y + h, x:x + w]
                            face_resized = cv2.resize(face, (200, 200))
                            
                            # Save temporarily
                            temp_file_path = os.path.join(person_folder, f"{count}.jpg")
                            cv2.imwrite(temp_file_path, face_resized)
                            
                            # Upload to Firebase Storage
                            firebase_path = f"{dataset_path}/{class_name}/{name}/{count}.jpg"
                            blob = bucket.blob(firebase_path)
                            blob.upload_from_filename(temp_file_path)
                            
                            print(f"✅ Uploaded image {count+1}/{images_to_capture} to Firebase: {firebase_path}")
                            count += 1
                            break  # Only use the first face detected
                    else:
                        print("❌ No face detected during capture!")
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("⛔ Capture canceled by user.")
            break
            
        elif key == ord('c') and not capturing and face_detected:
            capturing = True
            countdown = countdown_time
            last_capture_time = time.time()
            print(f"📸 Capturing in {countdown} seconds...")
        
        # Auto-start capturing if preferred
        elif key == ord('a') and not capturing:
            print("🔄 Auto-capture mode enabled. Press 'q' to stop.")
            capturing = True
            countdown = countdown_time
            last_capture_time = time.time()

    # Clean up
    camera.release()
    cv2.destroyAllWindows()
    
    # Report status
    if count >= images_to_capture:
        print(f"✅ Successfully captured all {count} images for {name} in class {class_name}!")
    else:
        print(f"⚠️ Captured {count}/{images_to_capture} images for {name} in class {class_name}.")
    
    # Clean up temporary files - properly handle nested directories
    try:
        # Remove all the image files
        for file in os.listdir(person_folder):
            os.remove(os.path.join(person_folder, file))
        
        # Remove the directory structure
        os.rmdir(person_folder)
        
        # Try to remove the class directory if empty
        class_dir = os.path.join(temp_dataset_path, class_name)
        if os.path.exists(class_dir) and not os.listdir(class_dir):
            os.rmdir(class_dir)
            
        # Remove the temporary root directory
        if os.path.exists(temp_dataset_path):
            shutil.rmtree(temp_dataset_path, ignore_errors=True)
            
        print("🧹 Temporary files cleaned up.")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {str(e)}")
    
    # Instructions for next steps
    print("\n📋 Next steps:")
    print("1. Run the main facial recognition system (v2.py)")
    print("2. The system will automatically detect and recognize the newly added face")
    print("3. If recognition doesn't work well, try adding more images with different poses and lighting")

if __name__ == "__main__":
    main()
