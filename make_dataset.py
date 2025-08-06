"""
Face Dataset Creation Tool - Optimized Edition
----------------------------------------------
This tool captures face images for the facial recognition system and
uploads them to Firebase Storage for use in the attendance system.
Features intelligent face detection and optimized image processing.
"""

import cv2
import os
import time
import tempfile
import shutil
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np

def main():
    # Initialize Firebase (only if not already initialized)
    if not firebase_admin._apps:
        cred = credentials.Certificate("facial-attendance-binus-firebase-adminsdk-fbsvc-663eb05a63.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'facial-attendance-binus.firebasestorage.app'
        })

    bucket = storage.bucket()

    # Create a temporary folder for face images
    temp_dataset_path = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dataset_path}")

    dataset_path = "face_dataset"

    # Open camera with optimized settings
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Error: Unable to access the camera.")
        return
    
    # Optimize camera settings
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("❌ Error: Failed to load face detection model.")
        camera.release()
        return

    # Ask for the person's information
    name = input("Enter your name: ").strip()
    class_name = input("Enter your class name: ").strip()
    
    # PIN entry with validation
    pin = ""
    while True:
        pin = input("Enter your PIN (4-6 digits): ").strip()
        if not pin:
            print("❌ Error: PIN cannot be empty.")
            continue
        if not pin.isdigit():
            print("❌ Error: PIN must contain only numbers.")
            continue
        if not (4 <= len(pin) <= 6):
            print("❌ Error: PIN must be between 4-6 digits.")
            continue
        break
        
    if not name or not class_name:
        print("❌ Error: Name and class cannot be empty.")
        camera.release()
        return
    
    # Create folder for the person's images
    person_folder = os.path.join(temp_dataset_path, class_name, name)
    os.makedirs(person_folder, exist_ok=True)
    
    # Store PIN in Firebase
    pin_path = f"{dataset_path}/{class_name}/{name}/pin.txt"
    pin_blob = bucket.blob(pin_path)
    pin_blob.upload_from_string(pin)
    print(f"✅ PIN stored securely in Firebase")

    # Configuration
    images_to_capture = 15  # Increased for better recognition
    countdown_time = 2
    count = 0
    quality_threshold = 100  # Minimum face area for quality check
    
    print(f"\n📸 We'll capture {images_to_capture} high-quality images for facial recognition.")
    print("Position yourself in front of the camera with good lighting.")
    print("Try different angles and expressions for better recognition.")
    print("Press 'c' to capture an image when ready, or 'q' to quit.")
    
    # Main loop with enhanced quality control
    capturing = False
    countdown = 0
    last_capture_time = 0
    captured_positions = []  # Track face positions to encourage variety
    
    while count < images_to_capture:
        success, frame = camera.read()
        if not success:
            print("❌ Error: Failed to capture frame.")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Detect faces with enhanced parameters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100),  # Larger minimum size for better quality
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Enhanced face quality assessment
        best_face = None
        best_quality = 0
        
        for (x, y, w, h) in faces:
            # Calculate face quality score
            face_area = w * h
            center_x = x + w // 2
            center_y = y + h // 2
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            
            # Distance from center (prefer centered faces)
            center_distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
            max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
            center_score = 1 - (center_distance / max_distance)
            
            # Size score (prefer larger faces)
            max_area = frame.shape[0] * frame.shape[1] * 0.25  # 25% of frame
            size_score = min(face_area / max_area, 1.0)
            
            # Overall quality score
            quality_score = (face_area * 0.4) + (center_score * 100) + (size_score * 100)
            
            if quality_score > best_quality and face_area > quality_threshold:
                best_quality = quality_score
                best_face = (x, y, w, h)
        
        # Draw rectangle around best face
        face_detected = False
        if best_face is not None:
            x, y, w, h = best_face
            
            # Color based on quality
            if best_quality > 200:
                color = (0, 255, 0)  # Green for excellent quality
                quality_text = "Excellent"
            elif best_quality > 150:
                color = (0, 255, 255)  # Yellow for good quality
                quality_text = "Good"
            else:
                color = (0, 165, 255)  # Orange for acceptable quality
                quality_text = "Acceptable"
            
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, f"Quality: {quality_text}", 
                       (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            face_detected = True
        
        # Progress and status information
        cv2.putText(display_frame, f"Progress: {count}/{images_to_capture}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(display_frame, f"Unique positions: {len(set(captured_positions))}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                   
        # Status messages
        if not face_detected:
            cv2.putText(display_frame, "No quality face detected! Position yourself properly.", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif capturing and countdown > 0:
            cv2.putText(display_frame, f"Capturing in: {countdown}", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Press 'c' to capture or 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("High-Quality Face Capture", display_frame)
        
        # Handle countdown for automatic capture
        if capturing:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0:
                countdown -= 1
                last_capture_time = current_time
                
                if countdown <= 0:
                    capturing = False
                    
                    if best_face is not None:
                        x, y, w, h = best_face
                        
                        # Add some padding around the face
                        padding = 20
                        x_start = max(0, x - padding)
                        y_start = max(0, y - padding)
                        x_end = min(frame.shape[1], x + w + padding)
                        y_end = min(frame.shape[0], y + h + padding)
                        
                        # Crop and enhance face image
                        face = frame[y_start:y_end, x_start:x_end]
                        
                        # Resize to consistent size with high quality
                        face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
                        
                        # Enhance image quality
                        # Histogram equalization for better contrast
                        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                        face_eq = cv2.equalizeHist(face_gray)
                        face_enhanced = cv2.cvtColor(face_eq, cv2.COLOR_GRAY2BGR)
                        
                        # Blend original and enhanced
                        face_final = cv2.addWeighted(face_resized, 0.7, face_enhanced, 0.3, 0)
                        
                        # Save temporarily
                        temp_file_path = os.path.join(person_folder, f"{count:03d}.jpg")
                        
                        # Save with high quality
                        cv2.imwrite(temp_file_path, face_final, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        # Upload to Firebase Storage
                        firebase_path = f"{dataset_path}/{class_name}/{name}/{count:03d}.jpg"
                        blob = bucket.blob(firebase_path)
                        blob.upload_from_filename(temp_file_path)
                        
                        # Track position for variety
                        center_x = x + w // 2
                        center_y = y + h // 2
                        position_key = f"{center_x//50}_{center_y//50}"  # Grid-based position
                        captured_positions.append(position_key)
                        
                        print(f"✅ Uploaded high-quality image {count+1}/{images_to_capture} to Firebase")
                        print(f"   Quality score: {best_quality:.1f}, Position: {position_key}")
                        count += 1
                    else:
                        print("❌ No suitable face detected during capture!")
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("⛔ Capture canceled by user.")
            break
            
        elif key == ord('c') and not capturing and face_detected:
            capturing = True
            countdown = countdown_time
            last_capture_time = time.time()
            print(f"📸 Capturing high-quality image in {countdown} seconds...")
        
        # Auto-capture mode
        elif key == ord('a') and not capturing and face_detected:
            print("🔄 Auto-capture mode enabled. Press 'q' to stop.")
            capturing = True
            countdown = countdown_time
            last_capture_time = time.time()

    # Clean up
    camera.release()
    cv2.destroyAllWindows()
    
    # Report status
    if count >= images_to_capture:
        unique_positions = len(set(captured_positions))
        print(f"✅ Successfully captured all {count} high-quality images for {name} in class {class_name}!")
        print(f"📊 Image variety: {unique_positions} unique positions captured")
        
        # Quality assessment
        if unique_positions >= images_to_capture * 0.7:
            print("🎯 Excellent variety! This should provide very good recognition accuracy.")
        elif unique_positions >= images_to_capture * 0.5:
            print("👍 Good variety! Recognition should work well.")
        else:
            print("⚠️ Consider adding more images from different angles for better accuracy.")
    else:
        print(f"⚠️ Captured {count}/{images_to_capture} images for {name} in class {class_name}.")
    
    # Clean up temporary files
    try:
        if os.path.exists(temp_dataset_path):
            shutil.rmtree(temp_dataset_path, ignore_errors=True)
        print("🧹 Temporary files cleaned up.")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {str(e)}")
    
    # Instructions for next steps
    print("\n📋 Next steps:")
    print("1. Run the main facial recognition system.")
    print("2. The system will automatically detect and recognize the newly added face")
    print("3. If recognition doesn't work well, try adding more images with different poses and lighting")

if __name__ == "__main__":
    main()
