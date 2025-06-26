# simple_test_main.py - Test without graph system
import cv2
from time import time

def simple_head_pose_detection(frame):
    """Simple head pose detection"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    if len(faces) == 0:
        return "neutral"
    
    # Use largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Draw face rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Calculate face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    
    # Draw centers
    cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)  # Face center (green)
    cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)  # Frame center (red)
    
    # Calculate offsets
    x_offset = face_center_x - frame_center_x
    y_offset = face_center_y - frame_center_y
    
    # More sensitive thresholds
    x_threshold = frame.shape[1] * 0.08  # 8% of frame width
    y_threshold = frame.shape[0] * 0.08  # 8% of frame height
    
    # Determine direction
    direction = "neutral"
    if abs(x_offset) > abs(y_offset):
        if x_offset > x_threshold:
            direction = "right"
        elif x_offset < -x_threshold:
            direction = "left"
    else:
        if y_offset > y_threshold:
            direction = "down"
        elif y_offset < -y_threshold:
            direction = "up"
    
    # Add debug info to frame
    cv2.putText(frame, f"Direction: {direction}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Offset: ({x_offset:.0f}, {y_offset:.0f})", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return direction

def simple_eye_tracking(frame):
    """Simple eye tracking"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    
    if len(faces) == 0:
        return "unknown"
    
    # Use first face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h//2, x:x+w]  # Upper half of face
    roi_color = frame[y:y+h//2, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(10, 10))
    
    if len(eyes) == 0:
        return "unknown"
    
    # Use largest eye
    largest_eye = max(eyes, key=lambda eye: eye[2] * eye[3])
    (ex, ey, ew, eh) = largest_eye
    
    # Draw eye rectangle
    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Simple gaze detection
    eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
    
    if ew < 15 or eh < 10:
        return "unknown"
    
    # Divide eye into left and right halves
    left_side = eye_roi[:, :ew//2]
    right_side = eye_roi[:, ew//2:]
    
    # Count dark pixels
    left_dark = cv2.countNonZero(cv2.threshold(left_side, 50, 255, cv2.THRESH_BINARY_INV)[1])
    right_dark = cv2.countNonZero(cv2.threshold(right_side, 50, 255, cv2.THRESH_BINARY_INV)[1])
    
    threshold = max(10, min(ew * eh * 0.1, 30))
    
    if abs(left_dark - right_dark) < threshold:
        return "center"
    elif left_dark > right_dark:
        return "left"
    else:
        return "right"

def simple_test_main():
    """Test both head pose and eye tracking without graph system"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("🚀 Simple test running. Press 'q' to quit.")
    print("📹 Testing both head pose and eye tracking.")
    
    # Simple state tracking
    state = {
        "head_data": {
            "tilt_counts": {"left": 0, "right": 0, "up": 0, "down": 0, "neutral": 0},
            "total_samples": 0
        },
        "eye_data": {
            "gaze_counts": {"left": 0, "right": 0, "center": 0, "unknown": 0},
            "total_samples": 0
        },
        "start_time": time()
    }
    
    frame_count = 0
    last_print_time = time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Process every 5th frame to reduce noise
            if frame_count % 5 == 0:
                # Head pose detection
                head_direction = simple_head_pose_detection(frame)
                if head_direction in state["head_data"]["tilt_counts"]:
                    state["head_data"]["tilt_counts"][head_direction] += 1
                state["head_data"]["total_samples"] += 1
                
                # Eye tracking
                eye_direction = simple_eye_tracking(frame)
                if eye_direction in state["eye_data"]["gaze_counts"]:
                    state["eye_data"]["gaze_counts"][eye_direction] += 1
                state["eye_data"]["total_samples"] += 1
                
                print(f"Frame {frame_count}: Head={head_direction}, Eye={eye_direction}")
            
            # Display stats on frame
            y_pos = 110
            cv2.putText(frame, f"Head Samples: {state['head_data']['total_samples']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25
            
            for direction, count in state["head_data"]["tilt_counts"].items():
                cv2.putText(frame, f"H-{direction}: {count}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_pos += 20
            
            # Eye stats on right side
            x_pos = frame.shape[1] - 200
            y_pos = 110
            cv2.putText(frame, f"Eye Samples: {state['eye_data']['total_samples']}", 
                       (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
            
            for direction, count in state["eye_data"]["gaze_counts"].items():
                cv2.putText(frame, f"E-{direction}: {count}", (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                y_pos += 20
            
            # Print stats every 10 seconds
            current_time = time()
            if current_time - last_print_time >= 10:
                print(f"\n📊 Stats after {current_time - state['start_time']:.1f}s:")
                print(f"Head: {state['head_data']['tilt_counts']}")
                print(f"Eye: {state['eye_data']['gaze_counts']}")
                last_print_time = current_time
            
            cv2.imshow("Simple Test - Head & Eye Tracking", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Stop after 2 minutes
            if time() - state["start_time"] >= 2 * 60:
                print("⏹️  Test completed after 2 minutes.")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print("🏁 SIMPLE TEST RESULTS:")
        print(f"Total frames: {frame_count}")
        print(f"Head tracking samples: {state['head_data']['total_samples']}")
        print(f"Head directions: {state['head_data']['tilt_counts']}")
        print(f"Eye tracking samples: {state['eye_data']['total_samples']}")
        print(f"Eye directions: {state['eye_data']['gaze_counts']}")

if __name__ == "__main__":
    simple_test_main()