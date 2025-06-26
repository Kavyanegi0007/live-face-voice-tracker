import cv2
from graph.state import AgentState

# Load pre-trained OpenCV eye detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def estimate_gaze_direction(eye_roi):
    """Fixed gaze direction estimation"""
    if eye_roi is None or eye_roi.size == 0:
        return "unknown"
    
    # Check if eye is large enough for analysis
    h, w = eye_roi.shape
    if w < 10 or h < 5:
        return "unknown"
    
    # Divide eye ROI into left and right halves
    left_side = eye_roi[:, :w//2]
    right_side = eye_roi[:, w//2:]

    # Count dark pixels (the pupil is dark)
    left_dark = cv2.countNonZero(cv2.threshold(left_side, 50, 255, cv2.THRESH_BINARY_INV)[1])
    right_dark = cv2.countNonZero(cv2.threshold(right_side, 50, 255, cv2.THRESH_BINARY_INV)[1])

    # Adaptive threshold based on eye size
    threshold = max(10, min(w * h * 0.1, 50))
    
    if abs(left_dark - right_dark) < threshold:
        return "center"
    elif left_dark > right_dark:
        return "left"   # FIXED: More dark pixels on left = looking left
    else:
        return "right"  # FIXED: More dark pixels on right = looking right

def eye_node(state: AgentState) -> AgentState:
    """Fixed eye tracking node"""
    try:
        frame = state.get("frame")
        if frame is None:
            return state
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # More lenient face detection parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

        direction = "unknown"
        
        for (x, y, w, h) in faces:
            # Focus on upper half of face for better eye detection
            roi_gray = gray[y:y+h//2, x:x+w]
            
            # More lenient eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(10, 10))
            
            if len(eyes) == 0:
                direction = "unknown"
                break

            # Use the largest eye (most reliable detection)
            largest_eye = max(eyes, key=lambda eye: eye[2] * eye[3])
            (ex, ey, ew, eh) = largest_eye
            
            # Check if eye is large enough
            if ew < 15 or eh < 10:
                direction = "unknown"
                break
            
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            direction = estimate_gaze_direction(eye_roi)
            break  # Process only the first face

        # Initialize eye_data if it doesn't exist
        if "eye_data" not in state:
            state["eye_data"] = {
                "gaze_direction_log": [],
                "total_samples": 0,
                "blink_count": 0,
                "gaze_counts": {"left": 0, "right": 0, "center": 0, "unknown": 0}
            }

        # Update state
        state["eye_data"]["gaze_direction_log"].append(direction)
        state["eye_data"]["total_samples"] += 1

        # Count gaze directions
        if direction in state["eye_data"]["gaze_counts"]:
            state["eye_data"]["gaze_counts"][direction] += 1

        # Simple blink detection based on no eye detection
        if direction == "unknown" and len(faces) > 0:
            state["eye_data"]["blink_count"] += 1

        print(f"[DEBUG] Eye direction: {direction}, Total samples: {state['eye_data']['total_samples']}")
        
    except Exception as e:
        print(f"Eye node error: {e}")
        # Initialize state if it doesn't exist
        if "eye_data" not in state:
            state["eye_data"] = {
                "gaze_direction_log": [],
                "total_samples": 0,
                "blink_count": 0,
                "gaze_counts": {"left": 0, "right": 0, "center": 0, "unknown": 0}
            }
        state["eye_data"]["total_samples"] += 1

    return state