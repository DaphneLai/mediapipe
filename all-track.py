import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Known width of a hand in cm (adjust based on your hand size)
KNOWN_HAND_WIDTH = 8.0  # Approximate hand width in cm
FOCAL_LENGTH = 500  # Adjust this value by testing

def calculate_distance(known_width, focal_length, perceived_width):
    """Estimate distance using the formula: Distance = (Known Width * Focal Length) / Perceived Width"""
    if perceived_width == 0:
        return 0
    return (known_width * focal_length) / perceived_width

def count_fingers(hand_landmarks):
    """
    Detects the number of extended fingers.
    Uses landmark positions to determine which fingers are open.
    """
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    thumb_tip = 4
    finger_count = 0

    # Get wrist and finger positions
    wrist_y = hand_landmarks.landmark[0].y  # Wrist y-coordinate
    
    # Count extended fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    # Check if thumb is extended
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[3].x:  
        finger_count += 1

    return finger_count

def recognize_gesture(finger_count):
    """ Recognizes gestures based on the number of extended fingers """
    if finger_count == 5:
        return "Open Palm 🖐️"
    elif finger_count == 0:
        return "Closed Fist ✊"
    # elif finger_count == 1:
    #     return "Pointing ☝️"
    # elif finger_count == 2 and hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
    #     return "Thumbs Up 👍"
    return "Unknown Gesture: " + str(finger_count)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count extended fingers and recognize gesture
            finger_count = count_fingers(hand_landmarks)
            gesture = recognize_gesture(finger_count)

            # Get bounding box around the hand
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            # Convert to pixel values
            h, w, _ = frame.shape
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)

            # Compute hand width in pixels
            hand_width_pixels = x_max - x_min

            # Estimate distance
            distance = calculate_distance(KNOWN_HAND_WIDTH, FOCAL_LENGTH, hand_width_pixels)

            # Find center position of the hand
            hand_x = (x_min + x_max) // 2
            hand_y = (y_min + y_max) // 2

            # Draw bounding box and center point
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.circle(frame, (hand_x, hand_y), 5, (0, 0, 255), -1)

            # Display position and distance
            cv2.putText(frame, f"X: {hand_x}, Y: {hand_y}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Distance: {int(distance)} cm", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Display gesture text
            cv2.putText(frame, gesture, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 255), 2, cv2.LINE_AA) 

    # Show the frame
    cv2.imshow("Hand Tracking with Distance & Position", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
