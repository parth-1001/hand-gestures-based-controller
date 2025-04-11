import cv2
import mediapipe as mp
import pyautogui

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

prev_action = None
enter_pressed = False  # Flag to avoid repeat enter press

def is_open_palm(landmarks):
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]
    distance = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5
    return distance > 0.2

def is_closed_fist(landmarks):
    wrist_y = landmarks[0].y
    return all(landmarks[i].y > wrist_y for i in [4, 8, 12, 16, 20])

def is_left_hand(center_x, width):
    return center_x < width // 2

def fingers_up(landmarks):
    # Determine which fingers are up (1) or down (0)
    tips = [4, 8, 12, 16, 20]
    fingers = []
    
    # Thumb: horizontal check
    fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    
    # Other fingers: vertical check
    for tip in tips[1:]:
        fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)
    
    return fingers  # [thumb, index, middle, ring, pinky]

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    results = hands.process(rgb)
    action = "normal"

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[0]
            center_x = int(wrist.x * width)
            landmarks = hand_landmarks.landmark
            fingers = fingers_up(landmarks)
            hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'

            # --- ENTER GESTURE (right hand, index  up only)
            if hand_label == "Right" and fingers == [0, 1, 0, 0, 0]:
                if not enter_pressed:
                    pyautogui.press('enter')
                    enter_pressed = True
            else:
                enter_pressed = False  # Reset when fingers change

            # --- GAS/BRAKE DETECTION
            if is_open_palm(landmarks):
                if is_left_hand(center_x, width):
                    action = "brake"
                else:
                    action = "gas"
            elif is_closed_fist(landmarks):
                action = "normal"

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    # Handle gas/brake keys only when action changes
    if action != prev_action:
        pyautogui.keyUp('left')
        pyautogui.keyUp('right')
        if action == 'gas':
            pyautogui.keyDown('right')
        elif action == 'brake':
            pyautogui.keyDown('left')
        prev_action = action

    # Show action on screen
    cv2.putText(frame, f'Action: {action}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hill Climb Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pyautogui.keyUp('left')
pyautogui.keyUp('right')
