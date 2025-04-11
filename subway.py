import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe and drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Timing
prev_action_time = 0
action_delay = 0.2  # seconds
current_action = ""

# Helper: which fingers are up
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: right hand - tip < base; left hand - tip > base
    fingers.append(1 if abs(hand_landmarks.landmark[tips_ids[0]].x - hand_landmarks.landmark[tips_ids[0] - 2].x) > 0.02 else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y else 0)

    return fingers

# Helper: determine left or right hand
def get_hand_label(handedness):
    return handedness.classification[0].label  # "Left" or "Right"

# Helper: determine if palm is facing camera using thumb direction
def is_palm_facing(hand_landmarks, hand_label):
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_base_x = hand_landmarks.landmark[2].x

    if hand_label == "Right":
        return thumb_tip_x < thumb_base_x  # Thumb points left
    elif hand_label == "Left":
        return thumb_tip_x > thumb_base_x  # Thumb points right
    return False

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    current_time = time.time()

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = get_hand_label(handedness)
            fingers = fingers_up(hand_landmarks)
            palm_facing = is_palm_facing(hand_landmarks, label)

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_time - prev_action_time > action_delay:
                # Closed fist
                if fingers == [0, 0, 0, 0, 0]:
                    current_action = "No Action"

                # Open palm with palm facing
                elif fingers == [1, 1, 1, 1, 1] and palm_facing:
                    if label == "Right":
                        pyautogui.press("right")
                        current_action = "Move Right"
                        prev_action_time = current_time
                    elif label == "Left":
                        pyautogui.press("left")
                        current_action = "Move Left"
                        prev_action_time = current_time

                # Two fingers up (index + middle)
                elif fingers == [1, 1, 1, 0, 0]:
                    if label == "Right":
                        pyautogui.press("up")
                        current_action = "Jump"
                        prev_action_time = current_time
                    elif label == "Left":
                        pyautogui.press("down")
                        current_action = "Roll"
                        prev_action_time = current_time

    # Display current action
    cv2.putText(image, f"Action: {current_action}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    cv2.imshow("Subway Surfers Gesture Control", image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
