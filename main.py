import cv2
import mediapipe as mp
import pyautogui as p
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Gesture recognition coordinates
finger_coordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_coordinate = (4, 2)

last_action_time = time.time()  # Prevents multiple keypress spam

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # Mirror the camera
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    multi_landmarks = results.multi_hand_landmarks

    if multi_landmarks:
        hand_points = []
        for hand_lms in multi_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            for lm in hand_lms.landmark:
                h, w, _ = img.shape
                hand_points.append((int(lm.x * w), int(lm.y * h)))

        # Check number of raised fingers
        up_count = sum(1 for coord in finger_coordinates if hand_points[coord[0]][1] < hand_points[coord[1]][1])
        if hand_points[thumb_coordinate[0]][0] > hand_points[thumb_coordinate[1]][0]:  
            up_count += 1  # Thumb check

        # Display count
        cv2.putText(img, f"Fingers: {up_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Prevent multiple keypress spam
        if time.time() - last_action_time > 1:
            if up_count == 1:
                p.press("space")  # Play/Pause
                cv2.putText(img, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif up_count == 2:
                p.press("up")  # Volume Up
                cv2.putText(img, "Volume Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif up_count == 3:
                p.press("down")  # Volume Down
                cv2.putText(img, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif up_count == 4:
                p.press("right")  # Forward
                cv2.putText(img, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif up_count == 5:
                p.press("left")  # Backward
                cv2.putText(img, "Backward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            last_action_time = time.time()  # Reset timer

    # Show output
    cv2.imshow("Hand Gesture Control", img)

    # Exit when 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
