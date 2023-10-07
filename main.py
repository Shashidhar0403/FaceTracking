import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

double_click_threshold = 0.7  # Increased threshold for double click
double_click_time = 4  # 4 seconds for double click

last_click_time = time.time()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        if (left[0].y - left[1].y) < double_click_threshold:
            current_time = time.time()
            if current_time - last_click_time > double_click_time:
                pyautogui.doubleClick()
                last_click_time = current_time
        if 0 <= screen_x <= screen_w and 0 <= screen_y <= screen_h:
            pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)
