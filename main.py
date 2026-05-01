import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from face_mesh import draw_landmarks_on_image
from smooth import smooth_landmarks


alpha = 0.4

capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("not able to open cam")
    exit()

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=2,
    output_face_blendshapes=False
)

detector = vision.FaceLandmarker.create_from_options(options)

prev_smoothed = None

while True:
    ret, frame = capture.read()
    if not ret:
        print("not able to capture frame")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)

    #mediapipe output
    result = detector.detect_for_video(mp_image, timestamp_ms)
    cur_landmarks = result.face_landmarks

    if cur_landmarks:
        num_faces = len(cur_landmarks)
        num_points = len(cur_landmarks[0])

        # Convert → NumPy (faces, 468, 2)
        curr_np = np.zeros((num_faces, num_points, 2), dtype=np.float32)

        for f in range(num_faces):
            for i in range(num_points):
                curr_np[f, i, 0] = cur_landmarks[f][i].x
                curr_np[f, i, 1] = cur_landmarks[f][i].y

        # Smoothing
        smoothed, prev_smoothed = smooth_landmarks(
            curr_np,
            prev_smoothed,
            alpha
        )

        annotated = draw_landmarks_on_image(rgb_frame, smoothed)
        annotated_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)

    else:
        prev_smoothed = None
        annotated_bgr = frame

    cv.imshow("output", annotated_bgr)

    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()