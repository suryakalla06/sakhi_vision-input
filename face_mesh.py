#@markdown We implemented some functions to visualize the face landmark detection results.
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt

FLC = vision.FaceLandmarksConnections

# give input in RGB format cause the input frame is also in rgb format.
green   = (0, 255, 0)
blue    = (0, 0, 255)
red     = (255, 0, 0)
cyan    = (0, 255, 255)
yellow  = (255, 255, 0)
white   = (255, 255, 255)


def draw_landmarks_on_image(rgb_image, face_landmarks_list):

  
  annotated_image = np.copy(rgb_image)

  y_dist = rgb_image.shape[0]
  x_dist = rgb_image.shape[1]

  eyes = FLC.FACE_LANDMARKS_LEFT_EYE + FLC.FACE_LANDMARKS_RIGHT_EYE
  eye_brows = FLC.FACE_LANDMARKS_LEFT_EYEBROW + FLC.FACE_LANDMARKS_RIGHT_EYEBROW
  iris = FLC.FACE_LANDMARKS_LEFT_IRIS + FLC.FACE_LANDMARKS_RIGHT_IRIS

  regions = [
    # (FLC.FACE_LANDMARKS_TESSELATION,white),
    (eyes, green),
    (iris, cyan),
    (eye_brows, yellow),
    (FLC.FACE_LANDMARKS_LIPS, red),
    (FLC.FACE_LANDMARKS_NOSE, blue),
    (FLC.FACE_LANDMARKS_FACE_OVAL, white)
  ]

  # Loop through the detected faces to visualize.
  for face_landmarks in face_landmarks_list:

    # Draw the face landmarks.
    for connections,color in regions:
      for pair in connections:
        if pair.start<len(face_landmarks) and pair.end < len(face_landmarks):
          pt1 = face_landmarks[pair.start]
          pt2 = face_landmarks[pair.end]
          y1 = int(y_dist*pt1[1])
          x1 = int(x_dist*pt1[0])
          y2 = int(y_dist*pt2[1])
          x2 = int(x_dist*pt2[0])
          cv.line(annotated_image,(x1,y1),(x2,y2),color,1,cv.LINE_AA)

  return annotated_image