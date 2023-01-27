import cv2
import mediapipe as mp

vect = mp.solutions.drawing_utils
position = mp.solutions.pose
pose = position.Pose()
target = cv2.VideoCapture(0)

while True:
  success, frame = target.read()
  # adicionar cores em coordenadas no frame
  frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  res = pose.process(frameRGB)
  
  if res.pose_landmarks:
    vect.draw_landmarks(frame, res.pose_landmarks, position.POSE_CONNECTIONS)
      
  cv2.imshow("Image", frame)
  cv2.waitKey(1)