import cv2
import mediapipe as mp

# START CLASS detectar posicoes
class PoseDetect:
  def __init__(self, mode=False, upper=False, smooth=True, min_detect=0.5, max_detect=0.5):
      self.mode = mode
      self.upper = upper
      self.smooth = smooth
      self.min_detect = min_detect
      self.max_detect = max_detect
    
      self.vect = mp.solutions.drawing_utils
      self.position = mp.solutions.pose
      self.pose = self.position.Pose(self.mode, self.upper, self.smooth, self.min_detect, self.max_detect )

  def getPose(self, frame):
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = self.pose.process(frameRGB)
    
    if res.pose_landmarks:
      self.vect.draw_landmarks(frame, res.pose_landmarks, self.position.POSE_CONNECTIONS)
    
    return frame
# END CLASS detectar posicoes


def main():
  target = cv2.VideoCapture(0)
  detect = PoseDetect()
  while True:
    success, frame = target.read()
    frame = detect.getPose(frame)
    
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
  
  
if __name__ == "__main__":
    main()