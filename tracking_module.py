import cv2
import numpy as np
import mediapipe as mp


class HandDetector:

    def __init__(self,
                 mode: bool = False,
                 max_hands: int = 2,
                 detection_conf: float = 0.5,
                 tracking_conf: float = 0.5) -> None:

        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.tracking_conf
            )
        self.draw = mp.solutions.drawing_utils
        self.finger_ids = [4, 8, 12, 16, 20]

    def find_hands(self,
                   frame: np.array,
                   draw_hands: bool = True) -> np.array:

        rgb_frame = cv2.cvtColor(frame,
                                 cv2.COLOR_BGR2RGB)

        self.res = self.hands.process(rgb_frame)

        if self.res.multi_hand_landmarks:            
            for landmark in self.res.multi_hand_landmarks:
                if draw_hands:
                    self.draw.draw_landmarks(frame, landmark,
                                             self.mphands.HAND_CONNECTIONS
                                             )
                    
        return frame
    
    def find_position(self,
                      frame: str,
                      hand_number: int = 0,
                      draw: bool = True) -> list[list[int, float, float]]:
        
        self.landmarks =  []

        if self.res.multi_hand_landmarks:

            hand_landmarks = self.res.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmarks.append([id, cx, cy])

                if draw and (id == 8 or id == 12):
                    cv2.circle(frame, (cx, cy), 7, (255, 255, 0), 5)

        return self.landmarks
    
    def fingers_up(self) -> list[int]:

        fingers = []

        # большой палец
        if (self.landmarks[self.finger_ids[0]][1] < \
        self.landmarks[self.finger_ids[0] - 1][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        # остальные 4 пальца
        for i in range(1, 5):
            if (self.landmarks[self.finger_ids[i]][2] < \
                self.landmarks[self.finger_ids[i] - 2][2]):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main() -> None:

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:

        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        _ = detector.find_position(frame)

        cv2.imshow('Camera', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
