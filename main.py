import cv2
import numpy as np

from tracking_module import HandDetector

BRUSH_COLOR = (255, 0, 0)
BRUSH_THICKNESS = {
    0: 75,
    1: 15
}


def main(brush_color: tuple = BRUSH_COLOR,
         brush_thickness: dict = BRUSH_THICKNESS,
         flag: bool = True) -> None:

    xp, yp = 0, 0
    draw_color = brush_color
    header = cv2.imread('images/header.png')
    img_canvas = np.ones((720, 1280, 3)) * 255

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(max_hands=1)

    while True:

        _, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw_hands=True)
        landmarks = detector.find_position(img, draw=False)

        if landmarks:
            x1, y1 = landmarks[8][1:]
            x2, y2 = landmarks[12][1:]

            fingers = detector.fingers_up()

            # режим выбора
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < 216:
                    if 0 < x1 < 640:
                        draw_color = brush_color
                        flag = True
                    elif 640 < x1 < 1280:
                        draw_color = (255, 255, 255)
                        flag = False
                cv2.rectangle(
                    img, (x1, y1 - 25), (x2, y2 + 25),
                    draw_color, cv2.FILLED
                    )

            # режим рисования
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp, = x1, y1
                cv2.line(
                    img_canvas, (xp, yp), (x1, y1),
                    draw_color, brush_thickness[flag]
                    )
                xp, yp, = x1, y1

        img[:216, :] = header

        cv2.imshow('Canvas', img_canvas)
        cv2.imshow('Camera', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
