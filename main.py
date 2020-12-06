import threading
from concurrent.futures.thread import ThreadPoolExecutor
import cv2
from imutils.video import FPS

from yolo_detection import yolo_detection
from drawing import Drawing


class GestureLock:
    def __init__(self):
        self.gesture = "No gesture"
        self.action = "No action"
        self.lock = threading.Lock()
        self.pairs = {"OK": "Yellow", "Palm": "Erasing", "Fist": "Green", "Two": "Brown", "Five": "Blue"}

    def get_gesture(self):
        with self.lock:
            gesture = self.gesture
            action = self.action
        return gesture, action

    def set_gesture(self, gesture):
        with self.lock:
            self.gesture = gesture
            self.action = self.pairs[gesture]


def main_cam(gesture_lock):
    cap = cv2.VideoCapture(0)
    address = "http://192.168.1.193:8080/video"
    cap.open(address)

    fps = FPS().start()
    fps_count = 0.0
    drawing = Drawing()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Video file finished.')
            break
        frame = cv2.flip(frame, 1)
        gesture, action = gesture_lock.get_gesture()
        text = action + " (" + gesture + ")"
        cv2.putText(frame, text=text, org=(30, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                    color=(0, 255, 20), thickness=5)
        frame = drawing.process_frame(frame, action)

        fps.update()
        if fps._numFrames == 25:
            fps.stop()
            fps_count = fps.fps()
            fps = FPS().start()
        cv2.putText(frame, text=str(round(fps_count, 1)), org=(1750, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2, color=(0, 255, 20), thickness=3)

        cv2.imshow('Cam', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lock = GestureLock()
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(main_cam, lock)
        executor.submit(yolo_detection, lock)
