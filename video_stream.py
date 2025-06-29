import cv2
from threading import Thread, Lock

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.stream.isOpened():
            raise RuntimeError("Could not open camera.")
        self.grabbed, self.frame = self.stream.read()
        self.lock = Lock()
        self.stopped = False
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        # Flush 2 stale frames to reduce lag
        for _ in range(2):
            self.stream.grab()
        with self.lock:
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()
