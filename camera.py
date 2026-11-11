
import cv2
import threading
import time


class Camera:

    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720
    ):

        self.camera = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            raise RuntimeError("Cannot access camera!")

        self.camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )

        self.thread.start()

        # Wait until the first frame is available
        while self.frame is None:
            time.sleep(0.01)

        print("Camera initialized successfully!")

    def _update(self):

        while self.running:

            success, frame = self.camera.read()

            if success:

                with self.lock:
                    self.frame = frame

    def capture_raw(self):

        with self.lock:

            if self.frame is None:
                raise RuntimeError("No camera frame available!")

            return self.frame.copy()

    def capture(self):

        return self.capture_raw()

    def release(self):

        self.running = False
        self.thread.join(timeout=2)

        self.camera.release()
        cv2.destroyAllWindows()
