from camera import Camera
import cv2

camera = Camera(
    camera_index=0,
    width=1920,
    height=1080
)

try:
    while True:
        image = camera.capture()

        cv2.imshow("Camera Test", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()