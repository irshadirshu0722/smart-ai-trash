
from detector import BottleDetector
from camera import Camera

import serial
import time


SERIAL_PORT = "COM10" 
BAUD_RATE = 9600


def main():

    detector = BottleDetector(
        mode="high",
        confidence=0.25,
        save=True
    )

    camera = Camera(camera_index=0)

    arduino = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print("Python AI Waste Detector Started")
    print("Waiting for Arduino commands...")

    try:

        while True:

            if arduino.in_waiting > 0:

                command = arduino.readline().decode(
                    "utf-8"
                ).strip()

                print(f"\nArduino Command: {command}")

                if command == "CAPTURE":

                    print("Capturing image...")

                    image = camera.capture()

                    print("Running AI detection...")

                    result = detector.detect(image)


                    if result["arduino_value"]:
                        response = "BOTTLE"
                    else:
                        response = "NORMAL_TRASH"

                    arduino.write(
                        f"{response}\n".encode("utf-8")
                    )

                    print("Sent to Arduino:", response)

    except KeyboardInterrupt:

        print("\nStopping application...")

    finally:

        camera.release()
        arduino.close()


if __name__ == "__main__":
    main()
