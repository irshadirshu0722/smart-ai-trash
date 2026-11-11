
import serial
import time


SERIAL_PORT = "COM11"
BAUD_RATE = 9600
WAIT_AFTER_RESPONSE = 5


def main():

    arduino = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print("Virtual Arduino Started!")
    print("Waiting to communicate with Python AI...\n")

    try:

        while True:

            print("[Arduino] Sending: CAPTURE")

            arduino.write(b"CAPTURE\n")

            print("[Arduino] Waiting for AI response...")

            response = ""

            while not response:

                response = arduino.readline().decode(
                    "utf-8"
                ).strip()

            print(f"[Arduino] Received: {response}")

            if response ==  "BOTTLE":

                print("[Arduino] Action: Bottle detected!")

            elif response == "NORMAL_TRASH":

                print("[Arduino] Action: Normal trash detected!")

            else:

                print("[Arduino] Unknown response!")

            print(f"\n[Arduino] Next capture in {WAIT_AFTER_RESPONSE} seconds...\n")

            input("Enter to continue")


    except KeyboardInterrupt:

        print("\nVirtual Arduino stopped.")

    finally:

        arduino.close()


if __name__ == "__main__":
    main()
