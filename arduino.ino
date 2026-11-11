
#include <Servo.h>

// PIN CONFIGURATION

const int ENTRY_TRIG = 2;
const int ENTRY_ECHO = 3;

const int NORMAL_TRIG = 4;
const int NORMAL_ECHO = 5;

const int BOTTLE_TRIG = 6;
const int BOTTLE_ECHO = 7;

const int NORMAL_SERVO_PIN = 8;
const int BOTTLE_SERVO_PIN = 9;

const int RED_LED = 10;
const int GREEN_LED = 11;

const int RESET_BUTTON = 12;


// CONFIGURATION

const int ENTRY_DISTANCE = 15;
const int BIN_DISTANCE = 12;

const int SERVO_CLOSED = 0;
const int SERVO_OPEN = 90;

const unsigned long MAX_OPEN_TIME = 10000;
const unsigned long CLOSE_DELAY = 2000;

const unsigned long SERIAL_TIMEOUT = 20000;
const unsigned long ULTRASONIC_TIMEOUT = 25000;


// SERVO OBJECTS

Servo normalServo;
Servo bottleServo;


// SYSTEM STATES

enum SystemState {

    IDLE,

    WAITING_FOR_AI,

    OPEN_NORMAL_BIN,

    OPEN_BOTTLE_BIN,

    WAITING_FOR_TRASH,

    CLOSING_BIN,

    WAITING_FOR_CLEAR

};

SystemState currentState = IDLE;


// TIMERS

unsigned long stateStartTime = 0;
unsigned long detectionTime = 0;

unsigned long lastButtonTime = 0;

bool binDetectionStarted = false;
bool resetButtonHandled = false;


// DISTANCE MEASUREMENT

float getDistance(int trigPin, int echoPin) {

    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);

    digitalWrite(trigPin, LOW);

    unsigned long duration = pulseIn(
        echoPin,
        HIGH,
        ULTRASONIC_TIMEOUT
    );

    if (duration == 0) {
        return -1;
    }

    float distance = duration * 0.0343 / 2.0;

    return distance;
}


// LED CONTROL

void setReadyLED() {

    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);

}


void setBusyLED() {

    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);

}


// SERVO CONTROL

void closeAllBins() {

    normalServo.write(SERVO_CLOSED);
    bottleServo.write(SERVO_CLOSED);

    Serial.println("[SERVO] All bins closed");

}


void openNormalBin() {

    normalServo.write(SERVO_OPEN);
    bottleServo.write(SERVO_CLOSED);

    Serial.println("[SERVO] Normal trash bin opened");

}


void openBottleBin() {

    bottleServo.write(SERVO_OPEN);
    normalServo.write(SERVO_CLOSED);

    Serial.println("[SERVO] Bottle bin opened");

}


// EMERGENCY RESET

void emergencyReset() {

    Serial.println();
    Serial.println("!!! EMERGENCY RESET !!!");

    closeAllBins();

    setReadyLED();

    binDetectionStarted = false;

    stateStartTime = millis();

    currentState = IDLE;

    while (Serial.available() > 0) {
        Serial.read();
    }

    Serial.println("[SYSTEM] Reset completed");
    Serial.println("[SYSTEM] Waiting for object");

}


// BUTTON MONITORING

void checkEmergencyButton() {

    bool buttonPressed = (
        digitalRead(RESET_BUTTON) == LOW
    );

    if (!buttonPressed) {

        resetButtonHandled = false;

        return;
    }

    if (
        buttonPressed &&
        !resetButtonHandled &&
        millis() - lastButtonTime > 300
    ) {

        resetButtonHandled = true;

        lastButtonTime = millis();

        emergencyReset();

    }

}


// SERIAL COMMUNICATION

void sendCaptureCommand() {

    Serial.println("CAPTURE");

    Serial.println(
        "[SERIAL] CAPTURE command sent to Python"
    );

}


int receiveAIResponse() {

    if (Serial.available() <= 0) {
        return -1;
    }

    String response = Serial.readStringUntil('\n');

    response.trim();
    response.toUpperCase();

    Serial.print("[AI] Received response: ");
    Serial.println(response);

    if (
        response == "1" ||
        response == "BOTTLE"
    ) {

        return 1;

    }

    if (
        response == "0" ||
        response == "NORMAL_TRASH"
    ) {

        return 0;

    }

    Serial.println("[AI] Invalid response");

    return -1;

}


// INITIALIZATION

void setup() {

    Serial.begin(9600);

    Serial.setTimeout(50);

    pinMode(ENTRY_TRIG, OUTPUT);
    pinMode(ENTRY_ECHO, INPUT);

    pinMode(NORMAL_TRIG, OUTPUT);
    pinMode(NORMAL_ECHO, INPUT);

    pinMode(BOTTLE_TRIG, OUTPUT);
    pinMode(BOTTLE_ECHO, INPUT);

    pinMode(RED_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);

    pinMode(
        RESET_BUTTON,
        INPUT_PULLUP
    );

    normalServo.attach(NORMAL_SERVO_PIN);
    bottleServo.attach(BOTTLE_SERVO_PIN);

    closeAllBins();

    setReadyLED();

    Serial.println();
    Serial.println("==============================");
    Serial.println(" SMART WASTE SORTING SYSTEM");
    Serial.println("==============================");

    Serial.println("[SYSTEM] Initialization completed");
    Serial.println("[SYSTEM] Waiting for object");

    currentState = IDLE;

    stateStartTime = millis();

}

void loop() {

    checkEmergencyButton();

    switch (currentState) {



        case IDLE: {

            setReadyLED();

            float distance = getDistance(
                ENTRY_TRIG,
                ENTRY_ECHO
            );

            if (
                distance > 0 &&
                distance <= ENTRY_DISTANCE
            ) {

                Serial.println();
                Serial.println("[ENTRY] Object detected");

                Serial.print("[ENTRY] Distance: ");
                Serial.print(distance);
                Serial.println(" cm");

                setBusyLED();

                sendCaptureCommand();

                stateStartTime = millis();

                currentState = WAITING_FOR_AI;

            }

            break;

        }



        case WAITING_FOR_AI: {

            if (
                millis() - stateStartTime >
                SERIAL_TIMEOUT
            ) {

                Serial.println("[ERROR] AI response timeout");

                closeAllBins();

                currentState = WAITING_FOR_CLEAR;

                break;

            }

            int result = receiveAIResponse();

            if (result == 1) {

                Serial.println("[AI] Classification: BOTTLE");

                openBottleBin();

                stateStartTime = millis();

                binDetectionStarted = false;

                currentState = OPEN_BOTTLE_BIN;

            }

            else if (result == 0) {

                Serial.println(
                    "[AI] Classification: NORMAL TRASH"
                );

                openNormalBin();

                stateStartTime = millis();

                binDetectionStarted = false;

                currentState = OPEN_NORMAL_BIN;

            }

            break;

        }



        case OPEN_NORMAL_BIN: {

            currentState = WAITING_FOR_TRASH;

            Serial.println(
                "[SYSTEM] Monitoring normal bin sensor"
            );

            break;

        }


        case OPEN_BOTTLE_BIN: {

            currentState = WAITING_FOR_TRASH;

            Serial.println(
                "[SYSTEM] Monitoring bottle bin sensor"
            );

            break;

        }



        case WAITING_FOR_TRASH: {

            float distance = -1;

            if (
                millis() - stateStartTime >=
                MAX_OPEN_TIME
            ) {

                Serial.println(
                    "[TIMEOUT] Maximum opening time reached"
                );

                currentState = CLOSING_BIN;

                break;

            }

            if (normalServo.read() == SERVO_OPEN) {

                distance = getDistance(
                    NORMAL_TRIG,
                    NORMAL_ECHO
                );

            }

            else {

                distance = getDistance(
                    BOTTLE_TRIG,
                    BOTTLE_ECHO
                );

            }

            if (
                !binDetectionStarted &&
                distance > 0 &&
                distance <= BIN_DISTANCE
            ) {

                Serial.println(
                    "[BIN] Trash arrival detected"
                );

                Serial.print("[BIN] Distance: ");
                Serial.print(distance);
                Serial.println(" cm");

                binDetectionStarted = true;

                detectionTime = millis();

            }

            if (binDetectionStarted) {

                if (
                    millis() - detectionTime >=
                    CLOSE_DELAY
                ) {

                    Serial.println(
                        "[BIN] 2-second grace period completed"
                    );

                    currentState = CLOSING_BIN;

                }

            }

            break;

        }



        case CLOSING_BIN: {

            closeAllBins();

            setBusyLED();

            Serial.println("[SYSTEM] Bin operation completed");

            stateStartTime = millis();

            currentState = WAITING_FOR_CLEAR;

            break;

        }



        case WAITING_FOR_CLEAR: {

            float distance = getDistance(
                ENTRY_TRIG,
                ENTRY_ECHO
            );

            if (
                distance < 0 ||
                distance > ENTRY_DISTANCE
            ) {

                Serial.println(
                    "[SYSTEM] Entry area clear"
                );

                setReadyLED();

                currentState = IDLE;

                Serial.println(
                    "[SYSTEM] Ready for next object"
                );

            }

            break;

        }

    }

}
