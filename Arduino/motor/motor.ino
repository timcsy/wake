#include "Secret.h"
#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// WebSocket server IP
const char* serverIP = SERVER_IP;
const uint16_t serverPort = SERVER_PORT;
WebSocketsClient webSocket;

// Motor control pins
const int motorPin1 = D1; // IN1 for PWM
const int motorPin2 = D2; // IN2 should be LOW

// Function to connect to WiFi
void connectWiFi() {
    Serial.print("Connecting to WiFi");
    if (password == "") WiFi.begin(ssid);
    else WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println(" connected!");
}

// Function to control the motor
void controlMotor(int speed) {
    if (speed > 0) {
        digitalWrite(motorPin2, LOW); // Ensure IN2 is LOW
        analogWrite(motorPin1, speed);  // Set PWM to specified speed
        Serial.printf("Motor ON at %d speed\n", speed);
    } else {
        analogWrite(motorPin1, 0);      // Stop PWM
        Serial.println("Motor OFF");
    }
}

// Function to handle WebSocket messages
void handleWebSocketMessage(const char* message) {
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, message);

    if (error) {
        Serial.print("deserializeJson() failed: ");
        Serial.println(error.c_str());
        return;
    }

    const char* device = doc["device"];
    const char* command = doc["command"];

    if (strcmp(device, "motor") == 0) {
        if (strcmp(command, "on") == 0) {
            controlMotor(255); // Turn on the motor at full speed
        } else if (strcmp(command, "off") == 0) {
            controlMotor(0);   // Turn off the motor
        } else if (strcmp(command, "set_speed") == 0) {
            int percentage = doc["parameters"]["speed"];
            int speed = map(percentage, 0, 100, 0, 255);
            controlMotor(speed); // Set the motor speed
        }
    }
}

// Setup function
void setup() {
    Serial.begin(115200);
    connectWiFi();

    // Set motor control pins
    pinMode(motorPin1, OUTPUT);
    pinMode(motorPin2, OUTPUT);
    digitalWrite(motorPin2, LOW); // Ensure IN2 is LOW

    webSocket.begin(serverIP, serverPort);
    webSocket.onEvent([](WStype_t type, uint8_t* payload, size_t length) {
        if (type == WStype_TEXT) {
            payload[length] = '\0'; // Null-terminate the received message
            handleWebSocketMessage((const char*)payload);
        }
    });
}

// Loop function
void loop() {
    webSocket.loop();
}
