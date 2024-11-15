#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "lab";
const char* password = "0912106664";

// WebSocket server IP
const char* serverIP = "172.16.1.147";
const uint16_t serverPort = 8765;
WebSocketsClient webSocket;

// Light control pin
const int lightPin = D1; // 連接燈光的 GPIO 引腳

// Timing variables for non-blocking flash
unsigned long flashStartTime = 0;
bool isFlashing = false;
bool lightState = false;
unsigned long flashInterval = 250; // Flashing interval in milliseconds

void connectWiFi() {
    Serial.print("Connecting to WiFi");
    WiFi.begin("S24");
    // WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println(" connected!");
}

void controlLight(bool isOn) {
    digitalWrite(lightPin, isOn ? HIGH : LOW);
    lightState = isOn;
    Serial.println(isOn ? "Light ON" : "Light OFF");
}

void setBrightness(int percentage) {
    int brightness = map(percentage, 0, 100, 0, 255);
    analogWrite(lightPin, brightness);
    Serial.printf("Light brightness set to %d%%\n", percentage);
}

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

    if (strcmp(device, "light") == 0) {
        if (strcmp(command, "on") == 0) {
            controlLight(true);
            isFlashing = false; // Stop flashing if turned on
        } else if (strcmp(command, "off") == 0) {
            controlLight(false);
            isFlashing = false; // Stop flashing if turned off
        } else if (strcmp(command, "flash") == 0) {
            isFlashing = true; // Start flashing
        } else if (strcmp(command, "set_brightness") == 0) {
            int percentage = doc["parameters"]["brightness"];
            if (percentage < 0) percentage = 0;
            if (percentage > 100) percentage = 100;
            setBrightness(percentage);
        }
    }
}

void handleFlashing() {
    if (isFlashing) {
        unsigned long currentTime = millis();
        if (currentTime - flashStartTime >= flashInterval) {
            flashStartTime = currentTime;
            lightState = !lightState; // Toggle light state
            controlLight(lightState);
        }
    }
}

void setup() {
    Serial.begin(115200);
    connectWiFi();

    pinMode(lightPin, OUTPUT);
    controlLight(false); // Ensure light is off at startup

    webSocket.begin(serverIP, serverPort);
    webSocket.onEvent([](WStype_t type, uint8_t* payload, size_t length) {
        if (type == WStype_TEXT) {
            payload[length] = '\0'; // Null-terminate the received message
            handleWebSocketMessage((const char*)payload);
        }
    });
}

void loop() {
    webSocket.loop();
    handleFlashing(); // Check for flashing state
}