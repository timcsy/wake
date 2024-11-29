# AI 喚醒助手

簡報：https://www.canva.com/design/DAGSHvYgQrA/2Nc-o4nzHSURe0rOtom0QQ/view

## Setup

### Python

Install Python 3
```
pip install -r requirements.txt
```

### Node.js Server

Install Node.js LTS version
```
cd server
npm install
```

### Arduino

Setup [ESP8266](https://github.com/esp8266/Arduino).

Install Libraries:
- [WebSockets](https://github.com/Links2004/arduinoWebSockets)
- [ArduinoJson](https://arduinojson.org/?utm_source=meta&utm_medium=library.properties)

Change Your WiFi SSID, password, and server IP address and port.

To do so, please copy the `Secret_template.h` in the `Arduino/motor` and the `Arduino/light` and rename to `Secret.h`, and edit the variables inside.


## Usage

Server
```
cd server
node server.js
```

Python
```
python run.py
```

WebSocket Connection
```
ws://<your server address and port>/

{
	// JSON API
}
```

HTTP send command
```
POST http://<your server address and port>/

{
	// JSON API
}
```

HTTP polling
```
POST http://<your server address and port>/hook
```