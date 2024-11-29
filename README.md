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

Using ESP8266

Change Your WiFi SSID, password, and server IP address.


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