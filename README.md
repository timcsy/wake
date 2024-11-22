# AI 喚醒助手

簡報：https://www.canva.com/design/DAGSHvYgQrA/2Nc-o4nzHSURe0rOtom0QQ/view

## Usage

Server
```
cd ./server
npm install koa koa-router koa-bodyparser @koa/cors ws
node server.js
```

Python
```
python run.py
```

HTTP polling
```
POST http://<your server address and port>/hook

{
	// JSON API
}
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