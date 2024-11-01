const WebSocket = require('ws');

// 创建一个 WebSocket 服务器，监听端口 8765
const wss = new WebSocket.Server({ port: 8765 });

wss.on('connection', (ws) => {
    console.log('新客户端已连接');

    ws.on('message', (message) => {
        console.log(`收到消息: ${message}`);

        // 将消息广播给所有其他客户端
        wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
                client.send(message.toString());
            }
        });
    });

    ws.on('close', () => {
        console.log('客户端已断开连接');
    });
});

console.log('WebSocket 服务器在 ws://localhost:8765 上运行');
