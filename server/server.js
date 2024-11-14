const Koa = require('koa');
const Router = require('koa-router');
const bodyParser = require('koa-bodyparser');
const cors = require('@koa/cors');  // 引入 koa-cors 中間件
const WebSocket = require('ws');

const app = new Koa();
const router = new Router();
app.use(bodyParser());
app.use(cors());  // 啟用 CORS 中間件

// WebSocket 伺服器
let wss;

// POST 路由，用於接收 JSON 訊息並轉發給 WebSocket 客戶端
router.post('/', (ctx) => {
    const message = ctx.request.body.message; // 從 JSON 中提取 message

    if (message) {
        console.log(`收到消息: ${message}`);
        // 將消息轉發給所有的 WebSocket 客戶端
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
        ctx.status = 200;
        ctx.body = 'Message sent to WebSocket clients';
    } else {
        ctx.status = 400;
        ctx.body = 'No message specified in JSON';
    }
});

// 使用 router 中間件
app.use(router.routes()).use(router.allowedMethods());

// 創建 HTTP 伺服器並將其與 Koa 應用程式綁定
const server = app.listen(8765, () => {
    console.log('伺服器在 http://localhost:8765 上運行');
});

// 創建 WebSocket 伺服器並綁定到 HTTP 伺服器
wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('新客戶端已連接');

    ws.on('message', (message) => {
        console.log(`收到消息: ${message}`);

        // 將消息廣播給所有其他客戶端
        wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
                client.send(message.toString());
            }
        });
    });

    ws.on('close', () => {
        console.log('客戶端已斷開連接');
    });
});
