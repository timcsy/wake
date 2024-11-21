const Koa = require('koa');
const Router = require('koa-router');
const bodyParser = require('koa-bodyparser');
const cors = require('@koa/cors');  // 引入 koa-cors 中間件
const WebSocket = require('ws');

const app = new Koa();
const router = new Router();
app.use(bodyParser());
app.use(cors());  // 啟用 CORS 中間件

// 儲存所有 HTTP 輪詢請求
let httpClients = [];

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

        // 將消息發送給所有 HTTP 輪詢的客戶端
        httpClients.forEach(client => {
            client.resolve({ message });  // 解決 Promise，返回消息
        });
        httpClients = [];  // 清空已處理的 HTTP 輪詢請求

        ctx.status = 200;
        ctx.body = 'Message sent to WebSocket clients';
    } else {
        ctx.status = 400;
        ctx.body = 'No message specified in JSON';
    }
});

// /hook 路由，用於長輪詢
router.post('/hook', (ctx) => {
    // 創建一個 Promise，並將它存儲在 httpClients 陣列中
    return new Promise(resolve => {
        httpClients.push({ ctx, resolve });
        console.log('新增輪詢請求');
    }).then(response => {
        // 當 WebSocket 發送消息時，HTTP 輪詢的客戶端將收到回應
        ctx.status = 200;
        ctx.body = response;  // 回應 WebSocket 發送的消息
    });
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

        // 將消息廣播給所有其他 WebSocket 客戶端
        wss.clients.forEach((client) => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
                client.send(message.toString());
            }
        });

        // 將消息發送給所有 HTTP 輪詢的客戶端
        httpClients.forEach(client => {
            client.resolve({ message });  // 解決 Promise，並將消息發送給 HTTP 輪詢客戶端
        });
        httpClients = [];  // 清空已處理的 HTTP 輪詢請求
    });

    ws.on('close', () => {
        console.log('客戶端已斷開連接');
    });
});
