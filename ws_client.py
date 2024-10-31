import websocket
import threading

class WebSocketClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.thread = None

    def on_message(self, ws, message):
        print(f"收到服务器消息: {message}")


    def on_error(self, ws, error):
        print(f"出现错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### WebSocket 连接关闭 ###")

    def on_open(self, ws):
        print("WebSocket 连接已开启")
        # 可以添加发送初始消息的代码，或留空

    def start(self):
        # 创建 WebSocket 应用
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        # 启动 WebSocket 的线程
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True  # 让线程在主程序结束时自动关闭
        self.thread.start()

    def send(self, message):
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(message)
        else:
            print("无法发送消息：WebSocket 未连接")

    def close(self):
        if self.ws:
            self.ws.close()
