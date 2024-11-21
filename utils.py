import json
from ws_client import WebSocketClient
import time

ws = WebSocketClient('ws://localhost:8765')
ws.start()
time.sleep(1)

def ws_send(data):
    msg = json.dumps(data)
    ws.send(msg)

def light_on():
    ws_send({
        "device": "light",
        "command": "on"
    })

def light_off():
    ws_send({
        "device": "light",
        "command": "off"
    })

def light_brightness(brightness: int):
    # brightness: percentage
    ws_send({
        "device": "light",
        "command": "set_brightness",
        "parameters": { "brightness": brightness }
    })

def light_flash():
    ws_send({
        "device": "light",
        "command": "flash"
    })

def motor_on():
    ws_send({
        "device": "motor",
        "command": "on"
    })

def motor_off():
    ws_send({
        "device": "motor",
        "command": "off"
    })

def motor_speed(speed: int):
    # speed: percentage
    ws_send({
        "device": "motor",
        "command": "set_speed",
        "parameters": { "speed" : speed }
    })

def clock_on():
    ws_send({
        "device": "clock",
        "command": "on"
    })

def clock_off():
    ws_send({
        "device": "clock",
        "command": "off"
    })

def get_up():
    ws_send({
        "device": "you",
        "command": "up"
    })

def not_get_up():
    ws_send({
        "device": "you",
        "command": "down"
    })

def sleep_again(duration=300):
    # duration: 賴床時間，以秒為單位
    ws_send({
        "device": "you",
        "command": "sleep_again",
        "parameters": {
            "duration": duration
        }
    })

def system_off():
    light_off()
    motor_off()

def close():
    ws.close()