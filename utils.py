from enum import Enum
import json
from ws_client import WebSocketClient
import time

###############################################
#                    States                   #
###############################################
class Stage(Enum):
    NONE = 0
    EYE = 1
    COVERED = 2
    POSE = 3
    MIXED = 4

CLOCK = False
STATE = Stage.NONE
SPEECH = False
GET_UP = False

###############################################
#              WebSocket Commands             #
###############################################
ws = WebSocketClient('ws://localhost:8765')
ws.start()
time.sleep(1)

def ws_send(data):
    msg = json.dumps(data)
    ws.send(msg)

def refresh_state():
    while True:
        if not ws.inbox.empty():
            msg = ws.inbox.get()
            data = json.loads(msg)
            if data['device'] == 'clock':
                if data['command'] == 'on':
                    clock_on()
                elif data['command'] == 'off':
                    clock_off()
            elif data['device'] == 'you':
                if data['command'] == 'state':
                    if GET_UP:
                        get_up(force=True)
                    else:
                        not_get_up(force=True)

def close():
    ws.close()

###############################################
#               Arduino Commands              #
###############################################
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

###############################################
#               Control Commands              #
###############################################
def clock_on():
    global CLOCK
    CLOCK = True
    ws_send({
        "device": "clock",
        "command": "on"
    })

def clock_off():
    global CLOCK, STATE, SPEECH
    CLOCK = False
    STATE = Stage.NONE
    light_off()
    motor_off()
    ws_send({
        "device": "clock",
        "command": "off"
    })

def get_up(force=False):
    global GET_UP
    if force or not GET_UP:
        GET_UP = True
        ws_send({
            "device": "you",
            "command": "up"
        })

def not_get_up(force=False):
    global GET_UP
    if force or GET_UP:
        GET_UP = False
        ws_send({
            "device": "you",
            "command": "down"
        })

def get_state():
    ws_send({
        "device": "you",
        "command": "state"
    })

def sleep_again(duration=300):
    # duration: 賴床時間，以秒為單位
    # deprecated
    ws_send({
        "device": "you",
        "command": "sleep_again",
        "parameters": {
            "duration": duration
        }
    })

###############################################
#               Feature Commands              #
###############################################
def eye_on():
    global STATE
    STATE = Stage.EYE
    ws_send({
        "device": "eye",
        "command": "on"
    })

def eye_off():
    global STATE
    STATE = Stage.NONE
    light_off()
    ws_send({
        "device": "eye",
        "command": "off"
    })

def covered_on():
    global STATE
    STATE = Stage.COVERED
    ws_send({
        "device": "pillow",
        "command": "on"
    })

def covered_off():
    global STATE
    STATE = Stage.NONE
    motor_off()
    ws_send({
        "device": "pillow",
        "command": "off"
    })

def pose_on():
    global STATE
    STATE = Stage.POSE
    ws_send({
        "device": "pose",
        "command": "on"
    })

def pose_off():
    global STATE
    STATE = Stage.NONE
    clock_off()
    get_up()
    ws_send({
        "device": "pose",
        "command": "off"
    })

def speech_on():
    global SPEECH
    SPEECH = True
    ws_send({
        "device": "speech",
        "command": "on"
    })

def speech_off():
    global SPEECH
    SPEECH = False
    clock_off()
    get_up()
    ws_send({
        "device": "speech",
        "command": "off"
    })

###############################################
#            Future Feature Commands          #
###############################################
def feather_on():
    ws_send({
        "device": "feather",
        "command": "on"
    })

def feather_off():
    ws_send({
        "device": "feather",
        "command": "off"
    })

def push_on():
    ws_send({
        "device": "push",
        "command": "on"
    })

def push_off():
    ws_send({
        "device": "push",
        "command": "off"
    })

def stamp_on():
    ws_send({
        "device": "stamp",
        "command": "on"
    })

def stamp_off():
    ws_send({
        "device": "stamp",
        "command": "off"
    })

if __name__ == "__main__":
    refresh_state()