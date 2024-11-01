import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
import openai
import utils

# Azure OpenAI GPT-4o 設定
openai.api_key = '07e0bd4794824daf8fe45508d132ab99'
openai.api_base = 'https://ccsh.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2024-08-01-preview'
GPT_DEPLOYMENT_NAME = 'gpt4o'

# 初始化 Vosk 模型
# 去 https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip 下載，然後解壓縮
MODEL_PATH = "./vosk-model-small-cn-0.22"  # 替換為你下載的模型路徑
model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)

# 音訊佇列初始化
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """將音訊片段放入佇列"""
    if status:
        print(f"錄音狀態錯誤：{status}")
    audio_queue.put(bytes(indata))

def recognize_speech():
    """使用 Vosk 辨識語音，轉成文字"""
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        print("開始錄音，請說話...")

        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                print(f"辨識結果：{text}")
                if analyze_intent(text):
                    utils.system_off()
                    print("偵測到停止指令，停止叫起床程序。")
                    break

def analyze_intent(text):
    """使用 GPT-4o 分析語句的語意"""
    prompt = f"情境是我做了一個自動叫起床的機器人，會用棍棒打你、強光照你，直到你爬起來，我想要的是假設你有起來了，會說一些話，例如說我已經起來了、不要再照了、不要再打了之類的話，讓這個系統可以停止。以下句子是否表示使用者希望停止叫起床的行為？回答'是'或'否'：\n「{text}」"

    response = openai.ChatCompletion.create(
        deployment_id='gpt4o',
        api_version='2024-08-01-preview',
        messages=[
            {"role": "system", "content": "你是一個能理解自然語言的助理。"},
            {"role": "user", "content": prompt}
        ]
    )

    result = response['choices'][0]['message']['content'].strip()
    print(f"GPT-4o 分析結果：{result}")
    return result == "是"

def main():
    try:
        utils.motor_on()
        recognize_speech()
    except KeyboardInterrupt:
        print("程式已停止")
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    main()