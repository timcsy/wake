import os
import queue
import tempfile
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import openai
#import utils
import wave

# Azure OpenAI GPT-4o 設定
openai.api_key = '07e0bd4794824daf8fe45508d132ab99'
openai.api_base = 'https://ccsh.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2024-08-01-preview'
GPT_DEPLOYMENT_NAME = 'gpt4o'

# 音訊佇列初始化
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """將音訊片段放入佇列"""
    if status:
        print(f"錄音狀態錯誤：{status}")
    audio_queue.put(indata.copy())

def recognize_speech():
    """使用 SpeechRecognition 辨識語音，轉成文字並自動斷句"""
    recognizer = sr.Recognizer()
    with sd.InputStream(samplerate=16000, blocksize=8000, dtype='float32', channels=1, callback=audio_callback):
        print("開始錄音，請說話...")

        accumulated_audio = []
        while True:
            # 將佇列中的音訊數據取出並保存
            audio_data = audio_queue.get()
            accumulated_audio.append(audio_data)

            # 當音訊累積足夠時進行處理
            combined_audio = np.concatenate(accumulated_audio, axis=0)
            # 檢查是否有停頓（音量低於閾值）
            volume = np.linalg.norm(combined_audio) / len(combined_audio)
            if volume < 0.01:  # 設定音量閾值，表示停頓
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    tmp_filename = tmpfile.name
                    # 將音訊數據轉換為 16-bit PCM 格式並保存
                    with wave.open(tmp_filename, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        combined_audio = (combined_audio * 32767).astype(np.int16)
                        wf.writeframes(combined_audio.tobytes())

                # 使用 SpeechRecognition 進行語音識別
                with sr.AudioFile(tmp_filename) as source:
                    audio = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio, language="zh-TW")
                    except sr.UnknownValueError:
                        text = ""
                    except sr.RequestError as e:
                        print(f"無法請求 Google Speech Recognition 服務; {e}")
                        text = ""

                # 刪除臨時檔案
                os.unlink(tmp_filename)

                # 打印並分析辨識結果
                print(f"辨識結果：{text}")
                if analyze_intent(text):
                    #utils.system_off()
                    print("偵測到停止指令，停止叫起床程序。")
                    break

                # 重置累積的音訊
                accumulated_audio = []

def analyze_intent(text):
    """使用 GPT-4o 分析語句的語意"""
    prompt = f"情境是我做了一個自動叫起床的機器人，會用棍棒打你、強光照你，直到你爬起來，我想要的是假設你有起來了，會說一些話，例如說我已經起來了、不要再照了、不要再打了之類的話，讓這個系統可以停止。以下句子是否表示使用者希望停止叫起床的行為？回答'是'或'否'：\n「{text}」"

    response = openai.ChatCompletion.create(
        engine=GPT_DEPLOYMENT_NAME,
        deployment_id=GPT_DEPLOYMENT_NAME,
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
        #utils.motor_on()
        recognize_speech()
    except KeyboardInterrupt:
        print("程式已停止")
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    main()
