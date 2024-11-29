import sounddevice as sd
import speech_recognition as sr
import openai
import utils

# Azure OpenAI GPT-4o configuration
openai.api_key = '07e0bd4794824daf8fe45508d132ab99'
openai.api_base = 'https://ccsh.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2024-08-01-preview'
GPT_DEPLOYMENT_NAME = 'gpt4o'

def record_audio(duration=3, samplerate=16000):
    """
    Record audio using sounddevice and return the audio data.
    :param duration: Recording duration in seconds
    :param samplerate: Sampling rate in Hz
    :return: NumPy array of recorded audio
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    return audio.flatten()

def recognize_speech():
    recognizer = sr.Recognizer()

    while utils.CLOCK and utils.SPEECH:
        print("請說話 (錄音進行中)...")

        # Record audio using sounddevice
        audio_data = record_audio()

        try:
            # Convert NumPy array to an AudioData object
            audio = sr.AudioData(audio_data.tobytes(), 16000, 2)

            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"辨識結果：{text}")

            # Analyze intent
            if text and analyze_intent(text):
                print("偵測到停止指令，停止叫起床程序。")
                return
        except sr.UnknownValueError:
            print("無法辨識語音，請再試一次。")
        except sr.RequestError as e:
            print(f"語音辨識服務出錯：{e}")

def analyze_intent(text):
    """Analyze intent using GPT-4o"""
    prompt = (
        f"情境是我做了一個自動叫起床的機器人，會用棍棒打你、強光照你，直到你爬起來。"
        f"假設你有起來了，會說一些話，例如說我已經起來了、不要再照了、不要再打了之類的話，"
        f"讓這個系統可以停止。以下句子是否表示使用者希望停止叫起床的行為？"
        f"回答'是'或'否'：\n「{text}」"
    )

    try:
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
    except Exception as e:
        print(f"GPT 分析出錯：{e}")
        return False

def main():
    try:
        utils.speech_on()
        recognize_speech()
        utils.speech_off()
    except KeyboardInterrupt:
        print("程式已停止")
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    utils.clock_on()
    utils.light_flash()
    utils.motor_on()
    main()
