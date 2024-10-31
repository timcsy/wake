# run.py
import threading
import eye
import body
import speech

# 定義執行 eye、body 和 speech 的主函式
def run_eye():
    eye.main()

def run_body():
    body.main()

def run_speech():
    speech.main()

if __name__ == "__main__":
    # 創建執行緒
    eye_thread = threading.Thread(target=run_eye)
    body_thread = threading.Thread(target=run_body)
    speech_thread = threading.Thread(target=run_speech)

    # 啟動執行緒
    eye_thread.start()
    body_thread.start()
    speech_thread.start()

    # 等待執行緒結束
    eye_thread.join()
    body_thread.join()
    speech_thread.join()

    print("All threads have finished execution.")
