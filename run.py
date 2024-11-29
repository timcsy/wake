# run.py
import threading
import eye
import body
import speech
import utils

def run_refresh_state():
    utils.refresh_state()

# 定義執行 eye、body 和 speech 的主函式
def run_eye_covered_pose():
    last_clock = False
    last_state = utils.Stage.NONE
    while True:
        if not last_clock and utils.CLOCK:
            last_clock = True
            utils.clock_on()
        elif last_clock and not utils.CLOCK:
            last_clock = False
            last_state = utils.Stage.NONE
            utils.clock_off()
            body.idle()
        
        if utils.CLOCK and utils.STATE == utils.Stage.NONE:
            if last_state == utils.Stage.NONE:
                last_state = utils.Stage.EYE
                utils.eye_on()
                eye.main()
            elif last_state == utils.Stage.EYE:
                last_state = utils.Stage.COVERED
                utils.covered_on()
                body.main()
            elif last_state == utils.Stage.COVERED:
                last_state = utils.Stage.POSE
                utils.pose_on()
                body.main()

def run_speech():
    while True:
        if utils.CLOCK:
            speech.main()

if __name__ == "__main__":
    # 創建執行緒
    main_thread = threading.Thread(target=run_eye_covered_pose)
    speech_thread = threading.Thread(target=run_speech)
    refresh_thread = threading.Thread(target=run_refresh_state)

    # 啟動執行緒
    main_thread.start()
    speech_thread.start()
    refresh_thread.start()

    # 等待執行緒結束
    main_thread.join()
    speech_thread.join()
    refresh_thread.join()

    print("All threads have finished execution.")