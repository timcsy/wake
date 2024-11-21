import cv2
import os

file_dir = 'data'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

# 啟動攝像頭
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 確認攝像頭是否成功開啟
if not cap.isOpened():
    print("無法開啟攝像頭")
    exit()

# 初始化照片計數器
photo_count = 251 

print("按下 'N' 拍攝無被子的照片，按下 'P' 拍攝有被子的照片。按下 'Q' 離開程式。")

while True:
    # 讀取影像
    ret, frame = cap.read()

    # 檢查是否成功讀取影像
    if not ret:
        print("無法讀取影像")
        break

    # 顯示影像
    cv2.imshow("Capture", frame)

    # 等待按鍵
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 'N' 拍攝無被子的照片
    if key == ord('n'):
        filename = f"data/{photo_count:05d}_N.jpg"  # 五位數的編號，N 表示無被子
        cv2.imwrite(filename, frame)
        print(f"已儲存圖片為 {filename}")

    # 如果按下 'P' 拍攝有被子的照片
    elif key == ord('p'):
        filename = f"data/{photo_count:05d}_P.jpg"  # 五位數的編號，P 表示有被子
        cv2.imwrite(filename, frame)
        print(f"已儲存圖片為 {filename}")
        # 增加計數器
        photo_count += 1

    # 如果按下 'Q' 離開程式
    elif key == ord('q'):
        print("結束拍攝")
        break

# 釋放攝像頭並關閉視窗
cap.release()
cv2.destroyAllWindows()
