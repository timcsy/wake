import cv2
import mediapipe as mp
import math
import torch
import torch.nn as nn
from torchvision import transforms
import utils

# 初始化MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 計算兩點之間的角度
def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang)

def sit_up(rgb_frame, frame):
    # 使用MediaPipe偵測姿勢
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # 提取頭部、肩膀、臀部的地標
        head = [landmarks[0].x * frame.shape[1], landmarks[0].y * frame.shape[0]]        # 頭部
        shoulder = [landmarks[11].x * frame.shape[1], landmarks[11].y * frame.shape[0]]  # 左肩
        hip = [landmarks[23].x * frame.shape[1], landmarks[23].y * frame.shape[0]]       # 左臀

        # 畫出關鍵點和肩膀到臀部的向量
        cv2.circle(frame, (int(head[0]), int(head[1])), 5, (0, 0, 255), -1)    # 頭部
        cv2.circle(frame, (int(shoulder[0]), int(shoulder[1])), 5, (0, 255, 0), -1)  # 肩膀
        cv2.circle(frame, (int(hip[0]), int(hip[1])), 5, (255, 0, 0), -1)      # 臀部
        cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (0, 255, 255), 3)

        # 計算肩膀-臀部-頭部的角度
        body_angle = calculate_angle(head, shoulder, hip)

        # 顯示角度
        cv2.putText(frame, f'Angle: {int(body_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if body_angle < 135:
            utils.system_off()
        elif body_angle > 135:
            utils.motor_on()

    # 顯示影像
    cv2.imshow('Sit-up', frame)


# 定義 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # 調整尺寸
        self.fc2 = nn.Linear(512, 2)  # 二分類

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 載入模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
model.load_state_dict(torch.load('blanket_classifier_model.pth', map_location=device))
model.eval()

# 準備資料前處理（與訓練時一致）
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

last_state = None

def covered(img_rgb):
    global last_state
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # 正規化處理
    input_tensor = preprocess(img_resized).unsqueeze(0).to(device)

    result = torch.argmax(model(input_tensor)).item()

    if result == 1:
        if last_state != 1:
            print('有被子')
        last_state = 1
        utils.motor_on()
    elif result == 0:
        if last_state != 0:
            print('沒被子')
        last_state = 0
        utils.motor_off()

def main():
    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    utils.motor_on()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 將影像轉換為RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        sit_up(rgb_frame, frame)
        covered(rgb_frame)

        if cv2.waitKey(5) & 0xFF == 27:  # 按 "ESC" 鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()