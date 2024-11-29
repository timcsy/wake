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

def sit_up(rgb_frame, frame, idle=False):
    # 使用 MediaPipe 偵測姿勢
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        def get_landmark(idx_right, idx_left):
            """
            計算左右肩或左右腳的平均點
            """
            right = [landmarks[idx_right].x * frame.shape[1], landmarks[idx_right].y * frame.shape[0]] \
                if landmarks[idx_right].visibility > 0.5 else None
            left = [landmarks[idx_left].x * frame.shape[1], landmarks[idx_left].y * frame.shape[0]] \
                if landmarks[idx_left].visibility > 0.5 else None

            if right and left:
                # 返回平均點
                return [(right[0] + left[0]) / 2, (right[1] + left[1]) / 2]
            return right if right else left

        # 提取雙肩和雙腳的平均點
        shoulder_avg = get_landmark(12, 11)  # 右肩和左肩
        foot_avg = get_landmark(28, 27)     # 右腳踝和左腳踝

        # 提取臀部位置
        hip = [landmarks[23].x * frame.shape[1], landmarks[23].y * frame.shape[0]] \
            if landmarks[23].visibility > 0.5 else None

        # 檢查必要點是否存在
        if shoulder_avg is None or foot_avg is None or hip is None:
            cv2.putText(frame, 'Missing key landmarks!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # 畫出關鍵點與連線
            cv2.circle(frame, (int(shoulder_avg[0]), int(shoulder_avg[1])), 5, (0, 255, 0), -1)  # 雙肩平均點
            cv2.circle(frame, (int(hip[0]), int(hip[1])), 5, (255, 0, 0), -1)                   # 臀部
            cv2.circle(frame, (int(foot_avg[0]), int(foot_avg[1])), 5, (0, 255, 255), -1)       # 雙腳平均點
            cv2.line(frame, (int(shoulder_avg[0]), int(shoulder_avg[1])), (int(hip[0]), int(hip[1])), (0, 255, 255), 2)
            cv2.line(frame, (int(hip[0]), int(hip[1])), (int(foot_avg[0]), int(foot_avg[1])), (255, 0, 255), 2)

            # 計算雙肩平均點-臀部-雙腳平均點的夾角
            body_angle = calculate_angle(shoulder_avg, hip, foot_avg)

            # 顯示角度
            cv2.putText(frame, f'Angle: {int(body_angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 根據角度執行動作
            if not idle:
                if body_angle < 135:
                    utils.pose_off()
                elif body_angle > 135:
                    utils.light_flash()
                    utils.motor_on()
            else:
                if body_angle < 135:
                    utils.get_up()
                elif body_angle > 135:
                    utils.not_get_up()

    # 顯示影像
    cv2.imshow('Pose with Shoulders and Feet', frame)


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
    global last_state, count
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
        count += 1
        if count > 10:
            utils.covered_off()

def main():
    global count
    count = 0

    # 開啟攝影機
    cap = cv2.VideoCapture(0)

    while utils.CLOCK and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 將影像轉換為RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if utils.STATE == utils.Stage.POSE:
            sit_up(rgb_frame, frame)
        elif utils.STATE == utils.Stage.COVERED:
            covered(rgb_frame)
        elif utils.STATE == utils.Stage.MIXED:
            sit_up(rgb_frame, frame)
            covered(rgb_frame)
        else:
            break

        if cv2.waitKey(5) & 0xFF == 27:  # 按 "ESC" 鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()

def idle():
    # 開啟攝影機
    cap = cv2.VideoCapture(0)

    while utils.CLOCK and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 將影像轉換為RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not utils.CLOCK:
            sit_up(rgb_frame, frame, idle=True)
        else:
            break

        if cv2.waitKey(5) & 0xFF == 27:  # 按 "ESC" 鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    utils.clock_on()
    # utils.covered_on()
    utils.pose_on()
    main()