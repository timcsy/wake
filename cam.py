import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

# 使用 Grad-CAM++ 進行視覺化
def visualize_grad_cam(model, image_path, target_layer, target_class=None):
    # 讀取圖片
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # 正規化處理
    input_tensor = preprocess(img_resized).unsqueeze(0).to(device)

    # 設定 Grad-CAM++ 計算
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    
    # 設定目標類別，如果沒有提供則為模型預測的類別
    if target_class is None:
        target_class = torch.argmax(model(input_tensor)).item()
    targets = [ClassifierOutputTarget(target_class)]

    # 產生 Grad-CAM++ 視覺化
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True)

    # 顯示結果
    cv2.imshow("Grad-CAM++ Visualization", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 指定模型的最後一個卷積層作為 Grad-CAM++ 的目標層
target_layer = model.conv3

# 執行視覺化
image_path = 'data/00008_P.jpg'  # 替換為你想要視覺化的圖片路徑
visualize_grad_cam(model, image_path, target_layer)
