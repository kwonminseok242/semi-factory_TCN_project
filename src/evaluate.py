import torch
import torch.nn as nn
import numpy as np
import os
from model import TCN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

# 검증 데이터 로드
X_val = np.load(os.path.join(RESULTS_DIR, "VX_windows.npy"))
y_val = np.load(os.path.join(RESULTS_DIR, "Vy_labels.npy"))

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# DataLoader는 사용하지 않고 전체를 한 번에 평가
X_val = X_val.to("cpu")
y_val = y_val.to("cpu")

# 모델 정의 (훈련 때와 동일한 구조로 정의해야 함)
INPUT_SIZE = X_val.shape[2]  # feature 수
OUTPUT_SIZE = len(torch.unique(y_val))  # 클래스 수
NUM_CHANNELS = [32, 64, 128, 128]  # 예시로 설정했던 구조
KERNEL_SIZE = 7

model = TCN(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE,
            num_channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE)

model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "tcn_model_b_paramas.pth"), map_location=torch.device('cpu')))
model.eval()

# 예측 수행
with torch.no_grad():
    outputs = model(X_val)
    _, predicted = torch.max(outputs, 1)

# 평가 출력
print("[Classification Report]")
print(classification_report(y_val.numpy(), predicted.numpy()))

print("[Confusion Matrix]")
cm = confusion_matrix(y_val.numpy(), predicted.numpy())
print(cm)

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.show()