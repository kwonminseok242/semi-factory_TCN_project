import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import TCN
import os

# ---------------------- SequenceDataset 정의 ---------------------- #
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------- 설정 ---------------------- #
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.000443939
INPUT_LENGTH = 50         # 시퀀스 길이
NUM_FEATURES = 7          # 센서 개수 (삭제된 feature 기준)
NUM_CLASSES = 4           # 예측할 상태 (state 0~3)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 경로 설정 ---------------------- #
# 이 스크립트 기준 경로: src/train.py
current_dir = os.path.dirname(os.path.abspath(__file__))
X_path = os.path.join(current_dir, "..", "results", "X1_windows.npy")
y_path = os.path.join(current_dir, "..", "results", "y1_labels.npy")
model_save_path = os.path.join(current_dir, "..", "results", "tcn_model_b_paramas.pth")

# ---------------------- 데이터 로딩 ---------------------- #
X = np.load(X_path)
y = np.load(y_path)

dataset = SequenceDataset(X, y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------- 클래스 가중치 직접 입력 ---------------------- #
class_weights = [1.0, 3.895926805322783, 1.0, 1.0]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# ---------------------- 모델 정의 ---------------------- #
model = TCN(input_size=NUM_FEATURES,
            output_size=NUM_CLASSES,
            num_channels=[32, 64, 128, 128],
            kernel_size=7,
            dropout=0.101497).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------- Early Stopping 설정 ---------------------- #
best_acc = 0.0
patience = 5
counter = 0


# ---------------------- 학습 루프 ---------------------- #
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        acc = 100 * correct / total
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {running_loss:.4f} | Accuracy: {acc:.4f}%")

    # Early Stopping 조건 체크
    if acc > best_acc:
        best_acc = acc
        counter = 0
        # 모델도 이 시점에서 저장
        torch.save(model.state_dict(), model_save_path)
    else:
        counter += 1
        if counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}. No improvement in accuracy for {patience} epochs.")
            break

# ---------------------- 모델 저장 ---------------------- #
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ 모델 저장 완료 → {os.path.abspath(model_save_path)}")