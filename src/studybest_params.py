import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import optuna
from model import TCN  # model.py에 정의된 TCN 모델

# ---------------------- 데이터셋 클래스 정의 ---------------------- #
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------- 경로 설정 및 데이터 로드 ---------------------- #
current_dir = os.path.dirname(os.path.abspath(__file__))
X_path = os.path.join(current_dir, "..", "results", "X_windows.npy")
y_path = os.path.join(current_dir, "..", "results", "y_labels.npy")

X = np.load(X_path)
y = np.load(y_path)
input_size = X.shape[2]
output_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- DataLoader 구성 ---------------------- #
dataset = SequenceDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------------- 클래스 가중치 직접 입력 ---------------------- #
# 클래스별 샘플 수: [17368, 11367, 11404, 4146]
# 총 샘플 수: 44285
# 계산된 가중치 (boost 미적용 상태)
# class_weights = [1, 3.895926805322783, 3.8832865657766322, 1]
class_weights = [1.0, 3.895926805322783, 1, 1.0]

weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ---------------------- Optuna 목적 함수 정의 ---------------------- #
def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    num_channels = trial.suggest_categorical("num_channels", 
                                             [[64, 64, 64, 64],
                                              [32, 64, 64, 64],
                                              [32, 64, 128, 128],
                                              [64, 128, 128, 64],
                                              [128, 128, 64, 32],
                                              [32, 32, 64, 64]])

    model = TCN(input_size=input_size, output_size=output_size,
                num_channels=num_channels, kernel_size=7, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 🔹 학습 반복 횟수 추가
    EPOCHS = 5
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # 🔹 전체 학습 데이터로 평가
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = torch.argmax(model(X_batch), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    return f1_score(all_targets, all_preds, average="macro")

# ---------------------- Optuna 튜닝 실행 ---------------------- #
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("✅ Best Hyperparameters:", study.best_params)
