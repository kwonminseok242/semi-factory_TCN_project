import pandas as pd
import numpy as np
import os

def find_continuous_sequences(df, interval_sec=1, tolerance=0.1, min_length=50): # 허용범위 1초 이하
    """
    일정한 시간 간격을 갖는 연속된 시계열 구간을 찾습니다.
    - df: timestamp 컬럼 포함된 DataFrame
    - interval_sec: 기대 시간 간격 (초)
    - tolerance: 허용 오차 비율
    - min_length: 구간 최소 길이

    Returns:
    - List of (start_idx, end_idx, 시작시간, 끝시간, 길이)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    lower = interval_sec * (1 - tolerance)
    upper = interval_sec * (1 + tolerance)

    sequences = []
    start_idx = 0

    for i in range(1, len(df)):
        if not (lower <= df.loc[i, 'time_diff'] <= upper):
            if i - start_idx >= min_length:
                sequences.append((
                    start_idx,
                    i - 1,
                    df.loc[start_idx, 'timestamp'],
                    df.loc[i - 1, 'timestamp'],
                    i - start_idx
                ))
            start_idx = i

    if len(df) - start_idx >= min_length:
        sequences.append((
            start_idx,
            len(df) - 1,
            df.loc[start_idx, 'timestamp'],
            df.loc[len(df) - 1, 'timestamp'],
            len(df) - start_idx
        ))

    return sequences


def generate_windows_from_sequences(df, sequences, window_size=50,stride=1):
    """
    주어진 연속 구간 목록에서 슬라이딩 윈도우 시퀀스를 생성합니다.
    - df: timestamp, state 포함된 전체 DataFrame
    - sequences: find_continuous_sequences() 출력 리스트
    - window_size: 시퀀스 길이

    Returns:
    - X: (n_samples, window_size, n_features) numpy array
    - y: (n_samples,) numpy array
    """
    X, y = [], []

    for start, end, _, _, _ in sequences:
        sub_df = df.iloc[start:end + 1].reset_index(drop=True)
        for i in range(len(sub_df) - window_size):
            window = sub_df.iloc[i:i + window_size]
            features = window.drop(columns=['timestamp', 'state']).values
            label = window.iloc[-1]['state']
            X.append(features)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def save_sliding_window_data(X, y):
    # 항상 src/ 기준으로 상위 폴더의 results/에 저장되도록 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "..", "results")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "VX1_windows.npy"), X)
    np.save(os.path.join(save_dir, "Vy1_labels.npy"), y)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "data", "V_f_t_d.csv")

    # ✅ 파일 존재 여부 점검
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[❌] 파일이 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    sequences = find_continuous_sequences(df, interval_sec=1, tolerance=0.1, min_length=50)
    X, y = generate_windows_from_sequences(df, sequences, window_size=50, stride=1)
    save_sliding_window_data(X, y)

