# # src/predict_manual_input.py

# import streamlit as st
# import numpy as np
# import torch
# import sys
# import os

# # ---------- 경로 설정 ----------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.append(current_dir)

# from model import TCN

# # ---------- 설정 ----------
# INPUT_FEATURES = ['PM2.5', 'NTC', 'CT1', 'CT2', 'CT3', 'CT4', 'temp_max']
# WINDOW_SIZE = 30
# MODEL_PATH = os.path.join(root_dir, "results", "tcn_model_best.pth")

# # ---------- 모델 로드 ----------
# @st.cache_resource
# def load_model():
#     model = TCN(input_size=len(INPUT_FEATURES), output_size=4)
#     model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
#     model.eval()
#     return model

# # ---------- Streamlit 앱 ----------
# st.title("🧪 센서값 수동 입력 → OHT 상태 예측")

# st.write("최근 30개 시점의 센서 데이터를 입력해주세요.")

# # ---------- 입력창 ----------
# input_window = []
# for i in range(WINDOW_SIZE):
#     st.markdown(f"### 🔢 시점 {i + 1}")
#     values = []
#     for feature in INPUT_FEATURES:
#         val = st.number_input(f"{feature} (시점 {i + 1})", key=f"{feature}_{i}")
#         values.append(val)
#     input_window.append(values)

# # ---------- 예측 ----------
# if st.button("🚀 예측 실행"):
#     input_array = np.array(input_window).reshape(1, WINDOW_SIZE, len(INPUT_FEATURES))  # (1, win, feat)
#     input_tensor = torch.tensor(input_array, dtype=torch.float32)

#     model = load_model()
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_class = torch.argmax(output, dim=1).item()
#         st.success(f"✅ 예측된 상태: **state {predicted_class}**")
