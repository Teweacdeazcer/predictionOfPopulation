import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


H, W = 72, 49  
TIME_WINDOW = 24
PREDICT_HOURS = 4

def load_heatmap_data(data_dir):
    male_data = []
    female_data = []
    day_idx = []
    hour_idx = []
    
    for day in range(1, 28):
        for hour in range(24):
            date_str = f"202202{day:02d}"
            try:
                male = pd.read_csv(os.path.join(data_dir, f"{date_str}_m_2529_P_T_{hour:02d}.csv"), header=None).values
                female = pd.read_csv(os.path.join(data_dir, f"{date_str}_w_2529_P_T_{hour:02d}.csv"), header=None).values
                male_data.append(male)
                female_data.append(female)
                day_idx.append(day)
                hour_idx.append(hour)
            except:
                continue

    return np.array(male_data), np.array(female_data), np.array(day_idx), np.array(hour_idx)

def multi_hot_time_day(hour, day):
    vec = np.zeros(24 + 7)
    vec[hour] = 1
    vec[24 + ((day - 1) % 7)] = 1  # 0:월 ~ 6:일 기준
    return vec  # shape: (31,)


# Load
data_dir = './miniData'

# input/output
male_data, female_data, days, hours = load_heatmap_data(data_dir)
total_frames = male_data.shape[0]

male_data = male_data / np.max(male_data)
female_data = female_data / np.max(female_data)

X = []
y = []
LIMIT = 50 

for i in range(min(LIMIT, total_frames - TIME_WINDOW - PREDICT_HOURS + 1)):
    # 공간 채널 구성: (male, female)
    male_seq = male_data[i:i+TIME_WINDOW]     # (24, 72, 49)
    female_seq = female_data[i:i+TIME_WINDOW] # (24, 72, 49)
    male_seq = np.transpose(male_seq, (1, 2, 0))   # (72, 49, 24)
    female_seq = np.transpose(female_seq, (1, 2, 0)) # (72, 49, 24)

    spatial_channel = np.concatenate([male_seq, female_seq], axis=-1)  # (72, 49, 48)

    # 시간+요일 멀티핫: (24, 31) → broadcast to (72, 49, 24×31)
    time_day_encodings = [multi_hot_time_day(hours[i+j], days[i+j]) for j in range(TIME_WINDOW)]
    time_day_encodings = np.array(time_day_encodings).T  # (31, 24)
    time_day_map = np.tile(time_day_encodings[np.newaxis, np.newaxis, :, :], (72, 49, 1, 1))  # (72, 49, 31, 24)
    time_day_map = np.reshape(time_day_map, (72, 49, 31 * 24))  # (72, 49, 744)

    # 최종 입력
    full_input = np.concatenate([spatial_channel, time_day_map], axis=-1)  # (72, 49, 792)
    X.append(full_input)

    # 타겟 y
    y.append(male_data[i+TIME_WINDOW:i+TIME_WINDOW+PREDICT_HOURS] + 
             female_data[i+TIME_WINDOW:i+TIME_WINDOW+PREDICT_HOURS])  # 합산값을 예측

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32) 

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Model definition
model = models.Sequential([
    layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(72, 49, 792)),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(PREDICT_HOURS, (1, 1), activation=None),
    layers.Permute((3, 1, 2))
])
model.compile(optimizer='adam', loss='mse')

model.summary()


model.fit(X_train, y_train, epochs=10, batch_size=1, validation_split=0.1)


# 마지막 24시간 입력
male_input_28 = male_data[-TIME_WINDOW:]     # (24, 72, 49)
female_input_28 = female_data[-TIME_WINDOW:] # (24, 72, 49)

male_input_28 = np.transpose(male_input_28, (1, 2, 0))    # (72, 49, 24)
female_input_28 = np.transpose(female_input_28, (1, 2, 0))  # (72, 49, 24)

spatial_input = np.concatenate([male_input_28, female_input_28], axis=-1)  # (72, 49, 48)

predictions = []

# 마지막 24시간 입력 준비
male_input_28 = male_data[-TIME_WINDOW:]     # (24, 72, 49)
female_input_28 = female_data[-TIME_WINDOW:] # (24, 72, 49)
final_hours = hours[-TIME_WINDOW:]
final_days = days[-TIME_WINDOW:]

male_input_28 = np.transpose(male_input_28, (1, 2, 0))     # (72, 49, 24)
female_input_28 = np.transpose(female_input_28, (1, 2, 0)) # (72, 49, 24)
spatial_input = np.concatenate([male_input_28, female_input_28], axis=-1)  # (72, 49, 48)

# 시간+요일 멀티핫 채널 만들기
time_day_encodings = [multi_hot_time_day(h, d) for h, d in zip(final_hours, final_days)]
time_day_encodings = np.array(time_day_encodings).T  # (31, 24)
time_day_map = np.tile(time_day_encodings[np.newaxis, np.newaxis, :, :], (H, W, 1, 1))  # (72, 49, 31, 24)
time_day_map = np.reshape(time_day_map, (H, W, 31 * 24))  # (72, 49, 744)

# 전체 입력 (72, 49, 792)
current_input = np.concatenate([spatial_input, time_day_map], axis=-1)

# 28일 24시간 예측 (4시간씩 6번)
next_hour = final_hours[-1]
next_day = final_days[-1]

for _ in range(6):
    pred_4h = model.predict(current_input[np.newaxis, ...])[0]  # (4, 72, 49)
    predictions.extend(pred_4h)

    # 다음 입력용 new heatmaps
    next_heatmaps = np.transpose(pred_4h, (1, 2, 0))  # (72, 49, 4)

    # 슬라이딩: spatial
    prev_spatial = current_input[:, :, :48]  # (72, 49, 48)
    prev_mh = current_input[:, :, 48:]       # (72, 49, 744)

    prev_male = prev_spatial[:, :, :24][:, :, 4:]  # (72, 49, 20)
    prev_female = prev_spatial[:, :, 24:][:, :, 4:]  # (72, 49, 20)

    new_male = next_heatmaps / 2  # 단순 예: 남녀 절반씩 분배 (임시)
    new_female = next_heatmaps / 2

    male_channel = np.concatenate([prev_male, new_male], axis=2)     # (72, 49, 24)
    female_channel = np.concatenate([prev_female, new_female], axis=2)  # (72, 49, 24)

    new_spatial = np.concatenate([male_channel, female_channel], axis=2)  # (72, 49, 48)

    # 시간/요일 업데이트 생성
    hour_updates = []
    day_updates = []

    for _ in range(4):
        next_hour = (next_hour + 1) % 24
        if next_hour == 0:
            next_day += 1
        hour_updates.append(next_hour)
        day_updates.append(next_day)

    # 시간/요일도 슬라이딩 & 생성
    final_hours = list(final_hours[4:]) + hour_updates
    final_days = list(final_days[4:]) + day_updates

    time_day_encodings = [multi_hot_time_day(h, d) for h, d in zip(final_hours, final_days)]  # 24개
    time_day_encodings = np.array(time_day_encodings).T  # (31, 24)
    time_day_map = np.tile(time_day_encodings[np.newaxis, np.newaxis, :, :], (H, W, 1, 1))  # (72, 49, 31, 24)
    time_day_map = np.reshape(time_day_map, (H, W, 744))  # (72, 49, 744)

    current_input = np.concatenate([new_spatial, time_day_map], axis=-1)  # (72, 49, 792)

for set_idx in range(6):  # 6세트 (0~3시, 4~7시, ...)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 가로 4개짜리 subplot
    
    for i in range(4):
        hour_idx = set_idx * 4 + i
        ax = axes[i]
        ax.imshow(predictions[hour_idx], cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f'Predicted 02-28 {hour_idx:02d}:00')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()