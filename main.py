import os
import pandas as pd
import random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
data_dir = './predictionOfPopulation/miniData'

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

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=1,
    validation_split=0.1,
    callbacks=[early_stop]
)

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

y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"\nMean Squared Error(MSE): {mse:.6f}")
mae = np.mean(np.abs(y_test - y_pred))
print(f"Mean absolute error(MAE): {mae:.6f}")

# boundary.csv 불러오기(마스크)
boundary_mask = pd.read_csv('./predictionOfPopulation/miniData/boundary.csv', header=None).values  # (72, 49)

# Tensor로 변환 및 broadcast
boundary_mask_tensor = tf.convert_to_tensor(boundary_mask, dtype=tf.float32)  # (72, 49)
boundary_mask_tensor = tf.expand_dims(boundary_mask_tensor, axis=0)  # (1, 72, 49)
boundary_mask_tensor = tf.expand_dims(boundary_mask_tensor, axis=0) # (1, 1, 72, 49)
boundary_mask_tensor = tf.broadcast_to(boundary_mask_tensor, y_test.shape)    # (N, 4, 72, 49)

masked_y_test = y_test * boundary_mask_tensor
masked_y_pred = y_pred * boundary_mask_tensor

masked_y_test_flat = masked_y_test.numpy().flatten()
masked_y_pred_flat = masked_y_pred.numpy().flatten()


EPSILON = 1e-6
safe_divisor = np.where(np.abs(masked_y_test_flat) < EPSILON, EPSILON, masked_y_test_flat)
mape_masked = np.mean(np.abs((masked_y_test_flat - masked_y_pred_flat) / safe_divisor))

rmse_masked = np.sqrt(mean_squared_error(masked_y_test_flat, masked_y_pred_flat))
r2_masked = r2_score(masked_y_test_flat, masked_y_pred_flat)

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
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))  # 위: 원본 / 아래: 마스크 적용

    for i in range(4):
        hour_idx = set_idx * 4 + i

        # 1. 원본데이터 예측
        ax_raw = axes[0, i]
        ax_raw.imshow(predictions[hour_idx], cmap='RdYlGn_r', vmin=0, vmax=1)
        ax_raw.set_title(f"[Raw] 02-28 {hour_idx:02d}:00")
        ax_raw.axis('off')

        # 2. 마스크 적용 예측
        ax_masked = axes[1, i]
        masked_pred = predictions[hour_idx] * boundary_mask
        ax_masked.imshow(masked_pred, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax_masked.set_title(f"[Masked] 02-28 {hour_idx:02d}:00")
        ax_masked.axis('off')
    
    plt.tight_layout()
    plt.show()

# 학습 그래프
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training & Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 모델 성능 평가지표
avg_abs_error_map = np.mean(np.abs(y_test - y_pred), axis=(0, 1))
plt.figure(figsize=(6, 4))
plt.imshow(avg_abs_error_map, cmap='RdYlGn_r', vmin=0, vmax=1)
plt.title("Mean absolute error(MAE)")
plt.colorbar()
plt.tight_layout()
plt.show()

y_test_flat = y_test.reshape(-1)
y_pred_flat = y_pred.reshape(-1)

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)
mape = mean_absolute_percentage_error(y_test_flat, y_pred_flat)

print(f"RMSE (Root Mean Squared Error):             {rmse:.6f}")
print(f"R² Score (coefficient of determination):   {r2:.6f}")
print(f"MAPE (Mean Absolute % Error):   {mape * 100:.2f}%")

print("\n[Boundary Masked 평가 지표]")
print(f"MAPE (Masked): {mape_masked * 100:.2f}%")
print(f"RMSE (Masked): {rmse_masked:.6f}")
print(f"R²   (Masked): {r2_masked:.6f}")

masked_abs_error = np.abs(masked_y_test - masked_y_pred)  # (N, 4, 72, 49)
avg_masked_error = np.mean(masked_abs_error, axis=(0, 1))  # (72, 49)

plt.figure(figsize=(6, 4))
plt.imshow(avg_masked_error, cmap='RdYlGn_r', vmin=0, vmax=1)
plt.title("Masked Mean Absolute Error Map")
plt.colorbar()
plt.tight_layout()
plt.show()