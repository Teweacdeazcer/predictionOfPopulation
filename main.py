import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from sklearn.model_selection import train_test_split

data_dir = './miniData'

input_data = []  
labels = []      

for day in range(1, 28):  
    date_str = f"202202{day:02d}"  # '20220201', '20220202', ...
    
    for hour in range(24):  
        male_file = f"{date_str}_m_2529_P_T_{hour:02d}.csv"
        female_file = f"{date_str}_w_2529_P_T_{hour:02d}.csv"
        
        male_path = os.path.join(data_dir, male_file)
        female_path = os.path.join(data_dir, female_file)
        
        male_df = pd.read_csv(male_path, header=None)
        female_df = pd.read_csv(female_path, header=None)
        
        # 남자+여자 합산
        total_df = male_df + female_df
        
        input_data.append(total_df.values.flatten())
        
        labels.append([int(date_str), hour])

input_data = np.array(input_data)
labels = np.array(labels)

test_data = []
test_labels = []
date_str = "20220228"
for hour in range(24):
    male_file = f"{date_str}_m_2529_P_T_{hour:02d}.csv"
    female_file = f"{date_str}_w_2529_P_T_{hour:02d}.csv"

    male_path = os.path.join(data_dir, male_file)
    female_path = os.path.join(data_dir, female_file)

    male_df = pd.read_csv(male_path, header=None)
    female_df = pd.read_csv(female_path, header=None)

    total_df = male_df + female_df

    test_data.append(total_df.values.flatten())
    test_labels.append([int(date_str), hour])

test_data = np.array(test_data)
test_labels = np.array(test_labels)

def extract_features(date_int, hour):
    date = pd.to_datetime(str(date_int))
    weekday = date.weekday()  # 월=0, 일=6
    is_weekend = 1 if weekday >= 5 else 0
    
    weekday_onehot = np.eye(7)[weekday]  # 7차원
    hour_onehot = np.eye(24)[hour]       # 24차원
    
    feature_vec = np.concatenate([weekday_onehot, hour_onehot, [is_weekend]])
    return feature_vec

grid_input = Input(shape=(49, 72, 1))
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(grid_input)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)

# Conv2DTranspose
x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(16, (3,3), strides=(1,1), padding='same', activation='relu')(x)

feature_input = Input(shape=(32,))
f = layers.Dense(64, activation='relu')(feature_input)
f = layers.Dense(49*72, activation='relu')(f)
f = layers.Reshape((49,72,1))(f)

combined = layers.Concatenate(axis=-1)([x, f])

output = layers.Conv2D(1, (3,3), padding='same', activation='linear')(combined)

model = Model(inputs=[grid_input, feature_input], outputs=output)

model.compile(optimizer='adam', loss='mse')

input_data_reshaped = input_data.reshape((-1, 49, 72, 1))
test_data_reshaped = test_data.reshape((-1, 49, 72, 1))

feature_data = np.array([extract_features(date, hour) for date, hour in labels])
test_feature_data = np.array([extract_features(date, hour) for date, hour in test_labels])


X_train_grid, X_test_grid, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
    input_data_reshaped, feature_data, input_data_reshaped, test_size=0.2, random_state=42)

train_idx = labels[:,0] < 20220228
test_idx = labels[:,0] == 20220228

X_train_grid = input_data_reshaped[train_idx]
X_test_grid = input_data_reshaped[test_idx]

X_train_feat = feature_data[train_idx]
X_test_feat = feature_data[test_idx]

y_train = input_data_reshaped[train_idx]
y_test = input_data_reshaped[test_idx]


history = model.fit(
    [input_data_reshaped, feature_data], input_data_reshaped,
    validation_data=([test_data_reshaped, test_feature_data], test_data_reshaped),
    epochs=30,
    batch_size=8
)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

pred = model.predict([test_data_reshaped, test_feature_data], batch_size=8)

plt.imshow(test_data_reshaped[0].squeeze().T, cmap='RdYlGn_r')
plt.title('Input')
plt.colorbar()

plt.figure()
plt.imshow(pred[0].squeeze().T, cmap='RdYlGn_r')
plt.title('Prediction')
plt.colorbar()

plt.show()

x = layers.Conv2DTranspose(32, (3,3), strides=(1,1), padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(16, (3,3), strides=(1,1), padding='same', activation='relu')(x)
output = layers.Conv2D(1, (3,3), padding='same', activation='linear')(x)