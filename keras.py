import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow import keras
import tensorflow as tf

'''
# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 사용 가능:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("❌ GPU를 인식하지 못했습니다.")
'''

# 데이터 불러오기
df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')

X = df[['Daily_Screen_Time(hrs)','Stress_Level(1-10)', 'Sleep_Quality(1-10)']]
y = df['Happiness_Index(1-10)']

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 복잡한 모델 구성
model = keras.Sequential([
    keras.layers.Dense(512, input_dim=3, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # 출력층
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# EarlyStopping
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

# 학습
model.fit(X_scaled, y, epochs=3000, callbacks=[early_stop], verbose=1)

# 예측 및 성능 계산
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
print(f"R² 결정계수: {r2:.4f}")
print(f"평균제곱오차(MSE): {mse:.4f}")

# 파일명에 성능 포함하여 저장 (매번 새로운 파일 생성)
filename = f"my_model_R2_{r2:.4f}_MSE_{mse:.4f}.keras"
model.save(filename)

print(f"✅ 모델 저장 완료: {filename}")
