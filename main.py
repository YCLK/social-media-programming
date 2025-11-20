# pip install tensorflow==2.20.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow import keras
import tensorflow as tf


# 데이터 불러오기
df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')
X = df[['Daily_Screen_Time(hrs)','Stress_Level(1-10)','Days_Without_Social_Media']]
y = df['Happiness_Index(1-10)']

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = keras.models.load_model('my_model.keras')

y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("결정계수:", r2)
print("평균제곱오차:", mse)

print(model.predict(np.array([[10, 10, 2]])))
print(model.predict(np.array([[1, 1, 2]])))

residuals = y - y_pred.reshape(-1)
plt.scatter(y_pred, residuals, marker='^', alpha=0.2)
plt.axhline(0, color='red', linestyle='--')
plt.show()