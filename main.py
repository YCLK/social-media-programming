# pip install tensorflow==2.20.0
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np


df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')
X = df[['Daily_Screen_Time(hrs)','Stress_Level(1-10)','Sleep_Quality(1-10)']]
y = df['Happiness_Index(1-10)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr = LinearRegression() #선형회귀분석을 위한 객체 생성
lr.fit(X_scaled, y) #LinearRegression 클래스 내의 fit() 함수를 통해 두 변수 간 회귀분석 실행

print("회귀모델의 결정계수 :", r2_score(y, lr.predict(X_scaled)))
print("회귀모델의 MSE :", mean_squared_error(y, lr.predict(X_scaled)))


model = keras.models.load_model('my_model_R2_0.7892_MSE_0.4818.keras')

y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("결정계수:", r2)
print("평균제곱오차:", mse)

print("[스크린타임 7시간 | 스트레스 7 | 수면의 질 2 → 행복도 예측]", (model.predict(np.array([[7, 7, 2]])))[0][0])
print("[스크린타임 2시간 | 스트레스 2 | 수면의 질 7 → 행복도 예측]", (model.predict(np.array([[2, 2, 7]])))[0][0])

residuals = y - y_pred.reshape(-1)

plt.subplot(2, 1, 1)
plt.scatter(lr.predict(X_scaled), y-lr.predict(X_scaled), marker='^', alpha=0.2)
plt.axhline(0, color='red', linestyle='--')

plt.subplot(2, 1, 2)
plt.scatter(y_pred, residuals, marker='^', alpha=0.2)
plt.axhline(0, color='red', linestyle='--')
plt.show()
