# ============================================================
# ADVANCED TIME SERIES FORECASTING WITH N-BEATS (FULL CODE)
# ============================================================

# -------------------------
# 1. IMPORTS & SETUP
# -------------------------
import numpy as np
import pandas as pd
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# -------------------------
# 2. SYNTHETIC DATA GENERATION
# -------------------------
def generate_time_series(
    n_steps=2000,
    daily_period=24,
    weekly_period=168,
    noise_std=0.3
):
    t = np.arange(n_steps)

    trend = 0.005 * t
    daily = 2 * np.sin(2 * np.pi * t / daily_period)
    weekly = 1.5 * np.sin(2 * np.pi * t / weekly_period)
    noise = np.random.normal(0, noise_std, size=n_steps)

    series = trend + daily + weekly + noise
    return series, trend, daily + weekly

series, true_trend, true_seasonality = generate_time_series()

# -------------------------
# 3. WINDOWING
# -------------------------
INPUT_SIZE = 168
HORIZON = 24

def create_windows(series, input_size, horizon):
    X, y = [], []
    for i in range(len(series) - input_size - horizon):
        X.append(series[i:i+input_size])
        y.append(series[i+input_size:i+input_size+horizon])
    return np.array(X), np.array(y)

X, y = create_windows(series, INPUT_SIZE, HORIZON)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -------------------------
# 4. N-BEATS BLOCKS
# -------------------------
class TrendBlock(nn.Module):
    def __init__(self, input_size, horizon, hidden_size=256, degree=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.theta = nn.Linear(hidden_size, degree + 1)
        self.degree = degree
        self.input_size = input_size
        self.horizon = horizon

    def forward(self, x):
        theta = self.theta(self.fc(x))

        t_back = torch.linspace(0, 1, self.input_size).to(x.device)
        t_fore = torch.linspace(0, 1, self.horizon).to(x.device)

        backcast = sum(theta[:, i:i+1] * t_back**i for i in range(self.degree+1))
        forecast = sum(theta[:, i:i+1] * t_fore**i for i in range(self.degree+1))

        return backcast, forecast


class SeasonalityBlock(nn.Module):
    def __init__(self, input_size, horizon, hidden_size=256, harmonics=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.theta = nn.Linear(hidden_size, 2 * harmonics)
        self.harmonics = harmonics
        self.input_size = input_size
        self.horizon = horizon

    def forward(self, x):
        theta = self.theta(self.fc(x))
        p = torch.arange(self.harmonics).float().to(x.device)

        def seasonality(t):
            cos = torch.cos(2 * math.pi * p * t)
            sin = torch.sin(2 * math.pi * p * t)
            return torch.cat([cos, sin], dim=0)

        back_t = torch.linspace(0, 1, self.input_size).to(x.device)
        fore_t = torch.linspace(0, 1, self.horizon).to(x.device)

        S_back = torch.stack([seasonality(t) for t in back_t])
        S_fore = torch.stack([seasonality(t) for t in fore_t])

        backcast = torch.matmul(S_back, theta.unsqueeze(-1)).squeeze(-1).T
        forecast = torch.matmul(S_fore, theta.unsqueeze(-1)).squeeze(-1).T

        return backcast, forecast

# -------------------------
# 5. FULL N-BEATS MODEL
# -------------------------
class NBEATS(nn.Module):
    def __init__(self, input_size, horizon):
        super().__init__()
        self.trend = TrendBlock(input_size, horizon)
        self.seasonality = SeasonalityBlock(input_size, horizon)

    def forward(self, x):
        residual = x
        forecast = torch.zeros((x.size(0), HORIZON)).to(x.device)

        for block in [self.trend, self.seasonality]:
            backcast, f = block(residual)
            residual = residual - backcast
            forecast = forecast + f

        return forecast

# -------------------------
# 6. TRAINING
# -------------------------
model = NBEATS(INPUT_SIZE, HORIZON)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

EPOCHS = 25

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# -------------------------
# 7. EVALUATION
# -------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()

y_true = y_test.numpy()

rmse = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()))
mae = mean_absolute_error(y_true.flatten(), preds.flatten())
mape = np.mean(np.abs((y_true - preds) / y_true)) * 100

print("\nN-BEATS RESULTS")
print("RMSE:", rmse)
print("MAE :", mae)
print("MAPE:", mape)

# -------------------------
# 8. SARIMA BENCHMARK
# -------------------------
sarima = SARIMAX(
    series[:split + INPUT_SIZE],
    order=(2,1,2),
    seasonal_order=(1,1,1,24)
)

sarima_fit = sarima.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=len(y_true.flatten()))

sarima_rmse = np.sqrt(mean_squared_error(y_true.flatten(), sarima_forecast))
sarima_mae = mean_absolute_error(y_true.flatten(), sarima_forecast)

print("\nSARIMA RESULTS")
print("RMSE:", sarima_rmse)
print("MAE :", sarima_mae)

# -------------------------
# 9. INTERPRETABILITY
# -------------------------
sample = X_test[0:1]

with torch.no_grad():
    trend_back, trend_fore = model.trend(sample)
    seas_back, seas_fore = model.seasonality(sample)

plt.figure(figsize=(12,4))
plt.plot(trend_fore[0], label="Trend Forecast")
plt.plot(seas_fore[0], label="Seasonality Forecast")
plt.legend()
plt.title("N-BEATS Forecast Decomposition")
plt.show()

# -------------------------
# 10. ACTUAL vs PREDICTED
# -------------------------
plt.figure(figsize=(12,4))
plt.plot(y_true[0], label="Actual")
plt.plot(preds[0], label="N-BEATS Prediction")
plt.legend()
plt.title("Actual vs Forecast")
plt.show()
