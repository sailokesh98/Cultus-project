# Advanced Time Series Forecasting with N-BEATS and Explainability

## 📌 Project Overview

This project implements **Advanced Time Series Forecasting using Neural Networks with Explainability**, based on the **N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting)** architecture.

The goal is to **design, implement, and rigorously evaluate** an N-BEATS model for **multi-horizon time series forecasting**, while leveraging its inherent interpretability to decompose forecasts into **trend and seasonality components**.
The model is built **from scratch using PyTorch**, without relying on high-level forecasting libraries such as Prophet or Darts.

A **traditional statistical benchmark (SARIMA)** is used for comparison, and performance is evaluated using standard error metrics.

---

## 🎯 Project Objectives

- Generate a **synthetic time series dataset** with:
  - Clear long-term trend
  - Multiple seasonalities (daily and weekly)
  - Controlled noise
- Implement **N-BEATS architecture from scratch**
- Perform **multi-horizon forecasting**
- Decompose forecasts into **trend and seasonality**
- Compare performance with a **benchmark model (SARIMA)**
- Evaluate using **RMSE, MAE, and MAPE**
- Provide both **quantitative and qualitative analysis**

---

## 📊 Dataset Description

The dataset is **programmatically generated** using NumPy to ensure full control over its structure.

### Components:
- **Trend**: Linear increasing trend
- **Seasonality**:
  - Daily seasonality (period = 24)
  - Weekly seasonality (period = 168)
- **Noise**: Gaussian noise

This setup enables clear evaluation of the model’s ability to learn and separate underlying patterns.

---

## 🧠 Model Architecture: N-BEATS

N-BEATS is a deep neural network architecture composed of **fully connected layers**, designed specifically for time series forecasting.

### Key Concepts:
- Backcast–Forecast mechanism
- Residual learning across blocks
- Explicit basis expansion for interpretability
- No recurrent or convolutional layers

### Implemented Blocks

#### 🔹 Trend Block
- Polynomial basis expansion
- Captures long-term behavior
- Interpretable polynomial coefficients

#### 🔹 Seasonality Block
- Fourier (sine and cosine) basis expansion
- Captures repeating patterns
- Supports multiple seasonalities

---

## ⏱ Forecasting Setup

| Parameter | Value |
|--------|-------|
| Lookback Window | 168 |
| Forecast Horizon | 24 |
| Train/Test Split | 80% / 20% |

The model predicts **multiple future time steps simultaneously**.

---

## 📉 Benchmark Model

A **SARIMA (Seasonal ARIMA)** model is used as a traditional statistical benchmark.

- Seasonal period: 24
- Trained on the same data
- Evaluated on the same forecast horizon

This comparison highlights the advantages of neural forecasting on nonlinear patterns.

---

## 📏 Evaluation Metrics

The following metrics are used for evaluation:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

---

## 🔍 Interpretability & Explainability

One of the core strengths of N-BEATS is its **inherent interpretability**.

This project explicitly extracts and visualizes:
- Trend forecast
- Seasonality forecast
- Actual vs predicted values

---

## 📂 Project Structure

```
.
├── ADVANCED TIME SERIES FORECASTING.py   # Complete end-to-end implementation
├── README.md                # Project documentation
```

---

## 🛠 Technologies Used

- Python
- NumPy
- PyTorch
- Matplotlib
- Scikit-learn
- Statsmodels (SARIMA)

---

## ▶️ How to Run

### 1. Install dependencies
```
pip install numpy torch matplotlib scikit-learn statsmodels
```

### 2. Run the project
```
ADVANCED TIME SERIES FORECASTING.py
```

---

## 📚 Key Learnings

- Fully connected networks can be highly effective for time series forecasting
- N-BEATS achieves strong performance without recurrence or attention
- Basis expansions enable **transparent and interpretable forecasts**
- Neural models can outperform traditional statistical methods on complex patterns

---

## ✅ Conclusion

This project demonstrates a **complete, interpretable, and benchmarked implementation** of the N-BEATS architecture for time series forecasting.
It fulfills all requirements of an advanced forecasting assignment, including **from-scratch modeling, explainability, evaluation, and comparison with classical methods**.

---

## 📌 Author

**Sai Lokesh**

---

## 📜 License

This project is intended for **educational and academic use**.
