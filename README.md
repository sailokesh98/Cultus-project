Advanced Time Series Forecasting with N-BEATS and Explainability
Project Overview

This project implements Advanced Time Series Forecasting using Neural Networks with Explainability, based on the N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) architecture.

The objective is to design, implement, and evaluate N-BEATS from scratch for multi-horizon forecasting, while explicitly decomposing forecasts into trend and seasonality components.
The model is evaluated against a traditional statistical benchmark (SARIMA) using standard error metrics.

This project follows the exact expectations described in the provided project brief.

Key Objectives

Programmatically generate a complex synthetic time series with:

Clear long-term trend

Multiple seasonalities (daily & weekly)

Controlled noise

Implement N-BEATS from scratch using PyTorch

Avoid high-level forecasting libraries (e.g., Prophet, Darts)

Perform multi-horizon forecasting

Leverage N-BEATS interpretability to extract:

Trend component

Seasonality component

Compare performance with SARIMA

Evaluate using RMSE, MAE, and MAPE

Provide qualitative and quantitative analysis

Dataset Description

The dataset is synthetically generated using NumPy to ensure full control over its components.

Components:

Trend: Linear increasing trend

Seasonality:

Daily seasonality (24-step period)

Weekly seasonality (168-step period)

Noise: Gaussian noise

This design allows clear evaluation of the model’s ability to learn and decompose patterns.

Model Architecture: N-BEATS

N-BEATS is a deep neural architecture built using fully connected layers, unlike RNNs or LSTMs.

Architecture Highlights

Backcast–Forecast mechanism

Sequential blocks that:

Explain parts of the input (backcast)

Produce forecasts

Residual learning across blocks

Explicit trend and seasonality decomposition

Implemented Blocks
1. Trend Block

Uses polynomial basis expansion

Captures long-term movement

Interpretable coefficients

2. Seasonality Block

Uses Fourier (sine & cosine) basis

Captures repeating patterns

Supports multiple seasonalities

Forecasting Setup

Input window (lookback): 168 time steps

Forecast horizon: 24 time steps

Train/Test split: 80% / 20%

The model predicts multiple future steps simultaneously.

Benchmark Model

To validate performance, a SARIMA (Seasonal ARIMA) model is implemented as a traditional statistical benchmark.

Seasonal period: 24

Evaluated on the same test horizon

Compared using RMSE and MAE

Evaluation Metrics

The following standard metrics are used:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

These metrics provide both scale-dependent and relative performance evaluation.

Interpretability & Explainability

One of the key strengths of N-BEATS is its inherent interpretability.

This project explicitly extracts:

Trend forecast

Seasonality forecast

Visualizations are generated for:

Trend vs Seasonality decomposition

Actual vs Predicted forecasts

This enables qualitative assessment of learned components.

Project Structure
├── ADVANCED TIME SERIES FORECASTING.py   # Complete end-to-end implementation
├── README.md                # Project documentation


The entire logic is contained in a single Python script for clarity and reproducibility.

Technologies Used

Python

NumPy

PyTorch

Matplotlib

Scikit-learn

Statsmodels (SARIMA)

How to Run

Install required packages:

pip install numpy torch matplotlib scikit-learn statsmodels


Run the script:

ADVANCED TIME SERIES FORECASTING.py


Outputs:

Training loss logs

Evaluation metrics

Forecast plots

Trend & seasonality decomposition plots

Key Learnings

N-BEATS can outperform traditional models on nonlinear patterns

Fully connected networks can be powerful for time-series forecasting

Explicit basis functions enable model interpretability

Neural models can provide both accuracy and explainability

Conclusion

This project demonstrates a complete, interpretable, and rigorous implementation of N-BEATS for time series forecasting.
It fulfills all requirements of the advanced project brief, including from-scratch modeling, explainability, benchmarking, and evaluation.
