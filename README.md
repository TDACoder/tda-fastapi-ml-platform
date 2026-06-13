# TDA-ML Platform

An advanced, asynchronous time-series inference pipeline that combines **Topological Data Analysis (TDA)** with Deep Learning (**PyTorch**) and serves it via **FastAPI**.

## Features
- **TDA Feature Engineering:** Uses Takens Embedding and Vietoris-Rips Persistence (via `giotto-tda`) to extract topological features from time-series data.
- **Asynchronous Architecture:** Utilizes Python's `asyncio` and a `ProcessPoolExecutor` to handle heavy mathematical computations without blocking the FastAPI event loop.
- **Deep Learning Inference:** A Convolutional Novel Network (CNN) built in PyTorch that processes the persistence images.
- **Data Validation:** Strict input validation with Pydantic v2.

## How to run (Example)
1. Install requirements: `pip install -r requirements.txt`
2. Run the server: `python main.py`
