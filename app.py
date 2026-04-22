from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{stock}")
def predict(stock: str):
    data = yf.download(stock, period="1y")

    if data.empty:
        return {"error": "Invalid stock symbol"}

    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close_prices = data['Close']

    df = close_prices.to_frame()
    df['Prediction'] = df['Close'].shift(-1)

    X = np.array(df[['Close']][:-1])
    y = np.array(df['Prediction'][:-1])

    model = LinearRegression()
    model.fit(X, y)

    last_price = np.array(df[['Close']].tail(1))
    prediction = model.predict(last_price)

    prices = close_prices.tail(30).tolist()

    signal = "BUY" if prediction[0] > last_price[0][0] else "SELL"

    return {
        "stock": stock,
        "current_price": float(last_price[0][0]),
        "predicted_price": float(prediction[0]),
        "signal": signal,
        "prices": prices
    }
