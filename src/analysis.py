from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from ta import add_all_ta_features
from io import BytesIO
import numpy as np

app = FastAPI()

# Global variable to store stock data
stocks_data = {}
symbols = ['AAPL', 'MSFT', 'META', 'GOOG', 'AMZN']


@app.on_event("startup")
async def load_data():
    """
    Load stock data into memory on app startup
    """
    global stocks_data
    base_dir = "./Dataset/"
    for symbol in symbols:
        stocks_data[symbol] = pd.read_csv(os.path.join(base_dir, f"{symbol}.csv")).dropna()


@app.get("/")
def home():
    return {"message": "Welcome to the Stock Analysis API!"}


@app.get("/basic-info/{symbol}")
def get_basic_data(symbol: str):
    """
    Returns basic data info for a given stock
    """
    if symbol in stocks_data:
        stock_df = stocks_data[symbol]
        info = {
            "description": stock_df.describe().to_dict(),
            "info": stock_df.info(buf=None)
        }
        return JSONResponse(content=info)
    else:
        return JSONResponse(content={"error": "Stock symbol not found."}, status_code=404)


@app.get("/price-change/{symbol}")
def price_change(symbol: str):
    """
    Plot stock price change for a given symbol
    """
    if symbol in stocks_data:
        stock_df = stocks_data[symbol].copy()
        stock_df.set_index('Date', inplace=True)
        stock_df['Adj Close'].plot(legend=True, figsize=(12, 5))
        plt.title(f"{symbol} Stock Price")
        return plot_to_image()
    else:
        return JSONResponse(content={"error": "Stock symbol not found."}, status_code=404)


@app.get("/moving-averages/{symbol}")
def moving_averages(symbol: str):
    """
    Plot moving averages for a stock symbol
    """
    if symbol in stocks_data:
        stock_df = stocks_data[symbol].copy()
        mov_avg_days = [10, 20, 50]
        for day in mov_avg_days:
            column_name = f"MA for {day} days"
            stock_df[column_name] = stock_df['Adj Close'].rolling(window=day, center=False).mean()
        stock_df.set_index('Date', inplace=True)
        stock_df[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize=(12, 5))
        plt.title(f"{symbol} Moving Averages")
        return plot_to_image()
    else:
        return JSONResponse(content={"error": "Stock symbol not found."}, status_code=404)


@app.get("/daily-return/{symbol}")
def daily_return(symbol: str):
    """
    Plot daily return for a stock symbol
    """
    if symbol in stocks_data:
        stock_df = stocks_data[symbol].copy()
        stock_df['Daily Return'] = stock_df['Adj Close'].pct_change()
        plot = sns.displot(stock_df['Daily Return'].dropna(), bins=50, color='blue')
        plt.title(f"{symbol} Daily Return")
        return plot_to_image()
    else:
        return JSONResponse(content={"error": "Stock symbol not found."}, status_code=404)


@app.get("/correlation-matrix")
def correlation_matrix():
    """
    Plot the correlation matrix of the stock symbols
    """
    stock_list = [stocks_data[symbol].copy() for symbol in symbols]
    df_pivot, corr_df = compute_correlation_matrix(stock_list, symbols)
    plt.figure(figsize=(13, 8))
    sns.heatmap(corr_df, annot=True, cmap="RdYlGn")
    plt.title("Correlation Matrix")
    return plot_to_image()


@app.get("/stock-returns")
def stock_returns():
    """
    Plot the stock returns for all stocks
    """
    stock_list = [stocks_data[symbol].copy() for symbol in symbols]
    df_pivot, _ = compute_correlation_matrix(stock_list, symbols)
    df_pivot.plot(figsize=(10, 4))
    plt.ylabel('Price')
    plt.title("Price Plot for all Stocks")
    return plot_to_image()


def compute_correlation_matrix(stock_list, symbol_list):
    data = []
    for stock, symbol in zip(stock_list, symbol_list):
        stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce')
        stock['Adj Close'] = pd.to_numeric(stock['Adj Close'], errors='coerce')
        stock_data = stock[['Date', 'Adj Close']].copy()
        stock_data['Symbol'] = symbol
        data.append(stock_data)
    df = pd.concat(data)
    df = df.dropna(subset=['Date', 'Adj Close'])
    df_pivot = df.pivot(index='Date', columns='Symbol', values='Adj Close').dropna(axis=0)
    corr_df = df_pivot.corr(method='pearson')
    return df_pivot, corr_df


def plot_to_image():
    """
    Convert the matplotlib plot to an image that can be returned as a response
    """
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return FileResponse(buffer, media_type='image/png')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
