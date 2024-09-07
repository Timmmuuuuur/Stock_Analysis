from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from databases import Database
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime

# FastAPI and SQLAlchemy setup
app = FastAPI()

SQLALCHEMY_DATABASE_URL = "sqlite:///./stocks.db"
database = Database(SQLALCHEMY_DATABASE_URL)

Base = declarative_base()
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Stock Model for SQLAlchemy
class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    symbol = Column(String, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    adj_close = Column(Float)
    volume = Column(Integer)

# Create the database table
Base.metadata.create_all(bind=engine)


@app.on_event("startup")
async def startup():
    """
    Initialize the database and stock data on startup
    """
    await database.connect()
    load_data()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


def get_db():
    """
    Dependency to get the SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Global variable to store stock data in memory
stocks_data = {}
symbols = ['AAPL', 'MSFT', 'META', 'GOOG', 'AMZN']


def load_data():
    """
    Load stock data from CSV files into memory and database
    """
    base_dir = "./Dataset/"
    for symbol in symbols:
        stock_df = pd.read_csv(os.path.join(base_dir, f"{symbol}.csv")).dropna()
        stocks_data[symbol] = stock_df
        # Save stock data to the database
        save_to_database(stock_df, symbol)


def save_to_database(stock_df, symbol: str, db: Session = None):
    """
    Save stock data into the SQL database
    """
    with SessionLocal() as db:
        for index, row in stock_df.iterrows():
            stock = Stock(
                date=datetime.strptime(row['Date'], "%Y-%m-%d"),
                symbol=symbol,
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                adj_close=row['Adj Close'],
                volume=row['Volume'],
            )
            db.add(stock)
        db.commit()


@app.get("/basic-info/{symbol}")
def get_basic_data(symbol: str, db: Session = Depends(get_db)):
    """
    Fetch basic data for a stock from the SQL database
    """
    if symbol not in symbols:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    result = db.query(Stock).filter(Stock.symbol == symbol).all()
    if not result:
        raise HTTPException(status_code=404, detail="No data found for the stock")

    df = pd.DataFrame([{
        'date': stock.date,
        'open': stock.open_price,
        'high': stock.high_price,
        'low': stock.low_price,
        'close': stock.close_price,
        'adj_close': stock.adj_close,
        'volume': stock.volume,
    } for stock in result])

    info = {
        "description": df.describe().to_dict(),
        "info": str(df.info(buf=None))
    }
    return JSONResponse(content=info)


@app.get("/price-change/{symbol}")
def price_change(symbol: str, db: Session = Depends(get_db)):
    """
    Plot stock price change from SQL database
    """
    if symbol not in symbols:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    result = db.query(Stock).filter(Stock.symbol == symbol).all()
    if not result:
        raise HTTPException(status_code=404, detail="No data found for the stock")

    df = pd.DataFrame([{
        'date': stock.date,
        'adj_close': stock.adj_close,
    } for stock in result])

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['adj_close'].plot(legend=True, figsize=(12, 5))
    plt.title(f"{symbol} Stock Price")
    return plot_to_image()


@app.get("/moving-averages/{symbol}")
def moving_averages(symbol: str, db: Session = Depends(get_db)):
    """
    Plot moving averages using NumPy for calculations
    """
    if symbol not in symbols:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    result = db.query(Stock).filter(Stock.symbol == symbol).all()
    if not result:
        raise HTTPException(status_code=404, detail="No data found for the stock")

    df = pd.DataFrame([{
        'date': stock.date,
        'adj_close': stock.adj_close,
    } for stock in result])

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Calculate moving averages using NumPy
    mov_avg_days = [10, 20, 50]
    for day in mov_avg_days:
        df[f"MA_{day}"] = np.convolve(df['adj_close'], np.ones(day)/day, mode='valid')

    df[['adj_close', 'MA_10', 'MA_20', 'MA_50']].plot(subplots=False, figsize=(12, 5))
    plt.title(f"{symbol} Moving Averages")
    return plot_to_image()


@app.get("/daily-return/{symbol}")
def daily_return(symbol: str, db: Session = Depends(get_db)):
    """
    Plot daily returns using NumPy for percentage change
    """
    if symbol not in symbols:
        raise HTTPException(status_code=404, detail="Stock symbol not found")

    result = db.query(Stock).filter(Stock.symbol == symbol).all()
    if not result:
        raise HTTPException(status_code=404, detail="No data found for the stock")

    df = pd.DataFrame([{
        'date': stock.date,
        'adj_close': stock.adj_close,
    } for stock in result])

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Calculate daily returns using NumPy
    df['daily_return'] = np.diff(df['adj_close'], prepend=np.nan) / df['adj_close']
    plot = sns.displot(df['daily_return'].dropna(), bins=50, color='blue')
    plt.title(f"{symbol} Daily Return")
    return plot_to_image()


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
