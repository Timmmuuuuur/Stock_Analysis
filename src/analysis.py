from dataclasses import dataclass
import shutil
from unittest import result
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta import add_all_ta_features
from ta.utils import dropna
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'

# link: https://medium.com/analytics-vidhya/python-for-stock-analysis-fcff252ca559

def get_basic_data(stock_df):
    # get basic stock infor by analyzing df using pandas
    print(f"{stock_df} stock information:")
    print(stock_df.describe())
    print(stock_df.info())


def price_change(stock_df, stock_name):
    # plots change in stock price over time based on the adjusted close value
    stock_df.set_index('Date',inplace=True)
    stock_df['Adj Close'].plot(legend=True,figsize=(12,5))
    plt.title(f"{stock_name} Stock Price")
    plt.show()
    # plot the change in traded volume for a particular stock
    stock_df.plot(legend=True,figsize=(12,5))
    plt.show()

def tech_indicators(stock_df):
    ta_data = add_all_ta_features(stock_df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    print(ta_data.columns)

def moving_averages(stock_df, stock_name):
    """
    Compute the moving averages for stocks over 10, 20, and 50 day period
    Used to identify possible indicators of trend changes when considering long and short positions
    """
    mov_avg_days = [10, 20, 50]
    for day in mov_avg_days:
        column_name = "MA for %s days" %(str(day))
        stock_df[column_name] = stock_df['Adj Close'].rolling(window=day,center=False).mean()
    # print(stock_df.tail())
    stock_df.set_index('Date',inplace=True)
    stock_df[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(12,5))
    plt.title(f"{stock_name} Moving Averages")
    plt.show()

def daily_return(stock_df, stock_name):
    """
    Indicate polarity of daily returns for a particular stock
    """
    stock_df['Daily Return'] = stock_df['Adj Close'].pct_change()
    # print(stock_df['Daily Return'])
    plot = sns.displot(stock_df['Daily Return'].dropna(),bins=50,color='blue')
    plt.title(f"{stock_name} Daily Return")
    plt.show()

def comp_daily_return_corr(stock_list, symbol_list):
    """ 
    Calculate the correlation of daily returns values from a list of different stocks.
    Combine all stock data into a single dataframe 
    Compute Pearson correlation coefficient
    """
    data = []
    for stock, symbol in zip(stock_list, symbol_list):
        # Ensure 'Date' is in datetime format
        stock['Date'] = pd.to_datetime(stock['Date'], errors='coerce')

        # Convert 'Adj Close' to numeric, coercing any errors
        stock['Adj Close'] = pd.to_numeric(stock['Adj Close'], errors='coerce')

        # Keep only the required columns
        stock_data = stock[['Date', 'Adj Close']].copy()
        stock_data['Symbol'] = symbol
        data.append(stock_data)

    # Concatenate all stock data into a single DataFrame
    df = pd.concat(data)
    df = df.dropna(subset=['Date', 'Adj Close'])  # Drop rows with missing values in 'Date' or 'Adj Close'
    
    # Pivot the data to create a table where each column is a stock symbol
    df_pivot = df.pivot(index='Date', columns='Symbol', values='Adj Close').dropna(axis=0)

    # Compute the Pearson correlation coefficient, excluding non-numeric columns
    corr_df = df_pivot.corr(method='pearson')

    # Plot the correlation matrix
    plt.figure(figsize=(13, 8))
    sns.heatmap(corr_df, annot=True, cmap="RdYlGn")
    plt.title("Correlation Matrix")
    plt.show()
  
    return df_pivot, corr_df


def stock_returns_plt(df_pivot):
    """
    Plot the return of a list of stocks using the Adjusted Close Price
    """
    # No need to set 'Date' as the index again; it already is the index
    df_pivot.plot(figsize=(10, 4))
    plt.ylabel('Price')
    plt.title(f"Price Plot for all Stocks 2021-2022")
    plt.show()

def normalizing_stocks(df_pivot):
    print(df_pivot.head())
    print(df_pivot.dtypes)
    returnfstart = df_pivot[['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']].apply(lambda x: x / x[0])
    # print(returnfstart.head())
    # returnfstart.plot(figsize=(10,4)).axhline(1, lw=1, color='black')
    # plt.ylabel('Return From Start Price')
    # plt.show()
    return

def daily_ret_percent(df_pivot):
    """
    Daily percentage return value for a list of stocks 
    """
    percent_diff_df = df_pivot[['AAPL', 'AMZN', 'GOOG', 'INTC', 'MSFT']].pct_change()
    percent_diff_df.insert(0, 'Date', df_pivot['Date'])
    percent_diff_df.set_index('Date',inplace=True)
    percent_diff_df.plot(figsize=(10,4))
    plt.axhline(0, color='black', lw=1)
    plt.ylabel('Daily Percentage Return')
    plt.title("Daily Percentage Return")
    plt.show()
    # for most plots fix xaxis so it shows the date
    return
    
def investment_risk_val(corr_df):
    """
    Quantify risk of investing in a particular stock by comparing the 
    expected return with that standard deviation of daily returns 
    """
    risk_df = corr_df.dropna()
    plt.figure(figsize=(8,5))
    plt.scatter(risk_df.mean(),risk_df.std(),s=25)
    plt.xlabel('Expected Return')
    plt.ylabel('Risk')

    # add annotations to the scatterplot
    for label,x,y in zip(risk_df.columns,risk_df.mean(),risk_df.std()):
        plt.annotate(
        label,
        xy=(x,y),xytext=(-120,20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad=-0.5'))
    plt.show()
    return

def get_val_at_risk(stock_df):
    """
    Compute the value at risk (amount of money we expect to lose of a given confidence interval) 
    """
    stock_df['Daily Return'] = stock_df['Close'] - stock_df['Open']
    sns.displot(stock_df['Daily Return'].dropna(),bins=100,color='purple')
    plt.title("Daily Percentage Return")
    plt.show()
    return



def main():
    # read in stock csv files 
    AAPL = pd.read_csv("./Dataset/AAPL.csv").dropna()
    MSFT = pd.read_csv("./Dataset/MSFT.csv").dropna()
    META = pd.read_csv("./Dataset/META.csv").dropna()
    GOOG = pd.read_csv("./Dataset/GOOG.csv").dropna()
    AMZN = pd.read_csv("./Dataset/AMZN.csv").dropna()

    result_path = './Results'
    if os.path.exists(result_path):
        shutil.rmtree(result_path, ignore_errors=False, onerror=None)
    
    # create new directory 
    # os.makedir(result_path)

    symbols = ['AAPL', 'MSFT', 'META', 'GOOG', 'AMZN']
    stocks = [AAPL, MSFT, META, GOOG, AMZN]

    # create directories corresponding to each stock

    # price_change(AAPL, symbols[0])
    # tech_indicators(AAPL)
    # moving_averages(AAPL)
    # daily_return(AAPL)

    df_pivot, corr_df = comp_daily_return_corr(stocks, symbols)
    stock_returns_plt(df_pivot)
    # normalizing_stocks(df_pivot)
    # daily_ret_percent(df_pivot)
    # investment_risk_val(corr_df)
    # get_val_at_risk(AAPL)


main()