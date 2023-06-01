import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_portfolio_returns(df):
    """
    Calculates monthly returns and cumulative returns for a portfolio given a DataFrame with ticker names and allocations.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'Ticker' and 'Allocation' columns.
        start_date (str): Start date for the calculation in 'YYYY-MM-DD' format. Default is '2010-01-01'.
    
    Returns:
        pd.DataFrame: DataFrame with monthly returns and cumulative returns for the portfolio.
    """
    # Create an empty DataFrame to store the monthly returns
    portfolio_returns = pd.DataFrame()

    # Iterate through each ticker in the DataFrame
    for _, row in df.iterrows():
        ticker = row['Ticker']
        allocation = row['Allocation']

        # Get the historical data for the ticker from Yahoo Finance
        data = yf.download(ticker, start='2010-01-01')
        
        # Filter out rows with missing or zero values
        data = data[data['Adj Close'].notna() & (data['Adj Close'] > 0)]
        
        # Resample the data to monthly frequency
        data_monthly = data['Adj Close'].resample('M').ffill()
        
        # Calculate the monthly returns for the ticker
        returns = data_monthly.pct_change()
        
        # Add the returns to the portfolio_returns DataFrame, weighted by the allocation
        portfolio_returns[ticker] = returns * allocation

    # Calculate the total monthly returns for the portfolio
    portfolio_returns['Portfolio'] = portfolio_returns.sum(axis=1)

    # Calculate the cumulative returns
    cumulative_returns = portfolio_returns['Portfolio'].cumsum() * 100

    # Return the portfolio returns DataFrame
    return cumulative_returns