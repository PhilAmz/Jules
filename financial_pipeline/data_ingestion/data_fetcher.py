# This file will contain functions for fetching financial data.
import pandas as pd
import yfinance as yf

def fetch_financial_data(tickers, start_date, end_date, interval='1d'):
    """
    Fetches historical financial data for a given ticker or list of tickers from Yahoo Finance.

    Args:
        tickers (str or list): A single ticker symbol or a list of ticker symbols.
        start_date (str): The start date for the data (YYYY-MM-DD).
        end_date (str): The end date for the data (YYYY-MM-DD).
        interval (str, optional): The interval of the data (e.g., '1d', '1h', '1wk'). Defaults to '1d'.

    Returns:
        pandas.DataFrame: A DataFrame containing the historical financial data with a DatetimeIndex.
                          Returns an empty DataFrame if an error occurs or no data is found.
    """
    if not isinstance(tickers, (str, list)):
        print("Error: Tickers must be a string or a list of strings.")
        return pd.DataFrame()

    try:
        # If a single ticker is provided, yfinance returns a DataFrame.
        # If multiple tickers are provided, yfinance returns a Panel-like DataFrame (dict of DataFrames).
        # For simplicity, this initial version will focus on handling a single ticker.
        # Support for multiple tickers (e.g., stacking or prefixing columns) can be added later.
        if isinstance(tickers, list) and len(tickers) > 1:
            print("Warning: This version primarily supports single ticker requests for simplicity.")
            print("Fetching data for the first ticker in the list:", tickers[0])
            data = yf.download(tickers[0], start=start_date, end=end_date, interval=interval)
        elif isinstance(tickers, list) and len(tickers) == 1:
            data = yf.download(tickers[0], start=start_date, end=end_date, interval=interval)
        elif isinstance(tickers, str):
            data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        else: # Should not happen due to the initial check, but as a safeguard
            print("Error: Invalid ticker format.")
            return pd.DataFrame()


        if data.empty:
            print(f"No data found for ticker(s): {tickers} between {start_date} and {end_date}.")
            return pd.DataFrame()

        # Handle potential MultiIndex columns for single ticker case
        if isinstance(tickers, (str, list)) and (isinstance(tickers, str) or len(tickers) == 1):
            if isinstance(data.columns, pd.MultiIndex):
                # If single ticker, the top level is often the ticker name. We can drop it.
                # Or, if it's ('Price', 'Ticker'), we might want to handle it differently.
                # For yfinance, typical multi-index for single ticker might be [('Open', 'AAPL'), ('Close', 'AAPL')]
                # Or just [Open, Close, ...] if it's simple.
                # If the test showed [('Close', 'MSFT'), ...], this means the first level is 'Price' category.
                # Let's assume the structure is ('Category', 'Ticker') or just ('Category')
                # The error message showed: MultiIndex([( 'Close', 'MSFT'),...], names=['Price', 'Ticker'])
                # This means yf.download is returning columns like ('Close', 'MSFT')
                # We want to simplify this to just 'Close' if only one ticker was requested.

                # Get the actual ticker name requested (first one if list)
                actual_ticker_name = tickers if isinstance(tickers, str) else tickers[0]

                # Check if the second level of MultiIndex contains this ticker name
                # (it might not, if yfinance changes format again)
                # A simpler approach: if single ticker and MultiIndex, take the first level of column names.
                # This assumes the desired columns (Open, High, Low, Close) are in the first level.
                # Based on error: names=['Price', 'Ticker'], so data.columns.get_level_values(0) is 'Price'
                if len(data.columns.levels) == 2: # e.g. ('Close', 'MSFT')
                    # We want the first part of the tuple as column name
                    data.columns = data.columns.get_level_values(0)
                # else: could be a more complex MultiIndex or yf changed format.

        # Basic data cleaning
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # Ensure the index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        return data

    except Exception as e:
        print(f"Error fetching data for {tickers}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    # Single ticker
    ticker_symbol = 'AAPL'
    start = '2020-01-01'
    end = '2023-01-01'

    print(f"Fetching data for {ticker_symbol} from {start} to {end}")
    stock_data = fetch_financial_data(ticker_symbol, start, end)
    if not stock_data.empty:
        print("Data for", ticker_symbol, ":")
        print(stock_data.head())
        print("\nMissing values after cleaning:")
        print(stock_data.isnull().sum())
    else:
        print(f"Could not retrieve data for {ticker_symbol}")

    print("-" * 50)

    # Example with a list containing a single ticker
    ticker_list_single = ['GOOGL']
    print(f"Fetching data for {ticker_list_single} from {start} to {end}")
    stock_data_list_single = fetch_financial_data(ticker_list_single, start, end)
    if not stock_data_list_single.empty:
        print("Data for", ticker_list_single, ":")
        print(stock_data_list_single.head())
    else:
        print(f"Could not retrieve data for {ticker_list_single}")

    print("-" * 50)

    # Example with multiple tickers (current implementation will warn and pick the first)
    ticker_list_multiple = ['MSFT', 'TSLA']
    print(f"Fetching data for {ticker_list_multiple} from {start} to {end}")
    stock_data_list_multiple = fetch_financial_data(ticker_list_multiple, start, end)
    if not stock_data_list_multiple.empty:
        print("Data for the first ticker in", ticker_list_multiple, ":")
        print(stock_data_list_multiple.head())
    else:
        print(f"Could not retrieve data for the first ticker in {ticker_list_multiple}")

    print("-" * 50)

    # Example of an invalid ticker
    invalid_ticker = 'INVALIDTICKERXYZ'
    print(f"Fetching data for invalid ticker {invalid_ticker} from {start} to {end}")
    invalid_data = fetch_financial_data(invalid_ticker, start, end)
    if not invalid_data.empty:
        print(invalid_data.head())
    else:
        print(f"Data fetching failed for {invalid_ticker} as expected.")

    print("-" * 50)

    # Example of a valid ticker but potentially no data for a very short or future interval
    ticker_short_interval = 'AAPL'
    start_future = '2030-01-01'
    end_future = '2030-01-02'
    print(f"Fetching data for {ticker_short_interval} from {start_future} to {end_future}")
    no_data_for_interval = fetch_financial_data(ticker_short_interval, start_future, end_future)
    if no_data_for_interval.empty:
        print(f"No data found for {ticker_short_interval} in the future range, as expected.")
    else:
        print(no_data_for_interval.head())

    # Example with 1 hour interval for a shorter period
    ticker_hourly = 'NVDA'
    start_hourly = '2023-12-01'
    end_hourly = '2023-12-05' # yfinance might need at least 7 days for hourly for some tickers
    print(f"Fetching hourly data for {ticker_hourly} from {start_hourly} to {end_hourly}")
    # For hourly data, yfinance might have restrictions (e.g., data available only for the last 730 days)
    # Adjust start_hourly and end_hourly if no data is returned.
    # For instance, yf.download often requires the period to be within the last 730 days for 1h interval.
    # Let's try a recent period.
    from datetime import datetime, timedelta
    end_recent_hourly = datetime.now().strftime('%Y-%m-%d')
    start_recent_hourly = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

    hourly_data = fetch_financial_data(ticker_hourly, start_recent_hourly, end_recent_hourly, interval='1h')
    if not hourly_data.empty:
        print(f"Hourly data for {ticker_hourly}:")
        print(hourly_data.head())
    else:
        print(f"Could not retrieve hourly data for {ticker_hourly} for the period {start_recent_hourly} to {end_recent_hourly}.")
        print("Note: yfinance has limitations on historical extent for intraday data (e.g., often last 730 days).")

    # Example with a list of invalid tickers
    invalid_tickers_list = ["INVALID1", "INVALID2"]
    print(f"Fetching data for invalid tickers list {invalid_tickers_list} from {start} to {end}")
    invalid_data_list = fetch_financial_data(invalid_tickers_list, start, end)
    if invalid_data_list.empty:
        print(f"Data fetching failed for {invalid_tickers_list} as expected (processed first ticker).")
    else:
        print(invalid_data_list.head())
