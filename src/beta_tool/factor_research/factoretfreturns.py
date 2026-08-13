import requests
import pandas as pd
import io

FMP_API_KEY = "TJHPbSl9QgqQsLqUxY3bw9i54EjMo1KB"
T_API_KEY = "4f5afca3fdcfdf32e5e5224c5b4a10d221f69c76"

def get_stock_data(symbol, start_date=None, end_date=None):
    # url = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"

    # params = {
    #     "symbol": symbol,
    #     "apikey": FMP_API_KEY,
    # }

    # if start_date:
    #     params["from"] = start_date

    # if end_date:
    #     params["to"] = end_date

    # response = requests.get(url, params=params)

    # # Raise an error if the request failed
    # response.raise_for_status()

    # # Convert JSON response into Python objects
    # data = response.json()

    # # Convert list of dictionaries into a DataFrame
    # df = pd.DataFrame(data)

    # # Convert date strings into actual pandas dates
    # df["date"] = pd.to_datetime(df["date"])

    # # Sort oldest -> newest
    # df = df.sort_values("date").reset_index(drop=True)

    # return df

    #----------------------------------------------------------
    #       Tiingo API
    #
    # 80,000+ tickers (US Equities, ETFs, Mutual Funds, and Chinese A-Shares)
    #
    # adjusted close price that considers dividends and splits
    #
    #----------------------------------------------------------

    # Meta Data
    #https://api.tiingo.com/tiingo/daily/<ticker>

    # Latest Price
    #https://api.tiingo.com/tiingo/daily/<ticker>/prices

    # Historical Prices
    #https://api.tiingo.com/tiingo/daily/<ticker>/prices?startDate=2012-1-1&endDate=2016-1-1 

    ticker = symbol.lower()
    start = start_date
    end = end_date

    interval_set = {"daily","weekly","monthly","annually"}
    #the data will expand start and end to fit your entire interval windows; for example if you set start date on a wednesday, but interval is weekly, the start date will be rolled back to monday. end date will roll forward likewise
    interval = "daily"

    headers = {
        'Content-Type': 'application/json',
        'Authorization' : f'Token {T_API_KEY}'
        }
    
    params = {
        "startDate": start,
        "endDate": end,
        "resampleFreq": interval,
        "sort": "date",
        "format": "csv",
        "columns": ["date","adjClose"]
        }

    requestResponse = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                                    headers=headers, params=params)
    df = pd.read_csv(io.BytesIO(requestResponse.content), encoding="utf-8")

    return df

df = get_stock_data("iwm","2026-07-01")
print(df.tail())
