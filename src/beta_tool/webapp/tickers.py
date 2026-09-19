import requests
import zipfile
import io
import bisect
import pandas as pd

def is_valid_tiingo_key(api_key: str) -> bool:
    url = "https://api.tiingo.com/api/test/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}"
    }
    
    if not api_key or not isinstance(api_key, str):
        return False
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # Valid keys return a 200 status code
        if response.status_code == 200:
            # Expected payload: {'message': 'You successfully sent a request'}
            return True
            
        # Invalid keys usually return 401 Unauthorized
        return False
        
    except requests.RequestException:
        # Handle network or connection errors safely
        return False


def get_tickers():
    url = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    response = requests.get(url)
    response.raise_for_status()

    # Extract the CSV file from the ZIP archive
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find the CSV file inside the zip
        csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]

        with z.open(csv_filename) as f:
            df = pd.read_csv(f)

    # Normalize ticker symbols
    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Sorted unique ticker list for fast autocomplete
    ticker_list = tuple(
        sorted(df["ticker"].dropna().unique())
    )


    return df, ticker_list


def find_tickers(tickers, query, limit=10):
    query = query.strip().upper()

    if len(query) < 2:
        return []

    # First ticker >= query
    start = bisect.bisect_left(tickers, query)

    # First ticker after the query's prefix
    end = bisect.bisect_left(
        tickers,
        query + "\uffff"
    )

    return list(tickers[start:end][:limit])

