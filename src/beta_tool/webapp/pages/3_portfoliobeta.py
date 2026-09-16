import streamlit as st
import requests
import io
import zipfile
import pandas as pd

@st.cache_data
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

    return df


df = get_tickers()

if "selected_items" not in st.session_state:
    st.session_state.selected_items = []


search_query = st.text_input("Search")

if len(search_query) >= 2:
    filtered_df = df[df['ticker'].str.contains(search_query, case=False,na=False)]
    dropdown_options = filtered_df['ticker'].head(10).tolist()

else:
    dropdown_options = []


options_pool = list(set(dropdown_options + st.session_state.selected_items))

updated_selection = st.multiselect(
    "select ticker",
    options=options_pool,
    default=st.session_state.selected_items

)

if updated_selection != st.session_state.selected_items:
    st.session_state.selected_items = updated_selection
    st.rerun()



