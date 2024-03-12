import pandas as pd
import requests
from datetime import datetime

def obtener_datos():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=max'
    response = requests.get(url)
    data = pd.DataFrame(response.json())
    return data 

def procesar_datos(df):
    df['Date'] = pd.to_datetime(df[0], unit='ms')
    df.set_index('Date', inplace=True)
    df['date'] = pd.to_datetime(df[0], unit='ms')
    df = df.drop(0, axis=1)
    df = df.rename(columns={1: 'Open', 2: 'High', 3: 'Low', 4: 'Close'})
    return df

# def agregar_caracteristicas(data):
#     data['MMS30'] = data['Close'].rolling(window=30).mean()
#     data['MMS60'] = data['Close'].rolling(window=60).mean()
#     return data.dropna()

btc_data = obtener_datos()
btc_data = procesar_datos(btc_data)