import pandas as pd
import requests

def obtener_datos():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=max'
    response = requests.get(url)
    data = pd.DataFrame(response.json())
    return data 

def procesar_datos(df):
    df['Date'] = pd.to_datetime(df[0], unit='ms')
    df.set_index('Date', inplace=True)
    df = df.drop(0, axis=1)
    df = df.rename(columns={1: 'Open', 2: 'High', 3: 'Low', 4: 'Close'})
    return df

btc_data = obtener_datos()
btc_data = procesar_datos(btc_data)

btc_data.to_csv('btc_data.csv')