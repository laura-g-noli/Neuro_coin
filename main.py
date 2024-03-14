from fastapi import FastAPI
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


modelo_entrenado = load_model("modelo_entrenado.keras")

btc_data = pd.read_csv("btc_data.csv")

app = FastAPI()

@app.get("/")
def index():
    return "Bienvenidos a NeuroCoin. Para obtener el dato de la predicción diríjase a la ruta => /prediccion"

def obtener_datos_mas_recientes():
    return btc_data.iloc[-60:, 3:4].values 

def hacer_prediccion():
    datos_entrada = obtener_datos_mas_recientes()

    sc = MinMaxScaler(feature_range=(0, 1))
    datos_entrada_escalados = sc.fit_transform(datos_entrada)

    X_prediccion = np.reshape(datos_entrada_escalados, (1, datos_entrada_escalados.shape[0], 1))

    prediccion = modelo_entrenado.predict(X_prediccion)

    prediccion_desnormalizada = sc.inverse_transform(prediccion)

    precio_predicho = float(prediccion_desnormalizada[0][0])

    return precio_predicho

@app.get("/prediccion")
async def obtener_prediccion():
    precio_prediccion = hacer_prediccion()
    
    return {"Predicción para el día siguiente": precio_prediccion}