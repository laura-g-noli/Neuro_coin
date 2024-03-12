from fastapi import FastAPI
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# Cargar el modelo entrenado
modelo_entrenado = load_model("modelo_entrenado.keras")

# Cargar los datos
btc_data = pd.read_csv("btc_data.csv")

# Crear la aplicación FastAPI
app = FastAPI()

@app.get("/")
def index():
    return "Mensaje de prueba"

# Función para obtener los datos más recientes disponibles
def obtener_datos_mas_recientes():
    return btc_data.iloc[-60:, 3:4].values  # Obtener los últimos 60 precios de cierre

# Función para hacer la predicción
def hacer_prediccion():
    datos_entrada = obtener_datos_mas_recientes()

    # Normalizar los datos
    sc = MinMaxScaler(feature_range=(0, 1))
    datos_entrada_escalados = sc.fit_transform(datos_entrada)

    # Preparar los datos para la predicción
    X_prediccion = np.reshape(datos_entrada_escalados, (1, datos_entrada_escalados.shape[0], 1))

    # Hacer la predicción
    prediccion = modelo_entrenado.predict(X_prediccion)

    # Desnormalizar la predicción
    prediccion_desnormalizada = sc.inverse_transform(prediccion)

    # Convertir la predicción a un tipo de dato float
    precio_predicho = float(prediccion_desnormalizada[0][0])

    return precio_predicho

# Endpoint de predicción
@app.get("/prediccion")
async def obtener_prediccion():
    # Hacer la predicción
    precio_predicho = hacer_prediccion()
    
    # Devolver la predicción como parte de la respuesta
    return {"precio_predicho": precio_predicho}