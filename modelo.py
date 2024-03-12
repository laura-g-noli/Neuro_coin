from data import btc_data
import pandas as pd
import numpy as np
from keras.models import save_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def dividir_datos(data):
    set_entrenamiento = data[:"2022"].iloc[:,1:2]
    set_validacion = data[:"2024"].iloc[:,1:2]

    return set_entrenamiento, set_validacion

def entrenar_modelo(set_entrenamiento):
    sc = MinMaxScaler(feature_range=(0,1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)

    for i in range(time_step, m):
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
        Y_train.append(set_entrenamiento_escalado[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    modelo = Sequential()
    modelo.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
    modelo.add(Dense(units=1))
    modelo.compile(optimizer="rmsprop", loss="mse")
    modelo.fit(X_train, Y_train, epochs=50, batch_size=32)
    
    return modelo, sc

def predecir(modelo, set_validacion, sc):
    x_test = sc.transform(set_validacion)
    X_test = []

    for i in range(60, len(x_test)):
        X_test.append(x_test[i-60:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)

    return prediccion


set_entrenamiento, set_validacion = dividir_datos(btc_data)
modelo, sc = entrenar_modelo(set_entrenamiento)
prediccion = predecir(modelo, set_validacion, sc)

save_model(modelo, "modelo_entrenado.keras")

# from sklearn.metrics import mean_squared_error

# Calcular el error cuadrático medio (MSE)
# mse = mean_squared_error(set_validacion.values[60:], prediccion)
# print(f'Error cuadrático medio (MSE): {mse}')