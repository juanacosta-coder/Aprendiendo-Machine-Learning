# Importamos las librerías requeridas

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Ruta del archivo
iowa_file_path = "../input/home-data-for-ml-course/train.csv"

# Cargamos el conjunto de datos
home_data = pd.read_csv(iowa_file_path)

# Creamos la variable objetivo
y = home_data["SalePrice"]

# Seleccionamos las características que utilizará el modelo para realizar las predicciones
features = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

X = home_data[features]


# Dividimos los datos en conjuntos de entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=1
)

# Creamos el modelo de Random Forest
iowa_model = RandomForestRegressor(random_state=1)

# Entrenamos el modelo
iowa_model.fit(train_X, train_y)

# Realizamos las predicciones y calculamos el MAE
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print(
    "MAE sin especificar max_leaf_nodes: {:,.0f}".format(val_mae)
)

# Repetimos el procedimiento limitando la cantidad máxima
# de hojas de cada árbol del Random Forest
iowa_model = RandomForestRegressor(
    max_leaf_nodes=100,
    random_state=1
)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print(
    "MAE especificando max_leaf_nodes=100: {:,.0f}".format(val_mae)
)

# Conclusión:
# En esta prueba, el Random Forest sin especificar max_leaf_nodes
# obtuvo un MAE menor que el modelo limitado a 100 hojas.
