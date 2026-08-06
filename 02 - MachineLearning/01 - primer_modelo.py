import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Ruta del conjunto de datos
iowa_file_path = "../input/home-data-for-ml-course/train.csv"

# Cargar el conjunto de datos
home_data = pd.read_csv(iowa_file_path)

# Seleccionar la variable objetivo (precio de venta)
y = home_data["SalePrice"]

# Seleccionar las características que utilizará el modelo
feature_names = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

# Crear el conjunto de datos de entrada (features)
X = home_data[feature_names]

# Crear el modelo de Árbol de Decisión
iowa_model = DecisionTreeRegressor(random_state=1)

# Entrenar el modelo con los datos
iowa_model.fit(X, y)

# Realizar predicciones
predictions = iowa_model.predict(X)

# Comparar los primeros valores reales con las primeras predicciones
print("Primeros precios reales:")
print(y.head())

print("\nPrimeras predicciones:")
print(predictions[:5])
