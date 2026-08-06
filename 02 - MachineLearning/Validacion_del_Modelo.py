# Importar las librerías necesarias
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Ruta del conjunto de datos
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Cargar el conjunto de datos
home_data = pd.read_csv(iowa_file_path)

# Seleccionar la variable objetivo (precio de venta)
y = home_data["SalePrice"]

# Seleccionar las características (features) que utilizará el modelo
feature_columns = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

# Crear el conjunto de datos de entrada (features)
X = home_data[feature_columns]

# Dividir los datos en entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=0
)

# Crear el modelo de Árbol de Decisión
iowa_model = DecisionTreeRegressor()

# Entrenar el modelo con los datos de entrenamiento
iowa_model.fit(train_X, train_y)

# Realizar predicciones sobre los datos de validación
val_predictions = iowa_model.predict(val_X)

# Calcular el Error Absoluto Medio (MAE)
mae = mean_absolute_error(val_y, val_predictions)

# Mostrar el resultado
print(f"MAE: {mae:.2f}")
