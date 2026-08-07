import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Ruta del conjunto de datos
iowa_file_path = "../input/home-data-for-ml-course/train.csv"

# Cargar el conjunto de datos
home_data = pd.read_csv(iowa_file_path)

# Variable objetivo (precio de venta)
y = home_data["SalePrice"]

# Características (features) utilizadas por el modelo
feature_columns = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

# Crear el conjunto de entrada
X = home_data[feature_columns]

# Dividir los datos en entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=0
)

# Función para calcular el MAE según la cantidad máxima de hojas
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    iowa_model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=0
    )

    # Entrenar el modelo
    iowa_model.fit(train_X, train_y)

    # Realizar predicciones
    predictions = iowa_model.predict(val_X)

    # Calcular el error absoluto medio
    mae = mean_absolute_error(val_y, predictions)

    return mae


# Comparar distintos valores de max_leaf_nodes
for max_leaf_nodes in [5, 50, 100, 250, 500]:
    my_mae = get_mae(
        max_leaf_nodes,
        train_X,
        val_X,
        train_y,
        val_y
    )

    print(f"Max Leaf Nodes: {max_leaf_nodes:<5}\t\tMAE: {my_mae:.0f}")

# Conclusión:
# El menor MAE se obtuvo con max_leaf_nodes = 50.
# Este fue el modelo que mejor generalizó sobre los datos de validación.
