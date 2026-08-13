import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Leer los datos
data = pd.read_csv('../input/train.csv', index_col='Id')
data_full = pd.read_csv('../input/test.csv', index_col='Id')

# Eliminar filas con el objetivo faltante, separar el objetivo de los predictores
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
data.drop(['SalePrice'], axis=1, inplace=True)

# Para simplificar, usamos solo predictores numéricos
X = data.select_dtypes(exclude=['object'])
X_test = data_full.select_dtypes(exclude=['object'])

# Separar el conjunto de validación de los datos de entrenamiento
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Obtener los nombres de las columnas con datos faltantes
cols_con_falta = [col for col in X_train.columns 
                  if X_train[col].isnull().any()]

# Eliminar columnas con datos faltantes en entrenamiento y validación
final_X_train = X_train.drop(cols_con_falta, axis=1)
final_X_valid = X_valid.drop(cols_con_falta, axis=1)

# Definir y entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Obtener predicciones de validación y el MAE
preds_valid = model.predict(final_X_valid)
print(mean_absolute_error(y_valid, preds_valid))

# Preprocesar los datos de prueba
final_X_test = X_test.drop(cols_con_falta, axis=1)
final_X_test = final_X_test.fillna(0)

# Obtener predicciones de prueba
preds_test = model.predict(final_X_test)
