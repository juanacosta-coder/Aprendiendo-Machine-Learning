import pandas as pd

# Ruta al conjunto de datos de viviendas de Iowa
iowa_file_path = "../input/home-data-for-ml-course/train.csv"

# Cargar el conjunto de datos en un DataFrame
home_data = pd.read_csv(iowa_file_path)

# Calcular el tamaño promedio de los terrenos
avg_lot_size = round(home_data["LotArea"].mean())

# Calcular la antigüedad de la vivienda más reciente
newest_home_age = 2026 - home_data["YearBuilt"].max()

# Mostrar los resultados obtenidos
print(f"Tamaño promedio del terreno: {avg_lot_size}")
print(f"Antigüedad de la vivienda más nueva: {newest_home_age} años")
