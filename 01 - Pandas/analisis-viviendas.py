import pandas as pd

# Ruta al conjunto de datos
iowa_file_path = "../input/home-data-for-ml-course/train.csv"

# Leer el archivo CSV
home_data = pd.read_csv(iowa_file_path)

# Calcular el tamaño promedio del terreno
avg_lot_size = round(home_data["LotArea"].mean())

# Calcular la antigüedad de la vivienda más nueva
newest_home_age = 2026 - home_data["YearBuilt"].max()

# Mostrar los resultados
print(f"Tamaño promedio del terreno: {avg_lot_size}")
print(f"Antigüedad de la vivienda más nueva: {newest_home_age} años")
