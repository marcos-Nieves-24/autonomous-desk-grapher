# librerias
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib 

# cargar datos
file_path = pathlib.Path("data/database.csv")
data = pd.read_csv(file_path, encoding='utf-8', sep=';')
# mostrar las primeras filas del dataset
print(data.head())
# resumen estadístico
print(data.describe())
# verificar valores nulos
print(data.isnull().sum())
# distribución de la variable objetivo
sns.barplot(x='porcentaje_humedad', y='muestra', data=data)
plt.title('Distribución de la Variable Objetivo: Porcentaje de Humedad')
plt.xlabel('Porcentaje de Humedad')
plt.ylabel('Frecuencia')

plt.savefig('data/figuras/distribucion_variable_objetivo.png')
