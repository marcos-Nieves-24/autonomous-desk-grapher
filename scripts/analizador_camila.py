# librerias
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib 
import scikit_posthocs as sp

# cargar datos
file_path = pathlib.Path("data_camila/database_camila.csv")
data = pd.read_csv(file_path, encoding='utf-8', sep=';')
# mostrar las primeras filas del dataset
print(data.head())
# resumen estadístico
print(data.describe())
# verificar valores nulos
print(data.isnull().sum())
# distribución de la variable objetivo
sns.barplot(x='muestra', y='porcentaje_humedad', data=data)

plt.title('Contenido de humedad')
plt.xlabel('Muestra')
plt.ylabel('Porcentaje de Humedad')

plt.savefig('data_camila/figuras/distribucion_variable_objetivo.png')

# análisis de varianza (ANOVA)
groups = [group["porcentaje_humedad"].values for name, group in data.groupby("grupo")]
f_statistic, p_value = stats.f_oneway(*groups)
print(f"ANOVA F-statistic: {f_statistic}, p-value: {p_value}")

# prueba de Tukey para comparaciones múltiples
if p_value < 0.05:
    print("\nRunning Tukey test...\n")
    
    tukey = pairwise_tukeyhsd(
        endog=data["porcentaje_humedad"],
        groups=data["grupo"],
        alpha=0.05
    )
    
    print(tukey)
    tukey.plot_simultaneous()
    plt.title('Tukey HSD Test')
    plt.xlabel('Mean Difference')
    plt.savefig('data_camila/figuras/tukey_test.png')
else:
    print("No significant differences found among groups.")

# gráfico de cajas para visualizar las diferencias entre grupos

stats = data.groupby("grupo")["porcentaje_humedad"].agg(["mean", "std"])

# estadísticas para asignar letras
groups = stats.index
means = stats["mean"]
stds = stats["std"]

# Tukey
tukey = pairwise_tukeyhsd(
    data["porcentaje_humedad"],
    data["grupo"],
    alpha=0.05
)

print(tukey)

groups = tukey.groupsunique
means = data.groupby("grupo")["porcentaje_humedad"].mean().loc[groups]
stds = data.groupby("grupo")["porcentaje_humedad"].std().loc[groups]

reject = tukey.reject

# Default letters
letters = ['a'] * len(groups)

# Simple logic for 3 groups
if len(groups) == 3:
    if all(reject):
        letters = ['a', 'b', 'c']
    elif not any(reject):
        letters = ['a', 'a', 'a']
    else:
        letters = ['a', 'a', 'b']
# Plot
plt.figure()

bars = plt.bar(
    groups,
    means,
    yerr=stds,
    capsize=6
)

# Colors
colors = ["#1f77b4", "orange", "#bcbd22"]
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Add Tukey letters above bars
for i, (mean, std, letter) in enumerate(zip(means, stds, letters)):
    plt.text(i, mean + std + 1, letter, ha='center', fontsize=14, fontweight='bold')

plt.title('Humedad relativa')
plt.xlabel('Grupo')
plt.ylabel('Perdida de humedad (%)')

plt.tight_layout()
plt.savefig('data_camila/figuras/boxplot_letras.png', dpi=600)

