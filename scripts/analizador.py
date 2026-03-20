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
file_path = pathlib.Path("data/database.csv")
data = pd.read_csv(file_path, encoding='utf-8', sep=';')
# mostrar las primeras filas del dataset
print(data.head())
# resumen estadístico
table_des = data.describe() 

fig, ax = plt.subplots()
ax.axis('off')

table = ax.table(
    cellText=table_des.round(2).values,
    colLabels=table_des.columns,
    rowLabels=table_des.index,
    loc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.savefig("data/figuras/resumen_estadistico.png", bbox_inches='tight', dpi=600)
plt.close()

# verificar valores nulos
print(data.isnull().sum())
# distribución de la variable objetivo
# configuración de estilo
sns.set(style="whitegrid")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 14

# Plot
fig, ax = plt.subplots()

sns.barplot(
    x='muestra',
    y='porcentaje_humedad',
    data=data,
    ax=ax
)

ax.set_title('Contenido de humedad')
ax.set_xlabel('Muestras')
ax.set_ylabel('Porcentaje de humedad (%)')

plt.savefig('data/figuras/distribucion_variable_objetivo.png')

# t-test
group1 = data[data["grupo"] == "A"]["porcentaje_humedad"]
group2 = data[data["grupo"] == "B"]["porcentaje_humedad"]

# t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

# Stats
means = data.groupby("grupo")["porcentaje_humedad"].mean()
stds = data.groupby("grupo")["porcentaje_humedad"].std()

# Plot
plt.figure()
bars = plt.bar(means.index, means.values, yerr=stds, capsize=6)

# color
colors = ["#4C72B0", "salmon"] 

for bar, color in zip(bars, colors):
    bar.set_color(color)

# Add significance line
x1, x2 = 0, 1
y = max(means + stds) + 2

plt.plot([x1, x1, x2, x2], [y, y+1, y+1, y], lw=1.5)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 14

# Add asterisk
if p_value < 0.001:
    text = "***"
elif p_value < 0.01:
    text = "**"
elif p_value < 0.05:
    text = "*"
else:
    text = "ns"

plt.text((x1+x2)/2, y+1.2, text, ha='center', fontsize=16)

plt.xlabel("Grupo", fontsize=16)
plt.ylabel("perdida de humedad (%)", fontsize=16)

# y axis
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('data/figuras/boxplot_letras.png', dpi=600)

