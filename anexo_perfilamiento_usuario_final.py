# -*- coding: utf-8 -*-
"""Perfilamiento_integrador.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dtBd_R3mxR1o-RBSHJli6wnGRKbZEFKw
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


df = pd.read_csv('base_datos_clientes.csv')

# Seleccionar solo las columnas de puntajes (aquellas que terminan en '_puntaje'),
# excluyendo 'edad_puntaje' y 'puntaje_total'
df_filtered = df[[col for col in df.columns if col.endswith('_puntaje') and col != 'puntaje_total']].copy()

#  Normalizar los datos (opcional, pero recomendado para K-means)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtered)

# Determinar el número óptimo de clusters usando el método del codo
inertia = []
for k in range(1, 11):  # Probar entre 1 y 10 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.show()

#  Calcular el Índice de Silueta para elegir el mejor K
silhouette_scores = []
for k in range(2, 11):  # El valor de K debe ser al menos 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

# Graficar el Índice de Silueta
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Índice de Silueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Índice de Silueta')
plt.show()

# Elegir el número de clusters basado en el codo o en el índice de silueta
# Aquí asumimos que el número de clusters es 5
kmeans = KMeans(n_clusters=5, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(df_scaled)  # Asignar la columna 'Cluster'

# Visualización de los clusters usando PCA (reducción de dimensionalidad)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_scaled)

# Crear un DataFrame con los componentes principales y los clusters
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df_filtered['Cluster']

# Graficar los clusters en un plano 2D
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis')
plt.title('Clusters identificados por K-means')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()

# Asignar un perfil a cada cluster basado en las características promedio
profile_mapping = {
    0: 'Conservador',
    1: 'Cauteloso',
    2: 'Equilibrado',
    3: 'Audaz',
    4: 'Visionario'
}

# Usar .loc[] para evitar el SettingWithCopyWarning y asignar perfiles de riesgo de manera segura
df_filtered.loc[:, 'Perfil de Riesgo'] = df_filtered['Cluster'].map(profile_mapping)

# Mostrar los resultados finales
print(df_filtered.head())

# Mostrar los clientes con los clusters asignados y sus perfiles de riesgo
clientes_con_clusters = df_filtered[['Cluster', 'Perfil de Riesgo']].drop_duplicates()

# Mostrar los resultados
clientes_con_clusters.head()

# Seleccionar solo las columnas numéricas (aquellas que contienen puntajes)
df_numeric = df_filtered.select_dtypes(include=['float64', 'int64']).copy()  # Seleccionar solo columnas numéricas

# Asegurarse de que la columna 'Cluster' está en el DataFrame
df_numeric['Cluster'] = df_filtered['Cluster']

# Calcular las características promedio de cada cluster
cluster_profiles = df_numeric.groupby('Cluster').mean()

# Mostrar los resultados promedios por cluster
print(cluster_profiles)

#  Asignar el perfil de riesgo basado en los puntajes promedio de cada cluster
def asignar_perfil(puntaje_promedio):
    if 15 <= puntaje_promedio <= 25:
        return 'Conservador'
    elif 26 <= puntaje_promedio <= 35:
        return 'Cauteloso'
    elif 36 <= puntaje_promedio <= 40:
        return 'Equilibrado'
    elif 51 <= puntaje_promedio <= 65:
        return 'Audaz'
    elif 66 <= puntaje_promedio <= 75:
        return 'Visionario'
    else:
        return 'Fuera de rango'

# Asignar el perfil de riesgo a cada cluster según el puntaje promedio de 'estrategia_inversion_puntaje'
cluster_profiles['Perfil de Riesgo'] = cluster_profiles['estrategia_inversion_puntaje'].apply(asignar_perfil)

# Mostrar los resultados con los perfiles asignados
print(cluster_profiles[['Perfil de Riesgo']])

# Verificar las columnas del DataFrame
print(df_filtered.columns)

# Crear la columna 'puntaje_total' sumando todas las columnas de puntajes
# Asegúrate de seleccionar las columnas correctas con puntajes (que terminan en '_puntaje')

puntaje_columnas = [col for col in df_filtered.columns if col.endswith('_puntaje')]  # Filtramos las columnas de puntajes
df_filtered['puntaje_total'] = df_filtered[puntaje_columnas].sum(axis=1)  # Sumar las columnas de puntajes

# Verificar si la columna 'puntaje_total' se ha creado correctamente
print(df_filtered[['puntaje_total']].head())

# Asignar el perfil de riesgo basado en el puntaje total
df_filtered['Perfil de Riesgo'] = df_filtered['puntaje_total'].apply(asignar_perfil)

# Mostrar los resultados finales con los perfiles de riesgo asignados
print(df_filtered[['Cluster', 'puntaje_total', 'Perfil de Riesgo']].head())

# Verificar el rango de los puntajes totales
min_puntaje = df_filtered['puntaje_total'].min()
max_puntaje = df_filtered['puntaje_total'].max()

print(f"Rango de puntajes totales: {min_puntaje} a {max_puntaje}")

# Ajustar los rangos para que se adapten al nuevo rango de puntajes totales (25-70)
def asignar_perfil(puntaje_total):
    if 25 <= puntaje_total <= 35:
        return 'Conservador'
    elif 36 <= puntaje_total <= 45:
        return 'Cauteloso'
    elif 46 <= puntaje_total <= 55:
        return 'Equilibrado'
    elif 56 <= puntaje_total <= 65:
        return 'Audaz'
    elif 66 <= puntaje_total <= 70:
        return 'Visionario'
    else:
        return 'Fuera de rango'

# Asignar el perfil de riesgo a cada cliente según el puntaje total
df_filtered['Perfil de Riesgo'] = df_filtered['puntaje_total'].apply(asignar_perfil)

# Mostrar los resultados finales con los perfiles de riesgo asignados
print(df_filtered[['Cluster', 'puntaje_total', 'Perfil de Riesgo']].head())