import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import openai
from dotenv import load_dotenv
from openai.error import RateLimitError
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import tensorflow
from tensorflow import keras
from keras import layers, models, callbacks



# Ruta local a la imagen
img_url = "/Users/israelgarciaruiz/Documents/GitHub/Proyecto-MBD/Logo.jpeg"

# img_url = "https://github.com/G-R-ISRAEL/Proyecto-MBD/blob/main/Logo.jpeg"


    
# Definir las columnas
col1, col2, col3 = st.columns(3)
    
with col1:
    st.image(str(img_url), width=100)  #  , caption="Equipo 5"
with col2:
    st.title("EasyShare")
with col3:
    # Diccionario para mapear las categorías numéricas a los nombres llamativos
    category_names = {
        0: 'Audaz',
        1: 'Confiado',
        2: 'Moderado',
        3: 'Cauteloso',
        4: 'Conservador'
    }

    # Filtro de categoría
    categoria_seleccionada = st.selectbox('Selecciona la categoría', list(category_names.keys()), format_func=lambda x: category_names[x])
   


st.divider()

















@st.cache_data
def get_sp500_tickers():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(sp500_url, header=0)
    df = table[0]
    return df['Symbol'].tolist()  # Retorna una lista de símbolos

# Función para obtener los precios históricos de las acciones
@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Función para descargar los datos de Yahoo Finance
@st.cache_data
def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    return data

@st.cache_data
def get_fundamentals(tickers):
    fundamentals = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            pb = info.get('priceToBook', np.nan)
            pe = info.get('trailingPE', np.nan)
            fundamentals[ticker] = {'P/B': pb, 'P/E': pe}
        except Exception as e:
            fundamentals[ticker] = {'P/B': np.nan, 'P/E': np.nan}
    return pd.DataFrame(fundamentals).T

def preprocess_data(data, sp500_tickers):
    price_data = pd.DataFrame()
    for ticker in sp500_tickers:
        try:
            price_data[ticker] = data[ticker]['Adj Close']
        except KeyError:
            print(f'No data for {ticker}, possibly delisted.')
    price_data = price_data.dropna(axis=1)

    if price_data.empty:
        raise ValueError("No hay datos disponibles para los tickers seleccionados.")
    
    returns_data = price_data.pct_change().dropna()
    volatility_data = returns_data.std()
    total_return_data = (price_data.iloc[-1] / price_data.iloc[0] - 1)

    fundamentals = get_fundamentals(sp500_tickers)

    final_data = pd.DataFrame({
        'Rendimiento Total': total_return_data,
        'Volatilidad': volatility_data,
        'P/B': fundamentals['P/B'],
        'P/E': fundamentals['P/E']
    })

    final_data['P/E'] = pd.to_numeric(final_data['P/E'], errors='coerce')
    final_data['P/B'] = pd.to_numeric(final_data['P/B'], errors='coerce')
    final_data = final_data.dropna(axis=0)

    final_data = final_data[(final_data['P/E'] < 1000) & (final_data['P/E'] > -1000)]
    final_data = final_data[(final_data['P/B'] < 1000) & (final_data['P/B'] > -1000)]
    final_data = final_data[(final_data['Volatilidad'] < 100) & (final_data['Volatilidad'] > 0)]
    final_data = final_data[(final_data['Rendimiento Total'] < 10) & (final_data['Rendimiento Total'] > -1)]
    
    return final_data

def apply_kmeans_15(final_data):
    X = final_data[['Volatilidad', 'Rendimiento Total', 'P/B', 'P/E']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=15, random_state=42)
    final_data['Categoria_15'] = kmeans.fit_predict(X_scaled)
    return final_data

def group_into_5_categories(final_data):
    category_counts = final_data['Categoria_15'].value_counts().sort_values()
    total_actions = final_data.shape[0]
    actions_per_group = total_actions // 5
    final_data['Categoria_5'] = -1
    current_group = 0
    current_count = 0

    for category in category_counts.index:
        category_data = final_data[final_data['Categoria_15'] == category]
        for i, row in category_data.iterrows():
            final_data.at[i, 'Categoria_5'] = current_group
            current_count += 1
            if current_count >= actions_per_group:
                current_group += 1
                current_count = 0
    return final_data

def filter_and_display_category(final_data, categoria_seleccionada, category_names):
    acciones_categoria = final_data[final_data['Categoria_5'] == categoria_seleccionada]
    acciones_categorias = acciones_categoria.drop(columns=['Categoria_5','Categoria_15','Categoria_Nombre'])
    st.write(f"**Acciones de la Categoría {category_names[categoria_seleccionada]}:**")
    st.dataframe(acciones_categorias)
    #st.dataframe(acciones_categoria)

def filter_and_get_category(final_data, categoria_seleccionada):
    acciones_categoria = final_data[final_data['Categoria_5'] == categoria_seleccionada]
    return acciones_categoria.index.tolist()  # Retornar los símbolos de las acciones en esta categoría


def plot_clusters(final_data, category_names):
    # Crear una columna con los nombres llamativos
    final_data['Categoria_Nombre'] = final_data['Categoria_5'].map(category_names)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Volatilidad', y='Rendimiento Total', 
                    hue='Categoria_Nombre', data=final_data, 
                    palette="Set1", s=100, alpha=0.8)
    plt.title('Distribución de Acciones del S&P 500 según Volatilidad y Rendimiento (5 Categorías)', fontsize=16)
    plt.xlabel('Volatilidad (Desviación Estándar)', fontsize=12)
    plt.ylabel('Rendimiento Total', fontsize=12)
    plt.legend(title='Categorías', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)

# Descargar y preprocesar datos antes de evaluar los botones
sp500_tickers = get_sp500_tickers()
start = '2010-01-01'
end = '2024-11-20'

data = download_data(sp500_tickers, start, end)
try:
    final_data = preprocess_data(data, sp500_tickers)
    final_data = apply_kmeans_15(final_data)
    final_data = group_into_5_categories(final_data)
    category_names = {
            0: 'Audaz',
            1: 'Confiado',
            2: 'Moderado',
            3: 'Cauteloso',
            4: 'Conservador'
        }
except ValueError as e:
    st.error(str(e))
    st.stop()

# Definir las columnas para los botones
col1, col2, col3 = st.columns(3)

# Guardar el estado de los botones
if 'show_technical' not in st.session_state:
    st.session_state['show_technical'] = False
if 'show_general' not in st.session_state:
    st.session_state['show_general'] = False
if 'show_forecast' not in st.session_state:
    st.session_state['show_forecast'] = False
if 'show_fundamental' not in st.session_state:
    st.session_state['show_fundamental'] = False
if 'show_profile' not in st.session_state:
    st.session_state['show_profile'] = False
if 'show_portafolio' not in st.session_state:
    st.session_state['show_portafolio'] = False

# Botones en la misma fila
if col1.button("Perfil de cliente", use_container_width=True):
    st.session_state['show_profile'] = True
    st.session_state['show_general'] = False
    st.session_state['show_fundamental'] = False
    st.session_state['show_technical'] = False
    st.session_state['show_forecast'] = False
    st.session_state['show_portafolio'] = False
        
if col2.button("Generales", use_container_width=True):
    st.session_state['show_profile'] = False
    st.session_state['show_general'] = True
    st.session_state['show_fundamental'] = False
    st.session_state['show_technical'] = False
    st.session_state['show_forecast'] = False
    st.session_state['show_portafolio'] = False
        
if col3.button("Fundamental", use_container_width=True):
    st.session_state['show_profile'] = False
    st.session_state['show_general'] = False
    st.session_state['show_fundamental'] = True
    st.session_state['show_technical'] = False
    st.session_state['show_forecast'] = False
    st.session_state['show_portafolio'] = False
    
if col1.button("Técnico", use_container_width=True):
    st.session_state['show_profile'] = False
    st.session_state['show_general'] = False
    st.session_state['show_fundamental'] = False
    st.session_state['show_technical'] = True
    st.session_state['show_forecast'] = False
    st.session_state['show_portafolio'] = False
    
if col2.button("Forecast", use_container_width=True):
    st.session_state['show_profile'] = False
    st.session_state['show_general'] = False
    st.session_state['show_fundamental'] = False
    st.session_state['show_technical'] = False
    st.session_state['show_forecast'] = True
    st.session_state['show_portafolio'] = False
        
if col3.button("Portafolio", use_container_width=True):
    st.session_state['show_profile'] = False
    st.session_state['show_general'] = False
    st.session_state['show_fundamental'] = False
    st.session_state['show_technical'] = False
    st.session_state['show_forecast'] = False
    st.session_state['show_portafolio'] = True
  
st.divider()





























# Mostrar gráfico técnico si se hace clic en "Perfil"
if st.session_state['show_profile']:

        # Definir las columnas para los botones
        col1, col2 = st.columns(2)

        # Guardar el estado de los botones
        if 'Clasificación de clientes' not in st.session_state:
            st.session_state['Clasificación de clientes'] = False
        if 'Clasificación de acciones' not in st.session_state:
            st.session_state['Clasificación de acciones'] = False

        # Botones en la misma fila
        if col1.button("Clasificación de clientes", use_container_width=True):
            st.session_state['Clasificación de clientes'] = True
            st.session_state['Clasificación de acciones'] = False
                
        if col2.button("Clasificación de acciones", use_container_width=True):
            st.session_state['Clasificación de clientes'] = False
            st.session_state['Clasificación de acciones'] = True
            
        st.divider()
        
        # Mostrar gráfico técnico si se hace clic en "Income Statement"
        if st.session_state['Clasificación de clientes']:

            import streamlit as st

            # Diccionario con los puntos para cada pregunta
            PUNTOS_2 = {
                "Básico": 6,
                "Medio Superior": 5,
                "Superior": 4,
                "Posgrado": 3,
                "Doctorado o más": 3
            }

            PUNTOS_3 = {
                "Conservadora": 1,
                "Moderada": 2,
                "Crecimiento": 3,
                "Agresiva": 5,
                "Muy Agresiva": 7,
                "No he invertido": 0
            }

            PUNTOS_4 = {
                "Sí": 0,
                "No": 3
            }

            PUNTOS_5 = {
                "Deuda bancaria": 1,
                "Deuda gubernamental": 1,
                "Divisas y valores en monedas diferentes al peso": 2,
                "Fondos de inversión en deuda": 2,
                "Fondos de inversión renta variable": 3,
                "Acciones y CPO’s": 4,
                "Valores corporativos": 3,
                "ETF’s y Tracks": 4,
                "Derivados": 5,
                "Valores extranjeros": 4,
                "Productos estructurados": 5,
                "Fibras": 3,
                "No he invertido": 0
            }

            PUNTOS_6 = {
                "Menos del 10%": 1,
                "Entre 10% y 20%": 2,
                "Entre 20% y 40%": 3,
                "Entre 40% y 60%": 4,
                "Más del 60%": 6
            }

            PUNTOS_7 = {
                "Es para incrementar mi patrimonio": 3,
                "Es para mi jubilación": 5,
                "Es para incrementar mi capital en una inversión de largo plazo": 3,
                "No pienso utilizar la mayor parte en el mediano plazo": 2,
                "Pienso utilizar la mayor parte en el corto plazo": 1,
                "Dependo de un flujo para mis gastos": 2
            }

            PUNTOS_8 = {
                "No estoy dispuesto": 1,
                "Sí estoy dispuesto": 5
            }

            PUNTOS_9 = {
                "Corto Plazo": 1,
                "Mediano Plazo": 3,
                "Largo Plazo": 5
            }

            PUNTOS_10 = {
                "Menor a 1 año": 1,
                "De 1 a 3 años": 2,
                "De 3 a 5 años": 3,
                "Más de 5 años": 5
            }

            PUNTOS_11 = {
                "100% en Deuda": 1,
                "80% en Deuda y 20% en Renta Variable": 2,
                "50% en Deuda y 50% en Renta Variable": 3,
                "20% en Deuda y 80% en Renta Variable": 5,
                "100% en Renta Variable": 7
            }

            PUNTOS_12 = {
                "No deseo invertir en Renta Variable": 1,
                "De 1% a 20%": 2,
                "Entre 21% y 50%": 3,
                "Más del 50%": 5
            }

            PUNTOS_13 = {
                "De 1% a 10%": 2,
                "De 11% a 20%": 3,
                "Entre 21% a 30%": 4,
                "Más de 30%": 5,
                "No estoy dispuesto": 1
            }

            PUNTOS_14 = {
                "No deseo invertir en valores corporativos": 1,
                "De 1% a 30%": 2,
                "De 31% a 60%": 3,
                "Más de 60%": 5
            }

            PUNTOS_15 = {
                "No estoy dispuesto": 1,
                "Hasta 10%": 2,
                "De 11% a 20%": 3,
                "De 21% a 30%": 4,
                "Más del 30%": 5
            }

            PUNTOS_16 = {
                "NO hay restricciones": 2,
                "NO deseo operar en mercado de capitales": 1,
                "NO deseo operar en mercado de derivados": 1
            }

            # Función para calcular el puntaje total
            def calcular_puntaje(respuestas):
                puntaje = 0
                puntaje += PUNTOS_2[respuestas['grado_estudio']]  # Grado máximo de estudio
                puntaje += PUNTOS_3[respuestas['estrategia_inversion']]  # Estrategia de inversión
                puntaje += PUNTOS_4[respuestas['conocimiento_servicios']]  # Conocimiento de los servicios de inversión
                puntaje += sum([PUNTOS_5[instrumento] for instrumento in respuestas['instrumentos']])  # Instrumentos invertidos
                puntaje += PUNTOS_6[respuestas['patrimonio_invertir']]  # Porcentaje de patrimonio que desea invertir
                puntaje += PUNTOS_7[respuestas['destino_inversion']]  # Destino de la inversión
                puntaje += PUNTOS_8[respuestas['disposicion_derivados']]  # Disposición a invertir en derivados
                puntaje += PUNTOS_9[respuestas['objetivo_inversion']]  # Objetivo de la inversión
                puntaje += PUNTOS_10[respuestas['plazo_disponible']]  # Plazo disponible para la inversión
                puntaje += PUNTOS_11[respuestas['portafolio']]  # Selección del portafolio
                puntaje += PUNTOS_12[respuestas['renta_variable']]  # Porcentaje en Renta Variable
                puntaje += PUNTOS_13[respuestas['accion_directa']]  # Disposición a invertir en una sola acción
                puntaje += PUNTOS_14[respuestas['valores_corporativos']]  # Porcentaje en valores corporativos
                puntaje += PUNTOS_15[respuestas['valores_corporativos_emisora']]  # Disposición a invertir en una sola emisora
                puntaje += PUNTOS_16[respuestas['restricciones']]  # Restricciones adicionales

                return puntaje

            # Función para determinar el perfil según el puntaje
            def determinar_perfil(puntaje):
                if puntaje >= 60:
                    return "Perfil Muy Alto (Audaz)"
                elif puntaje >= 45:
                    return "Perfil Alto (Confiado)"
                elif puntaje >= 30:
                    return "Perfil Medio (Moderado)"
                elif puntaje >= 20:
                    return "Perfil Bajo (Cauteloso)"
                else:
                    return "Perfil Muy Bajo (Conservador)"

            # Formulario para capturar respuestas
            st.title('Perfil de Riesgo')

            respuestas = {}

            # Preguntas del cuestionario
            respuestas['edad'] = st.selectbox("¿Cuál es tu rango de edad? (Esta pregunta no afecta el puntaje)", ['18 a 29 años', '30 a 39 años', '40 a 49 años', '50 a 59 años', '60 años o más'])
            respuestas['grado_estudio'] = st.selectbox("¿Cuál es tu grado máximo de estudio?", ['Básico', 'Medio Superior', 'Superior', 'Posgrado', 'Doctorado o más'])
            respuestas['estrategia_inversion'] = st.selectbox("¿Qué estrategia de inversión has utilizado?", ['Conservadora', 'Moderada', 'Crecimiento', 'Agresiva', 'Muy Agresiva', 'No he invertido'])
            respuestas['conocimiento_servicios'] = st.selectbox("¿Conoces los servicios de inversión?", ['Sí', 'No'])
            respuestas['instrumentos'] = st.multiselect("¿En cuáles de los siguientes instrumentos has invertido en los últimos 2 años?", 
                ['Deuda bancaria', 'Deuda gubernamental', 'Divisas y valores en monedas diferentes al peso', 
                'Fondos de inversión en deuda', 'Fondos de inversión renta variable', 'Acciones y CPO’s', 'Valores corporativos', 
                'ETF’s y Tracks', 'Derivados', 'Valores extranjeros', 'Productos estructurados', 'Fibras', 'No he invertido'])

            respuestas['patrimonio_invertir'] = st.selectbox("¿Lo que deseas invertir representa de tu patrimonio financiero?", 
                                                            ['Menos del 10%', 'Entre 10% y 20%', 'Entre 20% y 40%', 'Entre 40% y 60%', 'Más del 60%'])
            respuestas['destino_inversion'] = st.selectbox("¿Cuál es el destino de tu inversión?", 
                                                        ['Es para incrementar mi patrimonio', 'Es para mi jubilación', 
                                                            'Es para incrementar mi capital en una inversión de largo plazo', 
                                                            'No pienso utilizar la mayor parte en el mediano plazo', 
                                                            'Pienso utilizar la mayor parte en el corto plazo', 'Dependo de un flujo para mis gastos'])
            respuestas['disposicion_derivados'] = st.selectbox("¿Estás dispuesto a invertir en un instrumento financiero que contenga derivados?", 
                                                            ['No estoy dispuesto', 'Sí estoy dispuesto'])
            respuestas['objetivo_inversion'] = st.selectbox("¿Cuál es el objetivo de tu inversión?", 
                                                        ['Corto Plazo', 'Mediano Plazo', 'Largo Plazo'])
            respuestas['plazo_disponible'] = st.selectbox("¿En qué plazo podrías utilizar la mayor parte de los recursos de esta inversión?", 
                                                        ['Menor a 1 año', 'De 1 a 3 años', 'De 3 a 5 años', 'Más de 5 años'])
            respuestas['portafolio'] = st.selectbox("Si pudieras elegir uno de los siguientes portafolios, ¿Cuál elegirías?", 
                                                    ['100% en Deuda', '80% en Deuda y 20% en Renta Variable', '50% en Deuda y 50% en Renta Variable', 
                                                    '20% en Deuda y 80% en Renta Variable', '100% en Renta Variable'])
            respuestas['renta_variable'] = st.selectbox("¿Qué porcentaje deseas invertir en Renta Variable?", 
                                                    ['No deseo invertir en Renta Variable', 'De 1% a 20%', 'Entre 21% y 50%', 'Más del 50%'])
            respuestas['accion_directa'] = st.selectbox("¿Estás dispuesto a invertir en una sola acción en directo?", 
                                                    ['De 1% a 10%', 'De 11% a 20%', 'Entre 21% a 30%', 'Más de 30%', 'No estoy dispuesto'])
            respuestas['valores_corporativos'] = st.selectbox("¿Qué porcentaje deseas invertir en valores corporativos?", 
                                                            ['No deseo invertir en valores corporativos', 'De 1% a 30%', 'De 31% a 60%', 'Más de 60%'])
            respuestas['valores_corporativos_emisora'] = st.selectbox("¿Estás dispuesto a invertir en valores corporativos en una sola emisora?", 
                                                                    ['No estoy dispuesto', 'Hasta 10%', 'De 11% a 20%', 'De 21% a 30%', 'Más del 30%'])
            respuestas['restricciones'] = st.selectbox("¿Requiere restricciones adicionales a las que se derivan de su perfil de inversión?", 
                                                    ['NO hay restricciones', 'NO deseo operar en mercado de capitales', 'NO deseo operar en mercado de derivados'])

            # Calcular puntaje y mostrar resultado
            puntaje = calcular_puntaje(respuestas)
            perfil = determinar_perfil(puntaje)

            st.write(f"Tu puntaje total es: {puntaje}")
            st.write(f"Tu perfil es: {perfil}")


        # Mostrar gráfico técnico si se hace clic en "Balance Sheet"
        if st.session_state['Clasificación de acciones']:
            
            
            
            
            st.title('Clasificación de acciones')
            plot_clusters(final_data, category_names)
            filter_and_display_category(final_data, categoria_seleccionada, category_names)

            # acciones_categoria = filter_and_get_category(final_data, categoria_seleccionada) 
            # selected_ticker = st.selectbox('Selecciona el ticker:', acciones_categoria)



















    
    
    
    
    
    
    
    
# Mostrar gráfico técnico si se hace clic en "Perfil"
if st.session_state['show_general']:


    
    # Función para descargar todas las noticias de un ticker y guardarlas en un DataFrame
    @st.cache_data
    def download_full_news(tickers):
        all_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Verifica si news es una lista de diccionarios
            if isinstance(news, list):  
                for item in news:
                    item['Ticker'] = ticker  # Agrega el ticker a cada noticia
                all_data.extend(news)  # Agrega todas las noticias al total
            else:
                st.warning(f"No se encontraron noticias para el ticker: {ticker}")
        
        # Si no se encontró nada, previene que se devuelva un DataFrame vacío
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return df

    # Función para descargar toda la información de .info y guardarla en un DataFrame
    @st.cache_data
    def download_full_info(tickers):
        all_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            info['Ticker'] = ticker
            all_data.append(info)
        
        df = pd.DataFrame(all_data)
        return df
    
    # Uso de las funciones
    tickers = get_sp500_tickers()
    df = download_full_info(tickers)
    
    acciones_categoria = filter_and_get_category(final_data, categoria_seleccionada) 

    # Verificar si ya existen entradas de usuario previas en el estado de sesión
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'AAPL'  # Valor predeterminado
    
    # Ingresar el símbolo de la acción
    #ticker = st.selectbox('Selecciona el símbolo de la acción:', tickers)
    ticker = selected_ticker = st.selectbox('Selecciona el ticker:', acciones_categoria)
    st.session_state['ticker'] = ticker  # Guardar el símbolo de la acción  
    
    name_ticker = df.loc[df['Ticker'] == ticker, 'longName'].values[0]
    summary_ticker = df.loc[df['Ticker'] == ticker, 'longBusinessSummary'].values[0]

    sector_ticker = df.loc[df['Ticker'] == ticker, 'sector'].values[0]
    industry_ticker = df.loc[df['Ticker'] == ticker, 'industry'].values[0]
    currency_ticker = df.loc[df['Ticker'] == ticker, 'currency'].values[0]
    exchange_ticker = df.loc[df['Ticker'] == ticker, 'exchange'].values[0]

    # Crear widgets para mostrar los datos
    st.title(f"Información general de ")
    st.title(f"{name_ticker} ({ticker})")

    st.divider()

    st.subheader(f"Resumen ejecutivo:")
    st.markdown(summary_ticker)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Sector:")
        st.write(sector_ticker)
    with col2:
        st.subheader("Industry:")
        st.write(industry_ticker)
    with col3:
        st.subheader("Currency")
        st.write(currency_ticker)
    with col4:
        st.subheader("Exchange")
        st.write(exchange_ticker)

    st.divider()

    st.subheader(f"Gobierno corporativo:")
        
    audit_ticker = df.loc[df['Ticker'] == ticker, 'auditRisk'].values[0]
    board_ticker = df.loc[df['Ticker'] == ticker, 'boardRisk'].values[0]
    compensation_ticker = df.loc[df['Ticker'] == ticker, 'compensationRisk'].values[0]
    overll_ticker = df.loc[df['Ticker'] == ticker, 'overallRisk'].values[0]
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.subheader("Auditoría:")
        st.write(
            f"<div style='font-size:24px; color:black;'>{audit_ticker}</div>",
            unsafe_allow_html=True
        )
    with col6:
        st.subheader("Consejo:")
        st.write(
            f"<div style='font-size:24px; color:black;'>{board_ticker}</div>",
            unsafe_allow_html=True
        )
    with col7:
        st.subheader("Derechos de accionistas:")
        st.write(
            f"<div style='font-size:24px; color:black;'>{compensation_ticker}</div>",
            unsafe_allow_html=True
        )
    with col8:
        st.subheader("Compensación:")
        st.write(
            f"<div style='font-size:24px; color:black;'>{overll_ticker}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    st.subheader(f"Para mas información:")

    address1_ticker = df.loc[df['Ticker'] == ticker, 'address1'].values[0]
    city_ticker = df.loc[df['Ticker'] == ticker, 'city'].values[0]
    state_ticker = df.loc[df['Ticker'] == ticker, 'state'].values[0]
    zip_ticker = df.loc[df['Ticker'] == ticker, 'zip'].values[0]
    country_ticker = df.loc[df['Ticker'] == ticker, 'country'].values[0]

    phone_ticker = df.loc[df['Ticker'] == ticker, 'phone'].values[0]
    website_ticker = df.loc[df['Ticker'] == ticker, 'website'].values[0]

    col9, col10 = st.columns(2)
    with col9:
        st.subheader("Dirección")
        st.write(f"{address1_ticker} {city_ticker},{state_ticker} {zip_ticker} {country_ticker}.")
    with col10:
        st.subheader("Contacto")
        st.write(f"{phone_ticker}")
        st.write(f"{website_ticker}")

    st.divider()
    
    # Descargar las noticias de la acción seleccionada
    df = download_full_news([ticker])

    # Verificar si se descargaron noticias
    if not df.empty:
        # Crear dos columnas usando st.columns()
        col1, col2 = st.columns([1, 3])  # Ajustamos las proporciones de las columnas

        with col1:
            # Extraer los valores únicos de "publisher" para el filtro
            publishers = df['publisher'].unique()  # Extraemos los publicadores únicos
            publisher = st.selectbox('Selecciona un publicador:', publishers)  # Lista desplegable de publicadores

        with col2:
            # Filtrar las noticias por el publicador seleccionado
            filtered_df = df[df['publisher'] == publisher]

            # Mostrar los títulos con enlaces en lugar de un DataFrame
            st.write(f"**Noticias de {publisher}:**")
            
            # Iteramos sobre las noticias filtradas para mostrar los títulos con los enlaces
            if not filtered_df.empty:
                for index, row in filtered_df.iterrows():
                    title = row['title']
                    link = row['link']
                    
                    # Crear un enlace para cada título
                    st.markdown(f"[{title}]({link})")
            else:
                st.write("No hay noticias disponibles para este publicador.")
    else:
        st.warning(f"No se encontraron noticias para el ticker {ticker}.")




    st.divider()

    # Cargar las variables de entorno
    load_dotenv()

    # Obtener la API Key de OpenAI desde las variables de entorno

    if not os.getenv('api_key'):
        raise ValueError("API Key de OpenAI no encontrada en las variables de entorno")

    # Configurar tu API Key de OpenAI
    openai.api_key = os.getenv('api_key')

    # Título de la app
    st.title("Asesor IA")

    # Selección del ticker
    ticker = st.session_state['ticker']

    if ticker:
        # Usar un expander para mostrar el chat
        with st.expander("❓ ¿Necesitas ayuda?"):
            # Verificar si ya existe un historial de mensajes, si no lo hay, creamos uno con el contexto inicial
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US."},
                    {"role": "assistant", "content": f"Hola, soy tu asesor financiero. ¿En qué puedo ayudarte hoy ?"}
                ]
            
            # Actualizar el contexto del sistema con el nuevo ticker
            st.session_state["messages"][0] = {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US"}
            
            # Lista para mensajes visibles
            visible_messages = []

            # Mostrar los mensajes previos
            for msg in st.session_state["messages"]:
                if msg["role"] != "system":  # Ocultar los mensajes del sistema
                    visible_messages.append(msg)
                    st.chat_message(msg["role"]).write(msg["content"])

            # Capturar la entrada del usuario y responder con ChatGPT
            if user_input := st.chat_input():
                st.session_state["messages"].append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                try:
                    # Llamada a la nueva API de OpenAI para obtener la respuesta
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"]
                    )
                    
                    responseMessage = response['choices'][0]['message']['content']

                    # Mostrar la respuesta de ChatGPT
                    st.session_state["messages"].append({"role": "assistant", "content": responseMessage})
                    visible_messages.append({"role": "assistant", "content": responseMessage})
                    st.chat_message("assistant").write(responseMessage)
                
                except RateLimitError:
                    st.error("Has excedido tu cuota actual de uso de la API. Por favor, revisa tu plan y detalles de facturación en el dashboard de OpenAI.")



























# Mostrar gráfico técnico si se hace clic en "Fundamental"
if st.session_state['show_fundamental']:    

    # Función para obtener los tickers del S&P 500
    @st.cache_data
    def get_sp500_tickers():
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(sp500_url, header=0)
        df = table[0]
        return df['Symbol'].tolist()

    # Función para descargar diferentes tipos de datos financieros y guardarlos en un DataFrame
    @st.cache_data
    def download_financial_data(tickers, data_type, is_quarterly=False):
        all_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data = getattr(stock, data_type)
            if is_quarterly:
                data = data.quarterly
            if data is not None:
                df_data = data.copy()
                df_data['Ticker'] = ticker
                all_data.append(df_data)
            else:
                st.warning(f"No se encontró {data_type} para el ticker: {ticker}")

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, axis=0)
        return df

    # Uso de las funciones para descargar la información deseada
    tickers = get_sp500_tickers()
    income_stmt = download_financial_data(tickers, 'financials')
    balance_stmt = download_financial_data(tickers, 'balance_sheet')
    Cash_stmt = download_financial_data(tickers, 'cashflow')
    acciones_categoria = filter_and_get_category(final_data, categoria_seleccionada) 

    # Verificar si ya existen entradas de usuario previas en el estado de sesión
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'AAPL'  # Valor predeterminado
    
    
    if st.session_state.get('show_fundamental', False):
        
        # Ingresar el símbolo de la acción
        #ticker = st.selectbox('Selecciona el símbolo de la acción:', tickers)
        ticker = selected_ticker = st.selectbox('Selecciona el ticker:', acciones_categoria)
        st.session_state['ticker'] = ticker  # Guardar el símbolo de la acción
    
        # Definir las columnas para los botones
        col1, col2, col3 = st.columns(3)

        # Guardar el estado de los botones
        if 'Income Statement' not in st.session_state:
            st.session_state['Income Statement'] = False
        if 'Balance Sheet' not in st.session_state:
            st.session_state['Balance Sheet'] = False
        if 'Cash Flow' not in st.session_state:
            st.session_state['Cash Flow'] = False

        # Botones en la misma fila
        if col1.button("Income Statement", use_container_width=True):
            st.session_state['Income Statement'] = True
            st.session_state['Balance Sheet'] = False
            st.session_state['Cash Flow'] = False
                
        if col2.button("Balance Sheet", use_container_width=True):
            st.session_state['Income Statement'] = False
            st.session_state['Balance Sheet'] = True
            st.session_state['Cash Flow'] = False
                
        if col3.button("Cash Flow", use_container_width=True):
            st.session_state['Income Statement'] = False
            st.session_state['Balance Sheet'] = False
            st.session_state['Cash Flow'] = True
            
        st.divider()
        
        # Mostrar gráfico técnico si se hace clic en "Income Statement"
        if st.session_state['Income Statement']:
            # Filtrar el DataFrame por el ticker seleccionado
            filtered_income_stmt = income_stmt[income_stmt['Ticker'] == ticker]
            filtered_income_stmt = filtered_income_stmt.drop(columns=['Ticker'])

            st.title('Income Statement')
            income_stmt_cleaned = filtered_income_stmt.dropna(axis=1, how='all')
            st.dataframe(income_stmt_cleaned)

        # Mostrar gráfico técnico si se hace clic en "Balance Sheet"
        if st.session_state['Balance Sheet']:
            # Filtrar el DataFrame por el ticker seleccionado
            filtered_balance_stmt = balance_stmt[balance_stmt['Ticker'] == ticker]
            filtered_balance_stmt = filtered_balance_stmt.drop(columns=['Ticker'])

            st.title('Balance Sheet')
            balance_stmt_cleaned = filtered_balance_stmt.dropna(axis=1, how='all')
            st.dataframe(balance_stmt_cleaned)

        # Mostrar flujo de efectivo si se hace clic en "Cash Flow"
        if st.session_state['Cash Flow']:
            # Filtrar el DataFrame por el ticker seleccionado
            filtered_cash_flow_stmt = Cash_stmt[Cash_stmt['Ticker'] == ticker]
            filtered_cash_flow_stmt = filtered_cash_flow_stmt.drop(columns=['Ticker'])

            # st.title('Cash Flow Statement')

            # Definir las partidas para cada categoría
            operating_activities = filtered_cash_flow_stmt.loc[filtered_cash_flow_stmt.index.isin([
                'Net Income From Continuing Operations'
                ,'Cash Flow From Continuing Operating Activities'
                ,'Operating Cash Flow'
                ,'Change In Working Capital'
                ,'Change In Payables And Accrued Expense'
                ,'Change In Payable'
                ,'Change In Account Payable'
                ,'Change In Tax Payable'
                ,'Change In Income Tax Payable'
                ,'Change In Inventory'
                ,'Change In Receivables'
                ,'Changes In Account Receivables'
                ,'Stock Based Compensation'
                ,'Deferred Income Tax'
                ,'Deferred Tax'
                ,'Depreciation Amortization Depletion'
                ,'Depreciation And Amortization'
                ,'Operating Gains Losses'
                ,'Pension And Employee Benefit Expense'
                ,'Other Non Cash Items'
                ,'Interest Paid Supplemental Data'
                ,'Income Tax Paid Supplemental Data'
            ])]

            investing_activities = filtered_cash_flow_stmt.loc[filtered_cash_flow_stmt.index.isin([
                'Capital Expenditures', 'Investments', 'Acquisitions', 'Sale of Fixed Assets'
            ])]

            financing_activities = filtered_cash_flow_stmt.loc[filtered_cash_flow_stmt.index.isin([
                'Dividends Paid', 'Issuance of Debt', 'Repayment of Debt', 'Issuance/Repurchase of Stock'
            ])]

            other_activities = filtered_cash_flow_stmt.loc[filtered_cash_flow_stmt.index.isin([
                'Effect of Exchange Rate', 'Net Change in Cash'
            ])]

            # Mostrar las actividades agrupadas
            # st.subheader('Operating Activities')
            # st.dataframe(operating_activities.dropna(axis=1, how='all'))

            # st.subheader('Investing Activities')
            # st.dataframe(investing_activities.dropna(axis=1, how='all'))

            # st.subheader('Financing Activities')
            # st.dataframe(financing_activities.dropna(axis=1, how='all'))

            # st.subheader('Other')
            # st.dataframe(other_activities.dropna(axis=1, how='all'))


            # Filtrar el DataFrame por el ticker seleccionado
            filtered_Cash_stmt = Cash_stmt[Cash_stmt['Ticker'] == ticker]
            filtered_Cash_stmt = filtered_Cash_stmt.drop(columns=['Ticker'])

            st.title('Cash Flow')
            Cash_stmt_cleaned = filtered_Cash_stmt.dropna(axis=1, how='all')
            st.dataframe(Cash_stmt_cleaned)
    
    st.divider()

    # Cargar las variables de entorno
    load_dotenv()

    # Obtener la API Key de OpenAI desde las variables de entorno

    if not os.getenv('api_key'):
        raise ValueError("API Key de OpenAI no encontrada en las variables de entorno")

    # Configurar tu API Key de OpenAI
    openai.api_key = os.getenv('api_key')

    # Título de la app
    st.title("Asesor IA")

    # Selección del ticker
    ticker = st.session_state['ticker']

    if ticker:
        # Usar un expander para mostrar el chat
        with st.expander("❓ ¿Necesitas ayuda?"):
            # Verificar si ya existe un historial de mensajes, si no lo hay, creamos uno con el contexto inicial
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US."},
                    {"role": "assistant", "content": f"Hola, soy tu asesor financiero. ¿En qué puedo ayudarte hoy ?"}
                ]
            
            # Actualizar el contexto del sistema con el nuevo ticker
            st.session_state["messages"][0] = {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US"}
            
            # Lista para mensajes visibles
            visible_messages = []

            # Mostrar los mensajes previos
            for msg in st.session_state["messages"]:
                if msg["role"] != "system":  # Ocultar los mensajes del sistema
                    visible_messages.append(msg)
                    st.chat_message(msg["role"]).write(msg["content"])

            # Capturar la entrada del usuario y responder con ChatGPT
            if user_input := st.chat_input():
                st.session_state["messages"].append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                try:
                    # Llamada a la nueva API de OpenAI para obtener la respuesta
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"]
                    )
                    
                    responseMessage = response['choices'][0]['message']['content']

                    # Mostrar la respuesta de ChatGPT
                    st.session_state["messages"].append({"role": "assistant", "content": responseMessage})
                    visible_messages.append({"role": "assistant", "content": responseMessage})
                    st.chat_message("assistant").write(responseMessage)
                
                except RateLimitError:
                    st.error("Has excedido tu cuota actual de uso de la API. Por favor, revisa tu plan y detalles de facturación en el dashboard de OpenAI.")






   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
# Mostrar gráfico técnico si se hace clic en "Técnico"
if st.session_state['show_technical']:
    st.title('Gráfico de Precio de Acción con Medias Móviles Personalizadas')
    
    # Función para obtener los tickers del S&P 500
    @st.cache_data
    def get_sp500_tickers():
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(sp500_url, header=0)
        df = table[0]
        return df['Symbol'].tolist()

    # Función para obtener datos históricos de la acción
    @st.cache_data
    def obtener_datos_accion(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        return data

    # Función para calcular RSI
    def calcular_rsi(data, periodo=14):
        delta = data['Close'].diff()
        ganancia = delta.where(delta > 0, 0)
        perdida = -delta.where(delta < 0, 0)
        ganancia_media = ganancia.rolling(window=periodo).mean()
        perdida_media = perdida.rolling(window=periodo).mean()
        rs = ganancia_media / perdida_media
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Función para calcular MACD
    def calcular_macd(data):
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        return macd, macd_signal

    # Función para graficar el precio de cierre con medias móviles
    def graficar_precio(data, ticker, sma_short, sma_long):
        data[f'SMA_{sma_short}'] = data['Close'].rolling(window=sma_short).mean()
        data[f'SMA_{sma_long}'] = data['Close'].rolling(window=sma_long).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label='Precio de Cierre', color='black', linewidth=1)
        ax.plot(data.index, data[f'SMA_{sma_short}'], label=f'SMA {sma_short} días', color='blue', linestyle='--')
        ax.plot(data.index, data[f'SMA_{sma_long}'], label=f'SMA {sma_long} días', color='red', linestyle='--')
        ax.set_title(f'{ticker} - Precio de Cierre con Medias Móviles')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio ($)')
        ax.legend()
        st.pyplot(fig)

    # Función para descargar todas las recomendaciones de un ticker y guardarlas en un DataFrame
    @st.cache_data
    def download_full_recommendations(tickers):
        all_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            if recommendations is not None:  
                recommendations['Ticker'] = ticker
                all_data.append(recommendations)
            else:
                st.warning(f"No se encontraron recomendaciones para el ticker: {ticker}")
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data)
        return df

    # Función para descargar los upgrades y downgrades de un ticker y guardarlas en un DataFrame
    @st.cache_data
    def download_upgrades_downgrades(tickers):
        all_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            upgrades_downgrades = stock.upgrades_downgrades
            
            if upgrades_downgrades is not None:
                upgrades_downgrades['Ticker'] = ticker
                all_data.append(upgrades_downgrades)
            else:
                st.warning(f"No se encontraron upgrades/downgrades para el ticker: {ticker}")
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data)
        return df

    # Función para descargar toda la información de .info y guardarla en un DataFrame
    @st.cache_data
    def download_full_info(tickers):
        all_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            info['Ticker'] = ticker
            all_data.append(info)
        
        df = pd.DataFrame(all_data)
        return df

    # Uso de las funciones
    tickers = get_sp500_tickers()
    
    acciones_categoria = filter_and_get_category(final_data, categoria_seleccionada) 

    # Verificar si ya existen entradas de usuario previas en el estado de sesión
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'AAPL'
    if 'periodo' not in st.session_state:
        st.session_state['periodo'] = '1 mes'
    if 'sma_short' not in st.session_state:
        st.session_state['sma_short'] = 20
    if 'sma_long' not in st.session_state:
        st.session_state['sma_long'] = 50

    # Definir las dos columnas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Ingresar el símbolo de la acción
        #ticker = st.selectbox('Selecciona el símbolo de la acción:', tickers)
        ticker = selected_ticker = st.selectbox('Selecciona el ticker:', acciones_categoria)
        st.session_state['ticker'] = ticker  # Guardar el símbolo de la acción

    with col2:
        opciones = ['1 mes', '3 meses', '6 meses', '1 año', '5 años', 'Toda la historia']
        periodo = st.selectbox('Selecciona el rango de tiempo', opciones, index=opciones.index(st.session_state['periodo']))
        st.session_state['periodo'] = periodo

    with col3:
        sma_short = st.number_input('Número de días para la media móvil corta', min_value=1, max_value=200, value=st.session_state['sma_short'])
        st.session_state['sma_short'] = sma_short

    with col4:
        sma_long = st.number_input('Número de días para la media móvil larga', min_value=1, max_value=500, value=st.session_state['sma_long'])
        st.session_state['sma_long'] = sma_long

    # Calcular las fechas de inicio y fin según la opción seleccionada
    hoy = pd.to_datetime('today')

    if periodo == '1 mes':
        fecha_inicio = hoy - timedelta(days=30)
    elif periodo == '3 meses':
        fecha_inicio = hoy - timedelta(days=90)
    elif periodo == '6 meses':
        fecha_inicio = hoy - timedelta(days=180)
    elif periodo == '1 año':
        fecha_inicio = hoy - timedelta(days=365)
    elif periodo == '5 años':
        fecha_inicio = hoy - timedelta(days=1825)
    else:
        fecha_inicio = '1900-01-01'

    fecha_fin = hoy

    # Descargar los datos con las fechas seleccionadas
    data = obtener_datos_accion(ticker, fecha_inicio, fecha_fin)

    # Mostrar el gráfico de precios y medias móviles
    graficar_precio(data, ticker, sma_short, sma_long)

    st.divider()

    # Calcular indicadores técnicos
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI_14'] = calcular_rsi(data)
    data['MACD'], data['MACD_Signal'] = calcular_macd(data)

    # Gráfico de Velas con Medias Móviles
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )])

    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title=f"Gráfico de Velas y Medias Móviles para {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio ($)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)

    st.divider()

    # Gráfico RSI
    st.subheader("RSI de 14 días")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], mode='lines', name='RSI', line=dict(color='orange')))
    fig_rsi.update_layout(
        title=f"RSI de 14 días para {ticker}",
        xaxis_title="Fecha",
        yaxis_title="RSI",
        template="plotly_dark"
    )
    st.plotly_chart(fig_rsi)

    st.divider()

    # Gráfico MACD
    st.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='green')))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Línea de Señal', line=dict(color='red', dash='dash')))
    fig_macd.update_layout(
        title=f"MACD y Línea de Señal para {ticker}",
        xaxis_title="Fecha",
        yaxis_title="MACD",
        template="plotly_dark"
    )
    st.plotly_chart(fig_macd)
    
    st.divider()         
     
    # Descargar y mostrar las recomendaciones
    recommendations_df = download_full_recommendations(tickers)
    st.write(recommendations_df[recommendations_df['Ticker'] == ticker].drop(columns=['Ticker']))
    
    st.divider()

    #Descargar y mostrar las recomendaciones
    download_upgrades_downgrades_df = download_upgrades_downgrades(tickers)
    #st.write(download_upgrades_downgrades_df[download_upgrades_downgrades_df['Ticker'] == ticker].drop(columns=['Ticker']))

    #st.divider()

    # Crear dos columnas usando st.columns()
    col1, col2 = st.columns([1, 3])  # Ajustamos las proporciones de las columnas

    with col1:
        # Extraer los valores únicos de "publisher" para el filtro
        firms = download_upgrades_downgrades_df['Firm'].unique()  # Extraemos los publicadores únicos
        firm = st.selectbox('Selecciona un Firm:', firms)  # Lista desplegable de publicadores

    with col2:
        # Filtrar las noticias por el publicador seleccionado
        filtered_df = download_upgrades_downgrades_df[download_upgrades_downgrades_df['Firm'] == firm]
        
        st.write(f"**Noticias de {firm}:**")
            
        # Iteramos sobre las noticias filtradas para mostrar los títulos con los enlaces
        st.write(filtered_df.drop(columns=['Ticker','Firm']))


    st.divider()


    # Cargar las variables de entorno
    load_dotenv()

    # Obtener la API Key de OpenAI desde las variables de entorno

    if not os.getenv('api_key'):
        raise ValueError("API Key de OpenAI no encontrada en las variables de entorno")

    # Configurar tu API Key de OpenAI
    openai.api_key = os.getenv('api_key')

    # Título de la app
    st.title("Asesor IA")

    # Selección del ticker
    ticker = st.session_state['ticker']

    if ticker:
        # Usar un expander para mostrar el chat
        with st.expander("❓ ¿Necesitas ayuda?"):
            # Verificar si ya existe un historial de mensajes, si no lo hay, creamos uno con el contexto inicial
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US."},
                    {"role": "assistant", "content": f"Hola, soy tu asesor financiero. ¿En qué puedo ayudarte hoy ?"}
                ]
            
            # Actualizar el contexto del sistema con el nuevo ticker
            st.session_state["messages"][0] = {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US"}
            
            # Lista para mensajes visibles
            visible_messages = []

            # Mostrar los mensajes previos
            for msg in st.session_state["messages"]:
                if msg["role"] != "system":  # Ocultar los mensajes del sistema
                    visible_messages.append(msg)
                    st.chat_message(msg["role"]).write(msg["content"])

            # Capturar la entrada del usuario y responder con ChatGPT
            if user_input := st.chat_input():
                st.session_state["messages"].append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                try:
                    # Llamada a la nueva API de OpenAI para obtener la respuesta
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"]
                    )
                    
                    responseMessage = response['choices'][0]['message']['content']

                    # Mostrar la respuesta de ChatGPT
                    st.session_state["messages"].append({"role": "assistant", "content": responseMessage})
                    visible_messages.append({"role": "assistant", "content": responseMessage})
                    st.chat_message("assistant").write(responseMessage)
                
                except RateLimitError:
                    st.error("Has excedido tu cuota actual de uso de la API. Por favor, revisa tu plan y detalles de facturación en el dashboard de OpenAI.")


































# Mostrar gráfico técnico si se hace clic en "Forecast"
if st.session_state['show_forecast']:
    
    @st.cache_data
    def get_sp500_tickers2():
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(sp500_url, header=0)
        df = table[0]
        return df['Symbol'].tolist()

    @st.cache_data
    def preprocess_stock_data(data):
        """Preprocess stock data for training and scaling."""
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        time_step = 180

        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        train_data = scaled_data[:int(0.8 * len(scaled_data))]
        test_data = scaled_data[int(0.8 * len(scaled_data)):]

        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        return scaler, X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test

    @st.cache_resource
    def train_model(X_train, y_train):
        """Train and cache the LSTM model."""
        model = models.Sequential([
            layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            layers.LSTM(units=50, return_sequences=False),
            layers.Dense(units=25),
            layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2, callbacks=[early_stopping])
        return model

    # Función para obtener datos históricos de la acción
    @st.cache_data
    def obtener_datos_accion(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        return data
    
    def forecast_stock_with_variability(model, scaler, X_test, days_to_forecast, noise_factor=0.02):
        input_seq = X_test[-1:]  # Start with the last test sequence
        predictions = []
        lower_bound = []
        upper_bound = []

        for _ in range(days_to_forecast):
            # Predict the next value
            next_pred = model.predict(input_seq, verbose=0)

            # Add variability (random noise)
            next_pred_value = next_pred[0, 0]
            noise = np.random.normal(0, noise_factor * next_pred_value)
            noisy_pred = next_pred_value + noise

            # Append predictions and bounds
            predictions.append(noisy_pred)
            lower_bound.append(next_pred_value - abs(noise))
            upper_bound.append(next_pred_value + abs(noise))

            # Update the input sequence with the noisy prediction
            next_pred_reshaped = np.array([[noisy_pred]]).reshape((1, 1, -1))
            input_seq = np.append(input_seq[:, 1:, :], next_pred_reshaped, axis=1)

        # Inverse transform predictions and bounds
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        lower_bound = scaler.inverse_transform(np.array(lower_bound).reshape(-1, 1))
        upper_bound = scaler.inverse_transform(np.array(upper_bound).reshape(-1, 1))

        return predictions, lower_bound, upper_bound
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all but critical logs
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    default_start_date = datetime(2010, 1, 1)
    default_end_date = datetime(2024, 11, 1)

    sp500_tickers2 = get_sp500_tickers2()

    acciones_categoria = filter_and_get_category(final_data, categoria_seleccionada) 

    # Verificar si ya existen entradas de usuario previas en el estado de sesión
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'AAPL'  # Valor predeterminado
    
    # Ingresar el símbolo de la acción
    #ticker = st.selectbox('Selecciona el símbolo de la acción:', tickers)
    ticker = selected_ticker = st.selectbox('Selecciona el ticker:', acciones_categoria)
    st.session_state['ticker'] = ticker  # Guardar el símbolo de la acción  
   
    
    
    forecast_days = st.radio("Forecast Steps (Days)", [30, 90, 180], index=0)

    fecha_inicio = st.date_input("Start Date", default_start_date)
    fecha_fin = st.date_input("End Date", default_end_date)

    data = obtener_datos_accion(ticker, fecha_inicio, fecha_fin)    

    stock_data = data

    # Preprocess data and train model
    scaler, X_train, y_train, X_test, y_test = preprocess_stock_data(stock_data)
    model = train_model(X_train, y_train)

    # Forecast stock prices
    # Forecast stock prices with variability
    extended_forecast, lower_bound, upper_bound = forecast_stock_with_variability(model, scaler, X_test, 180)
    selected_forecast = extended_forecast[:forecast_days]
    selected_lower = lower_bound[:forecast_days]
    selected_upper = upper_bound[:forecast_days]

    # Forecast dates
    forecast_dates = pd.date_range(start=stock_data.index[-1], periods=forecast_days + 1, freq='B')[1:]

    # Plotting with confidence interval
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index, stock_data['Close'].values, label='True Stock Price')
    plt.plot(forecast_dates, selected_forecast, color='orange', label=f'{forecast_days}-Day Forecast')
    plt.fill_between(
        forecast_dates,
        selected_lower.flatten(),
        selected_upper.flatten(),
        color='gray',
        alpha=0.3,
        label='Confidence Interval'
    )
    plt.title(f'{ticker} Stock Price Prediction with {forecast_days}-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    


    st.divider()


    # Cargar las variables de entorno
    load_dotenv()

    # Obtener la API Key de OpenAI desde las variables de entorno

    if not os.getenv('api_key'):
        raise ValueError("API Key de OpenAI no encontrada en las variables de entorno")

    # Configurar tu API Key de OpenAI
    openai.api_key = os.getenv('api_key')

    # Título de la app
    st.title("Asesor IA")

    # Selección del ticker
    ticker = st.session_state['ticker']

    if ticker:
        # Usar un expander para mostrar el chat
        with st.expander("❓ ¿Necesitas ayuda?"):
            # Verificar si ya existe un historial de mensajes, si no lo hay, creamos uno con el contexto inicial
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US."},
                    {"role": "assistant", "content": f"Hola, soy tu asesor financiero. ¿En qué puedo ayudarte hoy ?"}
                ]
            
            # Actualizar el contexto del sistema con el nuevo ticker
            st.session_state["messages"][0] = {"role": "system", "content": f"Eres un asesor financiero especializado en temas bursátiles y actualmente estamos analizando acciones de US"}
            
            # Lista para mensajes visibles
            visible_messages = []

            # Mostrar los mensajes previos
            for msg in st.session_state["messages"]:
                if msg["role"] != "system":  # Ocultar los mensajes del sistema
                    visible_messages.append(msg)
                    st.chat_message(msg["role"]).write(msg["content"])

            # Capturar la entrada del usuario y responder con ChatGPT
            if user_input := st.chat_input():
                st.session_state["messages"].append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                try:
                    # Llamada a la nueva API de OpenAI para obtener la respuesta
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state["messages"]
                    )
                    
                    responseMessage = response['choices'][0]['message']['content']

                    # Mostrar la respuesta de ChatGPT
                    st.session_state["messages"].append({"role": "assistant", "content": responseMessage})
                    visible_messages.append({"role": "assistant", "content": responseMessage})
                    st.chat_message("assistant").write(responseMessage)
                
                except RateLimitError:
                    st.error("Has excedido tu cuota actual de uso de la API. Por favor, revisa tu plan y detalles de facturación en el dashboard de OpenAI.")
























# Mostrar gráfico técnico si se hace clic en "Portafolio"
if st.session_state['show_portafolio']:

    # Función para obtener los tickers del S&P 500 con nombres
    @st.cache_data
    def get_sp500_tickers():
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(sp500_url, header=0)
        df = table[0]
        df['Ticker-Name'] = df['Symbol'] + ' - ' + df['Security']
        return df[['Symbol', 'Ticker-Name']]

    # Función para obtener los precios históricos de las acciones
    @st.cache_data
    def get_stock_data(tickers, start_date, end_date):
        return yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Función para calcular la rentabilidad y riesgo de un portafolio
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return std, returns

    # Función para minimizar la volatilidad
    def minimize_volatility(weights, mean_returns, cov_matrix):
        return portfolio_performance(weights, mean_returns, cov_matrix)[0]

    # Función para maximizar los retornos
    def maximize_returns(weights, mean_returns, cov_matrix):
        return -portfolio_performance(weights, mean_returns, cov_matrix)[1]

    # Función para calcular retornos y volatilidad anualizada
    def annualize_performance(returns, std, period=252):
        annualized_return = (1 + returns)**period - 1
        annualized_std = std * np.sqrt(period)
        return annualized_return, annualized_std

    # Función para calcular el valor de la inversión a lo largo del tiempo
    def calculate_investment_value(initial_investment, returns, weights, stock_data):
        daily_returns = stock_data.pct_change().dropna()
        portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
        investment_value = initial_investment * (1 + portfolio_daily_returns).cumprod()
        return investment_value

    # Interfaz de usuario
    st.title('Optimización de Portafolio de Inversión')
    
    # Cuadro de texto para el monto a invertir
    investment_amount = st.number_input('Monto a invertir', min_value=0)

    # Selección de empresas del S&P 500
    tickers = get_sp500_tickers()
    selected_ticker_names = st.multiselect('Seleccione las empresas', tickers['Ticker-Name'])
    selected_tickers = [ticker.split(' - ')[0] for ticker in selected_ticker_names]

    # Selección de fecha de inicio
    start_date = st.date_input('Fecha de inicio', pd.to_datetime('2010-01-01'))
    end_date = pd.Timestamp.today()  # Fecha de fin siempre es la más reciente

    # Botón para calcular la optimización del portafolio
    if st.button('Calcular Optimización'):
        if selected_tickers and start_date < end_date.date():  # Corrección aquí
            stock_data = get_stock_data(selected_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            mean_returns = stock_data.pct_change().mean()
            cov_matrix = stock_data.pct_change().cov()
            
            num_assets = len(selected_tickers)
            args = (mean_returns, cov_matrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for asset in range(num_assets))

            # Portafolio con mínima volatilidad
            result_min_vol = minimize(minimize_volatility, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights_min_vol = result_min_vol.x
            std_min_vol, returns_min_vol = portfolio_performance(optimal_weights_min_vol, mean_returns, cov_matrix)

            # Portafolio Equilibrado (proporciones iguales)
            balanced_weights = np.array([1. / num_assets] * num_assets)
            std_balanced, returns_balanced = portfolio_performance(balanced_weights, mean_returns, cov_matrix)

            # Portafolio más óptimo (mejor relación riesgo-rendimiento)
            target_returns = np.linspace(returns_min_vol, mean_returns.max(), 50)
            frontier_volatility = []
            
            for ret in target_returns:
                constraints = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1] - ret},
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                result_optimal = minimize(minimize_volatility, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                frontier_volatility.append(result_optimal.fun)
            
            optimal_index = np.argmin(frontier_volatility)
            optimal_ret = target_returns[optimal_index]
            optimal_vol = frontier_volatility[optimal_index]
            
            constraints = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1] - optimal_ret},
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            result_opt = minimize(minimize_volatility, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_weights_opt = result_opt.x
            
            portfolios = [
                ('Mínima Volatilidad', optimal_weights_min_vol, returns_min_vol, std_min_vol),
                ('Portafolio Equilibrado', balanced_weights, returns_balanced, std_balanced),
                ('Portafolio Óptimo', optimal_weights_opt, optimal_ret, optimal_vol)
            ]
            
            st.write('### Tabla de Portafolios')
            for name, weights, ret, vol in portfolios:
                st.write(f'#### {name}')
                portfolio_data = {
                    'Empresa': selected_tickers,
                    'Peso (%)': [weight * 100 for weight in weights],
                    'Número de acciones': [investment_amount * weight / stock_data[ticker][-1] for ticker, weight in zip(selected_tickers, weights)],
                    'Monto invertido ($)': [investment_amount * weight for weight in weights]
                }
                df_portfolio = pd.DataFrame(portfolio_data)
                st.write(df_portfolio)
                
                # Resumen de rendimiento y volatilidad en tabla
                summary_data = {
                    'Período': ['1 año', '3 años', '5 años', '10 años'],
                    'Retorno Esperado (%)': [annualize_performance(ret, vol, period=252 * years)[0] * 100 for years in [1, 3, 5, 10]],
                    'Volatilidad Esperada (%)': [annualize_performance(ret, vol, period=252 * years)[1] * 100 for years in [1, 3, 5, 10]]
                }
                df_summary = pd.DataFrame(summary_data)
                st.write('Resumen de Retorno y Volatilidad')
                st.write(df_summary.set_index('Período'))
            
            # Graficar comportamiento histórico de la inversión
            fig = make_subplots()
            for name, weights, ret, vol in portfolios:
                investment_values = calculate_investment_value(investment_amount, ret, weights, stock_data)
                fig.add_trace(go.Scatter(x=investment_values.index, y=investment_values, mode='lines', name=name))
            
            fig.update_layout(title='Comportamiento Histórico de la Inversión',
                            xaxis_title='Fecha',
                            yaxis_title='Valor de la Inversión ($)',
                            hovermode='x unified')
            
            st.plotly_chart(fig)
        else:
            st.write('Por favor, seleccione al menos una empresa y asegúrese de que la fecha de inicio sea anterior a la fecha de fin.')























# Verificar si todos los estados de los botones son False
if not (st.session_state['show_profile'] or st.session_state['show_general'] or st.session_state['show_fundamental'] or st.session_state['show_technical'] or st.session_state['show_forecast'] or st.session_state['show_portafolio']):
    # Ruta local a la imagen
    img_url2 = "/Users/israelgarciaruiz/Documents/GitHub/Proyecto-MBD/Fondo.png"
    # img_url2 = "https://github.com/G-R-ISRAEL/Proyecto-MBD/blob/main/Fondo.png?raw=true"
    st.image(str(img_url2), caption="© 2024 EasyShare", width=700)
    
    st.divider()

    st.title('Integrantes:')
             
    # Definir las columnas para los botones
    col1, col2, col3, col4, col5 = st.columns(5)

    # Guardar el estado de los botones
    if 'Israel Garcia Ruiz' not in st.session_state:
        st.session_state['Israel Garcia Ruiz'] = False
    if 'Miguel Angel Rodríguez Huerta' not in st.session_state:
        st.session_state['Miguel Angel Rodríguez Huerta'] = False
    if 'Alfredo Estrada Mata' not in st.session_state:
        st.session_state['Alfredo Estrada Mata'] = False
    if 'Miguel Alejandro Ramírez Solís' not in st.session_state:
        st.session_state['Miguel Alejandro Ramírez Solís'] = False
    if 'Luis Ariel Valadés Ramírez' not in st.session_state:
        st.session_state['Luis Ariel Valadés Ramírez'] = False

    # Botones en la misma fila
    if col1.button("Israel Garcia Ruiz", use_container_width=True):
        st.session_state['Israel Garcia Ruiz'] = True
        st.session_state['Miguel Angel Rodríguez Huerta'] = False
        st.session_state['Alfredo Estrada Mata'] = False
        st.session_state['Miguel Alejandro Ramírez Solís'] = False
        st.session_state['Luis Ariel Valadés Ramírez'] = False
        
    if col2.button("Miguel Angel Rodríguez Huerta", use_container_width=True):
        st.session_state['Israel Garcia Ruiz'] = False
        st.session_state['Miguel Angel Rodríguez Huerta'] = True
        st.session_state['Alfredo Estrada Mata'] = False
        st.session_state['Miguel Alejandro Ramírez Solís'] = False
        st.session_state['Luis Ariel Valadés Ramírez'] = False
        
    if col3.button("Alfredo Estrada Mata", use_container_width=True):
        st.session_state['Israel Garcia Ruiz'] = False
        st.session_state['Miguel Angel Rodríguez Huerta'] = False
        st.session_state['Alfredo Estrada Mata'] = True
        st.session_state['Miguel Alejandro Ramírez Solís'] = False
        st.session_state['Luis Ariel Valadés Ramírez'] = False
        
    if col4.button("Miguel Alejandro Ramírez Solís", use_container_width=True):
        st.session_state['Israel Garcia Ruiz'] = False
        st.session_state['Miguel Angel Rodríguez Huerta'] = False
        st.session_state['Alfredo Estrada Mata'] = False
        st.session_state['Miguel Alejandro Ramírez Solís'] = True
        st.session_state['Luis Ariel Valadés Ramírez'] = False
        
    if col5.button("Luis Ariel Valadés Ramírez", use_container_width=True):
        st.session_state['Israel Garcia Ruiz'] = False
        st.session_state['Miguel Angel Rodríguez Huerta'] = False
        st.session_state['Alfredo Estrada Mata'] = False
        st.session_state['Miguel Alejandro Ramírez Solís'] = False
        st.session_state['Luis Ariel Valadés Ramírez'] = True

st.divider()

st.markdown("<h4 style='text-align: center;'>© 2024 EasyShare</h4>", unsafe_allow_html=True)