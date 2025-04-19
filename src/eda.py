import os
import sqlite3
import pandas as pd
from pathlib import Path

# Obtener la ruta al directorio de datos desde la variable de entorno
data_path = Path(os.getenv("DATA_PATH", "../data/processed"))

def conectar_bd(ruta_relativa):
    """
    Conecta a una base de datos SQLite y verifica su existencia.
    
    Args:
        ruta_relativa (Path): Ruta al archivo .sqlite
        
    Returns:
        sqlite3.Connection: Conexión a la base de datos
    """
    if not Path(ruta_relativa).exists():
        raise FileNotFoundError(f"❌ Archivo no encontrado: {ruta_relativa}")
    return sqlite3.connect(ruta_relativa)

def mostrar_tablas(conn):
    """
    Muestra las tablas disponibles en la base de datos.
    
    Args:
        conn (sqlite3.Connection): Conexión a la base de datos
    """
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("\nTablas disponibles:")
    print(tables)

def eda_resultados_generales():
    """
    Realiza EDA sobre los resultados de elecciones generales.
    """
    print("\n" + "="*50)
    print("EDA Nº1 - Resultados de elecciones generales")
    print("="*50)
    
    ruta = data_path / "resultados_elecciones.sqlite"
    conn = conectar_bd(ruta)
    mostrar_tablas(conn)
    
    query = "SELECT * FROM resultados_generales"
    df = pd.read_sql(query, conn)
    
    print("\nPrimeras filas:")
    print(df.head())
    print("\nÚltimas filas:")
    print(df.tail())
    print("\nDimensiones del DataFrame:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    df_porcentajes = df[df['Dato'].isin(['Votantes a las 14:00', 'Votantes a las 18:00', 'Votantes'])]
    stats = df_porcentajes.groupby('Dato')['Porcentaje'].describe()
    print("\nEstadísticas de participación:")
    print(stats)
    
    conn.close()

def eda_resultados_municipales():
    """
    Realiza EDA sobre los resultados de elecciones por municipio.
    """
    print("\n" + "="*50)
    print("EDA Nº2 - Resultados por municipio")
    print("="*50)
    
    ruta = data_path / "Elecciones_Consolidadas.sqlite"
    conn = conectar_bd(ruta)
    mostrar_tablas(conn)
    
    query = "SELECT * FROM resultados_municipales"
    df = pd.read_sql(query, conn)
    
    print("\nPrimeras filas:")
    print(df.head(10))
    print("\nÚltimas filas:")
    print(df.tail())
    print("\nParticipación electoral:")
    print(df['Participación'])
    print("\nDimensiones del DataFrame:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    conn.close()

def eda_renta_municipios():
    """
    Realiza EDA sobre los datos de renta por municipio.
    """
    print("\n" + "="*50)
    print("EDA Nº3 - Renta por municipio")
    print("="*50)
    
    ruta = data_path / "Datos_Renta_Municipios.sqlite"
    conn = conectar_bd(ruta)
    mostrar_tablas(conn)
    
    query = "SELECT * FROM Datos_Renta_Municipios"
    df = pd.read_sql(query, conn)
    
    print("\nPrimeras filas:")
    print(df.head())
    print("\nÚltimas filas:")
    print(df.tail())
    print("\nIndicadores de renta únicos:")
    print(df['Indicadores de renta media'].unique())
    print("\nDimensiones del DataFrame:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nDescripción general:")
    print(df.describe())
    
    conn.close()

def eda_renta_participacion():
    """
    Realiza EDA sobre la relación entre renta y participación electoral.
    """
    print("\n" + "="*50)
    print("EDA Nº4 - Renta y participación electoral")
    print("="*50)
    
    ruta = data_path / "Renta_y_Participacion.sqlite"
    conn = conectar_bd(ruta)
    mostrar_tablas(conn)
    
    query = "SELECT * FROM Renta_y_Participacion"
    df = pd.read_sql(query, conn)
    
    if df['Participación'].dtype == 'object':
        df['Participación'] = pd.to_numeric(df['Participación'], errors='coerce')
    
    print("\nDimensiones del DataFrame:", df.shape)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("\nEstadísticas descriptivas:")
    print(df.describe(include='all'))
    
    print("\nDistribución por cuartiles de renta media:")
    print(df['Cuartil_Renta_Media'].value_counts().sort_index())
    print("\nDistribución por cuartiles de renta mediana:")
    print(df['Cuartil_Renta_Mediana'].value_counts().sort_index())
    
    print("\nMatriz de correlaciones:")
    print(df[['Participación', 'Renta_Media', 'Renta_Mediana_Hogar']].corr())
    
    conn.close()

def main():
    """
    Función principal que ejecuta todos los análisis EDA.
    """
    print("="*50)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*50)
    
    eda_resultados_generales()
    eda_resultados_municipales()
    eda_renta_municipios()
    eda_renta_participacion()
    
    print("\n" + "="*50)
    print("EDA COMPLETADO")
    print("="*50)

if __name__ == "__main__":
    main()