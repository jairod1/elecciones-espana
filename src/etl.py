"""
Script ETL para procesar datos de elecciones generales españolas y renta por municipios.
Combina datos de PDFs, XLSX y CSV en bases de datos SQLite estructuradas.
"""

import os
import re
import time
import math
import sqlite3
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path


def configurar_rutas():
    """Configura las rutas de directorios del proyecto."""
    ROOT_DIR = Path().resolve()
    while not (ROOT_DIR / 'data').exists() and ROOT_DIR != ROOT_DIR.parent:
        ROOT_DIR = ROOT_DIR.parent
    
    return {
        'root': ROOT_DIR,
        'raw': ROOT_DIR / 'data' / 'raw',
        'processed': ROOT_DIR / 'data' / 'processed'
    }


def procesar_pdfs_elecciones(pdf_dir):
    """
    Procesa PDFs de resultados electorales y genera un DataFrame unificado.
    
    Args:
        pdf_dir (Path): Directorio con los PDFs de resultados electorales
        
    Returns:
        pd.DataFrame: DataFrame con los datos consolidados
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    
    def extraer_datos(pdf_path):
        doc = fitz.open(pdf_path)
        texto = "\n".join(page.get_text() for page in doc)

        # Extraer mes y año
        match_fecha = re.search(r"Congreso\s*:\s*(\w+)\s+(\d{4})", texto)
        mes = match_fecha.group(1).capitalize() if match_fecha else None
        año = match_fecha.group(2) if match_fecha else None
        año_int = int(año) if año else 0

        # Lista de datos generales a buscar
        patrones_datos = [
            "Población",
            "Votantes a las 14:00",
            "Votantes a las 18:00",
            "Votantes a las 20:00",
            "Total censo electoral",
            "Votantes",
            "Abstenciones",
        ]
        patron = rf"({'|'.join(patrones_datos)})\s*\n?([\d\.]+)"
        datos = re.findall(patron, texto)

        # Extraer porcentajes
        primera_pagina_texto = doc[0].get_text()

        match_ultimo_num = list(re.finditer(r"\b\d{1,2}\b\s*$", primera_pagina_texto, re.MULTILINE))
        if match_ultimo_num:
            inicio_bloque = match_ultimo_num[-1].end()
            bloque_final = primera_pagina_texto[inicio_bloque:]
        else:
            bloque_final = ""

        porcentajes = re.findall(r"(\d{1,3}[.,]?\d{1,2})%", bloque_final)
        porcentajes = [float(p.replace(",", ".")) for p in porcentajes]

        # Asignar datos - CORRECCIÓN AQUÍ
        datos_finales = []
        asignar_pct = False
        pct_index = 0

        for clave, valor in datos:
            porcentaje = None

            if "Votantes a las 14:00" in clave:
                asignar_pct = True

            if asignar_pct:
                if clave.strip() == "Total censo electoral":
                    if año_int in [1979, 1982]:
                        porcentaje = None
                    else:
                        porcentaje = None
                        pct_index += 1
                else:
                    if pct_index < len(porcentajes):
                        porcentaje = porcentajes[pct_index]
                        pct_index += 1

            datos_finales.append({
                "Mes": mes,
                "Año": año,
                "Dato": clave.strip(),
                "Valor": int(valor.replace(".", "")),
                "Porcentaje": porcentaje
            })

        return pd.DataFrame(datos_finales)

    # Procesar todos los PDFs
    dfs = [extraer_datos(path) for path in pdf_paths]
    return pd.concat(dfs, ignore_index=True)


def exportar_resultados_sqlite(df, output_path):
    """
    Exporta los resultados electorales a SQLite con información adicional.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos electorales
        output_path (Path): Ruta del archivo SQLite de salida
    """
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Guardar en tabla
    df.to_sql("resultados_generales", conn, if_exists="replace", index=False)

    # Añadir columnas adicionales
    try:
        cursor.execute('ALTER TABLE resultados_generales ADD COLUMN "¿GobiernoFormado?" TEXT')
    except Exception:
        pass

    try:
        cursor.execute('ALTER TABLE resultados_generales ADD COLUMN PartidoFormadorGobierno TEXT')
    except Exception:
        pass

    # Actualizar datos
    cursor.execute('UPDATE resultados_generales SET "¿GobiernoFormado?" = "Si"')
    cursor.execute('UPDATE resultados_generales SET "¿GobiernoFormado?" = "No" WHERE Año = 2016 OR Mes = "Abril"')

    cursor.execute("UPDATE resultados_generales SET PartidoFormadorGobierno = 'UCD' WHERE Año = 1979")
    cursor.execute("UPDATE resultados_generales SET PartidoFormadorGobierno = 'PSOE' WHERE Año IN (1982, 1986, 1989, 1993, 2004, 2008, 2023)")
    cursor.execute("UPDATE resultados_generales SET PartidoFormadorGobierno = 'PSOE' WHERE Año = 2019 AND Mes = 'Noviembre'")
    cursor.execute("UPDATE resultados_generales SET PartidoFormadorGobierno = 'PP' WHERE Año IN (1996, 2000, 2011, 2015)")

    conn.commit()
    conn.close()


def procesar_xlsx_municipios(xlsx_dir):
    """
    Procesa archivos XLSX con resultados por municipio.
    
    Args:
        xlsx_dir (Path): Directorio con los XLSX de resultados por municipio
        
    Returns:
        pd.DataFrame: DataFrame con los datos consolidados
    """
    xlsx_paths = sorted(xlsx_dir.glob("Generales_por_Municipio_*.xlsx"))
    
    def leer_archivos(archivos):
        dfs = []
        for archivo in archivos:
            try:
                df = pd.read_excel(archivo, engine='openpyxl', header=None)
                dfs.append(df)
            except Exception as e:
                print(f"Error en {archivo}: {str(e)}")
                continue

        df_total = pd.concat(dfs, ignore_index=True)

        # Insertar columnas de año y mes
        df_total.insert(0, "Año", None)
        df_total.insert(1, "Mes", None)

        # Detectar y propagar fechas
        año_actual = mes_actual = None
        for i in range(len(df_total)):
            valor = str(df_total.iloc[i, 2])
            if "congreso |" in valor.lower():
                partes = valor.split("|")
                if len(partes) >= 2:
                    fecha = partes[1].strip().split()
                    if len(fecha) == 2:
                        mes_actual = fecha[0].capitalize()
                        año_actual = fecha[1]

            if año_actual and mes_actual:
                df_total.at[i, "Año"] = año_actual
                df_total.at[i, "Mes"] = mes_actual

        df_total["Año"] = df_total["Año"].ffill()
        df_total["Mes"] = df_total["Mes"].ffill()

        return df_total

    return leer_archivos(xlsx_paths)


def exportar_municipios_sqlite(df, output_path):
    """
    Exporta los resultados por municipio a SQLite con columnas renombradas.
    Calcula y añade la columna de participación (Total_Votantes/Total_Censo_Electoral * 100).
    
    Args:
        df (pd.DataFrame): DataFrame con los datos por municipio
        output_path (Path): Ruta del archivo SQLite de salida
    """
    # --- CÁLCULO DE PARTICIPACIÓN ---
    def a_float(val):
        """Convierte valores a float, manejando formatos de texto."""
        if pd.isna(val):
            return None
        if isinstance(val, str):
            val = val.replace(".", "").replace(",", ".")
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # Calcular participación antes de exportar
    participacion = []
    for _, row in df.iterrows():
        try:
            censo = a_float(row[7])  # Total_Censo_Electoral
            votantes = a_float(row[8])  # Total_Votantes
            if censo and censo > 0 and votantes is not None:
                participacion.append(round((votantes / censo) * 100, 2))
            else:
                participacion.append(None)
        except Exception:
            participacion.append(None)

    # Insertar columna de participación en el DataFrame
    df.insert(11, "Participación", participacion)

    # --- EXPORTACIÓN A SQLITE ---
    conn = sqlite3.connect(output_path)
    
    # Exportar DataFrame con la nueva columna
    df.to_sql("resultados_municipales", conn, if_exists="replace", index=False)
    
    # Renombrar columnas numéricas
    column_renames = {
        "0": "Nombre_de_Comunidad",
        "1": "Codigo_de_Provincia",
        "2": "Nombre_de_Provincia",
        "3": "Codigo_de_Municipio",
        "4": "Nombre_de_Municipio",
        "5": "Poblacion",
        "6": "Numero_de_Mesas",
        "7": "Total_Censo_Electoral",
        "8": "Total_Votantes",
        # La columna 11 ya tiene el nombre correcto "Participación"
    }
    
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(resultados_municipales);")
    columnas_sql = [row[1] for row in cursor.fetchall()]
    
    for vieja, nueva in column_renames.items():
        if vieja in columnas_sql:
            try:
                cursor.execute(f'ALTER TABLE resultados_municipales RENAME COLUMN "{vieja}" TO "{nueva}"')
            except Exception:
                pass
    
    conn.commit()
    conn.close()
    
def procesar_renta_municipios(csv_path):
    """
    Procesa el CSV de renta por municipios y lo exporta a SQLite.
    
    Args:
        csv_path (Path): Ruta al archivo CSV de renta
        
    Returns:
        pd.DataFrame: DataFrame con los datos limpios de renta
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"El archivo CSV no existe: {csv_path}")
    
    # Leer y limpiar datos
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    conn = sqlite3.connect(csv_path.parent.parent / 'processed' / 'Datos_Renta_Municipios.sqlite')
    
    # Exportar y limpiar
    df.to_sql("Datos_Renta_Municipios", conn, if_exists='replace', index=False)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM Datos_Renta_Municipios WHERE Distritos IS NOT NULL;")
    conn.commit()
    
    # Leer solo columnas útiles
    query = """
        SELECT Municipios, [Indicadores de renta media], Periodo, Total
        FROM Datos_Renta_Municipios
    """
    df_limpio = pd.read_sql(query, conn)
    conn.close()
    
    return df_limpio


def crear_columna_municipio_limpio():
    """Crea la columna Municipio_Limpio en ambas bases de datos."""
    paths = configurar_rutas()
    
    # Conectar a las bases de datos
    conn_renta = sqlite3.connect(paths['processed'] / 'Datos_Renta_Municipios.sqlite')
    conn_elecciones = sqlite3.connect(paths['processed'] / 'Elecciones_Consolidadas.sqlite')

    # Procesar datos de renta
    conn_renta.execute("""
        CREATE TEMPORARY VIEW IF NOT EXISTS Renta_Limpio AS
        SELECT *,
               UPPER(TRIM(SUBSTR(Municipios, INSTR(Municipios, ' ') + 1))) AS Municipio_Limpio
        FROM Datos_Renta_Municipios;
    """)
    conn_renta.execute("DROP TABLE IF EXISTS Datos_Renta_Municipios_Cleaned;")
    conn_renta.execute("""
        CREATE TABLE Datos_Renta_Municipios_Cleaned AS
        SELECT * FROM Renta_Limpio;
    """)
    conn_renta.execute("DROP VIEW IF EXISTS Renta_Limpio;")

    # Procesar datos de elecciones
    conn_elecciones.execute("""
        CREATE TEMPORARY VIEW IF NOT EXISTS Elecciones_Limpio AS
        SELECT *,
               UPPER(TRIM(Nombre_de_Municipio)) AS Municipio_Limpio
        FROM resultados_municipales;
    """)
    conn_elecciones.execute("DROP TABLE IF EXISTS resultados_municipales_Cleaned;")
    conn_elecciones.execute("""
        CREATE TABLE resultados_municipales_Cleaned AS
        SELECT * FROM Elecciones_Limpio;
    """)
    conn_elecciones.execute("DROP VIEW IF EXISTS Elecciones_Limpio;")

    conn_renta.commit()
    conn_elecciones.commit()
    conn_renta.close()
    conn_elecciones.close()


def combinar_renta_participacion():
    """Combina datos de renta y participación electoral en una nueva base de datos."""
    paths = configurar_rutas()
    
    # Conectar a las bases de datos
    conn_renta = sqlite3.connect(paths['processed'] / 'Datos_Renta_Municipios.sqlite')
    conn_elecciones = sqlite3.connect(paths['processed'] / 'Elecciones_Consolidadas.sqlite')
    output_path = paths['processed'] / 'Renta_y_Participacion.sqlite'

    # Extraer datos de elecciones
    query_elecciones = """
    SELECT 
        Municipio_Limpio AS Municipio,
        Año,
        Mes AS Mes_Elecciones,
        Participación
    FROM resultados_municipales_Cleaned
    WHERE Municipio_Limpio IS NOT NULL AND Participación IS NOT NULL
    """
    df_elecciones = pd.read_sql_query(query_elecciones, conn_elecciones)
    
    # Convertir participación a float
    if df_elecciones['Participación'].dtype == 'object':
        df_elecciones['Participación'] = pd.to_numeric(
            df_elecciones['Participación'].str.replace(",", "."), errors='coerce'
        )

    # Extraer datos de renta
    query_renta = """
    SELECT 
        Municipio_Limpio,
        "Periodo" AS Año,
        "Indicadores de renta media" AS Indicador,
        Total
    FROM Datos_Renta_Municipios_Cleaned
    WHERE "Indicadores de renta media" IN (
        'Renta neta media por persona',
        'Mediana de la renta por unidad de consumo'
    )
    """
    df_renta = pd.read_sql_query(query_renta, conn_renta)

    # Pivotear datos de renta
    df_renta_pivot = df_renta.pivot_table(
        index=['Municipio_Limpio', 'Año'],
        columns='Indicador',
        values='Total',
        aggfunc='first'
    ).reset_index()
    df_renta_pivot.columns.name = None
    df_renta_pivot = df_renta_pivot.rename(columns={
        'Municipio_Limpio': 'Municipio',
        'Renta neta media por persona': 'Renta_Media',
        'Mediana de la renta por unidad de consumo': 'Renta_Mediana_Hogar'
    })

    # Limpiar valores de renta
    def limpiar_valor(val):
        if isinstance(val, str):
            val = val.replace(",", ".").replace(" ", "")
        return pd.to_numeric(val, errors='coerce') * 1000  # Convertir a euros

    df_renta_pivot['Renta_Media'] = df_renta_pivot['Renta_Media'].apply(limpiar_valor)
    df_renta_pivot['Renta_Mediana_Hogar'] = df_renta_pivot['Renta_Mediana_Hogar'].apply(limpiar_valor)

    # Combinar datos
    df_elecciones['Año'] = pd.to_numeric(df_elecciones['Año'], errors='coerce')
    df_renta_pivot['Año'] = pd.to_numeric(df_renta_pivot['Año'], errors='coerce')
    df_combinado = pd.merge(df_elecciones, df_renta_pivot, on=['Municipio', 'Año'], how='inner')

    # Calcular cuartiles por año
    def asignar_cuartiles_por_año(grupo):
        grupo['Cuartil_Renta_Media'] = pd.qcut(grupo['Renta_Media'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        grupo['Cuartil_Renta_Mediana'] = pd.qcut(grupo['Renta_Mediana_Hogar'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        grupo['Cuartil_Participacion'] = pd.qcut(grupo['Participación'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        return grupo

    df_combinado = df_combinado.groupby('Año', group_keys=False).apply(asignar_cuartiles_por_año)

    # Reordenar y guardar
    df_final = df_combinado[[
        'Municipio', 'Año', 'Mes_Elecciones', 'Participación',
        'Renta_Media', 'Renta_Mediana_Hogar',
        'Cuartil_Renta_Media', 'Cuartil_Renta_Mediana', 'Cuartil_Participacion'
    ]]

    conn_output = sqlite3.connect(output_path)
    df_final.to_sql("Renta_y_Participacion", conn_output, if_exists="replace", index=False)
    conn_output.close()
    conn_elecciones.close()
    conn_renta.close()


def main():
    """Función principal que ejecuta todo el pipeline ETL."""
    print("Iniciando proceso ETL...")
    rutas = configurar_rutas()
    
    # 1. Procesar PDFs de elecciones generales
    print("\nProcesando PDFs de resultados electorales...")
    df_pdf = procesar_pdfs_elecciones(rutas['raw'])
    exportar_resultados_sqlite(df_pdf, rutas['processed'] / "resultados_elecciones.sqlite")
    
    # 2. Procesar XLSX por municipio
    print("\nProcesando XLSX de resultados por municipio... (esto puede tardar)...")
    df_xlsx = procesar_xlsx_municipios(rutas['raw'])
    exportar_municipios_sqlite(df_xlsx, rutas['processed'] / "Elecciones_Consolidadas.sqlite")
    
    # 3. Procesar CSV de renta
    print("\nProcesando CSV de renta por municipio...")
    df_renta = procesar_renta_municipios(rutas['raw'] / "Datos_Renta_Municipios.csv")
    
    # 4. Crear columna Municipio_Limpio
    print("\nCreando columna Municipio_Limpio...")
    crear_columna_municipio_limpio()
    
    # 5. Combinar renta y participación
    print("\nCombinando datos de renta y participación...")
    combinar_renta_participacion()
    
    print("\n✅ Proceso ETL completado con éxito!")


if __name__ == "__main__":
    main()