"""
Script principal que ejecuta todo el pipeline ETL + EDA + Estadísticas de una sola vez.
Versión con manejo robusto de rutas cuando se ejecuta desde fuera de src.
"""

# Antes de lanzar el script, debemos de asegurarnos de tener instaladas las
# dependencias necesarias.
# Para ello, ejecutaremos el siguiente comando en la terminal:
# pip install -r requirements.txt

# Importar librerías necesarias
import os
import time
import sys
from pathlib import Path

# Configurar rutas base
ROOT_DIR = Path(__file__).parent.resolve()  # Directorio donde está main.py
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

# Añadir src al path para imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class PipelineManager:
    def __init__(self):
        self.rutas = {
            'root': ROOT_DIR,
            'data': DATA_DIR,
            'raw': DATA_DIR / "raw",
            'processed': DATA_DIR / "processed",
            'src': SRC_DIR
        }
        self.archivos_fuente_requeridos = [
            "Datos_Renta_Municipios.csv",
            "Municipios_-3498985394491007638.geojson"
        ]

        # Crear directorios si no existen
        os.makedirs(self.rutas['raw'], exist_ok=True)
        os.makedirs(self.rutas['processed'], exist_ok=True)

    def verificar_archivos_fuente(self):
        """Verifica que existan todos los archivos fuente requeridos."""
        faltantes = [
            archivo for archivo in self.archivos_fuente_requeridos 
            if not (self.rutas['raw'] / archivo).exists()
        ]

        if faltantes:
            print("\n❌ Archivos fuente faltantes:")
            for archivo in faltantes:
                print(f"- {archivo}")
            print(f"\nColócalos en: {self.rutas['raw']}")
            return False
        return True

    def ejecutar_etl(self):
        """Ejecuta el proceso ETL y verifica los archivos generados."""
        print("\n[ETL] Iniciando proceso de Extracción, Transformación y Carga...")
        start_time = time.time()

        try:
            # Cambiar al directorio src para ejecutar ETL
            os.chdir(self.rutas['src'])
            from etl import main as etl_main
            etl_main()
            os.chdir(ROOT_DIR)  # Volver al directorio original
        except Exception as e:
            print(f"\n❌ Error en ETL: {str(e)}")
            return False

        # Verificar archivos generados
        archivos_esperados = [
            "resultados_elecciones.sqlite",
            "Elecciones_Consolidadas.sqlite",
            "Datos_Renta_Municipios.sqlite",
            "Renta_y_Participacion.sqlite"
        ]

        faltantes = [
            archivo for archivo in archivos_esperados 
            if not (self.rutas['processed'] / archivo).exists()
        ]

        if faltantes:
            print("\n❌ El ETL no generó los archivos esperados:")
            for archivo in faltantes:
                print(f"- {archivo}")
            return False

        print(f"\n✅ ETL completado en {time.time() - start_time:.2f} segundos")
        return True

    def ejecutar_eda(self):
        """Ejecuta el análisis exploratorio de datos."""
        print("\n[EDA] Iniciando Análisis Exploratorio de Datos...")
        start_time = time.time()

        try:
            # Configurar variable de entorno con la ruta de datos
            os.environ['DATA_PATH'] = str(self.rutas['processed'])
            from eda import main as eda_main
            eda_main()
        except Exception as e:
            print(f"\n❌ Error en EDA: {str(e)}")
            return False

        print(f"\n✅ EDA completado en {time.time() - start_time:.2f} segundos")
        return True

    def ejecutar_estadisticas(self):
        """Ejecuta el análisis estadístico."""
        print("\n[ESTADÍSTICAS] Generando análisis estadístico...")
        start_time = time.time()

        try:
            from stats import main as stats_main
            stats_main()
        except Exception as e:
            print(f"\n❌ Error en Estadísticas: {str(e)}")
            return False

        print(f"\n✅ Estadísticas completadas en {time.time() - start_time:.2f} segundos")
        return True

    def ejecutar(self):
        """Ejecuta todo el pipeline."""
        print("\n" + "="*50)
        print(" INICIANDO PIPELINE COMPLETO ".center(50, '='))
        print("="*50)

        if not self.verificar_archivos_fuente():
            sys.exit(1)

        if not self.ejecutar_etl():
            sys.exit(1)

        if not self.ejecutar_eda():
            sys.exit(1)

        if not self.ejecutar_estadisticas():
            sys.exit(1)

        print("\n" + "="*50)
        print(" PIPELINE COMPLETADO CON ÉXITO ".center(50, '='))
        print("="*50)

def main():
    try:
        pipeline = PipelineManager()
        pipeline.ejecutar()
    except KeyboardInterrupt:
        print("\n🚫 Pipeline interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
