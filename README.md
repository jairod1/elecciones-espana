# 0 Descripción de hipótesis

En este trabajo práctico, haremos un analisis exploratorio de un predictor para elecciones generales en España.

Tenemos por cotejar dos hipótesis:

0) Hemos observado que, a primera vista, en las elecciones generales en las que más del 40% del censo electoral ha votado para las 14:00, la izquierda (PSOE) parece ser capaz de formar gobierno mucho más a menudo que la derecha (PP). De ser así, este sería un predictor valioso, que nos permitiría conocer anticipadamente los resultados de unas elecciones generales horas antes de que finalicen.

 Nuestra Hipótesis Nula (h0) en este caso, será la siguiente: Intentaremos demostrar que este dato -no- predice con certeza resultados electorales. Nuestra Hipótesis Alternativa (h1) será que si los predice.

   Para ello, utilizaremos coeficientes de correlacion y diagramas de dispersión.

-----------------------------------------------------------------------------------------------------------------------------------------
 
1) La segunda hipótesis es dependiente de la primera. Si en el anterior caso, h1 es cierta; es decir, si la izquierda gana más elecciones generales cuanta más gente participa antes de las 14:00, trataremos de correlacionarlo con el nivel económico de los diversos municipios de España.

 Nuestra Hipótesis Nula Prima (h0') será que NO existe una mayor participación en los municipios de menor renta en las elecciones generales en las que más del 40% de la población vota antes de las 14:00, ni en los resultados de dichas elecciones en particular. Nuestra Hipótesis Alternativa Prima (h1') será que si existe, y que hay mayor participación en los municipios de menor renta cuando más del 40% de la población vota antes de las 14:00.

 Observaremos también el comportamiento en estas elecciones de los municipios del cuartil superior de renta, y de los dos cuartiles medianos.

    Para ello, dividiremos los municipios españoles en cuartiles a partir de su renta, utilizando variables como a) la Renta neta media por persona, b) la Renta neta media por hogar o c) la Mediana de la renta por unidad de consumo.

    Sacaremos una media de las medias y medianas de cada uno de los cuartiles, y las relacionaremos con aquellas elecciones en las que formó gobierno la izquierda.

# 1 Proceso Pipeline

Llevaremos a cabo el proceso de transformación de datos de la siguiente manera:

0. Instalaremos el requirements.txt adjunto en la carpeta con el comando pip install -r requirements.txt

Al ejecutar el main.py:

1. etl.py: main.py lanzará primero el archivo etl.py, ubicado en src. Este archivo, de transformación de datos, hará lo siguiente,
   a. Leerá 15 pdfs (ubicados en la carpeta data/raw) de datos de elecciones generales en España, y extraerá de ellos datos y porcentajes electorales. Estos datos serán almacenados en una tabla SQL llamada "resultados_elecciones.sqlite", y servirán para comprobar la hipótesis h0.
   b. Leerá 15 archivos .xlsx de elecciones generales con resultados electorales concretos por municipio de España (ubicados en la carpeta data/raw), y extraerá de ellos datos de cada uno de esos municipios y elecciones, guardándolos en una tabla SQL llamada "Elecciones_Consolidadas.sqlite". Además, creará dos nuevas columnas de datos, una de porcentaje de participación por municipio y elección concreta, y otra de unificación de municipios, con el propósito de unir datos con los de "Datos_Renta_Municipios.sqlite".
   c. Leerá un archivo .csv de diversos datos de renta por municipio en España, y lo transformará en la tabla SQL "Datos_Renta_Municipios.sqlite", eliminando datos que no interesan para el proyecto. Además, añadirá una nueva columna de unificación de municipios, con el propósito de unir datos con los de "Elecciones_Consolidadas.sqlite".
   d. Hará una fusión de datos entre las tablas "Elecciones_Consolidadas.sqlite" y "Datos_Renta_Municipios.sqlite", de tal forma que se puedan consultar de forma unificada datos de participación electoral y de renta, para comprobar nuestra hipótesis h0'.
   Todas estas tablas SQLite serán guardadas en la carpeta data/processed.

2. eda.py: Una vez extraídos los datos, main.py lanzará el archivo eda.py, que hará una visualización de varios datos de las tablas SQL, para comprobar que todo está en orden.

3. stats.py: Una vez extraídos y visualizados en crudo los datos, los graficaremos en diversas visualizaciones estadísticas. Estas incluirán:
- En el terreno de la participación electoral:
   a. Dos gráficos de votantes, brutos y porcentuales, y sus respectivas correlaciones polinómicas a lo largo del tiempo.
   b. Un gráfico de votos y votantes a lo largo del tiempo coloreado con periodos electorales por partido ganador.
   c. Tres diagramas de barras correlacionando participación en un periodo y elecciones ganadas por partido.
   d. Tres mapas de calor de participación electoral.
   e. Dos correlaciones visuales de victorias por partido político.
- En el terreno de la participación por lugar y renta:
   f. Cuatro geomapas de distribución de renta (media y mediana) por municipio en España, en 2015 y 2019
   g. Dos geomapas de participación por municipio en España, en 2015 y 2019
   h. Tres diagramas de espaguetis de participación por municipio de España y renta (en cuartiles)
   i. Dos gráficos de densidad de participación electoral por municipios según renta

Con esto concluirá el proceso pipeline.

# 2. Notebooks y dashboards
En la carpeta 'notebooks', podremos visualizar los Jupyter Notebooks utilizados para realizar cada uno de los pasos del proceso Pipeline.

En la carpeta 'dashboards', incluiremos el archivo Proyecto1.pbix de PowerBI con las estadísticas generadas y varias transformaciones propias de la aplicación.
