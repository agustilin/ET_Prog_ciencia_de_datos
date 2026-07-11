# ET_Prog_ciencia_de_datos

## Descripción del proyecto

Este proyecto implementa un ecosistema de datos End-to-End para la segmentación de usuarios de una plataforma de streaming. La solución va más allá del simple modelado, integrando un pipeline robusto de Machine Learning con validación cruzada y despliegue automatizado.

La arquitectura orquestada integra:
- Múltiples fuentes de datos (CSV local y base de datos relacional PostgreSQL).
- Un pipeline ETL automatizado con tratamiento de valores atípicos (outliers).
- Entrenamiento de un modelo de aprendizaje no supervisado (K-Means) optimizado matemáticamente.
- Reducción de dimensionalidad (PCA) para visualización avanzada.
- Auditoría del modelo mediante un algoritmo supervisado (Random Forest) para validar la predictibilidad de los clústeres y extraer reglas de negocio interpretables.
- Exposición de inferencias a través de una API RESTful (FastAPI).
- Un dashboard interactivo (Streamlit) para el análisis dinámico del negocio.
- Contenerización completa y aislada mediante Docker y Docker Compose.

---

## Arquitectura de la solución

        usuarios_streaming.csv
                 |
                 v
            +---------------+
            |  ml-service   |
            |   train.py    |
            +---------------+
                 |  ^
        consulta |  | lee
        SQL a PG |  | CSV
                 |  |
        perfil_usuarios  
                 |
                 v
            +---------------+
            |  PostgreSQL   |
            | perfil_usuarios|
            +---------------+
                 |
                 v
  datos integrados, preprocesamiento y entrenamiento
                 |
                 v
            +---------------+      +--------------+
            |  FastAPI API  | ---> |  Streamlit   |
            |  ml-service   |      |  dashboard   |
            +---------------+      +--------------+

```

---

## Tecnologías utilizadas

### Lenguaje

* Python 3.11

### Machine Learning & Pipeline

* scikit-learn
* KMeans (Clustering)
* PCA (Reducción de Dimensionalidad)
* RandomForestClassifier (Modelo Supervisado de Validación)
* StandardScaler (Escalado de magnitudes)
* Silhouette Score y KneeLocator (Métricas de optimización matemática)

### Datos

* pandas
* NumPy
* PostgreSQL
* SQLAlchemy (psycopg2-binary)

### Despliegue y Backend

* FastAPI
* Uvicorn

### Visualización

* Streamlit
* Plotly
* matplotlib (Utilizado específicamente para renderizar y exportar el gráfico del árbol de decisión del modelo supervisado).

### Infraestructura

* Docker
* Docker Compose

---

## Estructura del proyecto

```text
Ev3_Prog_Ciencia_Datos/
│
├── docker-compose.yml
├── README.md
├── database/
│   ├── init.sql
│   └── perfil_usuarios.csv
├── data/
│   ├── outliers_comparacion.csv
│   ├── centroides.csv
│   ├── usuarios_data.csv
│   ├── usuarios_segmentados.csv
│   └── usuarios_streaming.csv
├── models/
│   ├── arbol_decision.png
│   ├── metricas.json
│   ├── modelo_kmeans.pkl
│   ├── modelo_rf.pkl
│   ├── pca.pkl
│   └── scaler.pkl
├── ml-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py
│   └── app.py
└── dashboard/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py

```

---

## Pipeline de Machine Learning (Entrenamiento)

El núcleo analítico reside en `ml-service/train.py`, el cual ejecuta el siguiente pipeline de forma secuencial y automatizada:

1. Extracción (Extract): Lectura del consumo desde CSV (`usuarios_streaming.csv`) y consulta SQL a PostgreSQL para extraer el perfilamiento.
2. Integración (Transform): Fusión de ambas fuentes mediante la llave primaria `id_cliente` (`usuarios_data.csv`).
3. Calidad de Datos: Identificación y tratamiento de Outliers (Clipping por Rangos Intercuartílicos - IQR), exportando un reporte del "Antes y Después".
4. Preprocesamiento: Escalado de variables numéricas utilizando `StandardScaler` para evitar sesgos de magnitud.
5. Búsqueda del K Óptimo: Iteración automatizada (K=2 a 10) utilizando `KneeLocator` sobre la inercia y validación con `Silhouette Score`.
6. Entrenamiento Principal: Ajuste del modelo K-Means con el K óptimo descubierto.
7. Reducción de Dimensionalidad: Aplicación de PCA (2 componentes) para permitir la visualización 2D de los clústeres multidimensionales.
8. Auditoría Supervisada e Interpretabilidad: Entrenamiento de un Random Forest Classifier utilizando los datos escalados como variables predictoras y los clústeres de K-Means como target. Se extrae un árbol de decisión representativo mediante `matplotlib` para explicar las reglas de negocio.
9. Serialización: Exportación física de los modelos, gráficos y métricas para su consumo por la API y el Dashboard.

---

## Artefactos Generados

Durante la ejecución del pipeline, el sistema genera dinámicamente los siguientes artefactos físicos en los volúmenes compartidos:

### Archivos de Datos (`/data`)

* `usuarios_segmentados.csv`: Dataset original enriquecido con la etiqueta del clúster final y los componentes PCA.
* `usuarios_data.csv`: Dataset crudo unificado (Merge entre CSV y BD) antes del escalado.
* `outliers_comparacion.csv`: Archivo de auditoría con la distribución de variables antes y después del clipping.
* `centroides.csv`: Coordenadas originales de los centroides para perfilamiento de negocio.

### Modelos y Visualizaciones (`/models`)

* `modelo_kmeans.pkl`: Modelo de clustering principal serializado.
* `modelo_rf.pkl`: Modelo supervisado (Random Forest) utilizado como proxy/auditor.
* `scaler.pkl`: Objeto de estandarización entrenado (vital para la inferencia en tiempo real).
* `pca.pkl`: Transformador de reducción de dimensionalidad entrenado.
* `arbol_decision.png`: Gráfico generado con `matplotlib` que visualiza las fronteras de decisión del modelo Random Forest.
* `metricas.json`: Diccionario centralizado que contiene el K óptimo, Silhouette Score, Varianza Explicada del PCA, y el Accuracy del validador.

---

## Ejecución del proyecto

### Requisitos previos

* Docker y Docker Compose instalados.

### Levantar la solución

Desde la raíz del proyecto, ejecuta:

```bash
docker compose up --build

```

Nota: Si se realizan cambios en `train.py` o los esquemas de base de datos, se recomienda ejecutar `docker compose down -v` antes de reconstruir para limpiar los volúmenes antiguos.

---

## Acceso a los servicios

### 1. Backend y API REST (FastAPI)

* URL: http://localhost:8000
* Documentación Swagger UI: http://localhost:8000/docs
* Endpoints principales:
* `GET /`: Health check del servicio.
* `GET /dashboard-data`: Retorna las métricas JSON, los centroides y datos integrados.
* `POST /predict`: Ingesta un JSON con datos de un nuevo usuario, aplica el `scaler.pkl` en memoria y devuelve la predicción del clúster en tiempo real.



### 2. Dashboard Interactivo (Streamlit)

* URL: http://localhost:8501
* Características principales:
* Panel de Métricas: Muestra el rendimiento del modelo no supervisado y la precisión del auditor Random Forest.
* Análisis de Calidad: Selector interactivo para visualizar diagramas de caja (Boxplots) demostrando el impacto del tratamiento de outliers.
* Interpretabilidad: Visualización del gráfico `arbol_decision.png` para traducir los clústeres matemáticos en reglas de negocio claras (If-Then).
* Mapa de Clústeres (PCA): Visualización bidimensional de la segmentación.
* Perfilamiento de Negocio: Tablas dinámicas extraídas desde el backend para la toma de decisiones basada en el promedio de cada segmento.



---

## Detener los servicios

Para bajar la arquitectura preservando los volúmenes:

```bash
docker compose down

```

Para realizar un reinicio limpio (destruye la base de datos temporal y artefactos generados):

```bash
docker compose down -v

```

"""

