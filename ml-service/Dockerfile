# ==============================================================================
# DOCKERFILE (ml-service)
# ==============================================================================
# Este archivo constituye la receta de construcción paso a paso para compilar la imagen
# del servicio de Machine Learning. Define un entorno Linux aislado y autocontenido,
# garantizando la reproducibilidad absoluta de la solución. Elimina de raíz el
# clásico problema de "en mi máquina funciona, pero en el entorno de evaluación no".
#
# Importancia Crítica para MLOps y la Solución End-to-End:
#   Actúa como el eslabón de empaquetado operativo. Convierte tus scripts de entrenamiento
#   y servicio (train.py y app.py) junto con sus dependencias exactas en una sola
#   unidad portable (contenedor) capaz de coordinarse con la base de datos PostgreSQL
#   y el dashboard en Streamlit mediante Docker Compose.
#
# Estrategia de Optimización de Capas (Docker Layer Caching):
#   El orden de las instrucciones está estructurado científicamente para maximizar la velocidad 
#   de reconstrucción. Se aíslan las dependencias pesadas (pip install) ANTES de copiar
#   el código fuente. Si modificas un print o un comentario en tus scripts, Docker reutilizará 
#   la caché de las librerías instaladas en segundos, evitando descargarlas de nuevo.
# ==============================================================================

# 1. IMAGEN BASE (Operating System & Runtime Environment)
# Selecciona la plantilla oficial optimizada para desarrollo estructurado en Python.
# Provee el intérprete 3.11 sobre una distribución Linux ligera, segura y estable.
FROM python:3.11

# 2. DIRECTORIO DE TRABAJO (Workspace Isolation)
# Crea y define la carpeta raíz absoluta dentro del sistema de archivos del contenedor.
# Toda instrucción posterior (COPY, RUN, CMD) se ejecutará posicionada físicamente aquí.
WORKDIR /app

# 3. AISLAMIENTO DE DEPENDENCIAS (Cache Pre-loading)
# Copia únicamente el manifiesto de librerías al directorio actual de trabajo interno.
# Este paso es aislado para indicarle al motor de Docker que si este archivo no cambia,
# la costosa instalación de paquetes de la siguiente capa puede ser totalmente cacheada.
COPY requirements.txt .

# 4. INSTALACIÓN DE LIBRERÍAS (Environment Provisioning)
# Descarga y monta los paquetes analíticos y web (Pandas, Scikit-Learn, FastAPI, Uvicorn, etc.).
# Garantiza que el contenedor posea exactamente el mismo ecosistema matemático que el equipo de desarrollo.
RUN pip install -r requirements.txt

# 5. INYECCIÓN DEL CÓDIGO FUENTE (Application Deployment)
# Copia todo el contenido del directorio local (scripts, configuraciones, directorios raíz) al contenedor.
# Nota: Se ejecuta después de instalar librerías para evitar invalidar la caché del paso anterior 
# cada vez que edites o guardes una línea de código en tus archivos .py.
COPY . .

# 6. DOCUMENTACIÓN DE PUERTOS (Network Interface Mapping)
# Informa explícitamente al motor de Docker y a los desarrolladores que el proceso 
# de la API web interna escuchará peticiones de red a través del puerto especificado.
EXPOSE 8000

# 7. FILOSOFÍA DE EJECUCIÓN MAESTRA (Sequential Entrypoint Execution)
# Define la tarea principal e inmutable que ejecutará el contenedor de forma automática al inicializarse.
#
# Lógica de Cortocircuito Operacional (Operador &&):
#   1. 'python train.py': Al encender el contenedor, arranca el pipeline de Machine Learning. 
#      Se conecta a Postgres, calcula el K-Óptimo (PCA + Silhouette) y guarda los artefactos (.pkl).
#   2. '&&': Barrera de seguridad crítica. Si el script de entrenamiento llegase a fallar por base de 
#      datos o código, el operador detiene el flujo, evitando levantar una API vacía sin modelos.
#   3. 'uvicorn app:app...': Si y solo si train.py finaliza con éxito (exit code 0), levanta de 
#      inmediato el servidor FastAPI en el host 0.0.0.0 para escuchar tráfico del Dashboard.
CMD python train.py && \
    uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000