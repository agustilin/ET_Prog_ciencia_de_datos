# ==============================================================================
# DOCKERFILE: Configuración del Contenedor para el Dashboard de Streaming
# ==============================================================================
# Este archivo define el entorno virtualizado para ejecutar la aplicación de 
# Streamlit. Utiliza una imagen base oficial de Python 3.11, establece el 
# directorio de trabajo, gestiona la caché de capas de Docker copiando e 
# instalando primero las dependencias, incluye el código fuente de la app, 
# e instruye al contenedor a exponer el puerto por defecto de Streamlit (8501) 
# garantizando que sea accesible de manera externa.
# ==============================================================================

# Define la imagen base oficial de Python (versión 3.11) optimizada para producción
FROM python:3.11

# Establece el directorio de trabajo interno del contenedor donde se ejecutará la app
WORKDIR /app

# Copia únicamente el archivo de dependencias al directorio actual del contenedor
COPY requirements.txt .

# Ejecuta el gestor de paquetes pip para instalar las librerías necesarias sin usar caché (ahorra espacio)
RUN pip install --no-cache-dir -r requirements.txt

# Copia la totalidad del código fuente y archivos del proyecto local hacia el contenedor
COPY . .

# Informa a Docker que el contenedor escuchará en el puerto 8501 en tiempo de ejecución
EXPOSE 8501

# Define el comando por defecto para iniciar Streamlit, vinculando la dirección a 0.0.0.0 para acceso externo
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]