"""API REST para servir los resultados del modelo de segmentación.

La aplicación expone endpoints para verificar el estado del servicio, entregar los
resultados históricos al dashboard y predecir el cluster de un nuevo usuario usando
los artefactos entrenados en el pipeline de Machine Learning.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException


def get_project_root() -> Path:
    """Obtiene la raíz del proyecto tanto en Docker como en ejecución local.

    Returns:
        Path: Objeto Path que apunta al directorio raíz para resolver rutas relativas.
    """
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "data").exists():
        return candidate
    return Path.cwd()


# ==========================================
# CONFIGURACIÓN DE INSTANCIAS Y DIRECTORIOS
# ==========================================
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Instanciación de FastAPI con metadatos estructurados para la documentación interactiva (/docs)
app = FastAPI(
    title="Servicio de Segmentación de Usuarios - Streaming",
    description="API REST para inferencia de clústeres y entrega de métricas de segmentación.",
    version="1.1.0",
)


def load_artifacts() -> tuple[pd.DataFrame, Any, Any, dict[str, Any], Any | None]:
    """Carga los artefactos más recientes escritos por el pipeline de entrenamiento.
    
    Esta función es el puente entre el entrenamiento (train.py) y el servicio activo.
    Captura la excepción FileNotFoundError para evitar que la API colapse si los 
    contenedores se levantan en frío y el modelo aún no se ha entrenado por primera vez.

    Returns:
        tuple:
            - usuarios (pd.DataFrame): Datos históricos de usuarios ya segmentados.
            - modelo (KMeans): Modelo K-Means serializado.
            - scaler (StandardScaler): Escalador con las medias y varianzas de entrenamiento.
            - metricas (dict): Metadatos, inercia, varianza explicada e hitos de negocio.
            - classifier (RandomForestClassifier | None): Modelo supervisado complementario.
    """
    try:
        # Recuperación de datos consolidados y binarios serializados
        usuarios = pd.read_csv(DATA_DIR / "usuarios_segmentados.csv")
        with open(MODELS_DIR / "modelo_kmeans.pkl", "rb") as handle:
            modelo = pickle.load(handle)
        with open(MODELS_DIR / "scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)
        with open(MODELS_DIR / "metricas.json", "r", encoding="utf-8") as handle:
            metricas = json.load(handle)

        # Carga opcional del clasificador de validación estructurada
        classifier = None
        if (MODELS_DIR / "classifier.pkl").exists():
            with open(MODELS_DIR / "classifier.pkl", "rb") as handle:
                classifier = pickle.load(handle)

        return usuarios, modelo, scaler, metricas, classifier
    except FileNotFoundError:
        # Fallback de contingencia ante falta de artefactos (Garantiza resiliencia)
        return pd.DataFrame(), None, None, {}, None


@app.get("/")
def inicio() -> dict[str, str]:
    """Endpoint de diagnóstico para verificar que la API está operativa.

    Returns:
        dict[str, str]: Mensaje de estado y código de confirmación.
    """
    return {"mensaje": "Servicio de Inferencia ML funcionando correctamente", "estado": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint de salud para monitoreo simple (Health Check).
    
    Utilizado por herramientas de orquestación como Docker o Kubernetes para 
    validar de forma automatizada si el contenedor sigue vivo.

    Returns:
        dict[str, str]: Estado actual del microservicio.
    """
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Entrega información resumida del modelo y del último entrenamiento.

    Permite consultar bajo demanda los metadatos de calidad del agrupamiento sin 
    necesidad de procesar todo el volumen de usuarios históricos.

    Returns:
        dict[str, Any]: Resumen de métricas incluyendo K óptimo, Silhouette y Takeaway.
    """
    _, _, _, metricas_actuales, _ = load_artifacts()
    return {
        "k_optimo": metricas_actuales.get("k_optimo"),
        "silhouette_score": metricas_actuales.get("silhouette_score"),
        "business_takeaway": metricas_actuales.get("business_takeaway"),
        "comparison_models": metricas_actuales.get("comparison_models", []),
    }


@app.get("/dashboard-data")
def dashboard_data() -> dict[str, Any]:
    """Entrega a Streamlit los usuarios segmentados, los centroides y las métricas.
    
    Este endpoint optimiza la comunicación de red (HTTP JSON) enviando en una única 
    petición todos los insumos históricos que el Dashboard necesita para renderizar 
    los reportes visuales de negocio.

    Returns:
        dict[str, Any]: Estructuras completas transformadas a diccionarios compatibles con JSON.
    """
    usuarios, _, _, metricas_actuales, _ = load_artifacts()
    
    # Lectura de centroides si están disponibles
    centroides = pd.read_csv(DATA_DIR / "centroides.csv") if (DATA_DIR / "centroides.csv").exists() else pd.DataFrame()
    
    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "metricas": metricas_actuales,
    }


@app.post("/predict")
def predict(datos: dict[str, Any]) -> dict[str, Any]:
    """Predice el cluster asociado a un usuario nuevo usando los artefactos entrenados.
    
    Toma un diccionario con las variables de comportamiento de un cliente en tiempo real,
    lo somete a las mismas reglas de limpieza, imputación y estandarización del pipeline
    original, y devuelve el clúster matemático calculado.

    Args:
        datos (dict[str, Any]): Parámetros del nuevo usuario enviados en el cuerpo del POST.

    Raises:
        HTTPException (400): Si los datos no son un diccionario válido o si faltan columnas.
        HTTPException (500): Si la API intenta procesar la inferencia sin modelos cargados.

    Returns:
        dict[str, Any]: Diccionario con el JSON de entrada y los clústeres asignados.
    """
    # 1. Validación estricta del tipo de payload recibido
    if not isinstance(datos, dict):
        raise HTTPException(status_code=400, detail="El cuerpo debe ser un objeto JSON.")

    usuarios_historicos, modelo, scaler, metricas_actuales, classifier = load_artifacts()
    
    # 2. Control ante peticiones prematuras (Falta de entrenamiento previo)
    if modelo is None or scaler is None:
        raise HTTPException(status_code=500, detail="Los modelos aún no están disponibles. Ejecute train.py primero.")

    # 3. Mapeo dinámico de variables esperadas según el entrenamiento
    feature_columns = metricas_actuales.get("feature_columns", [])
    if not feature_columns:
        # Fallback analítico: deducir las columnas ignorando los metadatos agregados
        feature_columns = [col for col in usuarios_historicos.columns if col not in {"id_cliente", "cluster", "pc1", "pc2"}]

    # 4. Asegurar la integridad estructural de la petición
    missing_columns = [col for col in feature_columns if col not in datos]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Faltan columnas obligatorias: {missing_columns}")

    # 5. Modelamiento temporal de la fila entrante mediante Pandas
    nueva_fila = pd.DataFrame([datos])
    nueva_fila = nueva_fila[feature_columns]  # Garantiza el orden idéntico de variables
    nueva_fila = nueva_fila.apply(pd.to_numeric, errors="coerce")
    nueva_fila = nueva_fila.fillna(0)  # Imputación de seguridad para evitar fallas matemáticas en producción

    # 6. REGLA DE ORO: Pasar los datos por el transform del scaler antes de predecir
    # K-Means colapsará o dará salidas erróneas si los datos no están en la escala original estándar.
    X = scaler.transform(nueva_fila)
    cluster_predicho = modelo.predict(X)

    # 7. Construcción estructurada de la respuesta
    respuesta: dict[str, Any] = {
        "usuario_recibido": datos,
        "cluster_asignado": int(cluster_predicho[0]), # Conversión nativa porque np.int64 no es serializable en JSON
    }

    # Validación cruzada si el clasificador supervisado Random Forest está en memoria
    if classifier is not None:
        respuesta["cluster_supervisado"] = int(classifier.predict(X)[0])

    return respuesta