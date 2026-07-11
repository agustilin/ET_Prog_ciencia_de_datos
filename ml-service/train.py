"""Pipeline de entrenamiento y preparación de datos para segmentación de usuarios.

Este módulo extrae los datos de un CSV y PostgreSQL, realiza un tratamiento de
calidad, entrena un modelo de clustering no supervisado con K-Means, calcula
métricas de validación, compara un modelo alternativo y exporta los artefactos
consumidos por la API y el dashboard.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto, compatible con Docker y ejecución local.
    
    Returns:
        Path: Objeto Path apuntando al directorio raíz del proyecto.
    """
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "data").exists():
        return candidate
    return Path.cwd()


# ==========================================
# CONFIGURACIÓN DE RUTAS Y VARIABLES GLOBALES
# ==========================================
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Creamos los directorios si no existen para evitar errores al guardar archivos
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# URI de conexión a la base de datos PostgreSQL (con fallback por defecto)
DB_URI = os.getenv("DATABASE_URL", "postgresql://admin:admin@postgres:5432/streaming_usuarios")


def validate_required_columns(data: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Valida que las columnas mínimas necesarias existan antes de procesar o entrenar.

    Args:
        data (pd.DataFrame): El DataFrame a validar.
        required_columns (list[str]): Lista de nombres de columnas que deben existir.

    Raises:
        ValueError: Si falta al menos una de las columnas requeridas.

    Returns:
        pd.DataFrame: Una copia del DataFrame validado.
    """
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")
    return data.copy()


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datos de comportamiento y el perfil de usuarios.
    
    Intenta extraer la información de los perfiles desde PostgreSQL. Si la conexión
    falla, utiliza un archivo CSV local de respaldo (fallback) garantizando que el
    pipeline no se detenga.

    Raises:
        SQLAlchemyError: Si falla la conexión a BD y tampoco existe el archivo de respaldo.
        ValueError: Si la tabla requerida está vacía.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame con datos de comportamiento (streaming).
            - DataFrame con datos de perfil de usuario.
    """
    usuarios_streaming = pd.read_csv(DATA_DIR / "usuarios_streaming.csv")
    try:
        # Intento de conexión a la base de datos PostgreSQL
        engine = create_engine(DB_URI)
        with engine.connect() as connection:
            perfil_usuarios = pd.read_sql("SELECT * FROM perfil_usuarios", connection)
        
        if perfil_usuarios.empty:
            raise ValueError("No se encontraron filas en la tabla perfil_usuarios")
        
        return usuarios_streaming, perfil_usuarios
    
    except (SQLAlchemyError, OSError, ValueError) as exc:
        # Plan de contingencia: Cargar CSV local si la BD falla
        fallback_path = PROJECT_ROOT / "database" / "perfil_usuarios.csv"
        if fallback_path.exists():
            perfil_usuarios = pd.read_csv(fallback_path)
            print(f"Usando fuente local de respaldo para perfil_usuarios por: {exc}")
            return usuarios_streaming, perfil_usuarios
        raise


def clean_and_integrate(usuarios_streaming: pd.DataFrame, perfil_usuarios: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Une ambas fuentes de datos, limpia valores inconsistentes y prepara el dataset.

    Este proceso incluye:
    - Inner Join entre bases por 'id_cliente'.
    - Eliminación de registros duplicados.
    - Imputación de valores nulos (usando la mediana).
    - Tratamiento de outliers mediante clipping con el rango intercuartílico (IQR).

    Args:
        usuarios_streaming (pd.DataFrame): Datos de comportamiento en la plataforma.
        perfil_usuarios (pd.DataFrame): Datos demográficos/perfil del usuario.

    Raises:
        ValueError: Si el dataset resultante queda vacío o sin columnas numéricas.

    Returns:
        tuple[pd.DataFrame, dict[str, Any]]: 
            - DataFrame limpio y listo para el modelado.
            - Diccionario (summary) con las métricas del proceso de limpieza.
    """
    # 1. Validación de llave primaria
    usuarios_streaming = validate_required_columns(usuarios_streaming, ["id_cliente"])
    perfil_usuarios = validate_required_columns(perfil_usuarios, ["id_cliente"])

    # 2. Integración de los datos (Inner Join)
    merged = usuarios_streaming.merge(perfil_usuarios, on="id_cliente", how="inner")
    
    # Identificar y eliminar duplicados basados en el id_cliente
    duplicate_mask = merged["id_cliente"].duplicated(keep=False)
    duplicate_rows = merged.loc[duplicate_mask].copy()
    rows_before_dedup = int(len(merged))
    data = merged.drop_duplicates(subset=["id_cliente"]).copy()

    if data.empty:
        raise ValueError("El dataset integrado está vacío tras la combinación de fuentes.")

    # 3. Selección y transformación de variables a formato numérico
    candidate_columns = [col for col in data.columns if col != "id_cliente"]
    for col in candidate_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    numeric_columns = [col for col in candidate_columns if pd.api.types.is_numeric_dtype(data[col])]
    if not numeric_columns:
        raise ValueError("No se encontraron columnas numéricas para entrenar el modelo.")

    # Diccionarios para registrar métricas de calidad de datos
    nulls_before: dict[str, int] = {}
    nulls_after: dict[str, int] = {}
    outlier_counts: dict[str, int] = {}
    outlier_comparison_frames: list[pd.DataFrame] = []

    # 4. Imputación de Nulos (Se usa la Mediana por ser robusta a outliers)
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        nulls_before[col] = int(data[col].isna().sum())

    for col in numeric_columns:
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)
        nulls_after[col] = int(data[col].isna().sum())

    # 5. Tratamiento de Outliers (Método del Rango Intercuartílico - IQR)
    for col in numeric_columns:
        before_values = data[col].copy()
        
        # Cálculo de los cuartiles y límites
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Conteo y aplicación de "Clipping" (acotar valores a los límites sin eliminarlos)
        outlier_counts[col] = int(((before_values < lower_bound) | (before_values > upper_bound)).sum())
        data[col] = before_values.clip(lower_bound, upper_bound)
        after_values = data[col].copy()

        # Guardar registro para comparar el antes y después de los outliers
        if outlier_counts[col] > 0:
            comparison_df = pd.DataFrame(
                {
                    "variable": [col] * (len(before_values) + len(after_values)),
                    "valor": pd.concat([before_values, after_values], ignore_index=True),
                    "etapa": ["antes"] * len(before_values) + ["despues"] * len(after_values),
                }
            )
            outlier_comparison_frames.append(comparison_df)

    # 6. Guardado de datasets intermedios y de auditoría
    data.to_csv(DATA_DIR / "usuarios_data.csv", index=False)

    if outlier_comparison_frames:
        outlier_comparison_df = pd.concat(outlier_comparison_frames, ignore_index=True)
    else:
        outlier_comparison_df = pd.DataFrame(columns=["variable", "valor", "etapa"])
    outlier_comparison_df.to_csv(DATA_DIR / "outliers_comparacion.csv", index=False)

    # 7. Resumen de Calidad de Datos (Data Quality Summary)
    summary = {
        "fuentes": ["data/usuarios_streaming.csv", "database/perfil_usuarios.csv"],
        "rows_before_dedup": rows_before_dedup,
        "rows_after_cleaning": int(len(data)),
        "duplicate_rows_count": int(len(duplicate_rows)),
        "duplicate_ids_preview": [int(value) for value in duplicate_rows["id_cliente"].drop_duplicates().head(10).tolist()],
        "columns_before_cleaning": int(len(merged.columns)),
        "columns_after_cleaning": int(len(data.columns)),
        "numeric_columns": numeric_columns,
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
        "outlier_counts": outlier_counts,
        "outlier_comparison_file": "data/outliers_comparacion.csv",
        "scaling_reason": "StandardScaler se aplicó porque K-Means usa distancias euclidianas y las variables con rangos distintos pueden dominar la segmentación.",
        "steps": [
            "Integración por id_cliente entre comportamiento y perfil de usuario.",
            "Eliminación de duplicados por id_cliente.",
            "Conversión de columnas numéricas y rellenado de nulos con la mediana.",
            "Recorte de outliers mediante el rango IQR.",
            "Estandarización de variables con StandardScaler para el clustering.",
            "Reducción de dimensionalidad a 2 componentes PCA para visualización.",
        ],
        "validation_checks": {
            "required_id_column_present": True,
            "numeric_columns_detected": len(numeric_columns),
        },
    }
    return data, summary


def train_clustering_model(X_scaled: np.ndarray, max_k: int = 10) -> tuple[KMeans, np.ndarray, list[dict[str, Any]], list[dict[str, Any]], int]:
    """Entrena el modelo K-Means buscando el valor óptimo de K dinámicamente.

    Utiliza el método del codo (Elbow Method) auxiliado por la librería 'kneed'
    para encontrar matemáticamente el mejor número de clusters. Además, calcula
    el índice Silhouette para medir la calidad de separación.

    Args:
        X_scaled (np.ndarray): Matriz de características ya estandarizadas.
        max_k (int, optional): Máximo número de clusters a evaluar. Defaults to 10.

    Raises:
        ValueError: Si los datos no tienen las dimensiones o muestras mínimas.

    Returns:
        tuple: 
            - final_model (KMeans): Objeto del modelo ya entrenado con K óptimo.
            - clusters (np.ndarray): Etiquetas asignadas a cada muestra.
            - inertia_by_k (list): Valores de inercia por cada K evaluado.
            - silhouette_by_k (list): Valores de silhouette por cada K evaluado.
            - k_optimo (int): Número de clusters seleccionado finalmente.
    """
    if X_scaled.ndim != 2:
        raise ValueError("X_scaled debe ser un array bidimensional")
    if len(X_scaled) < 2:
        raise ValueError("Se necesitan al menos dos muestras para evaluar clustering")

    max_k = min(max_k, len(X_scaled))
    k_values = list(range(2, max_k + 1))
    if len(k_values) < 2:
        raise ValueError("Se necesitan al menos dos valores de K para comparar")

    inertias: list[float] = []
    silhouettes: list[float] = []

    # Iteración sobre los posibles valores de K para evaluar métricas
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=29, n_init=10)
        model.fit(X_scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, model.labels_)))

    # Localización automática del punto óptimo de inflexión (Codo matemático)
    knee = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
    
    # Si KneeLocator falla, usa el K que maximiza el Silhouette Score
    k_optimo = int(knee.elbow) if knee.elbow is not None else int(k_values[int(np.argmax(silhouettes))])

    # Entrenamiento del modelo definitivo con el K seleccionado
    final_model = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
    clusters = final_model.fit_predict(X_scaled)

    inertia_by_k = [{"k": int(k), "inertia": float(value)} for k, value in zip(k_values, inertias)]
    silhouette_by_k = [{"k": int(k), "silhouette": float(value)} for k, value in zip(k_values, silhouettes)]
    
    return final_model, clusters, inertia_by_k, silhouette_by_k, k_optimo


def train_comparison_model(X_scaled: np.ndarray, n_clusters: int) -> dict[str, Any]:
    """Entrena un modelo de Clustering Jerárquico como punto de comparación (Challenger Model).

    Args:
        X_scaled (np.ndarray): Datos estandarizados.
        n_clusters (int): Número de clusters a utilizar (se iguala al K de K-Means).

    Returns:
        dict[str, Any]: Diccionario con el nombre del modelo y su métrica Silhouette.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)
    silhouette = float(silhouette_score(X_scaled, labels))
    return {"name": "AgglomerativeClustering", "silhouette": silhouette}


def train_supervised_classifier(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Entrena un Random Forest para predecir a qué cluster pertenece un usuario.

    Esto actúa como una validación de negocio: Si un modelo supervisado puede 
    predecir con alta precisión (accuracy) el cluster, significa que las agrupaciones 
    tienen patrones distinguibles y están bien separadas en el hiperespacio.

    Args:
        X (pd.DataFrame): Variables predictoras estandarizadas.
        y (pd.Series): Etiquetas del cluster (target).

    Returns:
        dict[str, Any]: Diccionario conteniendo el modelo clasificador y su accuracy.
    """
    if len(np.unique(y)) < 2:
        return {"model": None, "accuracy": 0.0}

    # Split estratificado si las clases están suficientemente representadas
    class_counts = pd.Series(y).value_counts()
    if min(class_counts.values) < 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=29)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=29, stratify=y)

    classifier = RandomForestClassifier(random_state=29, n_estimators=200)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    
    return {"model": classifier, "accuracy": accuracy}


def save_artifacts(
    data: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
    kmeans: KMeans,
    pca: PCA,
    metricas: dict[str, Any],
    classifier: RandomForestClassifier | None,
) -> None:
    """Serializa (guarda) los datos finales, métricas y modelos entrenados.

    Estos archivos (.pkl, .csv, .json) son los insumos que consumirán tanto 
    la API (para inferencia) como el Dashboard de Streamlit (para reportes visuales).

    Args:
        data (pd.DataFrame): Dataset final incluyendo columnas de clústeres y PCA.
        feature_columns (list[str]): Nombres de las variables usadas en el entrenamiento.
        scaler (StandardScaler): Objeto escalador ajustado.
        kmeans (KMeans): Modelo principal de clustering.
        pca (PCA): Objeto para reducción de dimensionalidad.
        metricas (dict[str, Any]): JSON con métricas y resumen de limpieza.
        classifier (RandomForestClassifier | None): Modelo clasificador de validación.
    """
    # Guardar CSV consolidado
    data.to_csv(DATA_DIR / "usuarios_segmentados.csv", index=False)

    # Inversión de estandarización para exportar los centroides en sus unidades originales
    centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centroides_df = pd.DataFrame(centroides_original, columns=feature_columns)
    centroides_df.to_csv(DATA_DIR / "centroides.csv", index=False)

    # Guardar métricas de evaluación
    with open(MODELS_DIR / "metricas.json", "w", encoding="utf-8") as handle:
        json.dump(metricas, handle, indent=4, ensure_ascii=False)

    # Guardado de artefactos .pkl (Librería pickle)
    with open(MODELS_DIR / "modelo_kmeans.pkl", "wb") as handle:
        pickle.dump(kmeans, handle)

    with open(MODELS_DIR / "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    with open(MODELS_DIR / "pca.pkl", "wb") as handle:
        pickle.dump(pca, handle)

    if classifier is not None:
        with open(MODELS_DIR / "classifier.pkl", "wb") as handle:
            pickle.dump(classifier, handle)


def main() -> None:
    """Ejecuta secuencialmente todo el ciclo de vida de los datos (ML Pipeline).
    
    Flujo:
    1. Extracción (Sources).
    2. Transformación y Limpieza (Data Cleaning).
    3. Estandarización de escalas (StandardScaler).
    4. Entrenamiento de modelos y búsqueda de K.
    5. Reducción a 2D (PCA) para despliegue visual.
    6. Exportación a volúmenes (Archivos .pkl y .csv).
    """
    # 1. Carga de datos
    usuarios_streaming, perfil_usuarios = load_source_data()
    
    # 2. Preprocesamiento
    data, preprocessing_summary = clean_and_integrate(usuarios_streaming, perfil_usuarios)

    feature_columns = [col for col in data.columns if col != "id_cliente"]
    X = data[feature_columns].copy()

    # 3. Estandarización obligatoria para modelos basados en distancias euclidianas (K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)

    # 4. Modelado y Agrupamiento
    kmeans, clusters, inertia_by_k, silhouette_by_k, k_optimo = train_clustering_model(X_scaled, max_k=min(8, len(X)))
    
    # Asignación del resultado al dataframe original
    data["cluster"] = clusters.astype(int)

    # 5. Reducción de Dimensionalidad para Dashboard (Visualización 2D)
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_scaled_df)
    data["pc1"] = componentes[:, 0]
    data["pc2"] = componentes[:, 1]

    # Modelos paralelos de validación
    classifier_result = train_supervised_classifier(X_scaled_df, data["cluster"])
    classifier = classifier_result.get("model")
    comparison_model = train_comparison_model(X_scaled, k_optimo)

    # 6. Empaquetado final de métricas de negocio
    silhouette_value = float(silhouette_score(X_scaled_df, data["cluster"]))
    business_takeaway = (
        f"El modelo seleccionó {k_optimo} segmentos con un silhouette de {silhouette_value:.3f}. "
        "La combinación de limpieza de datos, estandarización y PCA mejora la interpretabilidad y reduce ruido para decisiones de negocio."
    )
    
    metricas = {
        "k_optimo": int(k_optimo),
        "silhouette_score": silhouette_value,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_usuarios": int(len(data)),
        "n_clusters": int(k_optimo),
        "varianza_pca": float(pca.explained_variance_ratio_.sum()),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_.round(4)],
        "cumulative_variance_ratio": [float(value) for value in np.cumsum(pca.explained_variance_ratio_).round(4)],
        "inertia_by_k": inertia_by_k,
        "silhouette_by_k": silhouette_by_k,
        "classifier_accuracy": classifier_result["accuracy"],
        "feature_columns": feature_columns,
        "preprocessing_summary": preprocessing_summary,
        "comparison_models": [comparison_model],
        "business_takeaway": business_takeaway,
    }

    # 7. Serialización y Guardado
    save_artifacts(data, feature_columns, scaler, kmeans, pca, metricas, classifier)
    print(f"Modelo entrenado con K={k_optimo} y accuracy del clasificador complementario={classifier_result['accuracy']:.3f}")


if __name__ == "__main__":
    main()


# docker-compose up --build ml-service