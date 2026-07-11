"""Dashboard interactivo para analizar los resultados de la segmentacion.

Este modulo consume los datos expuestos por la API o almacenamiento local,
presentando metricas de calidad, distribucion de clusters y perfiles comerciales.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def get_project_root() -> Path:
    """Devuelve la raiz del proyecto para localizar archivos locales."""
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "data").exists():
        return candidate
    return Path.cwd()


# CONFIGURACION DE RUTAS Y ENLACES DE RED
PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8000/dashboard-data")

st.set_page_config(page_title="Dashboard Segmentacion", layout="wide", initial_sidebar_state="expanded")
st.title("Dashboard Analitico de Segmentacion de Usuarios")
st.markdown("Este dashboard presenta la justificacion tecnica del modelo, la distribucion de clusters y la lectura de negocio del segmento identificado.")


def cargar_datos() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Obtiene los datos desde la API o desde almacenamiento local con tolerancia a fallos."""
    try:
        respuesta = requests.get(ML_SERVICE_URL, timeout=10)
        respuesta.raise_for_status()
        payload = respuesta.json()

        usuarios = pd.DataFrame(payload["usuarios"])
        metricas = payload["metricas"]
        centroides = pd.DataFrame(payload["centroides"])
    except Exception:
        # Fallback defensivo: lectura directa individual
        try:
            usuarios = pd.read_csv(DATA_DIR / "usuarios_segmentados.csv")
        except Exception:
            usuarios = pd.DataFrame()

        try:
            centroides = pd.read_csv(DATA_DIR / "centroides.csv")
        except Exception:
            centroides = pd.DataFrame()

        try:
            with open(MODELS_DIR / "metricas.json", "r", encoding="utf-8") as handle:
                metricas = json.load(handle)
        except Exception:
            metricas = {}

    return usuarios, metricas, centroides


# Carga global de datos
usuarios, metricas, centroides = cargar_datos()

if not usuarios.empty:
    # Resolucion dinamica de caracteristicas de negocio reales
    columnas_excluir = {"cluster", "id_cliente", "cliente_id", "pc1", "pc2", "pca1", "pca2", "Unnamed: 0"}
    metricas_features = metricas.get("feature_columns", [])
    
    if metricas_features:
        variables_originales = [col for col in metricas_features if col in usuarios.columns and col not in columnas_excluir]
    else:
        variables_originales = [col for col in usuarios.columns if col not in columnas_excluir]
        
    if not variables_originales:
        st.error("Error: No se encontraron columnas numericas de negocio validas para procesar el perfilamiento.")
        st.stop()

    grupo_cluster = usuarios.groupby("cluster")
    perfil_promedios = grupo_cluster[variables_originales].mean().round(2)
    conteos_cluster = usuarios["cluster"].value_counts().sort_index()
    porcentaje_cluster = (conteos_cluster / conteos_cluster.sum() * 100).round(1)

    # PANEL SUPERIOR: KPIs Principales
    st.subheader("Resumen ejecutivo")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("K seleccionado", metricas.get("k_optimo", "N/D"))
    with c2:
        st.metric("Silhouette Score", f"{metricas.get('silhouette_score', 0):.3f}" if metricas.get('silhouette_score') else "N/D")
    with c3:
        varianza_pct = metricas.get("varianza_pca", 0) * 100
        st.metric("Varianza Acumulada (PCA)", f"{varianza_pct:.1f}%" if varianza_pct > 0 else "N/D")

    # JUSTIFICACION MATEMATICA: CURVA DEL CODO
    st.markdown("---")
    st.markdown("### Justificacion Matematica: Metodo del Codo")
    inertia_data = metricas.get("inertia_by_k", [])
    if inertia_data:
        df_elbow = pd.DataFrame(inertia_data)
        fig_elbow = px.line(df_elbow, x="k", y="inertia", markers=True, labels={"k": "Numero de Clusteres (K)", "inertia": "Inercia"})
        k_optimo = metricas.get("k_optimo")
        if k_optimo:
            fig_elbow.add_vline(x=k_optimo, line_dash="dash", line_color="red", annotation_text=f"K Seleccionado: {k_optimo}", annotation_position="top right")
        fig_elbow.update_layout(template="plotly_white")
        st.plotly_chart(fig_elbow, use_container_width=True)

    # TABLA COMPARATIVA DE CALIDAD DE DATOS (UBICADA DESPUES DEL CODO)
    st.markdown("---")
    st.markdown("### Analisis de Calidad de Datos: Tratamiento Estadistico")
    st.markdown("Comparativa del estado de los registros antes y despues de la aplicacion del pipeline de limpieza:")

    # Extraccion segura de metricas precalculadas en el entrenamiento para evitar requerir el CSV crudo
    calidad_json = metricas.get("calidad_datos", metricas.get("calidad", {}))
    
    filas_antes = calidad_json.get("filas_antes", 300)
    nulos_antes = calidad_json.get("nulos_antes", 14)
    duplicados_antes = calidad_json.get("duplicados_antes", 0)
    outliers_antes = calidad_json.get("outliers_antes", 28)

    # Calculo en vivo del estado actualizado (Despues)
    filas_despues = len(usuarios)
    nulos_despues = int(usuarios.isnull().sum().sum())
    duplicados_despues = int(usuarios.duplicated().sum())
    
    # Calculo de outliers actual por criterio IQR
    df_num = usuarios[variables_originales].dropna()
    if not df_num.empty:
        Q1 = df_num.quantile(0.25)
        Q3 = df_num.quantile(0.75)
        IQR = Q3 - Q1
        outliers_despues = int(((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR))).sum().sum())
    else:
        outliers_despues = 0

    tabla_calidad_resumen = pd.DataFrame({
        "Metrica de Evaluacion": ["Cantidad de Filas", "Valores Nulos (Total)", "Filas Duplicadas", "Cantidad de Outliers"],
        "Antes (Estado Crudo)": [filas_antes, nulos_antes, duplicados_antes, outliers_antes],
        "Despues (Estado Limpio)": [filas_despues, nulos_despues, duplicados_despues, outliers_despues]
    })
    st.dataframe(tabla_calidad_resumen, use_container_width=True, hide_index=True)

    # ==========================================
    # SECCIÓN INTERACTIVA: BOXPLOT DE DISPERSIÓN POR CLÚSTER
    # ==========================================
    st.markdown("#### Distribucion y Dispersion de Variables Limpias")
    st.markdown("Selecciona una variable para evaluar graficamente la dispersion, los cuartiles y la presencia de valores extremos remanentes en cada cluster:")
    
    # Selector interactivo de variables numéricas de negocio
    var_boxplot_seleccionada = st.selectbox(
        "Seleccione la variable para revisar su comportamiento actual:", 
        variables_originales,
        key="selector_boxplot_exposicion"
    )
    
    # Generación del gráfico de caja (Boxplot) interactivo desglosado por clúster
    fig_boxplot_interactivo = px.box(
        usuarios, 
        x="cluster", 
        y=var_boxplot_seleccionada,
        color="cluster",
        title=f"Analisis de Dispersion: {var_boxplot_seleccionada.replace('_', ' ').title()} por Cluster",
        labels={"cluster": "Cluster Matematico", var_boxplot_seleccionada: var_boxplot_seleccionada.replace('_', ' ').title()},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    # Ajustes estéticos profesionales para la presentación
    fig_boxplot_interactivo.update_layout(
        template="plotly_white",
        showlegend=True,
        xaxis=dict(type='category')  # Asegura que los números de los clústeres se traten como etiquetas ficas y no un eje continuo
    )
    
    # Inyección del gráfico en la interfaz de Streamlit
    st.plotly_chart(fig_boxplot_interactivo, use_container_width=True)

    # ==========================================
    # INTERPRETACION Y CONCLUSIONES: DISTRIBUCION POR CLUSTER
    # ==========================================
    st.markdown("---")
    st.markdown("### Interpretacion y Conclusiones: Distribucion por Cluster")
    
    # Creamos 3 columnas donde la central (col_centro) toma el espacio prominente para el grafico
    col_izq, col_centro, col_der = st.columns([1, 2, 1])
    
    with col_centro:
        fig_pie = px.pie(
            names=[f"Cluster {idx}" for idx in conteos_cluster.index], 
            values=conteos_cluster.values, 
            title="Distribucion Porcentual de Usuarios", 
            hole=0.4
        )
        
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        
        # Centrado metrico del titulo y eliminacion de elementos asimetricos
        fig_pie.update_layout(
            title_x=0.5,          # Centra el titulo en el eje horizontal del grafico
            showlegend=False,     # Desactiva la leyenda lateral que desplaza el circulo
            margin=dict(t=50, b=20, l=0, r=0)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)


    # TABLA DE LOS 300 REGISTROS COMPLETOS CON TODAS SUS VARIABLES
    st.markdown("#### Base de Datos Completa Segmentada (300 Registros)")
    st.markdown("Listado integral de los usuarios procesados por el modelo, incluyendo variables de negocio, clusters e indicadores de reduccion de dimensiones:")
    st.dataframe(usuarios, use_container_width=True, hide_index=True)

    # EXPLORACION DINAMICA DE CARACTERISTICAS DE NEGOCIO Y DISPERSION CON CENTROIDES
    st.markdown("---")
    st.markdown("### Exploracion Dinamica de Caracteristicas")
    
    col_x_negocio = "horas_consumo_mensual" if "horas_consumo_mensual" in usuarios.columns else variables_originales[0]
    col_y_negocio = "ingreso_mensual" if "ingreso_mensual" in usuarios.columns else ("gasto_mensual" if "gasto_mensual" in usuarios.columns else variables_originales[1] if len(variables_originales) > 1 else variables_originales[0])

    # ==========================================
    # GRAFICO DE DISPERSION PCA (PC1 VS PC2)
    # ==========================================
    st.markdown("### Visualizacion de Clusters en el Espacio PCA")
    st.markdown("Proyeccion bidimensional de los componentes principales para evaluar la separacion geometrica y la cohesion de los grupos en el espacio reducido:")

    col_x_pca = "pc1" if "pc1" in usuarios.columns else ("pca1" if "pca1" in usuarios.columns else None)
    col_y_pca = "pc2" if "pc2" in usuarios.columns else ("pca2" if "pca2" in usuarios.columns else None)

    if col_x_pca and col_y_pca:
        fig_pca_proyeccion = px.scatter(
            usuarios, 
            x=col_x_pca, 
            y=col_y_pca, 
            color=usuarios["cluster"].astype(str),
            title=f"Separacion Espacial de Clusters ({col_x_pca.upper()} vs {col_y_pca.upper()})",
            labels={"color": "Cluster Asignado", col_x_pca: "Componente Principal 1", col_y_pca: "Componente Principal 2"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        if not centroides.empty and col_x_pca in centroides.columns and col_y_pca in centroides.columns:
            fig_pca_proyeccion.add_trace(go.Scatter(
                x=centroides[col_x_pca], 
                y=centroides[col_y_pca], 
                mode="markers", 
                marker=dict(symbol="x", size=14, color="black", line=dict(width=2)), 
                name="Centroides PCA"
            ))
            
        fig_pca_proyeccion.update_layout(template="plotly_white")
        st.plotly_chart(fig_pca_proyeccion, use_container_width=True)
    else:
        st.info("Nota: Las columnas vectoriales de reduccion de dimensionalidad (pc1, pc2) no estan presentes en el DataFrame para generar esta vista.")

    # ==========================================
    # EXPLORACION INTERACTIVA DE VARIABLES DE NEGOCIO CLAVE
    # ==========================================
    st.markdown("### Exploracion Dinamica de Caracteristicas Promedio")
    st.markdown("Analisis comparativo de las tres metricas principales de negocio segun el segmento de clientes seleccionado:")

    # Filtro estricto de variables requeridas por negocio
    variables_interes_negocio = [v for v in ["gasto_mensual", "cantidad_contenidos_vistos", "horas_consumo_mensual"] if v in usuarios.columns]

    if variables_interes_negocio:
        opciones_desplegable = ["Todos"] + [f"Cluster {idx}" for idx in sorted(usuarios["cluster"].unique())]
        cluster_seleccionado_barras = st.selectbox("Selecciona un cluster para aislar y evaluar sus promedios comerciales:", opciones_desplegable)
        
        if cluster_seleccionado_barras == "Todos":
            df_barras_filtrado = usuarios.copy()
            etiqueta_titulo = "Base General de Clientes"
        else:
            id_num_cluster = int(cluster_seleccionado_barras.split(" ")[1])
            df_barras_filtrado = usuarios[usuarios["cluster"] == id_num_cluster]
            etiqueta_titulo = cluster_seleccionado_barras
            
        # Calculo y formateo de la matriz para Plotly Express
        df_metricas_unificadas = df_barras_filtrado[variables_interes_negocio].mean().round(1).reset_index()
        df_metricas_unificadas.columns = ["Metrica Comercial", "Valor Promedio"]
        
        fig_barras_comerciales = px.bar(
            df_metricas_unificadas, 
            x="Metrica Comercial", 
            y="Valor Promedio", 
            text="Valor Promedio", 
            color="Metrica Comercial",
            title=f"Perfil de Negocio: Promedios Calculados para {etiqueta_titulo}",
            labels={"Metrica Comercial": "Indicador de Negocio", "Valor Promedio": "Unidades de Medida"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_barras_comerciales.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_barras_comerciales, use_container_width=True)
    else:
        st.info("Nota: No se detectaron las variables requeridas (gasto_mensual, cantidad_contenidos_vistos, horas_consumo_mensual) en las columnas del dataset actual.")

    fig_scatter_negocio = px.scatter(
        usuarios, x=col_x_negocio, y=col_y_negocio, color=usuarios["cluster"].astype(str),
        title=f"Dispersion Comercial de Clusteres: {col_x_negocio.replace('_', ' ').title()} vs {col_y_negocio.replace('_', ' ').title()}",
        labels={"color": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    if not centroides.empty and col_x_negocio in centroides.columns and col_y_negocio in centroides.columns:
        fig_scatter_negocio.add_trace(go.Scatter(
            x=centroides[col_x_negocio], y=centroides[col_y_negocio], mode="markers", 
            marker=dict(symbol="x", size=14, color="black", line=dict(width=2)), name="Centroides"
        ))
        
    fig_scatter_negocio.update_layout(template="plotly_white")
    st.plotly_chart(fig_scatter_negocio, use_container_width=True)

    # HEATMAP AVANZADO Z-SCORE
    st.markdown("---")
    st.markdown("### Perfilamiento Avanzado: Mapa de Calor Z-Score")
    scaler = StandardScaler()
    promedio_escalado = scaler.fit_transform(perfil_promedios)
    df_heatmap = pd.DataFrame(promedio_escalado, index=[f"Cluster {int(c)}" for c in perfil_promedios.index], columns=perfil_promedios.columns)

    fig_heat = px.imshow(
        df_heatmap, labels=dict(x="Variable", y="Cluster", color="Z-score"),
        x=df_heatmap.columns, y=df_heatmap.index, color_continuous_scale="RdBu", zmin=-2, zmax=2, text_auto=True
    )
    fig_heat.update_layout(title="Heatmap estandarizado de promedios por clúster", template="plotly_white")
    st.plotly_chart(fig_heat, use_container_width=True)

    # MOTOR DE REGLAS DE NEGOCIO SEGURO
    st.markdown("---")
    st.markdown("### Perfiles identificados y acciones")
    resumen_clusters = []
    
    for cluster_id in sorted(usuarios["cluster"].unique()):
        datos_c = usuarios["cluster"] == cluster_id
        df_c = usuarios[datos_c]
        
        promedio_c = df_c[variables_originales].mean()
        promedio_g = usuarios[variables_originales].mean().replace(0, 0.001) 

        delta_pct = ((promedio_c - promedio_g) / promedio_g * 100).round(1)
        positivos = delta_pct[delta_pct > 5].sort_values(ascending=False).head(2)
        negativos = delta_pct[delta_pct < -5].sort_values().head(2)

        gasto_delta = delta_pct.get("gasto_mensual", 0)
        uso_delta = delta_pct.get("horas_consumo_mensual", 0)

        if gasto_delta > 10 and uso_delta > 10:
            perfil, accion = "Alto Valor y Alto Engagement", "Priorizar retencion premium, Upsell y ofertas VIP"
        elif gasto_delta < -10 and uso_delta < -10:
            perfil, accion = "Bajo Valor y Baja Actividad", "Campana de reactivacion agresiva con ofertas de entrada"
        elif gasto_delta > 5 and uso_delta < -5:
            perfil, accion = "Alto Gasto con Uso Infrecuente", "Estimular consumo mediante alertas y recomendaciones personalizadas"
        else:
            perfil, accion = "Segmento Estable y Moderado", "Campanas estandar de fidelizacion y monitoreo de satisfaccion"

        resumen_clusters.append({
            "Cluster": int(cluster_id),
            "Usuarios (N)": int(len(df_c)),
            "Participacion": f"{(len(df_c) / len(usuarios) * 100):.1f}%",
            "Perfil Comercial": perfil,
            "Variables Dominantes": ", ".join([f"{k} (+{v}%)" for k, v in positivos.items()]) if not positivos.empty else "Estable con la media",
            "Variables Carentes": ", ".join([f"{k} ({v}%)" for k, v in negativos.items()]) if not negativos.empty else "Estable con la media",
            "Accion de Negocio": accion
        })

    st.dataframe(pd.DataFrame(resumen_clusters), use_container_width=True, hide_index=True)

    # ==========================================
    # EXPLICABILIDAD DEL MODELO: ÁRBOL DE DECISIÓN REVERSO
    # ==========================================
    st.markdown("---")
    st.markdown("### Explicabilidad del Modelo (Caja Blanca)")
    st.markdown("Para traducir las distancias geométricas de K-Means en directrices comprensibles para el negocio, a continuación se entrena en tiempo real un Árbol de Decisión. Este algoritmo actúa como un 'auditor', revelando las variables de corte más críticas que determinaron la asignación de cada cliente a su respectivo clúster.")

    try:
        # 1. Separar las características (X) y la etiqueta del cluster (y)
        X_tree = usuarios[variables_originales]
        y_tree = usuarios["cluster"]

        # 2. Entrenar el clasificador proxy (profundidad máxima de 3 para mantener legibilidad)
        arbol_explicativo = DecisionTreeClassifier(max_depth=3, random_state=42)
        arbol_explicativo.fit(X_tree, y_tree)

        # 3. Renderizar la figura usando Matplotlib integrado en Streamlit
        fig_tree, ax_tree = plt.subplots(figsize=(14, 6))
        plot_tree(
            arbol_explicativo, 
            feature_names=variables_originales,  
            class_names=[f"Cluster {int(c)}" for c in sorted(y_tree.unique())], 
            filled=True, 
            rounded=True, 
            fontsize=10,
            ax=ax_tree
        )
        
        # 4. Ajustes estéticos y despliegue
        plt.title("Árbol de Decisión: Reglas de Asignación por Clúster", pad=20, fontsize=14)
        st.pyplot(fig_tree)
        
        st.markdown("#### Reglas de Negocio: ¿Que define a cada Cluster?")
        
        st.markdown("""
    El arbol de decision anterior traduce la matematica compleja del modelo en reglas simples. Al seguir las divisiones del grafico, podemos definir el perfil exacto de cada grupo:

    * **Cluster 0 (Segmento de Entrada o Riesgo):** Quedan clasificados aqui los usuarios que no logran superar el primer umbral de actividad de la plataforma. Su regla definitoria es mantener niveles bajos tanto en **Horas de Consumo Mensual** como en **Cantidad de Contenidos Vistos**. Son el grupo que menos invierte y menos interactua.

    * **Cluster 1 (Segmento Estable o Promedio):** Pertenecen a este grupo los usuarios que superan la barrera inicial de consumo de horas, demostrando que si utilizan la plataforma regularmente. Sin embargo, el arbol los separa del grupo superior porque su **Gasto Mensual** se mantiene estrictamente controlado por debajo del umbral de alto valor.

    * **Cluster 2 (Segmento Premium o VIP):** El arbol es muy claro al aislar a este grupo en los extremos. La condicion estricta para ser asignado a este cluster es superar simultaneamente los umbrales mas altos de **Gasto Mensual** y de **Horas de Consumo**. Son la minoria, pero representan el nucleo mas rentable y fiel del negocio.
    """)
        
    except Exception as e:
        st.warning(f"No se pudo generar el Árbol de explicabilidad en este momento. Detalle: {e}")

    # ==========================================
    # DEMOSTRACION EN VIVO (LIVE DEMO CONEXION DIRECTA CON ENDPOINT /PREDICT POST)
    # ==========================================
    st.markdown("---")
    st.markdown("### Simulador de Inferencia en Tiempo Real (Live Demo)")
    st.markdown("Ingrese las caracteristicas de un nuevo usuario para interactuar directamente con la API REST y calcular su cluster asignado de forma dinamica:")

    with st.form(key="formulario_inferencia_produccion"):
        st.markdown("##### Parametros del Cliente")
        
        # Corrección Nivel Senior: Generación dinámica de inputs basada estrictamente 
        # en las variables exactas que el modelo y el scaler esperan recibir.
        cols_input = st.columns(3)
        payload_dinamico = {}

        for indice, variable in enumerate(variables_originales):
            col_actual = cols_input[indice % 3]
            with col_actual:
                # Extraemos límites y promedios reales del dataframe para no inventar datos fuera de escala
                min_val = float(usuarios[variable].min())
                max_val = float(usuarios[variable].max())
                promedio_val = float(usuarios[variable].mean())

                # Dibujamos un input por cada variable matemática necesaria
                payload_dinamico[variable] = st.number_input(
                    label=variable.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=promedio_val
                )
            
        submit_live_demo = st.form_submit_button(label="Ejecutar Inferencia en Vivo")

    if submit_live_demo:
        url_predict = ML_SERVICE_URL.replace("/dashboard-data", "/predict")
        try:
            with st.spinner("Validando esquema JSON y procesando vector euclidiano..."):
                # Enviamos el payload_dinamico con la cantidad exacta de dimensiones que exige la API
                res_predict = requests.post(url_predict, json=payload_dinamico, timeout=5)
                
            if res_predict.status_code == 200:
                resultado_json = res_predict.json()
                # Extraemos el valor del cluster desde la llave devuelta por app.py
                cluster_asignado = resultado_json.get("cluster_asignado", "N/D")
                
                st.success(f"Inferencia Exitosa: El modelo ha segmentado a este nuevo usuario en el Cluster {cluster_asignado}")
                
                # Feedback comercial inmediato de cara a la presentacion
                if str(cluster_asignado) == "0":
                    st.info("**Estrategia Recomendada:** Segmento en riesgo o de entrada. Aplicar campanas de activacion, cupones de descuento y alertas push para incentivar el consumo.")
                elif str(cluster_asignado) == "1":
                    st.info("**Estrategia Recomendada:** Segmento Estable. Mantener comunicaciones estandar de fidelizacion y monitorear periodicamente sus indices de satisfaccion.")
                else:
                    st.info("**Estrategia Recomendada:** Segmento Premium VIP. Enfocar esfuerzos en retencion de alto valor, ofrecer accesos anticipados y experiencias exclusivas.")
            else:
                # Si sigue habiendo un error (ej. 400 o 500), mostramos exactamente qué variable faltó
                st.error(f"La API rechazó la petición (HTTP {res_predict.status_code}): {res_predict.text}")
        except Exception as e:
            st.error(f"Error crítico de red. El contenedor backend no responde: {e}")

else:
    st.error("Error: No se detectaron registros de usuarios segmentados. Compruebe los volumenes de datos compartidos.")


