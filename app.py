import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial de la página
st.set_page_config(
    page_title="Optimización Inferencial Agrícola - U4",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.markdown("**Proyecto Integrador Unidad 4** | Análisis Estadístico Inferencial y Regresión Lineal Múltiple")

# ==============================================================================
# FASE 1: INGESTA DETERMINISTA DE TELEMETRÍA EN DUCKDB (setseed = 0.42)
# ==============================================================================
@st.cache_data
def cargar_datos_telemetria():
    con = duckdb.connect(database=':memory:')
    con.execute("SELECT setseed(0.42);")
    
    query = """
    CREATE TABLE telemetria_invernadero AS
    WITH base AS (
        SELECT 
            range AS parcela_id,
            ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
            ROUND(50.0 + random() * 100.0, 2) AS fertilizante_kg,
            ROUND(18.0 + random() * 14.0, 2) AS temperatura_c,
            CASE WHEN random() < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
            -0.8 + random() * 1.6 AS ruido
        FROM range(1, 101)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        temperatura_c,
        tipo_tratamiento,
        ROUND(
            5.0 
            + (0.035 * volumen_riego) 
            + (0.020 * fertilizante_kg) 
            + (CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END)
            + ruido, 
        2) AS rendimiento_ton
    FROM base;
    """
    con.execute(query)
    df = con.execute("SELECT * FROM telemetria_invernadero").df()
    con.close()
    return df

df_raw = cargar_datos_telemetria()

# ==============================================================================
# BARRA LATERAL: CONTROLES INTERACTIVOS
# ==============================================================================
st.sidebar.header("🎛️ Filtros de Telemetría")
tratamientos_disp = df_raw['tipo_tratamiento'].unique().tolist()
tratamiento_sel = st.sidebar.multiselect(
    "Seleccionar Tipo de Tratamiento:",
    options=tratamientos_disp,
    default=tratamientos_disp
)

rango_riego = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    min_value=float(df_raw['volumen_riego'].min()),
    max_value=float(df_raw['volumen_riego'].max()),
    value=(float(df_raw['volumen_riego'].min()), float(df_raw['volumen_riego'].max()))
)

# Aplicar filtros
df = df_raw[
    (df_raw['tipo_tratamiento'].isin(tratamiento_sel)) &
    (df_raw['volumen_riego'].between(rango_riego[0], rango_riego[1]))
].copy()

if df.empty:
    st.warning("No hay datos disponibles con los filtros seleccionados. Ajusta los controles en la barra lateral.")
    st.stop()

# ==============================================================================
# FASE 2: MODELO DE REGRESIÓN LINEAL MÚLTIPLE (OLS)
# ==============================================================================
X = df[['volumen_riego', 'fertilizante_kg']]
X = sm.add_constant(X)
y = df['rendimiento_ton']
modelo_ols = sm.OLS(y, X).fit()
df['prediccion'] = modelo_ols.predict(X)
df['residuos'] = y - df['prediccion']
r2_ajustado = modelo_ols.rsquared_adj * 100

# ==============================================================================
# FASE 3: PRUEBA DE HIPÓTESIS t-STUDENT (TRATAMIENTO B vs TRATAMIENTO A)
# ==============================================================================
grupo_a = df_raw[df_raw['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_b = df_raw[df_raw['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']

# Prueba t unilateral (H1: mu_B > mu_A)
t_stat, p_val_bilateral = stats.ttest_ind(grupo_b, grupo_a, equal_var=False)
p_val_unilateral = p_val_bilateral / 2 if t_stat > 0 else 1 - (p_val_bilateral / 2)

# Intervalo de confianza al 95% para la diferencia de medias
diff_medias = grupo_b.mean() - grupo_a.mean()
se_diff = np.sqrt(grupo_b.var(ddof=1)/len(grupo_b) + grupo_a.var(ddof=1)/len(grupo_a))
gl = len(grupo_b) + len(grupo_a) - 2
ic_95 = stats.t.interval(0.95, df=gl, loc=diff_medias, scale=se_diff)

# ==============================================================================
# FASE 5: TARJETAS KPI SUPERIORES
# ==============================================================================
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric(
    label="🌾 Rendimiento Promedio Global",
    value=f"{df['rendimiento_ton'].mean():.2f} ton/ha",
    delta=f"N = {len(df)} parcelas"
)
col_kpi2.metric(
    label="📈 R² Ajustado del Modelo OLS",
    value=f"{r2_ajustado:.2f} %",
    delta=f"RMSE: {np.sqrt(modelo_ols.mse_resid):.2f} ton/ha"
)
col_kpi3.metric(
    label="🧪 P-Valor Prueba t (B vs A)",
    value=f"{p_val_unilateral:.4e}",
    delta="Significativo (p < 0.05)" if p_val_unilateral < 0.05 else "No significativo"
)

st.divider()

# ==============================================================================
# FASE 4: CUADRÍCULA 2x2 DE VISUALIZACIONES
# ==============================================================================
sns.set_theme(style="whitegrid")
fila1_col1, fila1_col2 = st.columns(2)

# Gráfica 1: Mapa de Calor de Correlación de Pearson
with fila1_col1:
    st.subheader("1. Mapa de Calor de Correlación (Pearson)")
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    cols_corr = ['volumen_riego', 'fertilizante_kg', 'temperatura_c', 'rendimiento_ton']
    corr_matrix = df[cols_corr].corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax1)
    ax1.set_title("Matriz de Correlación de Variables Agronómicas")
    st.pyplot(fig1)

# Gráfica 2: Dispersión con Línea de Regresión OLS
with fila1_col2:
    st.subheader("2. Dispersión y Ajuste Lineal OLS (Y vs Riego)")
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    sns.regplot(
        data=df, x='volumen_riego', y='rendimiento_ton',
        scatter_kws={'alpha': 0.7, 'color': '#2b7bba'},
        line_kws={'color': '#d95f02', 'label': 'Ajuste OLS'}, ax=ax2
    )
    ax2.set_xlabel("Volumen de Riego Acumulado (m³/ha)")
    ax2.set_ylabel("Rendimiento de Cosecha (ton/ha)")
    ax2.set_title("Rendimiento vs. Volumen de Riego (IC 95%)")
    ax2.legend()
    st.pyplot(fig2)

fila2_col1, fila2_col2 = st.columns(2)

# Gráfica 3: Diagrama de Cajas Comparativo (Boxplot Tratamientos)
with fila2_col1:
    st.subheader("3. Comparativa de Tratamientos (Boxplot)")
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(
        data=df_raw, x='tipo_tratamiento', y='rendimiento_ton',
        palette=['#8da0cb', '#66c2a5'], width=0.5, ax=ax3
    )
    ax3.set_xlabel("Tipo de Insumo / Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    ax3.set_title(f"Tratamiento A vs B (Diferencia media: +{diff_medias:.2f} ton/ha)")
    st.pyplot(fig3)

# Gráfica 4: Histograma y Densidad de Residuos
with fila2_col2:
    st.subheader("4. Diagnóstico de Normalidad de Residuos")
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    sns.histplot(df['residuos'], kde=True, color='#7570b3', bins=15, ax=ax4)
    ax4.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Media = 0')
    ax4.set_xlabel("Residuos (Y_real - Y_predicho)")
    ax4.set_ylabel("Frecuencia")
    ax4.set_title("Distribución de Errores del Modelo OLS")
    ax4.legend()
    st.pyplot(fig4)

# ==============================================================================
# SECCIÓN DE RESULTADOS ESTADÍSTICOS DETALLADOS Y VEREDICTO
# ==============================================================================
st.divider()
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.subheader("📋 Resumen del Modelo OLS (Statsmodels)")
    st.text(modelo_ols.summary().as_text())

with col_res2:
    st.subheader("⚖️ Veredicto Inferencial (Subtema 4.3)")
    st.markdown(f"""
    - **Media Tratamiento A (Estándar):** `{grupo_a.mean():.2f} ton/ha`
    - **Media Tratamiento B (Bio-estimulante):** `{grupo_b.mean():.2f} ton/ha`
    - **Diferencia de Medias ($\mu_B - \mu_A$):** `+{diff_medias:.2f} ton/ha`
    - **Intervalo de Confianza (95%):** `[{ic_95[0]:.2f}, {ic_95[1]:.2f}] ton/ha`
    - **Estadístico t:** `{t_stat:.4f}` | **Valor p (unilateral):** `{p_val_unilateral:.4e}`
    """)
    if p_val_unilateral < 0.05:
        st.success("**DECISIÓN TÉCNICA:** Dado que $p < 0.05$, se rechaza $H_0$. Existe evidencia estadística significativa de que el Tratamiento B incrementa el rendimiento. **Se recomienda autorizar la inversión para escalar el bio-estimulante.**")
    else:
        st.info("**DECISIÓN TÉCNICA:** Dado que $p \\ge 0.05$, no se rechaza $H_0$.")

with st.expander("🔍 Ver Tabla de Datos Simulados en DuckDB (N=100)"):
    st.dataframe(df, use_container_width=True)
