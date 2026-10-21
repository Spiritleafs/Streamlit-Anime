import streamlit as st
import pandas as pd

from tabs import entender


from tabs import limpiar
from tabs import descriptiva
from tabs import predictiva
from tabs import prescriptiva
from tabs import decisiones

# ----------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------

st.set_page_config(
    page_title="Marketing Analytics",
    page_icon="📊",
    layout="wide"
)


# ----------------------------------
# TÍTULO DE LA APLICACIÓN
# ----------------------------------

st.title("📊 Marketing Analytics")

st.markdown("""
Aplicación para analizar información de clientes
y convertir datos en decisiones de marketing.
""")


# ----------------------------------
# CARGAR DATASET
# ----------------------------------

df = pd.read_csv(
    "data/anime-dataset-2026.csv"
)


# ----------------------------------
# CREAR PESTAÑAS
# ----------------------------------

tab_entender, tab_limpiar, tab_descriptiva, tab_predictiva, tab_prescriptiva, tab_decisiones = st.tabs([
    "1. Entender",
    "2. Limpiar",
    "3. Descriptiva",
    "4. Predictiva",
    "5. Prescriptiva",
    "6. Decisiones"
])


# ----------------------------------
# 1. ENTENDER LOS DATOS
# ----------------------------------

with tab_entender:
    entender.mostrar(df)


# ----------------------------------
# PESTAÑAS PENDIENTES
# ----------------------------------

with tab_limpiar:
    df_limpio = limpiar.mostrar(df)

with tab_descriptiva:
    descriptiva.mostrar(df_limpio)



# ----------------------------------
# 4. ESTADÍSTICA PREDICTIVA
# ----------------------------------

with tab_predictiva:
    prediccion = predictiva.mostrar(df_limpio)


# ----------------------------------
# 5. ESTADÍSTICA PRESCRIPTIVA
# ----------------------------------

with tab_prescriptiva:
    plan = prescriptiva.mostrar(prediccion)


# ----------------------------------
# 6. TOMA DE DECISIONES
# ----------------------------------

with tab_decisiones:
    decisiones.mostrar(prediccion, plan)
