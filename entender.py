import pandas as pd
import streamlit as st

from utils import contar_marcadores, es_numerica_oculta, faltantes_totales


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("1. Entender los datos")

    st.markdown("""
    ### Pregunta inicial

    **¿Qué información tenemos sobre los anime?**
    """)

    # ----------------------------------
    # DIMENSIONES DEL DATASET
    # ----------------------------------

    marcadores = contar_marcadores(df)
    nulos = df.isnull().sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Número de anime", f"{len(df):,}")

    with col2:
        st.metric("Número de variables", len(df.columns))

    with col3:
        pct = faltantes_totales(df) / df.size * 100
        st.metric(
            "Celdas faltantes",
            f"{pct:.1f}%",
            help="Incluye los nulos reales y los valores escritos como 'UNKNOWN'."
        )

    # ----------------------------------
    # VISTA PREVIA
    # ----------------------------------

    st.subheader("Vista previa del dataset")

    st.dataframe(df.head(10), width="stretch")

    # ----------------------------------
    # DIAGNÓSTICO
    # ----------------------------------

    st.subheader("Diagnóstico del dataset")

    diagnostico = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str).values,
        "Valores únicos": df.nunique().values,
        "Nulos": nulos.values,
        "'UNKNOWN' u otros marcadores": marcadores.values,
        "% faltante total": (
            (nulos + marcadores) / len(df) * 100
        ).round(1).values,
    })

    st.dataframe(diagnostico, width="stretch", hide_index=True)

    if marcadores.sum() > 0:
        st.warning(
            "Este dataset no usa nulos reales para casi nada: escribe "
            "'UNKNOWN' como texto. Por eso, aunque casi no hay celdas vacías, "
            "faltan muchos datos. Variables como Score, Episodes, Rank y "
            "Scored By quedan guardadas como texto. Todo esto se corrige en "
            "la pestaña «Limpiar»."
        )

    # ----------------------------------
    # PROBLEMAS POTENCIALES
    # ----------------------------------

    st.subheader("Problemas potenciales")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Registros duplicados", int(df.duplicated().sum()))

    with col2:
        st.metric(
            "Variables con faltantes",
            int(((nulos + marcadores) > 0).sum())
        )

    with col3:
        st.metric(
            "Variables numéricas guardadas como texto",
            sum(es_numerica_oculta(df[c]) for c in df.columns)
        )

    # ----------------------------------
    # EXPLORADOR DE VARIABLES
    # ----------------------------------

    st.subheader("Explorar una variable")

    variable = st.selectbox(
        "Selecciona una variable:",
        df.columns,
        key="variable_entender"
    )

    # ----------------------------------
    # VARIABLE NUMÉRICA
    # ----------------------------------

    if pd.api.types.is_numeric_dtype(df[variable]):

        st.write("**Tipo:** Variable numérica")

        st.dataframe(
            df[variable].describe().round(2).to_frame(),
            width="stretch"
        )

    # ----------------------------------
    # VARIABLE CATEGÓRICA / TEXTO
    # ----------------------------------

    else:

        st.write("**Tipo:** Variable categórica / texto")

        if es_numerica_oculta(df[variable]):
            st.info(
                "Esta variable es numérica pero está guardada como texto. "
                "Se convertirá en la pestaña «Limpiar»."
            )

        n_unicos = df[variable].nunique()

        frecuencias = (
            df[variable]
            .value_counts(dropna=False)
            .reset_index()
        )
        frecuencias.columns = ["Valor", "Frecuencia"]

        # Variables como título o sinopsis tienen casi tantos valores
        # únicos como filas: se muestran solo los más frecuentes
        if n_unicos > 30:
            st.caption(
                f"Esta variable tiene {n_unicos:,} valores únicos. "
                "Se muestran los 20 más frecuentes."
            )
            frecuencias = frecuencias.head(20)

        st.dataframe(frecuencias, width="stretch")
