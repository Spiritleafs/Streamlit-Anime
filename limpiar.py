import numpy as np
import pandas as pd
import streamlit as st

import utils as u

OPC_NUM = [
    "Conservar nulos (se ignoran en los cálculos)",
    "Rellenar con la mediana",
    "Rellenar con la media",
    "Eliminar filas con nulo",
]

OPC_CAT = [
    "Rellenar con 'Desconocido'",
    "Conservar nulos",
    "Rellenar con la moda",
    "Eliminar filas con nulo",
]

# Texto libre o columnas de las que ya se derivó otra variable:
# rellenar con "Desconocido" no aporta nada
CONSERVAR_NULOS = {u.SINOPSIS, u.NOMBRE_EN, u.DECADA, u.EMISION, u.DURACION}

# Columnas que sugerimos descartar por defecto
DESCARTE_SUGERIDO = [u.IMAGEN, u.NOMBRE_OTRO]


def mostrar(df):
    """Devuelve el DataFrame limpio para usarlo en las siguientes pestañas."""

    st.header("2. Limpiar los datos")

    st.markdown("""
    ### Pregunta inicial

    **¿Qué tan confiables son los datos?**
    """)

    st.caption(
        "Cada decisión se aplica sobre una copia del dataset. "
        "Las pestañas siguientes trabajan con el resultado de esta limpieza."
    )

    decisiones = []
    limpio = df.copy()

    # ----------------------------------
    # PASO 1: FALTANTES ESCONDIDOS COMO TEXTO
    # ----------------------------------

    st.subheader("Paso 1. Faltantes escondidos como texto")

    marcadores = u.contar_marcadores(df)
    marcadores = marcadores[marcadores > 0].sort_values(ascending=False)

    if marcadores.empty:
        st.success("No se encontraron marcadores de faltante como texto.")
    else:
        st.dataframe(
            pd.DataFrame({
                "Variable": marcadores.index,
                "Celdas con marcador": marcadores.values,
                "% del total": (marcadores.values / len(df) * 100).round(1),
            }),
            width="stretch",
            hide_index=True
        )
        st.caption(
            "Marcadores: 'UNKNOWN', 'Not available', textos vacíos y la "
            "sinopsis genérica «No synopsis information has been added...»."
        )

        if st.checkbox(
            "Convertir esos marcadores en nulos reales",
            value=True,
            key="limpiar_marcadores"
        ):
            limpio = u.marcadores_a_nulos(limpio)
            decisiones.append((
                "Marcadores de faltante",
                f"{int(marcadores.sum()):,} celdas",
                "Convertidos en nulos reales"
            ))
        else:
            decisiones.append((
                "Marcadores de faltante",
                f"{int(marcadores.sum()):,} celdas",
                "Se dejaron como texto"
            ))

    # ----------------------------------
    # PASO 2: TIPOS Y VALORES INVÁLIDOS
    # ----------------------------------

    st.subheader("Paso 2. Tipos de datos y valores inválidos")

    for c in u.NUMERICAS_EN_TEXTO:
        limpio[c] = pd.to_numeric(limpio[c], errors="coerce")

    st.write(
        "Se convierten a número: "
        + ", ".join(f"`{c}`" for c in u.NUMERICAS_EN_TEXTO)
        + "."
    )
    decisiones.append((
        "Tipos de datos",
        ", ".join(u.NUMERICAS_EN_TEXTO),
        "Convertidos de texto a número"
    ))

    n_rank0 = int((limpio[u.RANKING] == 0).sum())
    n_pop0 = int((limpio[u.POPULARIDAD] == 0).sum())

    col1, col2 = st.columns(2)
    col1.metric("Rank = 0", n_rank0)
    col2.metric("Popularity = 0", n_pop0)

    st.caption(
        "Rank y Popularity son posiciones (1 = la mejor). Un 0 no es una "
        "posición válida: corresponde a títulos sin miembros en la comunidad."
    )

    if st.checkbox(
        "Tratar Rank = 0 y Popularity = 0 como nulos",
        value=True,
        key="limpiar_ceros"
    ):
        limpio[u.RANKING] = limpio[u.RANKING].replace(0, np.nan)
        limpio[u.POPULARIDAD] = limpio[u.POPULARIDAD].replace(0, np.nan)
        decisiones.append((
            "Rank / Popularity = 0",
            f"{n_rank0} / {n_pop0} registros",
            "Tratados como nulos"
        ))
    else:
        decisiones.append((
            "Rank / Popularity = 0",
            f"{n_rank0} / {n_pop0} registros",
            "Conservados"
        ))

    # ----------------------------------
    # PASO 3: VARIABLES DERIVADAS
    # ----------------------------------

    st.subheader("Paso 3. Variables derivadas")

    st.write(
        "`Premiered` viene vacío en la mayoría de los títulos, así que el "
        "año se obtiene de `Aired`. La duración se pasa a minutos."
    )

    limpio[u.ANIO] = u.anio_desde_emision(limpio[u.EMISION])
    limpio[u.DECADA] = (limpio[u.ANIO] // 10 * 10).apply(
        lambda a: f"{int(a)}s" if pd.notna(a) else np.nan
    )
    limpio[u.DURACION_MIN] = limpio[u.DURACION].map(u.duracion_a_minutos)

    col1, col2 = st.columns(2)
    col1.metric("Títulos con año", f"{int(limpio[u.ANIO].notna().sum()):,}")
    col2.metric(
        "Títulos con duración",
        f"{int(limpio[u.DURACION_MIN].notna().sum()):,}"
    )

    st.dataframe(
        limpio[[u.EMISION, u.ANIO, u.DECADA, u.DURACION, u.DURACION_MIN]].head(8),
        width="stretch",
        hide_index=True
    )

    decisiones.append((
        "Variables derivadas",
        f"{u.ANIO}, {u.DECADA}, {u.DURACION_MIN}",
        "Calculadas desde Aired y Duration"
    ))

    # ----------------------------------
    # PASO 4: DUPLICADOS
    # ----------------------------------

    st.subheader("Paso 4. Duplicados")

    n_dup = int(limpio.duplicated().sum())
    n_id = int(limpio.duplicated(subset=u.ID).sum())
    n_nom = int(limpio.duplicated(subset=u.NOMBRE).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicados exactos", n_dup)
    col2.metric("ID repetidos", n_id)
    col3.metric("Títulos repetidos", n_nom)

    dec_dup = st.radio(
        "¿Qué haces con los duplicados exactos?",
        ["Eliminarlos (conservar la primera aparición)", "Conservarlos"],
        index=0,
        key="dec_duplicados"
    )
    if dec_dup.startswith("Eliminarlos"):
        limpio = limpio.drop_duplicates()
    decisiones.append(("Duplicados exactos", f"{n_dup} registros", dec_dup))

    if n_id > 0:
        dec_id = st.radio(
            "¿Qué haces con los ID repetidos?",
            ["Eliminarlos (conservar la primera aparición)", "Conservarlos"],
            index=0,
            key="dec_ids"
        )
        if dec_id.startswith("Eliminarlos"):
            limpio = limpio.drop_duplicates(subset=u.ID)
        decisiones.append(("ID repetidos", f"{n_id} registros", dec_id))

    if n_nom > 0:
        with st.expander("Ver los títulos repetidos"):
            repetidos = limpio[limpio.duplicated(subset=u.NOMBRE, keep=False)]
            st.dataframe(
                repetidos.sort_values(u.NOMBRE)[
                    [u.ID, u.NOMBRE, u.TIPO, u.EPISODIOS, u.ANIO]
                ],
                width="stretch",
                hide_index=True
            )
            st.caption(
                "Tienen distinto ID, tipo o número de episodios: suelen ser "
                "obras distintas con el mismo nombre (canciones, remakes)."
            )

        dec_nom = st.radio(
            "¿Qué haces con los títulos repetidos?",
            ["Conservarlos", "Eliminarlos (conservar la primera aparición)"],
            index=0,
            key="dec_titulos"
        )
        if dec_nom.startswith("Eliminarlos"):
            limpio = limpio.drop_duplicates(subset=u.NOMBRE)
        decisiones.append(("Títulos repetidos", f"{n_nom} registros", dec_nom))

    # ----------------------------------
    # PASO 5: VALORES FALTANTES
    # ----------------------------------

    st.subheader("Paso 5. Valores faltantes")

    n = len(limpio)
    pct = limpio.isnull().mean() * 100

    resumen = pd.DataFrame({
        "Variable": limpio.columns,
        "Tipo": limpio.dtypes.astype(str).values,
        "Faltantes": limpio.isnull().sum().values,
        "% faltantes": pct.round(1).values,
    })
    resumen = resumen[resumen["Faltantes"] > 0].sort_values(
        "% faltantes", ascending=False
    )

    if resumen.empty:
        st.success("No hay valores faltantes en el dataset.")
    else:
        st.dataframe(resumen, width="stretch", hide_index=True)

        # 5a. Columnas a descartar
        st.markdown("**5a. Columnas a descartar**")

        candidatas = [c for c in limpio.columns if c not in u.PROTEGIDAS]

        umbral = st.slider(
            "Descartar columnas con más de este % de faltantes",
            0, 100, 60,
            key="umbral_faltantes"
        )

        por_umbral = [c for c in candidatas if pct[c] > umbral]

        extra = st.multiselect(
            "Descartar además (columnas que no usaremos en el análisis):",
            candidatas,
            default=[c for c in DESCARTE_SUGERIDO if c in candidatas],
            key="cols_extra"
        )

        descartar = sorted(set(por_umbral) | set(extra))

        if descartar:
            st.write("Se descartarán:", ", ".join(f"`{c}`" for c in descartar))
            limpio = limpio.drop(columns=descartar)
            decisiones.append((
                "Columnas descartadas",
                ", ".join(descartar),
                f"Más de {umbral}% faltantes o sin uso en el análisis"
            ))
        else:
            st.write("No se descartará ninguna columna.")

        st.caption(
            "Las variables que usan las pestañas siguientes están protegidas "
            "y no se pueden descartar."
        )

        # 5b. Decisión por variable
        restantes = [c for c in limpio.columns if limpio[c].isnull().any()]
        filas_a_eliminar = []

        if restantes:
            st.markdown("**5b. Decisión por variable**")
            st.caption(
                "En variables como Score se recomienda conservar los nulos: "
                "rellenarlos inventaría puntuaciones que nadie dio. Las "
                "puntuaciones faltan, por ejemplo, en todos los títulos que "
                "aún no se han emitido."
            )

            with st.expander("Elegir qué hacer con cada variable", expanded=True):
                for c in restantes:
                    es_num = pd.api.types.is_numeric_dtype(limpio[c])
                    opciones = OPC_NUM if es_num else OPC_CAT

                    if es_num or c in CONSERVAR_NULOS:
                        sugerida = "Conservar nulos"
                    else:
                        sugerida = "Rellenar con 'Desconocido'"
                    indice = next(
                        i for i, o in enumerate(opciones)
                        if o.startswith(sugerida)
                    )

                    n_nulos = int(limpio[c].isnull().sum())

                    eleccion = st.selectbox(
                        f"{c} — {n_nulos:,} faltantes ({n_nulos / n * 100:.1f}%)",
                        opciones,
                        index=indice,
                        key=f"nulos_{c}"
                    )

                    if eleccion == "Rellenar con la mediana":
                        limpio[c] = limpio[c].fillna(limpio[c].median())
                    elif eleccion == "Rellenar con la media":
                        limpio[c] = limpio[c].fillna(limpio[c].mean())
                    elif eleccion == "Rellenar con 'Desconocido'":
                        limpio[c] = limpio[c].fillna("Desconocido")
                    elif eleccion == "Rellenar con la moda":
                        moda = limpio[c].mode()
                        if not moda.empty:
                            limpio[c] = limpio[c].fillna(moda.iloc[0])
                    elif eleccion == "Eliminar filas con nulo":
                        filas_a_eliminar.append(c)

                    decisiones.append((
                        f"Nulos en {c}", f"{n_nulos:,} faltantes", eleccion
                    ))

            if filas_a_eliminar:
                limpio = limpio.dropna(subset=filas_a_eliminar)

    # ----------------------------------
    # RESULTADO
    # ----------------------------------

    st.subheader("Resultado de la limpieza")

    faltantes_antes = u.faltantes_totales(df)
    faltantes_despues = int(limpio.isnull().sum().sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Filas", f"{len(limpio):,}", delta=len(limpio) - len(df))
    col2.metric(
        "Columnas",
        len(limpio.columns),
        delta=len(limpio.columns) - len(df.columns)
    )
    col3.metric(
        "Celdas faltantes",
        f"{faltantes_despues:,}",
        delta=faltantes_despues - faltantes_antes,
        delta_color="inverse",
        help="Antes se cuentan los 'UNKNOWN' como faltantes; después, solo los nulos reales."
    )
    col4.metric(
        "Duplicados exactos",
        int(limpio.duplicated().sum()),
        delta=int(limpio.duplicated().sum() - df.duplicated().sum()),
        delta_color="inverse"
    )

    st.markdown("**Registro de decisiones**")
    st.dataframe(
        pd.DataFrame(decisiones, columns=["Paso", "Elemento", "Decisión"]),
        width="stretch",
        hide_index=True
    )

    return limpio
