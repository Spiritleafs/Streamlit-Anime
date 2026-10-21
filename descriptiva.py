import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils as u

# Variables numéricas disponibles para distribuciones y correlaciones
NUMERICAS = [
    u.PUNTUACION, u.EPISODIOS, u.DURACION_MIN, u.MIEMBROS, u.FAVORITOS,
    u.VOTANTES, u.POPULARIDAD, u.RANKING, u.ANIO,
]


def _etq(c):
    return u.ETIQUETAS.get(c, c)


def _fmt(valor, formato="{:,.2f}"):
    return "—" if pd.isna(valor) else formato.format(valor)


def _segmentar(df, columna, multiple, incluir_desconocido):
    """Crea la columna 'Segmento'. Si la variable tiene varios valores
    por fila (géneros, estudios), genera una fila por valor."""
    seg = df.copy()
    if multiple:
        seg["Segmento"] = seg[columna].apply(u.dividir_lista)
        seg = seg.explode("Segmento")
    else:
        seg["Segmento"] = seg[columna]
    seg = seg.dropna(subset=["Segmento"])
    seg["Segmento"] = seg["Segmento"].astype(str)
    if not incluir_desconocido:
        seg = seg[seg["Segmento"] != "Desconocido"]
    return seg


def mostrar(df):

    # ----------------------------------
    # ENCABEZADO
    # ----------------------------------

    st.header("3. Estadística descriptiva")

    st.markdown("""
    ### Pregunta inicial

    **¿Cómo se comportan los anime, qué segmentos existen y qué KPIs los describen mejor?**
    """)

    df = df.copy()
    df["_finalizado"] = df[u.ESTADO].eq("Finished Airing").astype(float)

    # ----------------------------------
    # KPIs
    # ----------------------------------

    st.subheader("KPIs generales")

    con_puntuacion = df[u.PUNTUACION].notna()

    fila1 = st.columns(4)
    fila1[0].metric("Anime analizados", f"{len(df):,}")
    fila1[1].metric(
        "Puntuación promedio",
        _fmt(df[u.PUNTUACION].mean()),
        help=f"Calculada sobre {int(con_puntuacion.sum()):,} títulos con puntuación."
    )
    fila1[2].metric("Episodios (mediana)", _fmt(df[u.EPISODIOS].median(), "{:,.0f}"))
    fila1[3].metric(
        "Duración (mediana, min)",
        _fmt(df[u.DURACION_MIN].median(), "{:,.0f}")
    )

    fila2 = st.columns(4)
    fila2[0].metric("% finalizados", f"{df['_finalizado'].mean() * 100:.1f}%")
    fila2[1].metric("% con puntuación", f"{con_puntuacion.mean() * 100:.1f}%")
    fila2[2].metric(
        "Miembros totales",
        f"{df[u.MIEMBROS].sum() / 1e6:,.1f} M",
        help="Suma de usuarios que tienen el título en su lista."
    )
    if df[u.ANIO].notna().any():
        fila2[3].metric(
            "Años cubiertos",
            f"{int(df[u.ANIO].min())}–{int(df[u.ANIO].max())}"
        )

    # ----------------------------------
    # DISTRIBUCIÓN
    # ----------------------------------

    st.subheader("Distribución de una variable numérica")

    numericas = [c for c in NUMERICAS if c in df.columns]

    var = st.selectbox(
        "Variable:", numericas, format_func=_etq, key="desc_var_num"
    )

    serie = df[var].dropna()

    col1, col2 = st.columns(2)
    usar_log = col1.checkbox(
        "Escala logarítmica (log10)",
        value=bool(serie.skew() > 3),
        key=f"desc_log_{var}",
        help="Recomendada para variables con valores extremos, como Miembros o Episodios."
    )
    bins = col2.slider("Número de intervalos", 5, 50, 20, key="desc_bins")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(serie.describe().round(2).to_frame(), width="stretch")

    with col2:
        valores = serie[serie > 0] if usar_log else serie
        if usar_log:
            valores = np.log10(valores)

        conteos, bordes = np.histogram(valores, bins=bins)
        hist = pd.DataFrame({
            "inicio": bordes[:-1],
            "fin": bordes[1:],
            "frecuencia": conteos,
        })

        titulo_x = f"log10({_etq(var)})" if usar_log else _etq(var)
        grafica = (
            alt.Chart(hist)
            .mark_bar()
            .encode(
                x=alt.X("inicio:Q", title=titulo_x),
                x2="fin:Q",
                y=alt.Y("frecuencia:Q", title="Frecuencia"),
                tooltip=["inicio", "fin", "frecuencia"],
            )
        )
        st.altair_chart(grafica, width="stretch")

    # ----------------------------------
    # SEGMENTOS
    # ----------------------------------

    st.subheader("Segmentos")

    segmentadores = {
        "Tipo": (u.TIPO, False),
        "Estado de emisión": (u.ESTADO, False),
        "Fuente original": (u.FUENTE, False),
        "Clasificación por edad": (u.CLASIFICACION, False),
        "Década de estreno": (u.DECADA, False),
        "Género / tema": (u.GENEROS, True),
        "Estudio": (u.ESTUDIOS, True),
    }
    segmentadores = {k: v for k, v in segmentadores.items() if v[0] in df.columns}

    col1, col2 = st.columns([2, 1])
    nombre = col1.selectbox(
        "Segmentar por:", list(segmentadores), key="desc_segmento"
    )
    incluir_desc = col2.checkbox(
        "Incluir «Desconocido»", value=False, key="desc_incluir_desc"
    )

    columna, multiple = segmentadores[nombre]
    seg = _segmentar(df, columna, multiple, incluir_desc)

    grupo = seg.groupby("Segmento")
    tabla = grupo.size().rename("Anime").to_frame()
    tabla["Puntuación promedio"] = grupo[u.PUNTUACION].mean().round(2)
    tabla["Miembros (mediana)"] = grupo[u.MIEMBROS].median()
    tabla["Favoritos (mediana)"] = grupo[u.FAVORITOS].median()
    tabla["Episodios (mediana)"] = grupo[u.EPISODIOS].median()
    tabla["% finalizados"] = (grupo["_finalizado"].mean() * 100).round(1)
    tabla = tabla.reset_index()

    col1, col2, col3 = st.columns(3)
    metrica = col1.selectbox(
        "Métrica:",
        [m for m in tabla.columns if m != "Segmento"],
        key="desc_metrica"
    )
    min_n = col2.slider(
        "Mínimo de anime por segmento", 1, 100, 10, key="desc_min_n"
    )
    top_n = col3.slider("Máximo de segmentos", 5, 30, 15, key="desc_top_n")

    vista = tabla[tabla["Anime"] >= min_n].dropna(subset=[metrica])

    if nombre == "Década de estreno":
        vista = vista.sort_values("Segmento").tail(top_n)
    else:
        vista = vista.sort_values(metrica, ascending=False).head(top_n)

    if vista.empty:
        st.warning("Ningún segmento cumple el mínimo de anime. Reduce el filtro.")
    else:
        datos_graf = vista[["Segmento", "Anime"]].copy()
        datos_graf["valor"] = vista[metrica]

        orden = list(datos_graf["Segmento"])
        grafica = (
            alt.Chart(datos_graf)
            .mark_bar()
            .encode(
                x=alt.X("valor:Q", title=metrica),
                y=alt.Y("Segmento:N", sort=orden, title=None),
                tooltip=[
                    "Segmento",
                    alt.Tooltip("valor:Q", title=metrica),
                    "Anime",
                ],
            )
        )
        st.altair_chart(grafica, width="stretch")

        if nombre != "Década de estreno":
            mejor = vista.iloc[0]
            st.caption(
                f"Mayor valor de «{metrica}»: **{mejor['Segmento']}** "
                f"({mejor[metrica]:,.2f}, sobre {int(mejor['Anime']):,} anime)."
            )

        with st.expander("Ver tabla completa de KPIs por segmento"):
            st.dataframe(
                tabla.sort_values("Anime", ascending=False),
                width="stretch",
                hide_index=True
            )

    if multiple:
        st.caption(
            "Un anime con varios valores cuenta una vez en cada uno, por lo "
            "que los totales pueden superar el número de anime. En «Género / "
            "tema» también aparecen demografías (Shounen, Seinen...) y temas "
            "(School, Space...)."
        )

    # ----------------------------------
    # EVOLUCIÓN POR AÑO
    # ----------------------------------

    if df[u.ANIO].notna().any():

        st.subheader("Evolución por año")

        a_min, a_max = int(df[u.ANIO].min()), int(df[u.ANIO].max())
        hoy = datetime.date.today().year

        rango = st.slider(
            "Rango de años",
            a_min, a_max,
            (max(a_min, 1960), min(a_max, hoy)),
            key="desc_rango_anios"
        )

        en_rango = df[df[u.ANIO].between(*rango)]
        por_anio = en_rango.groupby(en_rango[u.ANIO].astype(int))

        series = {
            "Cantidad de anime": por_anio.size(),
            "Puntuación promedio": por_anio[u.PUNTUACION].mean(),
            "Episodios (mediana)": por_anio[u.EPISODIOS].median(),
            "Miembros (mediana)": por_anio[u.MIEMBROS].median(),
        }

        sel = st.selectbox("Métrica:", list(series), key="desc_tendencia")
        st.line_chart(series[sel])

        st.caption(
            "Los años posteriores al actual son títulos anunciados. Los más "
            "recientes tienen pocos votantes, por lo que su puntuación es "
            "poco estable."
        )

    # ----------------------------------
    # MEJOR PUNTUADOS
    # ----------------------------------

    st.subheader("Los 10 mejor puntuados")

    min_votos = st.slider(
        "Mínimo de votantes",
        100, 50000, 5000, step=100,
        key="desc_min_votos",
        help="Evita que un título con pocos votos llegue al primer lugar."
    )

    top = (
        df[df[u.VOTANTES] >= min_votos]
        .nlargest(10, u.PUNTUACION)
        [[u.NOMBRE, u.TIPO, u.ANIO, u.PUNTUACION, u.VOTANTES, u.MIEMBROS]]
        .rename(columns=_etq)
    )
    top.index = range(1, len(top) + 1)
    st.dataframe(top, width="stretch")

    # ----------------------------------
    # CORRELACIONES
    # ----------------------------------

    st.subheader("Relación entre variables numéricas")

    corr = df[numericas].corr(method="spearman").round(2)
    corr = corr.rename(index=_etq, columns=_etq)

    largo = (
        corr.rename_axis("x").reset_index()
        .melt(id_vars="x", var_name="y", value_name="r")
    )

    base = alt.Chart(largo).encode(
        x=alt.X("x:N", title=None),
        y=alt.Y("y:N", title=None),
    )
    mapa = base.mark_rect().encode(
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"))
    )
    texto = base.mark_text().encode(
        text="r:Q",
        color=alt.condition(
            "abs(datum.r) > 0.6", alt.value("white"), alt.value("black")
        ),
    )

    st.altair_chart(mapa + texto, width="stretch")
    st.caption(
        "Correlación de Spearman (adecuada para variables con valores "
        "extremos). Ranking y Popularidad son posiciones (1 = mejor), por eso "
        "su correlación con Puntuación o Miembros sale negativa."
    )
