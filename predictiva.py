import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import utils as u

NUMERICAS = [
    "Episodios (log)", "Duración (log)", "Año",
    "N° de géneros", "N° de estudios",
]
CATEGORICAS = ["Tipo", "Fuente", "Clasificación"]


# ------------------------------------------------------------------
# Preparación de variables
# ------------------------------------------------------------------

def _listas(serie):
    """'Action, Comedy' -> ['Action', 'Comedy'] (sin 'Desconocido')."""
    return serie.apply(u.dividir_lista).apply(
        lambda lista: [x for x in lista if x != "Desconocido"]
    )


def _multi_hot(listas, top, prefijo):
    """Una columna 0/1 por cada valor frecuente (género o estudio)."""
    if not top:
        return pd.DataFrame(index=listas.index)
    ex = listas.explode().dropna()
    ex = ex[ex.isin(top)]
    m = pd.crosstab(ex.index, ex).reindex(
        index=listas.index, columns=top, fill_value=0
    )
    m.columns = [f"{prefijo}: {c}" for c in top]
    return (m > 0).astype(int)


@st.cache_data(show_spinner="Preparando variables del modelo...")
def preparar_features(base, k_generos, k_estudios):
    generos = _listas(base[u.GENEROS])
    estudios = _listas(base[u.ESTUDIOS])

    top_g = generos.explode().value_counts().head(k_generos).index.tolist()
    top_e = estudios.explode().value_counts().head(k_estudios).index.tolist()

    X = pd.DataFrame(index=base.index)
    X["Episodios (log)"] = np.log1p(base[u.EPISODIOS])
    X["Duración (log)"] = np.log1p(base[u.DURACION_MIN])
    X["Año"] = base[u.ANIO]
    X["N° de géneros"] = generos.map(len)
    X["N° de estudios"] = estudios.map(len)
    X["Tipo"] = base[u.TIPO]
    X["Fuente"] = base[u.FUENTE]
    X["Clasificación"] = base[u.CLASIFICACION]

    return pd.concat(
        [X, _multi_hot(generos, top_g, "Género"),
         _multi_hot(estudios, top_e, "Estudio")],
        axis=1
    )


# ------------------------------------------------------------------
# Modelo
# ------------------------------------------------------------------

def _nombre_variable(n):
    grupo, _, resto = n.partition("__")
    if grupo == "cat":
        col, _, valor = resto.partition("_")
        return f"{col} = {valor}"
    if grupo == "num":
        return f"{resto} (por desv. estándar)"
    return resto


@st.cache_data(show_spinner="Entrenando el modelo...")
def entrenar(X, y, etiquetado, C):
    """Regresión logística.

    - Probabilidades de los títulos con puntuación: validación cruzada
      (cada probabilidad la calcula un modelo que NO vio ese título).
    - Probabilidades del resto: modelo final entrenado con todos los
      títulos con puntuación.
    """
    binarias = [c for c in X.columns if c not in NUMERICAS + CATEGORICAS]

    preproceso = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("esc", StandardScaler()),
        ]), NUMERICAS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAS),
        ("bin", "passthrough", binarias),
    ])

    modelo = Pipeline([
        ("pre", preproceso),
        ("lr", LogisticRegression(C=C, max_iter=2000)),
    ])

    Xl = X.loc[etiquetado]
    yl = y.loc[etiquetado].astype(int)

    p_oof = cross_val_predict(
        modelo, Xl, yl,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba"
    )[:, 1]

    modelo.fit(Xl, yl)

    prob = pd.Series(np.nan, index=X.index)
    prob.loc[etiquetado] = p_oof
    if (~etiquetado).any():
        prob.loc[~etiquetado] = modelo.predict_proba(X.loc[~etiquetado])[:, 1]

    nombres = modelo.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({
        "Variable": [_nombre_variable(n) for n in nombres],
        "Coeficiente": modelo.named_steps["lr"].coef_[0],
    })
    coefs["Odds ratio"] = np.exp(coefs["Coeficiente"])

    return {
        "prob": prob,
        "p_oof": pd.Series(p_oof, index=Xl.index),
        "y": yl,
        "coefs": coefs,
        "auc": float(roc_auc_score(yl, p_oof)),
    }


# ------------------------------------------------------------------
# Pestaña
# ------------------------------------------------------------------

def mostrar(df):
    """Devuelve un diccionario con las probabilidades para las pestañas 5 y 6
    (o None si no hay datos suficientes para entrenar)."""

    st.header("4. Estadística predictiva")

    st.markdown("""
    ### Pregunta inicial

    **¿Qué características tienen los anime mejor puntuados y podemos predecir cuáles lo serán?**
    """)

    st.caption(
        "En este dataset no hay compradores. Usamos «lograr una puntuación alta» "
        "como el resultado a predecir: es el equivalente de «comprar en la última "
        "campaña»."
    )

    # ----------------------------------
    # DEFINIR EL OBJETIVO
    # ----------------------------------

    st.subheader("Qué vamos a predecir")

    col1, col2 = st.columns(2)

    umbral = col1.slider(
        "Una puntuación es «alta» si es mayor o igual a:",
        5.0, 9.0, 7.0, 0.1,
        key="pred_umbral_score"
    )
    min_votos = col2.slider(
        "Mínimo de votantes para confiar en la puntuación",
        100, 5000, 100, step=100,
        key="pred_min_votos",
        help="Con pocos votos la puntuación es inestable. Subirlo deja menos "
             "títulos para entrenar y los que quedan son más populares."
    )

    etiquetado = (
        df[u.PUNTUACION].notna()
        & (df[u.VOTANTES].fillna(0) >= min_votos)
    )
    y = (df[u.PUNTUACION] >= umbral).astype(float).where(etiquetado)

    n_et = int(etiquetado.sum())
    n_alta = int(y.sum())
    n_no_alta = n_et - n_alta

    col1, col2, col3 = st.columns(3)
    col1.metric("Títulos con puntuación fiable", f"{n_et:,}")
    col2.metric("% con puntuación alta", f"{n_alta / max(n_et, 1) * 100:.1f}%")
    col3.metric(
        "Títulos a predecir",
        f"{len(df) - n_et:,}",
        help="Sin puntuación o con menos votantes que el mínimo."
    )

    if min_votos > 100:
        st.caption(
            "Con este mínimo el modelo aprende solo de títulos con bastante "
            "audiencia. Sus predicciones para títulos poco conocidos pueden "
            "ser optimistas."
        )

    if min(n_alta, n_no_alta) < 100:
        st.error(
            "Hay muy pocos títulos de una de las dos clases para entrenar. "
            "Cambia el umbral o el mínimo de votantes, o revisa la limpieza "
            "(por ejemplo, rellenar `Score` elimina la diferencia entre títulos "
            "con y sin puntuación)."
        )
        return None

    # ----------------------------------
    # PERFIL: ALTA vs RESTO
    # ----------------------------------

    st.subheader("Características de los títulos con puntuación alta")

    et = df[etiquetado].copy()
    et["_alta"] = y[etiquetado].astype(int)

    cols_perfil = [
        c for c in [u.EPISODIOS, u.DURACION_MIN, u.ANIO, u.MIEMBROS, u.FAVORITOS]
        if c in et.columns
    ]
    perfil = et.groupby("_alta")[cols_perfil].median().T
    perfil.columns = ["No alta", "Alta"]
    perfil.index = [u.ETIQUETAS.get(c, c) for c in perfil.index]
    st.markdown("**Medianas**")
    st.dataframe(perfil.round(1), width="stretch")

    dimensiones = {
        "Tipo": (u.TIPO, False),
        "Fuente original": (u.FUENTE, False),
        "Clasificación por edad": (u.CLASIFICACION, False),
        "Género / tema": (u.GENEROS, True),
        "Estudio": (u.ESTUDIOS, True),
    }

    col1, col2 = st.columns([2, 1])
    nombre_dim = col1.selectbox(
        "Tasa de puntuación alta por:", list(dimensiones), key="pred_dimension"
    )
    min_n = col2.slider(
        "Mínimo de títulos", 10, 200, 30, key="pred_min_n_dim"
    )

    columna, multiple = dimensiones[nombre_dim]
    seg = et[[columna, "_alta"]].copy()
    if multiple:
        seg["Categoría"] = seg[columna].apply(u.dividir_lista)
        seg = seg.explode("Categoría")
    else:
        seg["Categoría"] = seg[columna]
    seg = seg.dropna(subset=["Categoría"])
    seg = seg[seg["Categoría"] != "Desconocido"]

    tasas = (
        seg.groupby("Categoría")["_alta"]
        .agg(Títulos="size", Tasa="mean")
        .reset_index()
    )
    tasas = tasas[tasas["Títulos"] >= min_n].sort_values("Tasa", ascending=False)
    tasas["Tasa (%)"] = (tasas["Tasa"] * 100).round(1)
    tasas = tasas.head(15)

    if tasas.empty:
        st.warning("Ninguna categoría cumple el mínimo de títulos.")
    else:
        orden = list(tasas["Categoría"])
        datos_graf = tasas[["Categoría", "Títulos", "Tasa (%)"]].rename(
            columns={"Tasa (%)": "tasa"}
        )
        grafica = (
            alt.Chart(datos_graf)
            .mark_bar()
            .encode(
                x=alt.X("tasa:Q", title="% con puntuación alta"),
                y=alt.Y("Categoría:N", sort=orden, title=None),
                tooltip=[
                    "Categoría",
                    "Títulos",
                    alt.Tooltip("tasa:Q", title="Tasa (%)"),
                ],
            )
        )
        st.altair_chart(grafica, width="stretch")
        st.caption(
            f"Tasa base: {n_alta / n_et * 100:.1f}%. Las categorías por encima "
            "de esa cifra concentran más títulos bien puntuados."
        )

    # ----------------------------------
    # MODELO
    # ----------------------------------

    st.subheader("Modelo de regresión logística")

    with st.expander("Configuración del modelo"):
        col1, col2, col3 = st.columns(3)
        k_generos = col1.slider(
            "Géneros incluidos (los más frecuentes)", 5, 60, 40, key="pred_k_gen"
        )
        k_estudios = col2.slider(
            "Estudios incluidos (los más frecuentes)", 0, 100, 30, key="pred_k_est"
        )
        C = col3.select_slider(
            "Regularización (C)",
            options=[0.01, 0.1, 1.0, 10.0],
            value=1.0,
            key="pred_C",
            help="Valores bajos simplifican el modelo; valores altos lo ajustan más a los datos."
        )

    st.caption(
        "Variables usadas: tipo, fuente, clasificación por edad, episodios, "
        "duración, año, número de géneros y estudios, y los géneros y estudios "
        "más frecuentes. Miembros, Favoritos y Votantes **no** se usan: son "
        "consecuencia de la popularidad y delatarían la respuesta."
    )

    columnas_base = [
        u.GENEROS, u.ESTUDIOS, u.EPISODIOS, u.DURACION_MIN, u.ANIO,
        u.TIPO, u.FUENTE, u.CLASIFICACION,
    ]
    X = preparar_features(df[columnas_base], k_generos, k_estudios)
    res = entrenar(X, y, etiquetado, C)

    # ----------------------------------
    # RESULTADOS
    # ----------------------------------

    st.subheader("¿Qué tan bien predice?")

    st.caption(
        "Validación cruzada de 5 particiones: cada título se evalúa con un "
        "modelo que no lo vio durante el entrenamiento."
    )

    yl, p = res["y"], res["p_oof"]

    umbral_dec = st.slider(
        "Umbral de decisión: se predice «alta» si la probabilidad es mayor o igual a:",
        0.05, 0.95, 0.40, 0.05,
        key="pred_umbral_decision"
    )
    pred = (p >= umbral_dec).astype(int)

    referencia = max(yl.mean(), 1 - yl.mean())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC", f"{res['auc']:.3f}", help="0,5 = azar; 1 = perfecto.")
    col2.metric(
        "Exactitud", f"{accuracy_score(yl, pred):.1%}",
        delta=f"{(accuracy_score(yl, pred) - referencia) * 100:+.1f} pts vs. referencia",
        help=f"Referencia: decir siempre «no alta» acierta {referencia:.1%}."
    )
    col3.metric(
        "Precisión", f"{precision_score(yl, pred, zero_division=0):.1%}",
        help="De los que el modelo marca como «alta», cuántos lo son."
    )
    col4.metric(
        "Recall", f"{recall_score(yl, pred, zero_division=0):.1%}",
        help="De los que realmente son «alta», cuántos detecta el modelo."
    )
    col5.metric("F1", f"{f1_score(yl, pred, zero_division=0):.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Matriz de confusión**")
        cm = confusion_matrix(yl, pred)
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Real: no alta", "Real: alta"],
                columns=["Predicho: no alta", "Predicho: alta"],
            ),
            width="stretch"
        )

        st.markdown("**¿Las probabilidades son confiables?**")
        niveles = pd.cut(
            p, [0, 0.1, 0.25, 0.5, 0.75, 1.0],
            include_lowest=True,
            labels=["0–10%", "10–25%", "25–50%", "50–75%", "75–100%"]
        )
        calibracion = (
            pd.DataFrame({"nivel": niveles, "real": yl.values, "prob": p.values})
            .groupby("nivel", observed=True)
            .agg(Títulos=("real", "size"),
                 **{"Probabilidad media": ("prob", "mean"),
                    "Tasa real": ("real", "mean")})
            .reset_index()
            .rename(columns={"nivel": "Probabilidad estimada"})
        )
        calibracion["Probabilidad media"] = (calibracion["Probabilidad media"] * 100).round(1)
        calibracion["Tasa real"] = (calibracion["Tasa real"] * 100).round(1)
        st.dataframe(calibracion, width="stretch", hide_index=True)
        st.caption(
            "Si el modelo es confiable, la tasa real se parece a la "
            "probabilidad media de cada nivel (ambas en %)."
        )

    with col2:
        st.markdown("**Curva ROC**")
        fpr, tpr, _ = roc_curve(yl, p)
        idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
        roc = pd.DataFrame({"FPR": fpr[idx], "TPR": tpr[idx]})
        diagonal = pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})

        curva = alt.Chart(roc).mark_line().encode(
            x=alt.X("FPR:Q", title="Falsos positivos"),
            y=alt.Y("TPR:Q", title="Verdaderos positivos"),
        )
        azar = alt.Chart(diagonal).mark_line(strokeDash=[4, 4]).encode(
            x="FPR:Q", y="TPR:Q"
        )
        st.altair_chart(curva + azar, width="stretch")

    # ----------------------------------
    # VARIABLES QUE MÁS PESAN
    # ----------------------------------

    st.subheader("Variables que más pesan")

    n_top = st.slider("Variables a mostrar por lado", 5, 25, 10, key="pred_n_top")

    coefs = res["coefs"].sort_values("Coeficiente")
    negativos = coefs.head(n_top).assign(Efecto="Disminuye la probabilidad")
    positivos = coefs.tail(n_top).assign(Efecto="Aumenta la probabilidad")
    top = pd.concat([negativos, positivos]).rename(columns={"Odds ratio": "odds"})

    grafica = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("Coeficiente:Q", title="Coeficiente (log-odds)"),
            y=alt.Y("Variable:N", sort=list(top["Variable"]), title=None),
            color=alt.Color(
                "Efecto:N",
                scale=alt.Scale(
                    domain=["Aumenta la probabilidad", "Disminuye la probabilidad"],
                    range=["#2e7d32", "#c62828"],
                ),
            ),
            tooltip=[
                "Variable",
                alt.Tooltip("Coeficiente:Q", format=".2f"),
                alt.Tooltip("odds:Q", title="Odds ratio", format=".2f"),
            ],
        )
    )
    st.altair_chart(grafica, width="stretch")

    st.caption(
        "Un odds ratio de 2 significa que la probabilidad de puntuación alta "
        "(en términos de odds) se duplica con esa característica. En las "
        "variables numéricas es por cada +1 desviación estándar. Las "
        "categorías «Desconocido» pesan en contra: los títulos con metadatos "
        "incompletos suelen ser poco conocidos. Son asociaciones, no causas."
    )

    # ----------------------------------
    # PREDICCIONES
    # ----------------------------------

    st.subheader("Predicciones para títulos sin puntuación fiable")

    out = df.copy()
    out[u.PROB] = res["prob"]
    out[u.ETIQUETADO] = etiquetado
    out[u.ALTA_REAL] = y

    pendientes = out[~out[u.ETIQUETADO]]

    if pendientes.empty:
        st.info("Todos los títulos tienen puntuación fiable: no hay nada que predecir.")
    else:
        mejores = (
            pendientes.nlargest(15, u.PROB)
            [[u.NOMBRE, u.TIPO, u.ESTADO, u.ANIO, u.PROB]]
            .copy()
        )
        mejores[u.PROB] = (mejores[u.PROB] * 100).round(1)
        mejores = mejores.rename(columns={u.PROB: "Prob. de puntuación alta (%)"})
        mejores.index = range(1, len(mejores) + 1)
        st.dataframe(mejores, width="stretch")
        st.caption(
            "Los 15 títulos sin puntuación fiable con mayor probabilidad. "
            "En la pestaña siguiente se convierten en un plan de acción."
        )

    return {
        "df": out,
        "umbral": umbral,
        "min_votos": min_votos,
        "tasa_base": float(yl.mean()),
        "auc": res["auc"],
        "coefs": res["coefs"],
    }
