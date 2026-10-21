import pandas as pd
import streamlit as st

import utils as u


class _Seguro(dict):
    """Evita errores si una plantilla pide un campo que no existe."""
    def __missing__(self, clave):
        return ""


def _primer_valor(texto, vacio):
    lista = [x for x in u.dividir_lista(texto) if x != "Desconocido"]
    return lista[0] if lista else vacio


def _mas_frecuentes(serie, n=3, multiple=False):
    if multiple:
        serie = serie.apply(u.dividir_lista).explode()
    serie = serie.dropna()
    serie = serie[serie != "Desconocido"]
    if serie.empty:
        return "sin datos"
    frec = serie.value_counts(normalize=True).head(n)
    return ", ".join(f"{k} ({v:.0%})" for k, v in frec.items())


def mostrar(pred, presc):

    st.header("6. Toma de decisiones")

    st.markdown("""
    ### Pregunta inicial

    **¿Qué mensaje deberíamos mostrarle a cada tipo de anime?**
    """)

    if pred is None or presc is None:
        st.info(
            "Para llegar aquí hace falta el modelo (pestaña 4) y el plan de "
            "acción (pestaña 5). Revisa los mensajes de esas pestañas."
        )
        return

    datos, plan = presc["datos"], presc["plan"]

    # ----------------------------------
    # SELECTOR DE SEGMENTO
    # ----------------------------------

    con_titulos = set(datos[u.SEGMENTO_PLAN].unique())
    segmentos = [
        s for s in plan.sort_values("Prioridad", kind="stable")["Segmento"]
        if s in con_titulos
    ]

    segmento = st.selectbox("Segmento:", segmentos, key="dec_segmento")

    fila = plan[plan["Segmento"] == segmento].iloc[0]
    sub = datos[datos[u.SEGMENTO_PLAN] == segmento]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Títulos", f"{len(sub):,}")
    col2.metric("Prob. media", f"{sub[u.PROB].mean():.0%}")
    col3.metric("Miembros (mediana)", f"{sub[u.MIEMBROS].median():,.0f}")
    col4.metric("Prioridad", int(fila["Prioridad"]))

    # ----------------------------------
    # ACCIÓN
    # ----------------------------------

    st.subheader("Acción")

    with st.container(border=True):
        st.markdown(fila[u.ACCION])

    # ----------------------------------
    # MENSAJE
    # ----------------------------------

    st.subheader("Mensaje")

    ejemplos = sub.sort_values(u.PROB, ascending=False).head(20).reset_index(drop=True)

    i = st.selectbox(
        "Título de ejemplo:",
        list(range(len(ejemplos))),
        format_func=lambda k: f"{ejemplos.loc[k, u.NOMBRE]}  ({ejemplos.loc[k, u.PROB]:.0%})",
        key=f"dec_titulo_{segmento}"
    )

    t = ejemplos.loc[i]
    estudio = _primer_valor(t[u.ESTUDIOS], "")
    contexto = _Seguro(
        titulo=t[u.NOMBRE],
        genero=_primer_valor(t[u.GENEROS], "anime"),
        estudio_txt=f", del estudio {estudio}" if estudio else "",
    )

    with st.container(border=True):
        st.markdown(f"> {fila['Mensaje'].format_map(contexto)}")

    st.caption(
        "El mensaje es una plantilla que se rellena con el género y el estudio "
        "del título elegido."
    )

    # ----------------------------------
    # JUSTIFICACIÓN
    # ----------------------------------

    st.subheader("Justificación")

    prob_media = sub[u.PROB].mean()
    tasa_base = pred["tasa_base"]
    mediana_universo = datos[u.MIEMBROS].median()
    mediana_segmento = sub[u.MIEMBROS].median()
    relacion = "por encima" if mediana_segmento > mediana_universo else "por debajo"

    st.markdown(
        f"- **Tamaño:** {len(sub):,} títulos, el {len(sub) / len(datos):.1%} del universo analizado.\n"
        f"- **Probabilidad:** {prob_media:.0%} de lograr una puntuación de "
        f"{pred['umbral']:.1f} o más, frente a {tasa_base:.0%} en el catálogo con "
        f"puntuación ({prob_media / tasa_base:.1f} veces la tasa base).\n"
        f"- **Alcance:** mediana de {mediana_segmento:,.0f} miembros, {relacion} de la "
        f"mediana del universo ({mediana_universo:,.0f}).\n"
        f"- **Perfil:** tipos más frecuentes: {_mas_frecuentes(sub[u.TIPO])}. "
        f"Géneros más frecuentes: {_mas_frecuentes(sub[u.GENEROS], multiple=True)}.\n"
        f"- **Por qué esta acción:** {fila['Por qué']}"
    )

    coefs = pred["coefs"].sort_values("Coeficiente")
    a_favor = ", ".join(coefs.tail(3)["Variable"][::-1])
    en_contra = ", ".join(coefs.head(3)["Variable"])

    with st.expander("Cómo se calculó la probabilidad"):
        st.markdown(
            f"El modelo es una regresión logística con AUC de {pred['auc']:.2f} "
            f"en validación cruzada. Las características que más suman son: "
            f"{a_favor}. Las que más restan: {en_contra}."
        )
        st.markdown(
            "La probabilidad estima la **puntuación** que tendrá un título, no "
            "sus resultados comerciales. Antes de ejecutar una acción conviene "
            "validarla con un test A/B."
        )

    # ----------------------------------
    # TÍTULOS DEL SEGMENTO
    # ----------------------------------

    st.subheader("Títulos del segmento")

    tabla = (
        sub.sort_values(u.PROB, ascending=False)
        .head(15)[[u.NOMBRE, u.TIPO, u.ESTADO, u.ANIO, u.MIEMBROS, u.PROB]]
        .copy()
    )
    tabla[u.PROB] = (tabla[u.PROB] * 100).round(1)
    tabla = tabla.rename(columns={u.PROB: "Prob. (%)"})
    tabla.index = range(1, len(tabla) + 1)
    st.dataframe(tabla, width="stretch")

    # ----------------------------------
    # PLAN COMPLETO
    # ----------------------------------

    with st.expander("Ver el plan completo de todos los segmentos"):
        st.dataframe(
            plan.sort_values("Prioridad", kind="stable")[
                ["Segmento", "Prioridad", u.ACCION, "Mensaje"]
            ],
            width="stretch",
            hide_index=True
        )
