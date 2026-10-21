import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils as u

# ------------------------------------------------------------------
# Segmentos de acción: nivel de probabilidad x alcance
# ------------------------------------------------------------------

SEGMENTOS = {
    "Éxito probable": dict(
        nivel="Alta", alcance="Alto", prioridad=1,
        definicion="Alta probabilidad de buena puntuación y gran audiencia",
        accion="Lanzamiento destacado: portada, tráiler en redes y campaña de estreno.",
        mensaje=(
            "{titulo} llega con todo: un anime de {genero}{estudio_txt} que promete "
            "ser de lo mejor de la temporada. No te lo pierdas."
        ),
        razon=(
            "La calidad esperada y el alcance ya coinciden. Aquí la inversión "
            "amplifica algo que probablemente funcionará."
        ),
    ),
    "Joya oculta": dict(
        nivel="Alta", alcance="Bajo", prioridad=1,
        definicion="Alta probabilidad de buena puntuación, pero pocos lo conocen",
        accion="Impulsar el descubrimiento: recomendaciones personalizadas y espacio en listas «para ti».",
        mensaje=(
            "Aún pocos lo conocen, pero {titulo} tiene todo para ser tu próximo "
            "favorito: {genero}{estudio_txt}. Descúbrelo antes que nadie."
        ),
        razon=(
            "El modelo espera buena recepción pero la audiencia es pequeña: es "
            "donde cada peso invertido en visibilidad puede rendir más."
        ),
    ),
    "Apuesta popular": dict(
        nivel="Media", alcance="Alto", prioridad=2,
        definicion="Probabilidad intermedia y gran audiencia",
        accion="Mantener visibilidad y probar variantes del tráiler con tests A/B.",
        mensaje=(
            "Todos hablan de {titulo}. Entra y decide por ti mismo si está a la "
            "altura: {genero}{estudio_txt}."
        ),
        razon=(
            "Ya hay interés, pero la calidad esperada no está garantizada. "
            "Conviene medir antes de aumentar la inversión."
        ),
    ),
    "Potencial por probar": dict(
        nivel="Media", alcance="Bajo", prioridad=2,
        definicion="Probabilidad intermedia y poca audiencia",
        accion="Prueba pequeña: mostrarlo a fans del mismo género y medir la respuesta.",
        mensaje=(
            "¿Te gusta el género {genero}? Prueba el primer episodio de {titulo}"
            "{estudio_txt} y cuéntanos qué piensas."
        ),
        razon=(
            "Hay incertidumbre en calidad y en alcance. Un experimento barato "
            "reduce el riesgo antes de comprometer presupuesto."
        ),
    ),
    "Popular sin garantía": dict(
        nivel="Baja", alcance="Alto", prioridad=3,
        definicion="Baja probabilidad de buena puntuación, pero gran audiencia",
        accion="Promoción dirigida a su comunidad de fans, sin apoyarse en la calidad.",
        mensaje=(
            "{titulo} ya tiene su propia comunidad de fans. Únete a la "
            "conversación y vive con ellos este anime de {genero}{estudio_txt}."
        ),
        razon=(
            "La audiencia existe, pero el modelo no espera buena puntuación. "
            "Conviene hablarle a quienes ya lo siguen y no prometer calidad."
        ),
    ),
    "Cola larga": dict(
        nivel="Baja", alcance="Bajo", prioridad=3,
        definicion="Baja probabilidad de buena puntuación y poca audiencia",
        accion="Sin inversión específica: disponible en búsqueda y listas por género.",
        mensaje=(
            "Para quienes buscan algo diferente: {titulo}, {genero}{estudio_txt}, "
            "disponible cuando quieras."
        ),
        razon=(
            "Probabilidad y alcance bajos: el retorno esperado de promocionarlo "
            "es pequeño frente a otras opciones."
        ),
    ),
}

NOMBRE_SEGMENTO = {(s["nivel"], s["alcance"]): n for n, s in SEGMENTOS.items()}

OPCIONES_UNIVERSO = [
    "Estrenos y títulos en emisión (sin puntuación fiable)",
    "Todos los títulos sin puntuación fiable",
    "Todo el catálogo",
]


def mostrar(pred):
    """Devuelve {'datos', 'plan'} para la pestaña 6 (o None si falta el modelo)."""

    st.header("5. Estadística prescriptiva")

    st.markdown("""
    ### Pregunta inicial

    **¿Qué acción deberíamos tomar con cada anime?**
    """)

    if pred is None:
        st.info(
            "Primero hay que entrenar el modelo de la pestaña 4. Revisa el "
            "mensaje de esa pestaña."
        )
        return None

    df = pred["df"]

    st.caption(
        "Cada título recibe una acción según dos cosas: la probabilidad de "
        "que obtenga una puntuación alta (modelo de la pestaña 4) y su alcance "
        "actual (cantidad de miembros)."
    )

    # ----------------------------------
    # UNIVERSO
    # ----------------------------------

    st.subheader("¿Sobre qué títulos decidimos?")

    universo = st.radio(
        "Universo:",
        OPCIONES_UNIVERSO,
        index=0,
        key="presc_universo",
        help="Los títulos que aún no se han estrenado o siguen en emisión son "
             "los únicos que todavía se pueden promocionar."
    )

    if universo == OPCIONES_UNIVERSO[0]:
        mascara = ~df[u.ETIQUETADO] & df[u.ESTADO].isin(u.ESTADOS_ACTIVOS)
    elif universo == OPCIONES_UNIVERSO[1]:
        mascara = ~df[u.ETIQUETADO]
    else:
        mascara = pd.Series(True, index=df.index)

    datos = df[mascara].copy()

    if datos.empty:
        st.warning("No hay títulos en este universo. Prueba con otra opción.")
        return None

    st.metric("Títulos en el universo", f"{len(datos):,}")

    # ----------------------------------
    # REGLAS
    # ----------------------------------

    st.subheader("Reglas para asignar segmentos")

    col1, col2 = st.columns(2)

    bajo, alto = col1.slider(
        "Nivel de probabilidad «Media» (entre estos valores)",
        0.0, 1.0, (0.25, 0.50), 0.05,
        key=f"presc_niveles_{pred['umbral']}",
        help="Por debajo del mínimo es «Baja»; desde el máximo, «Alta»."
    )
    pct_alto = col2.slider(
        "«Alto alcance»: el % superior del universo en miembros",
        5, 50, 25,
        key="presc_alcance"
    )

    corte_miembros = datos[u.MIEMBROS].quantile(1 - pct_alto / 100)

    datos[u.NIVEL] = np.select(
        [datos[u.PROB] >= alto, datos[u.PROB] >= bajo],
        ["Alta", "Media"],
        default="Baja"
    )
    datos[u.ALCANCE] = np.where(datos[u.MIEMBROS] >= corte_miembros, "Alto", "Bajo")
    datos[u.SEGMENTO_PLAN] = [
        NOMBRE_SEGMENTO[(n, a)] for n, a in zip(datos[u.NIVEL], datos[u.ALCANCE])
    ]

    st.caption(
        f"«Alto alcance» equivale a {corte_miembros:,.0f} miembros o más. "
        f"Tasa base de puntuación alta en el catálogo con puntuación: "
        f"{pred['tasa_base']:.0%}."
    )

    # ----------------------------------
    # DISTRIBUCIÓN
    # ----------------------------------

    st.subheader("Distribución de títulos")

    col1, col2 = st.columns(2)

    with col1:
        matriz = pd.crosstab(datos[u.NIVEL], datos[u.ALCANCE]).reindex(
            index=["Alta", "Media", "Baja"], columns=["Alto", "Bajo"], fill_value=0
        )
        matriz.index.name = "Probabilidad ↓ / Alcance →"
        st.dataframe(matriz, width="stretch")

    with col2:
        conteo = (
            datos[u.SEGMENTO_PLAN].value_counts()
            .reindex(list(SEGMENTOS), fill_value=0)
            .rename_axis("Segmento")
            .reset_index(name="Títulos")
        )
        grafica = (
            alt.Chart(conteo)
            .mark_bar()
            .encode(
                x=alt.X("Títulos:Q"),
                y=alt.Y("Segmento:N", sort=list(SEGMENTOS), title=None),
                tooltip=["Segmento", "Títulos"],
            )
        )
        st.altair_chart(grafica, width="stretch")

    # ----------------------------------
    # TABLA DE ACCIÓN POR SEGMENTO
    # ----------------------------------

    st.subheader("Acción recomendada por segmento")

    st.caption(
        "Las columnas «Acción recomendada» y «Prioridad» se pueden editar "
        "haciendo doble clic. La pestaña 6 usa lo que dejes aquí."
    )

    filas = []
    for nombre, s in SEGMENTOS.items():
        sub = datos[datos[u.SEGMENTO_PLAN] == nombre]
        filas.append({
            "Segmento": nombre,
            "Cómo se define": s["definicion"],
            "Títulos": len(sub),
            "Prob. media (%)": round(sub[u.PROB].mean() * 100, 1) if len(sub) else None,
            "Miembros (mediana)": float(sub[u.MIEMBROS].median()) if len(sub) else None,
            u.ACCION: s["accion"],
            "Prioridad": s["prioridad"],
        })
    plan_base = pd.DataFrame(filas)

    editado = st.data_editor(
        plan_base,
        key="presc_plan",
        hide_index=True,
        width="stretch",
        disabled=["Segmento", "Cómo se define", "Títulos",
                  "Prob. media (%)", "Miembros (mediana)"],
        column_config={
            u.ACCION: st.column_config.TextColumn(u.ACCION, width="large"),
            "Prioridad": st.column_config.NumberColumn(
                "Prioridad", min_value=1, max_value=3, step=1, format="%d",
                help="1 = más urgente"
            ),
            "Miembros (mediana)": st.column_config.NumberColumn(format="%d"),
        },
    )

    plan = editado.copy()
    plan["Mensaje"] = plan["Segmento"].map(lambda n: SEGMENTOS[n]["mensaje"])
    plan["Por qué"] = plan["Segmento"].map(lambda n: SEGMENTOS[n]["razon"])

    datos[u.ACCION] = datos[u.SEGMENTO_PLAN].map(
        plan.set_index("Segmento")[u.ACCION]
    )

    # ----------------------------------
    # DESCARGA
    # ----------------------------------

    st.subheader("Plan por título")

    prioridad = plan.set_index("Segmento")["Prioridad"]
    export = (
        datos.assign(_prioridad=datos[u.SEGMENTO_PLAN].map(prioridad))
        .sort_values(["_prioridad", u.PROB], ascending=[True, False])
        [[u.NOMBRE, u.TIPO, u.ESTADO, u.ANIO, u.MIEMBROS, u.PROB,
          u.NIVEL, u.ALCANCE, u.SEGMENTO_PLAN, u.ACCION]]
        .copy()
    )
    export[u.PROB] = export[u.PROB].round(3)

    st.dataframe(export.head(50), width="stretch", hide_index=True)
    st.caption("Se muestran los 50 primeros, ordenados por prioridad y probabilidad.")

    st.download_button(
        "Descargar el plan completo (CSV)",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="plan_de_accion_anime.csv",
        mime="text/csv",
    )

    return {"datos": datos, "plan": plan}
