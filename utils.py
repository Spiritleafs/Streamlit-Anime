import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Columnas del CSV (anime-dataset-2026.csv)
# ------------------------------------------------------------------

ID = "anime_id"
NOMBRE = "Name"
NOMBRE_EN = "English name"
NOMBRE_OTRO = "Other name"
PUNTUACION = "Score"
GENEROS = "Genres"
SINOPSIS = "Synopsis"
TIPO = "Type"
EPISODIOS = "Episodes"
EMISION = "Aired"
ESTRENO = "Premiered"
ESTADO = "Status"
PRODUCTORAS = "Producers"
LICENCIANTES = "Licensors"
ESTUDIOS = "Studios"
FUENTE = "Source"
DURACION = "Duration"
CLASIFICACION = "Rating"
RANKING = "Rank"
POPULARIDAD = "Popularity"
FAVORITOS = "Favorites"
VOTANTES = "Scored By"
MIEMBROS = "Members"
IMAGEN = "Image URL"

# Columnas derivadas (se crean en la pestaña "Limpiar")
ANIO = "Año"
DECADA = "Década"
DURACION_MIN = "Duración (min)"

# Vienen como texto en el CSV pero son numéricas
NUMERICAS_EN_TEXTO = [PUNTUACION, EPISODIOS, RANKING, VOTANTES]

# Columnas que usan las pestañas siguientes: no se pueden descartar
PROTEGIDAS = [
    ID, NOMBRE, PUNTUACION, GENEROS, TIPO, EPISODIOS, EMISION, ESTADO,
    ESTUDIOS, FUENTE, DURACION, CLASIFICACION, RANKING, POPULARIDAD,
    FAVORITOS, VOTANTES, MIEMBROS, ANIO, DECADA, DURACION_MIN,
]

# Etiquetas en español para gráficas y selectores
ETIQUETAS = {
    ID: "ID",
    NOMBRE: "Título",
    PUNTUACION: "Puntuación",
    EPISODIOS: "Episodios",
    DURACION_MIN: "Duración (min)",
    MIEMBROS: "Miembros",
    FAVORITOS: "Favoritos",
    VOTANTES: "Votantes",
    POPULARIDAD: "Popularidad (posición)",
    RANKING: "Ranking (posición)",
    ANIO: "Año",
    TIPO: "Tipo",
}

# ------------------------------------------------------------------
# Faltantes escondidos como texto
# ------------------------------------------------------------------

MARCADORES = {"UNKNOWN", "NOT AVAILABLE", ""}
PREFIJO_SIN_SINOPSIS = "No synopsis information has been added"


def es_texto(serie):
    return pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie)


def _mascara_marcadores(serie, columna=None):
    s = serie.str.strip()
    mascara = s.str.upper().isin(MARCADORES)
    if columna == SINOPSIS:
        mascara = mascara | s.str.startswith(PREFIJO_SIN_SINOPSIS, na=False)
    return mascara.fillna(False).astype(bool)


def contar_marcadores(df):
    """Cantidad de celdas 'UNKNOWN' / 'Not available' / sinopsis vacía por columna."""
    conteo = {}
    for c in df.columns:
        conteo[c] = int(_mascara_marcadores(df[c], c).sum()) if es_texto(df[c]) else 0
    return pd.Series(conteo)


def faltantes_totales(df):
    """Celdas faltantes: nulos reales + marcadores de texto."""
    return int(df.isnull().sum().sum() + contar_marcadores(df).sum())


def marcadores_a_nulos(df):
    out = df.copy()
    for c in out.columns:
        if es_texto(out[c]):
            s = out[c].str.strip()
            out[c] = s.mask(_mascara_marcadores(out[c], c))
    return out


def es_numerica_oculta(serie):
    """True si es texto pero casi todos sus valores válidos son números."""
    if not es_texto(serie):
        return False
    muestra = serie.head(5000)
    validos = muestra[~_mascara_marcadores(muestra)].dropna()
    if validos.empty:
        return False
    return pd.to_numeric(validos, errors="coerce").notna().mean() >= 0.95


# ------------------------------------------------------------------
# Variables derivadas
# ------------------------------------------------------------------


def anio_desde_emision(serie):
    """'Apr 3, 1998 to Apr 24, 1999' -> 1998."""
    return pd.to_numeric(serie.str.extract(r"(\d{4})")[0], errors="coerce")


def duracion_a_minutos(texto):
    """'24 min. per ep.' -> 24 | '1 hr. 55 min.' -> 115 | '30 sec.' -> 0.5."""
    if not isinstance(texto, str):
        return np.nan
    t = texto.lower()
    h = re.search(r"(\d+)\s*hr", t)
    m = re.search(r"(\d+)\s*min", t)
    s = re.search(r"(\d+)\s*sec", t)
    if not (h or m or s):
        return np.nan
    return (
        (int(h.group(1)) * 60 if h else 0)
        + (int(m.group(1)) if m else 0)
        + (int(s.group(1)) / 60 if s else 0)
    )


def dividir_lista(valor):
    """'Action, Comedy' -> ['Action', 'Comedy']."""
    if isinstance(valor, (list, tuple, set)):
        return [str(v).strip() for v in valor if str(v).strip()]
    if pd.isna(valor):
        return []
    partes = re.split(r"[,;|]", str(valor))
    return [p.strip() for p in partes if p.strip()]

PROB = "Prob. alta"            # probabilidad de puntuación alta
ETIQUETADO = "Etiquetado"      # True si tiene puntuación fiable (se usó para entrenar)
ALTA_REAL = "Alta real"        # 1/0 según la puntuación real (NaN si no hay)
NIVEL = "Nivel"                # Alta / Media / Baja probabilidad
ALCANCE = "Alcance"            # Alto / Bajo según miembros
SEGMENTO_PLAN = "Segmento de acción"
ACCION = "Acción recomendada"

# Títulos que todavía se pueden promocionar
ESTADOS_ACTIVOS = ["Not yet aired", "Currently Airing"]