# Streamlit-Anime
App en Streamlit que recorre el flujo completo de análisis de datos (entender, limpiar, describir, predecir, prescribir y decidir) con un dataset de 30,763 anime. Predice qué títulos tendrán puntuación alta y recomienda acciones y mensajes de marketing por segmento.
# 📊 Marketing Analytics con Streamlit

Aplicación web que guía un análisis de datos de principio a fin y lo convierte en decisiones de marketing. Usa el dataset [anime-dataset-2026](https://www.kaggle.com/datasets/darkbarrierhub/anime-dataset-2026) de Kaggle (30,763 títulos).

**Pregunta central:** ¿qué características tienen los anime mejor puntuados y podemos predecir cuáles lo serán para decidir cómo promocionarlos?

## Pestañas

| # | Pestaña | Qué hace |
|---|---------|----------|
| 1 | Entender | Dimensiones, vista previa, tipos de datos y diagnóstico de faltantes |
| 2 | Limpiar | Convierte los `UNKNOWN` en nulos, crea variables derivadas y registra cada decisión |
| 3 | Descriptiva | KPIs, distribuciones, segmentos, evolución por año y correlaciones |
| 4 | Predictiva | Regresión logística con validación cruzada y variables que más pesan |
| 5 | Prescriptiva | Segmenta por probabilidad y alcance, con una acción editable por segmento |
| 6 | Decisiones | Acción, mensaje y justificación para el segmento elegido |

## Cómo ejecutarla

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en macOS/Linux: source .venv/bin/activate)
pip install streamlit pandas numpy altair scikit-learn
streamlit run app.py
```

Coloca el CSV en `data/anime-dataset-2026.csv`. La app abre en `http://localhost:8501`.

## Estructura

```
├── app.py
├── utils.py
├── data/
│   └── anime-dataset-2026.csv
└── tabs/
    ├── entender.py
    ├── limpiar.py
    ├── descriptiva.py
    ├── predictiva.py
    ├── prescriptiva.py
    └── decisiones.py
```

## Alcance y límites

- El modelo predice la **puntuación** de un título, no sus ventas. Las acciones recomendadas deben validarse con pruebas A/B.
- Miembros, Favoritos y Votantes no entran al modelo porque reflejan la popularidad.
- Proyecto académico de la materia Ciencia de Datos, Universidad Anáhuac Puebla.
