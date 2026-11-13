# Predicción de combates Pokémon
**Hito 3 – Refinamiento, Modelado Avanzado e Integración** · 1ACC0057 Machine Learning · Grupo 03 (NRC 4773)

Proyecto de Machine Learning que predice **qué equipo de 6 Pokémon ganaría un combate** y con qué probabilidad, siguiendo la metodología CRISP-DM. Hasta este hito se cubren las fases 1 a 5 (comprensión del negocio y de los datos, preparación, modelado avanzado y evaluación) y se cuenta con un dashboard en Streamlit para probar equipos.

## Alineación con los ODS
- **ODS 4 – Educación de calidad:** caso didáctico reproducible de ML con datos abiertos.
- **ODS 9 – Industria, innovación e infraestructura:** metodología transferible para predecir resultados entre equipos a partir de atributos agregados.

## Problema de ML
- **Target:** `winner` → 0 si gana el primer equipo, 1 si gana el segundo.
- **Variables (24 en el modelo final):** diferencias *first − second* de HP, Attack, Defense, Sp. Atk, Sp. Def, Speed y n.º de legendarios, ventaja de velocidad, ventaja de tipos, variables de movimientos, roles de los Pokémon y estilo del equipo.

## Datos
| Fuente | Contenido |
|---|---|
| [Kaggle: pokemon-dataset-with-team-combat](https://www.kaggle.com/datasets/tuannguyenvananh/pokemon-dataset-with-team-combat) (con `kagglehub`) | `pokemon.csv` (800 Pokémon), `combats.csv` (50 000 duelos 1v1), `pokemon_id_each_team.csv` (100 equipos de 6) y `team_combat.csv` (10 000 combates de equipos) |
| [PokeAPI](https://pokeapi.co) | Efectividad de tipos y movimientos de cada Pokémon (700 movimientos únicos) |

Archivos generados en `datos/`: `modelo_df_enriquecido.csv` (tabla final de modelado), `features_movimientos_por_pokemon.csv` y `movimientos_pokeapi.csv`.

## Resultados
Conjunto de **prueba**:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Hito 2 (Reg. Logística, 9 variables) | 0.8513 | 0.9067 | 0.8440 | 0.8742 | 0.9314 |
| **Hito 3 (Gradient Boosting, 24 variables)** | **0.8802** | **0.9150** | **0.8868** | **0.9007** | **0.9473** |


## Hallazgos principales
- La **velocidad** domina la decisión: importancia por permutación de 0.3105, unas 48 veces la siguiente variable. Las variables nuevas que más aportan son los movimientos de prioridad y de estado.
- Los errores se concentran en combates con velocidades parecidas. Con diferencia de velocidad de 0 a 25 puntos el modelo acierta 75.5 %, frente a 58.9 % de la regla "gana el más rápido".
- Con **equipos nunca vistos** (20 equipos apartados, 10 repeticiones) el rendimiento baja a cerca de 82–83 % de accuracy; el 88 % de la prueba corresponde a equipos que el modelo ya vio en otros combates.

## Dashboard (Streamlit)
Permite elegir 6 Pokémon por equipo (o equipos aleatorios) y muestra la probabilidad de victoria. Se puede escoger entre el Gradient Boosting (24 variables) y la Regresión Logística base. La predicción se promedia en ambos órdenes, (A,B) y (B,A), para quitar el sesgo de posición del dataset. Incluye comparación de stats, ventaja de tipos, movimientos y estilo, la tabla de Pokémon de cada equipo con su rol y las variables más importantes.

### Cómo ejecutarlo
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar el notebook completo. 
3. Desde la carpeta del proyecto: `streamlit run app.py`

### Estructura de carpetas
```
Machine_Learning_1ACC0057_4773_GRUPO_03/
├── nb_1ACC0057_4773_GRUPO_03.ipynb
├── app.py                    # dashboard Streamlit
├── dashboard_core.py         # variables y predicción 
├── requirements.txt
├── artefactos/               # lo genera el notebook
│   ├── catalogo.csv
│   ├── tabla_tipos.csv
│   ├── estilos.joblib
│   ├── modelo_gb.joblib
│   ├── modelo_lr.joblib
│   └── meta.json
└── datos/
    ├── modelo_df_enriquecido.csv
    ├── features_movimientos_por_pokemon.csv
    └── movimientos_pokeapi.csv
```

## Limitaciones y ética
- Los movimientos son una aproximación (los 4 mejores de cada especie).

- Los datos son públicos y sin información personal. El modelo es una herramienta de apoyo, no una decisión automática.

## Próximos pasos
- **Hito 4:** presentación del dashboard en funcionamiento.
- Mejoras: mostrar el top 4 de ataques por Pokémon.

## Autores
Cesar Avalos · Peter Pacherres