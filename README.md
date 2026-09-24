# Predicción de combates 
**Hito 1 – Comprensión del Negocio y de los Datos** · Curso de Machine Learning

Proyecto de Machine Learning que busca predecir **qué equipo de 6 Pokémon ganaría un versus** y con qué probabilidad, siguiendo la metodología CRISP-DM. Este hito cubre las fases 1 (Business Understanding) y 2 (Data Understanding).

## Alineación con los ODS
- **ODS 4 – Educación de calidad:** caso didáctico reproducible de ML con datos abiertos.
- **ODS 9 – Industria, innovación e infraestructura:** metodología transferible para predecir resultados entre equipos a partir de atributos agregados.

## Problema de ML
- **Target:** `first_wins` → 1 si gana el primer equipo, 0 si gana el segundo.
- **Predictores:** diferencias *first − second* de HP, Attack, Defense, Sp. Atk, Sp. Def, Speed y n.º de legendarios.

## Datos
Dataset de Kaggle: [pokemon-dataset-with-team-combat](https://www.kaggle.com/datasets/tuannguyenvananh/pokemon-dataset-with-team-combat) (se descarga automáticamente con `kagglehub`).

| Archivo | Contenido |
|---|---|
| `pokemon.csv` | 800 Pokémon con tipos y stats |
| `combats.csv` | 50 000 duelos 1v1 |
| `pokemon_id_each_team.csv` | 100 equipos de 6 Pokémon |
| `team_combat.csv` | 10 000 combates equipo vs equipo |

## Hallazgos principales del EDA
- Datos limpios: sin duplicados ni IDs faltantes.
- Clases desbalanceadas: 38.6 % gana el primero / 61.4 % gana el segundo.
- La **velocidad** es la variable más predictiva, seguida de Attack y Sp. Atk.
- "Gana quien tiene más stats totales" acierta solo 63.63 %, casi igual que la clase mayoritaria (61.39 %).
- Riesgo para el modelado: los combates incluyen todos los pares en ambos órdenes y 100 autoenfrentamientos, por lo que la partición train/test debe hacerse por equipo.

## Próximos pasos
Fase 3: excluir autoenfrentamientos, definir la validación por equipo, escalar variables y evaluar variables de tipo. Luego modelado y evaluación.

## Autores
Cesar Avalos · Peter Pacherres
