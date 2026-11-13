"""Ejecutar:  streamlit run app.py"""
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from dashboard_core import Motor, STATS, NOMBRES_ROLES, NOMBRES_ESTILOS

st.set_page_config(page_title='Simulador de duelos Pokémon', page_icon='⚔️', layout='wide')

@st.cache_resource
def cargar():
    return Motor()
mot = cargar()
nombres = mot.cat['Name'].tolist()
n2id = {n: i for i, n in zip(mot.cat.index, mot.cat['Name'])}

st.title('⚔️ Simulador de duelos Pokémon')
st.caption('Grupo 03 · NRC 4773 · 1ACC0057 Machine Learning')

with st.sidebar:
    st.header('Modelo')
    modelo = st.radio('Modelo entrenado', list(mot.modelos), index=0)
    info = mot.meta[modelo]
    st.metric('Accuracy en test', f"{info['accuracy']:.2%}")
    st.metric('ROC-AUC en test', f"{info['roc_auc']:.4f}")
    st.caption(info['nota'])
    st.divider()
    if st.button('🎲 Ambos equipos aleatorios', use_container_width=True):
        for k in ('A', 'B'):
            st.session_state[k] = list(np.random.choice(nombres, 6, replace=False))

def selector(k, titulo):
    st.subheader(titulo)
    if st.button(f'🎲 {titulo} aleatorio', key=f'r{k}'):
        st.session_state[k] = list(np.random.choice(nombres, 6, replace=False))
    return st.multiselect('Elige 6 Pokémon', nombres, key=k, max_selections=6, label_visibility='collapsed')

c1, c2 = st.columns(2)
with c1: selA = selector('A', 'Equipo A')
with c2: selB = selector('B', 'Equipo B')

if len(selA) != 6 or len(selB) != 6:
    st.info('Elige 6 Pokémon en cada equipo (o usa los botones aleatorios) para ver la predicción.')
    st.stop()

A, B = [n2id[n] for n in selA], [n2id[n] for n in selB]
p, p_ab, p_ba = mot.predecir(modelo, A, B)
ganador, conf = ('A', p) if p >= .5 else ('B', 1 - p)

st.divider()
st.subheader(f'Ganaría el equipo {ganador} ({conf:.0%} de probabilidad)')
fig = go.Figure()
fig.add_bar(y=[''], x=[p * 100], orientation='h', name='Equipo A', marker_color='#2f5d8a', text=f'A {p:.0%}', textposition='inside')
fig.add_bar(y=[''], x=[(1 - p) * 100], orientation='h', name='Equipo B', marker_color='#e07b39', text=f'B {1-p:.0%}', textposition='inside')
fig.update_layout(barmode='stack', height=110, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

pa, pb = mot.perfil(A), mot.perfil(B)
st.caption(f'Se promedian ambos órdenes: P(A gana) = {p_ab:.1%} si A va primero y {p_ba:.1%} si A va segundo.')

t1, t2, t3 = st.tabs(['Comparación de equipos', 'Pokémon de cada equipo', 'Cómo decide el modelo'])
with t1:
    g = go.Figure()
    g.add_bar(x=STATS, y=[pa['sumas'][s] for s in STATS], name='Equipo A', marker_color='#2f5d8a')
    g.add_bar(x=STATS, y=[pb['sumas'][s] for s in STATS], name='Equipo B', marker_color='#e07b39')
    g.update_layout(barmode='group', height=330, yaxis_title='Suma de stats base', margin=dict(t=10))
    st.plotly_chart(g, use_container_width=True)
    tabla = pd.DataFrame({
        'Equipo A': [pa['leg'], mot.ventaja_tipos(A, B), pa['mv']['mv_top4_power'], pa['mv']['mv_n_priority'], NOMBRES_ESTILOS[pa['estilo']]],
        'Equipo B': [pb['leg'], mot.ventaja_tipos(B, A), pb['mv']['mv_top4_power'], pb['mv']['mv_n_priority'], NOMBRES_ESTILOS[pb['estilo']]]},
        index=['Legendarios', 'Ventaja de tipos (mult. promedio)', 'Poder efectivo top 4 (prom.)', 'Mov. de prioridad (prom.)', 'Estilo del equipo'])
    st.dataframe(tabla.astype(str), use_container_width=True)
with t2:
    ca, cb = st.columns(2)
    for col, ids, tit in [(ca, A, 'Equipo A'), (cb, B, 'Equipo B')]:
        d = mot.cat.loc[ids][['Name', 'Type 1', 'Type 2'] + STATS + ['rol']].copy()
        d['rol'] = d['rol'].map(NOMBRES_ROLES)
        col.markdown(f'**{tit}**'); col.dataframe(d.reset_index(drop=True), use_container_width=True, hide_index=True)
with t3:
    st.write('Variables más importantes del modelo (importancia por permutación, validación):')
    imp = pd.Series(mot.meta['importancia']).sort_values()
    h = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h', marker_color='#2f5d8a'))
    h.update_layout(height=340, margin=dict(t=10), xaxis_title='Caída de ROC-AUC')
    st.plotly_chart(h, use_container_width=True)
    st.write('La velocidad total domina la decisión. Los movimientos de prioridad y de estado son las variables nuevas que más aportan.')

