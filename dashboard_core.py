import json, os
import numpy as np
import pandas as pd
import joblib

STATS = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
MV = ['mv_top4_power', 'mv_coverage', 'mv_n_priority', 'mv_n_status']
NOMBRES_ROLES = {0: 'Básicos / En desarrollo', 1: 'Tanques defensivos', 2: 'Sweepers mixtos / Élite',
                 3: 'Bulky / Bruisers', 4: 'Atacantes veloces'}
NOMBRES_ESTILOS = {0: 'Ligero / En desarrollo', 1: 'Ofensivo veloz y prioridad (Tempo)',
                   2: 'Ofensiva pesada / Bulky', 3: 'Defensivo / Desgaste (Tank / Stall)'}


class Motor:
    def __init__(self, carpeta='artefactos'):
        self.cat = pd.read_csv(os.path.join(carpeta, 'catalogo.csv')).set_index('id')
        self.chart = pd.read_csv(os.path.join(carpeta, 'tabla_tipos.csv'), index_col=0)
        est = joblib.load(os.path.join(carpeta, 'estilos.joblib'))
        self.sc_estilo, self.km_estilo, self.cols_estilo = est['scaler'], est['kmeans'], est['cols']
        self.modelos = {n: joblib.load(os.path.join(carpeta, f)) for n, f in
                        [('Gradient Boosting (24 variables)', 'modelo_gb.joblib'),
                         ('Regresión Logística (base, 9 variables)', 'modelo_lr.joblib')]}
        with open(os.path.join(carpeta, 'meta.json'), encoding='utf8') as f:
            self.meta = json.load(f)
        self._tipos = {i: [t for t in (r['Type 1'], r['Type 2']) if isinstance(t, str)] for i, r in self.cat.iterrows()}
        self._ix = {t: k for k, t in enumerate(self.chart.index)}
        self._M = self.chart.values

    # ---------- variables de un equipo ----------
    def _off_adv(self, a, b):
        # mejor multiplicador que A puede lograr contra B con sus propios tipos
        return max(np.prod([self._M[self._ix[t], self._ix[d]] for d in self._tipos[b]]) for t in self._tipos[a])

    def ventaja_tipos(self, ta, tb):
        return float(np.mean([self._off_adv(a, b) for a in ta for b in tb]))

    def perfil(self, ids):
        c = self.cat.loc[ids]
        conteo = {f'rol_{k}': int((c['rol'] == k).sum()) for k in range(5)}
        mv = c[MV].mean()
        prof = pd.concat([c[STATS].mean(), mv, pd.Series(conteo)])
        estilo = int(self.km_estilo.predict(self.sc_estilo.transform(pd.DataFrame([prof[self.cols_estilo]])))[0])
        return dict(sumas=c[STATS].sum(), leg=int(c['Legendary'].sum()), mv=mv, roles=conteo, estilo=estilo)

    # ---------- fila de variables para (first, second) ----------
    def fila(self, first, second):
        a, b = self.perfil(first), self.perfil(second)
        f = {}
        for s, n in zip(STATS, ['HP', 'Attack', 'Defense', 'SpAtk', 'SpDef', 'Speed']):
            f[f'diff_sum_{n}'] = float(a['sumas'][s] - b['sumas'][s])
        f['diff_n_legendary'] = float(a['leg'] - b['leg'])
        d = f['diff_sum_Speed']
        f['speed_adv_first_mas_rapido'] = float(d > 0)
        f['speed_adv_second_mas_rapido'] = float(d < 0)
        f['diff_type_adv'] = self.ventaja_tipos(first, second) - self.ventaja_tipos(second, first)
        for m in MV:
            f[f'diff_{m}'] = float(a['mv'][m] - b['mv'][m])
        for k in range(4):
            f[f'diff_rol_{k}'] = float(a['roles'][f'rol_{k}'] - b['roles'][f'rol_{k}'])
        for k in (1, 2, 3):
            f[f'style_first_{k}'] = float(a['estilo'] == k)
            f[f'style_second_{k}'] = float(b['estilo'] == k)
        return f

    def _p_second(self, nombre, first, second):
        m = self.modelos[nombre]
        x = pd.DataFrame([self.fila(first, second)])[m['cols']].astype(float)
        x[m['num_cols']] = m['scaler'].transform(x[m['num_cols']])
        return float(m['modelo'].predict_proba(x)[0, 1])   # P(gana el segundo equipo)

    def predecir(self, nombre, A, B):
        """P(gana A), promediando ambos órdenes (A,B) y (B,A) para quitar el sesgo de posición."""
        p_ab = 1 - self._p_second(nombre, A, B)
        p_ba = self._p_second(nombre, B, A)
        return 0.5 * (p_ab + p_ba), p_ab, p_ba
