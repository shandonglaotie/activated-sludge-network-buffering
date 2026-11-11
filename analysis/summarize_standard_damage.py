from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[1]
O = R / 'results/damage_standard'
Q = O / 'seven_metric_output'
S = O / 'source_data'
V = O / 'SVG'
P = O / 'preview'
PRESS = ['Low temperature', 'Low C/N ratio', 'Low C/P ratio', 'High industrial proportion']
LAB = ['Low temperature', 'Low C/N', 'Low C/P', 'Industrial wastewater']
COL = ['#607FAE', '#7651A5', '#8D91B9', '#CF607E']
MET = [('bridge_direct', 'Bridge edges', 'Intercommunity direct support'), ('bridge_paths', 'Bridge edges', 'Alternative intercommunity paths'), ('within_direct', 'Assortative communities', 'Within-community direct support'), ('within_integrity', 'Assortative communities', 'Within-community module integrity'), ('cp_direct', 'Core–periphery backbone', 'Core–periphery direct support'), ('evenness', 'Abundance reorganization', 'Community evenness'), ('dominance', 'Abundance reorganization', 'Dominance balance')]
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 7, 'svg.fonttype': 'none', 'axes.linewidth': 0.6})
frames = []
b = pd.read_csv(Q / 'bridge/buffering_response_replicates_100_real_pressure_specific.csv')
for col, key in [('A_direct_intercommunity_support_retention_percent', 'bridge_direct'), ('B_alternative_intercommunity_path_retention_percent', 'bridge_paths')]:
    q = b.copy()
    q['metric'] = key
    q['retention'] = q[col]
    q['series'] = q.state.map({'Structure-preserved': 'Control', 'Connector structure damaged': 'Targeted'})
    q = q.rename(columns={'damage_intensity': 'intensity'})
    frames.append(q[['pressure', 'replicate', 'metric', 'intensity', 'series', 'retention']])
t = pd.read_csv(Q / 'topology/three_mechanism_new_replicates.csv')
for mech, col, key in [('assortative', 'metric1', 'within_direct'), ('assortative', 'metric2', 'within_integrity'), ('core_periphery', 'metric1', 'cp_direct')]:
    q = t[t.mechanism.eq(mech)].copy()
    q['metric'] = key
    q['retention'] = q[col]
    q['series'] = np.where(q.state.eq('Matched random node removal'), 'Control', 'Targeted')
    q = q.rename(columns={'damage_intensity': 'intensity'})
    frames.append(q[['pressure', 'replicate', 'metric', 'intensity', 'series', 'retention']])
a = pd.read_csv(Q / 'abundance_3v3.csv')
am = {'Low temp.': PRESS[0], 'Low C/N': PRESS[1], 'Low C/P': PRESS[2], 'Industrial wastewater': PRESS[3]}
a.pressure = a.pressure.map(am)
a.metric = a.metric.map({'V': 'evenness', 'D': 'dominance'})
a['series'] = a.state.map({'preserved': 'Control', 'damaged': 'Targeted'})
a = a.rename(columns={'repeat': 'replicate'})
frames.append(a[['pressure', 'replicate', 'metric', 'intensity', 'series', 'retention']])
long = pd.concat(frames, ignore_index=True)
assert len(long) == 4 * 7 * 100 * 21 * 2
long.to_csv(S / 'seven_metrics_final_curves.csv', index=False)
curves = long.groupby(['pressure', 'metric', 'intensity', 'series']).retention.agg(['mean', 'std']).reset_index()
w = long.pivot(index=['pressure', 'metric', 'replicate', 'intensity'], columns='series', values='retention').reset_index()
w['gap'] = w.Control - w.Targeted
ints = []
for (p, m, r), g in w.groupby(['pressure', 'metric', 'replicate']):
    g = g.sort_values('intensity')
    ints.append([p, m, r, np.trapezoid(g.gap, g.intensity)])
ii = pd.DataFrame(ints, columns=['pressure', 'metric', 'replicate', 'effect'])
rows = []
rng = np.random.default_rng(20260921)
for (p, m), g in ii.groupby(['pressure', 'metric']):
    x = g.effect.to_numpy()
    boot = x[rng.integers(0, len(x), (3000, len(x)))].mean(1)
    rows.append([p, m, x.mean(), *np.quantile(boot, [0.025, 0.975])])
res = pd.DataFrame(rows, columns=['pressure', 'metric', 'estimate', 'low95', 'high95'])
res.to_csv(S / 'Fig3b_final_summary.csv', index=False)
