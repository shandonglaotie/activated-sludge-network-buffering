from pathlib import Path
import math, json, sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'swi'
SVG = OUT / 'SVG'
PRE = OUT / 'preview'
SRC = OUT / 'source_data'
for p in (OUT, SVG, PRE, SRC):
    p.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'shared' / 'data'
meta = pd.read_csv(DATA / '04_Metadata' / '04C_plant_metadata_252.csv')
pca = pd.read_csv(DATA / '08_PCA_source_data' / '08A_PCA_input_and_scores_252_plants.csv')
d = meta.merge(pca, on=['Site_ID', 'strict_223_subset'], validate='one_to_one')
DIMS = list('EFCSVD')
ZCOLS = [f'robust_z_{x}' for x in DIMS]
PRESS = ['Low temperature', 'Low C/N ratio', 'Low C/P ratio', 'High industrial proportion']
LAB = ['Low temperature', 'Low C/N', 'Low C/P', 'Industrial wastewater']
COL = dict(zip(PRESS, ['#607FAE', '#7651A5', '#8D91B9', '#CF607E']))
SELECT = {'Low temperature': list('CSVD'), 'Low C/N ratio': list('VD'), 'Low C/P ratio': list('VD'), 'High industrial proportion': list('E')}
mpl.rcParams.update({'font.family': 'Times New Roman', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'savefig.facecolor': 'white'})

def groups(name):
    if name == 'Low temperature':
        v = d.dropna(subset=['MIT', *ZCOLS]).sort_values(['MIT', 'Site_ID'], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[n:len(v) - n])
    if name == 'Low C/N ratio':
        v = d[d.N_chemically_coherent.eq(True)].dropna(subset=['C_N_ratio', *ZCOLS]).sort_values(['C_N_ratio', 'Site_ID'], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[n:])
    if name == 'Low C/P ratio':
        v = d.dropna(subset=['C_P_ratio', *ZCOLS]).sort_values(['C_P_ratio', 'Site_ID'], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[n:])
    v = d[d.Industrial_pct.gt(0)].dropna(subset=['Industrial_pct', *ZCOLS]).sort_values(['Industrial_pct', 'Site_ID'], ascending=[False, True], kind='mergesort')
    n = math.ceil(0.2 * len(v))
    return (v.iloc[:n], v.iloc[-n:])

def empirical_percentile(values, ref):
    values = np.asarray(values, float)
    ref = np.asarray(ref, float)
    return np.array([100.0 * np.mean(ref <= x) for x in values])

def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    return float(np.mean([(pos > x).mean() + 0.5 * (pos == x).mean() for x in neg]))

def save(fig, name):
    fig.savefig(SVG / f'{name}.svg', format='svg', bbox_inches=None, facecolor='white')
    fig.savefig(PRE / f'{name}.pdf', format='pdf', bbox_inches=None, facecolor='white')
    fig.savefig(PRE / f'{name}.png', dpi=300, bbox_inches=None, facecolor='white')
    fig.savefig(PRE / f'{name}.tiff', dpi=600, bbox_inches=None, facecolor='white')
    plt.close(fig)
formula_rows = []
scores = d[['Site_ID', 'strict_223_subset']].copy()
group_rows = []
for p in PRESS:
    e, r = groups(p)
    center = r[ZCOLS].median().to_numpy(float)
    scale = (r[ZCOLS].quantile(0.75) - r[ZCOLS].quantile(0.25)).to_numpy(float)
    scale = np.where(scale > 1e-09, scale, 1.0)
    ze = (e[ZCOLS].to_numpy(float) - center) / scale
    zr = (r[ZCOLS].to_numpy(float) - center) / scale
    delta = ze.mean(0) - zr.mean(0)
    selected = np.array([x in SELECT[p] for x in DIMS])
    beta = np.where(selected, delta, 0.0)
    beta = beta / np.abs(beta).sum()
    allz = (d[ZCOLS].to_numpy(float) - center) / scale
    raw_all = allz @ beta
    raw_ref = zr @ beta
    swi_all = empirical_percentile(raw_all, raw_ref)
    exposed = scores.Site_ID.isin(e.Site_ID).to_numpy()
    state = np.full(len(scores), 'Not exposed', dtype=object)
    state[exposed & (swi_all < 75)] = 'Reference range'
    state[exposed & (swi_all >= 75) & (swi_all < 95)] = 'Concern'
    state[exposed & (swi_all >= 95)] = 'Warning'
    scores[f'{p}__exposed'] = exposed
    scores[f'{p}__directional_projection'] = raw_all
    scores[f'{p}__SWI'] = swi_all
    scores[f'{p}__state'] = state
    formula_rows.append([p, *beta, len(e), len(r), auc(empirical_percentile(ze @ beta, raw_ref), empirical_percentile(raw_ref, raw_ref))])
    group_rows.append([p, len(e), len(r)])
form = pd.DataFrame(formula_rows, columns=['condition', *DIMS, 'exposed_n', 'reference_n', 'apparent_full_data_AUC'])
form.to_csv(SRC / 'percentile_SWI_formulas.csv', index=False)
scores.to_csv(SRC / 'Supplementary_Data_6_percentile_SWI.csv', index=False)
pd.DataFrame(group_rows, columns=['condition', 'exposed_n', 'reference_n']).to_csv(SRC / 'group_contract.csv', index=False)
