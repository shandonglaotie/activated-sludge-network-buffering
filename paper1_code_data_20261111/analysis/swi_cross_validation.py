import sys
sys.modules['bottleneck'] = None
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'shared' / 'data'
OUT = ROOT / 'results/swi_cross_validation'
OUT.mkdir(exist_ok=True)
DIMS = list('EFCSVD')
ZCOLS = [f'robust_z_{x}' for x in DIMS]
CONDITIONS = ['Low temperature', 'Low C/N', 'Low C/P', 'Industrial wastewater']
SEED = 20260920
REPEATS = 200
FOLDS = 5
INNER_BOOT = 200
meta = pd.read_csv(DATA / '04_Metadata' / '04C_plant_metadata_252.csv')
pca = pd.read_csv(DATA / '08_PCA_source_data' / '08A_PCA_input_and_scores_252_plants.csv')
d = meta.merge(pca, on=['Site_ID', 'strict_223_subset'], validate='one_to_one')

def get_groups(name):
    if name == 'Low temperature':
        v = d.dropna(subset=['MIT', *ZCOLS]).sort_values(['MIT', 'Site_ID'], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[n:len(v) - n], 'sampled', len(v))
    if name == 'Low C/N':
        v = d[d.N_chemically_coherent.eq(True)].dropna(subset=['C_N_ratio', *ZCOLS]).sort_values(['C_N_ratio', 'Site_ID'], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[n:], 'sampled', len(v))
    if name == 'Low C/P':
        v = d.dropna(subset=['C_P_ratio', *ZCOLS]).sort_values(['C_P_ratio', 'Site_ID'], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[n:], 'sampled', len(v))
    if name == 'Industrial wastewater':
        v = d[d.Industrial_pct.gt(0)].dropna(subset=['Industrial_pct', *ZCOLS]).sort_values(['Industrial_pct', 'Site_ID'], ascending=[False, True], kind='mergesort')
        n = math.ceil(0.2 * len(v))
        return (v.iloc[:n], v.iloc[-n:], 'fixed', len(v))
    raise ValueError(name)

def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    return float(np.mean([(pos > x).mean() + 0.5 * (pos == x).mean() for x in neg]))

def fit_fold(xe, xc, rng):
    center = np.median(xc, axis=0)
    scale = np.quantile(xc, 0.75, axis=0) - np.quantile(xc, 0.25, axis=0)
    scale = np.where(scale > 1e-09, scale, 1.0)
    a = (xe - center) / scale
    b = (xc - center) / scale
    boots = []
    for _ in range(INNER_BOOT):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        boots.append(aa.mean(axis=0) - bb.mean(axis=0))
    boots = np.asarray(boots)
    med = np.median(boots, axis=0)
    modal = np.where(med >= 0, 1.0, -1.0)
    stability = np.mean(np.sign(boots) == modal, axis=0)
    selected = stability >= 0.8
    if not selected.any():
        return None
    raw = np.where(selected, np.abs(med), 0.0)
    weights = raw / raw.sum()
    return (center, scale, selected, weights, stability, med)

def scores(z, weights):
    rms = np.sqrt(np.sum(weights * z ** 2, axis=1))
    l1 = np.sum(weights * np.abs(z), axis=1)
    one = np.sum(weights * np.maximum(z, 0.0), axis=1)
    return {'Weighted RMS': rms, 'Weighted L1': l1, 'One-sided loss': one}
LOCKED = {'Low temperature': ['C', 'S', 'V', 'D'], 'Low C/N': ['V', 'D'], 'Low C/P': ['V', 'D'], 'Industrial wastewater': ['E']}
percentile_rows = []
rng = np.random.default_rng(SEED + 31)
for condition in CONDITIONS:
    exposed, pool, mode, valid_n = get_groups(condition)
    xe_all = exposed[ZCOLS].to_numpy(float)
    mask = np.array([dim in LOCKED[condition] for dim in DIMS])
    for rep in range(REPEATS):
        control = pool if mode == 'fixed' else pool.iloc[rng.choice(len(pool), len(exposed), replace=False)]
        xc_all = control[ZCOLS].to_numpy(float)
        efolds = np.array_split(rng.permutation(len(xe_all)), FOLDS)
        cfolds = np.array_split(rng.permutation(len(xc_all)), FOLDS)
        pe = np.full(len(xe_all), np.nan)
        pc = np.full(len(xc_all), np.nan)
        state = np.full(len(xe_all), -1, dtype=int)
        for fold in range(FOLDS):
            te, tc = (efolds[fold], cfolds[fold])
            tre = np.setdiff1d(np.arange(len(xe_all)), te)
            trc = np.setdiff1d(np.arange(len(xc_all)), tc)
            center = np.median(xc_all[trc], axis=0)
            scale = np.quantile(xc_all[trc], 0.75, axis=0) - np.quantile(xc_all[trc], 0.25, axis=0)
            scale = np.where(scale > 1e-09, scale, 1.0)
            a = (xe_all[tre] - center) / scale
            b = (xc_all[trc] - center) / scale
            delta = a.mean(axis=0) - b.mean(axis=0)
            beta = np.where(mask, delta, 0.0)
            beta = beta / np.abs(beta).sum()
            ref_raw = (xc_all[trc] - center) / scale @ beta
            e_raw = (xe_all[te] - center) / scale @ beta
            c_raw = (xc_all[tc] - center) / scale @ beta

            def ecdf(vals):
                return np.array([(np.sum(ref_raw < x) + 0.5 * np.sum(ref_raw == x)) / len(ref_raw) for x in vals])
            pe[te] = ecdf(e_raw)
            pc[tc] = ecdf(c_raw)
            state[te] = np.where(pe[te] >= 0.95, 2, np.where(pe[te] >= 0.75, 1, 0))
        percentile_rows.append({'condition': condition, 'repeat': rep, 'cv_auc': auc(pe, pc), 'exposed_median_percentile': np.median(pe), 'control_median_percentile': np.median(pc), 'fraction_exposed_above_reference_median': np.mean(pe > 0.5), 'reference_range_fraction': np.mean(state == 0), 'concern_fraction': np.mean(state == 1), 'warning_fraction': np.mean(state == 2)})
percentile = pd.DataFrame(percentile_rows)
percentile_summary = percentile.groupby('condition', as_index=False).agg(cv_auc_median=('cv_auc', 'median'), cv_auc_low2_5=('cv_auc', lambda x: x.quantile(0.025)), cv_auc_high97_5=('cv_auc', lambda x: x.quantile(0.975)), fraction_auc_above_0_5=('cv_auc', lambda x: np.mean(x > 0.5)), median_exposed_percentile=('exposed_median_percentile', 'median'), median_control_percentile=('control_median_percentile', 'median'), median_fraction_exposed_above_reference_median=('fraction_exposed_above_reference_median', 'median'), median_reference_range_fraction=('reference_range_fraction', 'median'), median_concern_fraction=('concern_fraction', 'median'), median_warning_fraction=('warning_fraction', 'median'))
percentile.to_csv(OUT / '09_directional_percentile_cv_runs.csv', index=False)
percentile_summary.to_csv(OUT / '10_directional_percentile_summary.csv', index=False)
print('\nDIRECTION-PRESERVING REFERENCE PERCENTILE')
print(percentile_summary.to_string(index=False))
