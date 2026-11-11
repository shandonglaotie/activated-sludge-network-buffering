import sys
sys.modules['bottleneck'] = None
import numpy as np, pandas as pd, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
(root / 'results/high_load_fingerprints').mkdir(parents=True, exist_ok=True)
b = root / 'shared/data'
s = pd.read_csv(b / '11_Figure_source_data/network_risk.csv')
m = pd.read_csv(b / '04_Metadata/04C_plant_metadata_252.csv')
assert list(s.Site_ID) == list(m.Site_ID)
cols = ['Edge_realization_rate_richness_adjusted_risk_z', 'Largest_component_fraction_richness_adjusted_risk_z', 'Connector_presence_fraction_richness_adjusted_risk_z', 'Abundance_weighted_intercommunity_support_richness_adjusted_risk_z', 'Community_abundance_evenness_richness_adjusted_risk_z', 'Community_abundance_dominance_richness_adjusted_risk_z']
x = s[cols].replace([np.inf, -np.inf], np.nan)
x = x.fillna(x.median())
z = (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)).replace(0, 1)
a = z.to_numpy() - z.mean().to_numpy()
u, sv, vt = np.linalg.svd(a, full_matrices=False)
score = a @ vt[:2].T
exp = sv[:2] ** 2 / (sv * sv).sum() * 100
valid = (m.AtInfTN > m.AtInfNH4N) & (m.AtInfNH4N >= 0)
rng = np.random.default_rng(20261101)
out = []
audit = []
for name, col in [('High COD', 'AtInfCOD'), ('High NH4-N', 'AtInfNH4N'), ('High TN', 'AtInfTN'), ('High TP', 'AtInfTP')]:
    eligible = valid.to_numpy() if col in ['AtInfTN', 'AtInfNH4N'] else np.isfinite(m[col].to_numpy())
    threshold = float(m.loc[eligible, col].quantile(0.8))
    e = eligible & (m[col].to_numpy() >= threshold)
    r = eligible & (m[col].to_numpy() < threshold)
    vals = []
    for c in cols:
        aa = s.loc[e, c].to_numpy()
        bb = s.loc[r, c].to_numpy()
        boots = aa[rng.integers(len(aa), size=(3000, len(aa)))].mean(1) - bb[rng.integers(len(bb), size=(3000, len(bb)))].mean(1)
        vals.append([float(aa.mean() - bb.mean()), *np.quantile(boots, [0.025, 0.975]).tolist()])

    def ellipse(pts):
        mu = pts.mean(0)
        cv = np.cov(pts.T)
        v, V = np.linalg.eigh(cv)
        t = np.linspace(0, 2 * np.pi, 81)
        return (mu[:, None] + V @ np.diag(np.sqrt(np.maximum(v, 0) * 5.991)) @ np.array([np.cos(t), np.sin(t)])).T.tolist()
    oldE = None
    if col == 'AtInfTN':
        joint = (m.loc[valid, 'AtInfTN'].rank(pct=True) + m.loc[valid, 'AtInfNH4N'].rank(pct=True)) / 2
        oldE = set(m.loc[joint.index[joint >= joint.quantile(0.8)], 'Site_ID'])
        newE = set(m.loc[e, 'Site_ID'])
        difference = {'old_joint_high_n': len(oldE), 'new_actual_TN_high_n': len(newE), 'removed_from_high_group': sorted(oldE - newE), 'added_to_high_group': sorted(newE - oldE)}
    else:
        difference = {}
    out.append({'name': name, 'effects': vals, 'exposed': score[e].tolist(), 'reference': score[r].tolist(), 'ellipse_exposed': ellipse(score[e]), 'ellipse_reference': ellipse(score[r])})
    audit.append({'name': name, 'column': col, 'threshold_q80': threshold, 'n_exposed': int(e.sum()), 'n_reference': int(r.sum()), 'n_eligible': int(eligible.sum()), 'correction': difference})
(root / 'results/high_load_fingerprints/high_load_data.json').write_text(json.dumps({'panels': out, 'explained': exp.tolist(), 'audit': audit}, ensure_ascii=False, indent=2), encoding='utf8')
pd.DataFrame([{**{k: v for k, v in audit[i].items() if k != 'correction'}, 'dimension': 'EFCSVD'[j], 'difference': d[0], 'low95': d[1], 'high95': d[2]} for i, p in enumerate(out) for j, d in enumerate(p['effects'])]).to_csv(root / 'results/high_load_fingerprints/high_load_fingerprints.csv', index=False, encoding='utf-8-sig')
