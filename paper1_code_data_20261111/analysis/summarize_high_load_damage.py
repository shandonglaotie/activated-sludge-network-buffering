import sys
sys.modules['bottleneck'] = None
from pathlib import Path
import pandas as pd, numpy as np
O = Path(__file__).resolve().parents[1] / 'results/damage_high_load'
Q = O / 'simulation'
b = pd.read_csv(Q / 'bridge/buffering_response_replicates_100_real_pressure_specific.csv')
t = pd.read_csv(Q / 'topology/three_mechanism_new_replicates.csv')
a = pd.read_csv(Q / 'abundance_3v3.csv')
frames = []
for metric, c in [('bridge_direct', 'A_direct_intercommunity_support_retention_percent'), ('bridge_paths', 'B_alternative_intercommunity_path_retention_percent')]:
    z = b[['pressure', 'replicate', 'state', 'damage_intensity', c]].rename(columns={'damage_intensity': 'intensity', c: 'retention'})
    z['metric'] = metric
    z['series'] = z.state.map({'Structure-preserved': 'Control', 'Connector structure damaged': 'Targeted'})
    frames.append(z.drop(columns='state'))
for mech, metric, c in [('assortative', 'within_direct', 'metric1'), ('assortative', 'within_integrity', 'metric2'), ('core_periphery', 'cp_direct', 'metric1')]:
    z = t.loc[t.mechanism.eq(mech), ['pressure', 'replicate', 'state', 'damage_intensity', c]].rename(columns={'damage_intensity': 'intensity', c: 'retention'})
    z['metric'] = metric
    z['series'] = np.where(z.state.eq('Matched random node removal'), 'Control', 'Targeted')
    frames.append(z.drop(columns='state'))
for m, metric in [('V', 'evenness'), ('D', 'dominance')]:
    z = a.loc[a.metric.eq(m), ['pressure', 'repeat', 'state', 'intensity', 'retention']].rename(columns={'repeat': 'replicate'})
    z['metric'] = metric
    z['series'] = z.state.map({'preserved': 'Control', 'damaged': 'Targeted'})
    frames.append(z.drop(columns='state'))
long = pd.concat(frames, ignore_index=True)
assert len(long) == 4 * 7 * 100 * 21 * 2
assert not long.isna().any().any()
assert not long.duplicated(['pressure', 'metric', 'replicate', 'intensity', 'series']).any()
assert np.allclose(long.loc[long.intensity.eq(0), 'retention'], 100)
assert long.retention.between(-1e-08, 100.000001).all()
long.to_csv(O / 'corrected_all_117600_replicate_values.csv', index=False)
mean = long.groupby(['pressure', 'metric', 'series', 'intensity'], as_index=False).retention.mean()
mean.to_csv(O / 'corrected_56_mean_curves.csv', index=False)
print('PASS: 117600 records, 56 mean curves, 21 points, 100 repeats, baseline 100%, all retention bounded 0-100%.')
