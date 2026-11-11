from pathlib import Path
import sys, runpy, types, shutil, json
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
R = Path(__file__).resolve().parents[1]
O = R / 'results/damage_standard'
S = O / 'source_data'
B = R / 'shared/simulation_engines'
I = O / 'seven_metric_input'
Q = O / 'seven_metric_output'
I.mkdir(parents=True, exist_ok=True)
Q.mkdir(parents=True, exist_ok=True)
S.mkdir(parents=True, exist_ok=True)
plant = pd.read_csv(R / 'shared/data/04_Metadata/04C_plant_metadata_252.csv').merge(pd.read_csv(R / 'shared/data/08_PCA_source_data/08A_PCA_input_and_scores_252_plants.csv'), on=['Site_ID', 'strict_223_subset'])
sg = pd.read_csv(R / 'shared/damage_standard/latest_sample_pressure_groups.csv')
dims = list('EFCSVD')
z = ['robust_z_' + x for x in dims]
spec = [('Low temperature', 'low_temp_flag', 'latest_low_temp_reference_flag'), ('Low C/N ratio', 'low_CN_flag', 'latest_low_CN_reference_flag'), ('Low C/P ratio', 'low_CP_flag', 'latest_low_CP_reference_flag'), ('High industrial proportion', 'industrial_flag', 'latest_industrial_reference_flag')]
params = pd.read_csv(B / 'topology/input/pressure_attack_parameters_extended.csv').set_index('pressure')
old = params.copy()
audit = []
for p, ef, rf in spec:
    e = plant[plant.Site_ID.isin(sg.loc[sg[ef].eq(1), 'wwtp_id'].unique())]
    r = plant[plant.Site_ID.isin(sg.loc[sg[rf].eq(1), 'wwtp_id'].unique())]
    delta = e[z].mean() - r[z].mean()
    for d, v in zip(dims, delta):
        params.loc[p, 'difference_' + d] = v
    audit.append([p, len(e), len(r), *delta])
for col, d, base, scale in [('random_edge_fraction', 'E', 0.01, 0.07), ('random_node_fraction', 'F', 0.005, 0.025), ('connector_node_fraction', 'C', 0, 0.03), ('intercommunity_edge_fraction', 'S', 0, 0.06)]:
    den = max(float(old['difference_' + d].max()), 1e-12)
    params[col] = base + scale * params['difference_' + d].clip(lower=0) / den
for f in ['node_annotations.csv', 'network_nodes.csv', 'network_edges.csv', 'pressure_sensitive_profiles.csv']:
    shutil.copy2(B / 'topology/input' / f, I / f)
params.reset_index().to_csv(I / 'pressure_attack_parameters_extended.csv', index=False)
basic = pd.read_csv(B / 'topology/input/pressure_attack_parameters.csv').columns
params.reset_index().loc[lambda x: ~x.pressure.eq('High industrial proportion'), basic].to_csv(I / 'pressure_attack_parameters.csv', index=False)
pd.DataFrame(audit, columns=['condition', 'exposed_n', 'reference_n', *['delta_' + x for x in dims]]).to_csv(S / 'seven_metric_parameter_differences.csv', index=False)
top = B / 'topology/simulate_and_plot_three_mechanisms_corrected_cp.py'
ns = runpy.run_path(str(top), run_name='final_topology')
g = ns['main'].__globals__
g['INPUT'] = I
g['OUTPUT'] = Q / 'topology'
g['MAIN_PRESSURES'] = [x[0] for x in spec]
g['parse_args'] = lambda: types.SimpleNamespace(repeats=100, x_points=21, pressure_set='main')
g['plot_grid'] = lambda *a, **k: None
g['plot_existing_bridge'] = lambda: None
ns['main']()
bridge = next((R / 'shared/bridge_engine').glob('*01_simulate_pressure_specific_damage.py'))
code = bridge.read_text(encoding='utf-8-sig')
code = code.replace('pressure_order = list(pressure_label)', "pressure_order = ['Low temperature','Low C/N ratio','Low C/P ratio','High industrial proportion']")
bn = {'__name__': 'final_bridge', '__file__': str(bridge)}
exec(compile(code, str(bridge), 'exec'), bn)
bn['parse_args'] = lambda: types.SimpleNamespace(repeats=100, x_points=21, input_dir=I, output_dir=Q / 'bridge')
bn['main']()
abns = runpy.run_path(str(B / 'abundance/simulate_and_plot_abundance_organization_damage.py'), run_name='final_abundance')
ab = pd.read_csv(R / 'shared/community_abundance/06_community_abundance.csv')
wide = ab.pivot(index='sample_id', columns='community_name', values='community_relative_abundance').reindex(columns=[f'C{i:02d}' for i in range(1, 9)])
rows = []
for pi, (p, ef, rf) in enumerate(spec):
    label = {'Low temperature': 'Low temp.', 'Low C/N ratio': 'Low C/N', 'Low C/P ratio': 'Low C/P', 'High industrial proportion': 'Industrial wastewater'}[p]
    pm = wide.loc[wide.index.intersection(sg.loc[sg[ef].eq(1), 'sample_id'])].to_numpy(float)
    rm = wide.loc[wide.index.intersection(sg.loc[sg[rf].eq(1), 'sample_id'])].mean().to_numpy(float)
    delta = pm.mean(0) - rm
    for rep in range(100):
        rng = np.random.default_rng(2026092100 + pi * 10000 + rep)
        p0 = abns['normalize'](pm[rng.integers(0, len(pm), len(pm))].mean(0))
        sink = int(np.argmax(p0))
        rank = np.asarray([i for i in np.argsort(-np.abs(delta)) if i != sink and abs(delta[i]) > 0])
        k = max(1, min(3, len(rank) // 2))
        for state, donors in [('preserved', rank[-k:]), ('damaged', rank[:k])]:
            sen = np.abs(delta[donors]) / max(float(np.abs(delta).max()), 1e-12)
            for x in np.linspace(0, 1, 21):
                q = p0.copy()
                m = x * q[donors] * sen
                q[donors] -= m
                q[sink] += m.sum()
                q = abns['normalize'](q)
                for metric, fn in [('V', abns['pielou']), ('D', abns['dominance_balance'])]:
                    rows.append([label, rep + 1, metric, state, x, 100 * fn(q) / fn(p0)])
pd.DataFrame(rows, columns=['pressure', 'repeat', 'metric', 'state', 'intensity', 'retention']).to_csv(Q / 'abundance_3v3.csv', index=False)
print('seven-metric recomputation complete')
