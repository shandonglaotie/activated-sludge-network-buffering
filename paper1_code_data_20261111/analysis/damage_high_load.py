from pathlib import Path
import sys, runpy, types, shutil, json
sys.modules['bottleneck'] = None
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
O = R / 'results/damage_high_load'
I = O / 'input'
Q = O / 'simulation'
I.mkdir(parents=True, exist_ok=True)
Q.mkdir(exist_ok=True)
B = R / 'shared/simulation_engines'
DATA = R / 'shared/data'
plant = pd.read_csv(DATA / '04_Metadata/04C_plant_metadata_252.csv').merge(pd.read_csv(DATA / '08_PCA_source_data/08A_PCA_input_and_scores_252_plants.csv'), on=['Site_ID', 'strict_223_subset'], validate='one_to_one')
valid = plant.AtInfTN.gt(plant.AtInfNH4N) & plant.AtInfNH4N.ge(0)
params = pd.read_csv(B / 'topology/input/pressure_attack_parameters_extended.csv').set_index('pressure')
old = params.copy()
specs = []
audit = []
for p, col in [('High TN', 'AtInfTN'), ('High NH4-N', 'AtInfNH4N'), ('High TP', 'AtInfTP'), ('High COD', 'AtInfCOD')]:
    eligible = valid if col in ['AtInfTN', 'AtInfNH4N'] else plant[col].notna()
    cut = plant.loc[eligible, col].quantile(0.8)
    em = eligible & plant[col].ge(cut)
    rm = eligible & plant[col].lt(cut)
    e = plant.loc[em]
    r = plant.loc[rm]
    delta = e[['robust_z_' + c for c in 'EFCSVD']].mean() - r[['robust_z_' + c for c in 'EFCSVD']].mean()
    for c, v in zip('EFCSVD', delta):
        params.loc[p, 'difference_' + c] = v
    specs.append((p, set(e.Site_ID), set(r.Site_ID)))
    audit.append(dict(pressure=p, cutoff=float(cut), exposed_n=len(e), reference_n=len(r), eligible_n=int(eligible.sum()), high_rule='concentration >= eligible Q80; all ties included', nitrogen_rule='TN > NH4N and NH4N >= 0', difference=dict(zip('EFCSVD', delta.to_numpy().tolist()))))
for col, c, base, scale in [('random_edge_fraction', 'E', 0.01, 0.07), ('random_node_fraction', 'F', 0.005, 0.025), ('connector_node_fraction', 'C', 0, 0.03), ('intercommunity_edge_fraction', 'S', 0, 0.06)]:
    den = max(float(old['difference_' + c].max()), 1e-12)
    for p, _, _ in specs:
        params.loc[p, col] = base + scale * max(float(params.loc[p, 'difference_' + c]), 0) / den
    assert params[col].between(0, 1).all()
for f in ['node_annotations.csv', 'network_nodes.csv', 'network_edges.csv', 'pressure_sensitive_profiles.csv']:
    shutil.copy2(B / 'topology/input' / f, I / f)
params.reset_index().to_csv(I / 'pressure_attack_parameters_extended.csv', index=False)
basic = pd.read_csv(B / 'topology/input/pressure_attack_parameters.csv').columns
params.reset_index().loc[lambda x: ~x.pressure.eq('High industrial proportion'), basic].to_csv(I / 'pressure_attack_parameters.csv', index=False)
(O / 'group_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
print('Prepared corrected four-pressure inputs', flush=True)
ns = runpy.run_path(str(B / 'topology/simulate_and_plot_three_mechanisms_corrected_cp.py'), run_name='four_influent_topology')
g = ns['main'].__globals__
g['INPUT'] = I
g['OUTPUT'] = Q / 'topology'
g['MAIN_PRESSURES'] = [a[0] for a in specs]
g['parse_args'] = lambda: types.SimpleNamespace(repeats=100, x_points=21, pressure_set='main')
g['plot_grid'] = lambda *a, **k: None
g['plot_existing_bridge'] = lambda: None
ns['main']()
print('Topology completed', flush=True)
bridge = next((R / 'shared/bridge_engine').glob('*01_simulate_pressure_specific_damage.py'))
code = bridge.read_text(encoding='utf-8-sig')
code = code.replace('pressure_order = list(pressure_label)', "pressure_order = ['High TN','High NH4-N','High TP','High COD']")
bn = {'__name__': 'four_influent_bridge', '__file__': str(bridge)}
exec(compile(code, str(bridge), 'exec'), bn)
bn['parse_args'] = lambda: types.SimpleNamespace(repeats=100, x_points=21, input_dir=I, output_dir=Q / 'bridge')
bn['main']()
print('Bridge completed', flush=True)
abns = runpy.run_path(str(B / 'abundance/simulate_and_plot_abundance_organization_damage.py'), run_name='four_influent_abundance')
ab = pd.read_csv(R / 'shared/community_abundance/06_community_abundance.csv')
wide = ab.pivot(index='sample_id', columns='community_name', values='community_relative_abundance').reindex(columns=[f'C{i:02d}' for i in range(1, 9)])
sg = pd.read_csv(R / 'shared/community_abundance/05_sample_pressure_groups.csv')
print('Group columns', sg.columns.tolist(), flush=True)
sitecol = next((c for c in ['wwtp_id', 'Site_ID', 'site_id'] if c in sg.columns))
rows = []
for pi, (p, eids, rids) in enumerate(specs):
    pm = wide.loc[wide.index.intersection(sg.loc[sg[sitecol].isin(eids), 'sample_id'])].to_numpy(float)
    rm = wide.loc[wide.index.intersection(sg.loc[sg[sitecol].isin(rids), 'sample_id'])].mean().to_numpy(float)
    assert len(pm) > 0 and np.isfinite(pm).all() and np.isfinite(rm).all()
    delta = pm.mean(0) - rm
    for rep in range(100):
        rng = np.random.default_rng(2026110200 + pi * 10000 + rep)
        p0 = abns['normalize'](pm[rng.integers(0, len(pm), len(pm))].mean(0))
        sink = int(np.argmax(p0))
        rank = np.asarray([i for i in np.argsort(-np.abs(delta)) if i != sink and abs(delta[i]) > 0])
        k = max(1, min(3, len(rank) // 2))
        for state, donors in [('preserved', rank[-k:]), ('damaged', rank[:k])]:
            sen = np.abs(delta[donors]) / max(float(np.abs(delta).max()), 1e-12)
            for x in np.linspace(0, 1, 21):
                qq = p0.copy()
                mass = x * qq[donors] * sen
                qq[donors] -= mass
                qq[sink] += mass.sum()
                qq = abns['normalize'](qq)
                for metric, fn in [('V', abns['pielou']), ('D', abns['dominance_balance'])]:
                    rows.append([p, rep + 1, metric, state, x, 100 * fn(qq) / fn(p0)])
pd.DataFrame(rows, columns=['pressure', 'repeat', 'metric', 'state', 'intensity', 'retention']).to_csv(Q / 'abundance_3v3.csv', index=False)
print('All simulations completed', flush=True)
