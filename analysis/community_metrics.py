from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/community_metrics'
OUT.mkdir(parents=True, exist_ok=True)
nodes = pd.read_csv(ROOT / 'results/roles_and_resampling/node_roles.csv')
edges = pd.read_csv(ROOT / 'shared/data/10_Raw_source_data/RMT_ASV_0.00001_0.01_edge.csv')
ids = set(nodes.Id)
edges = edges[edges.Source.isin(ids) & edges.Target.isin(ids) & edges.Source.ne(edges.Target)]
communities = nodes.set_index('Id').community
edges['community_1'] = edges.Source.map(communities)
edges['community_2'] = edges.Target.map(communities)
sizes = nodes.community.value_counts()
density = {c: len(edges[(edges.community_1 == c) & (edges.community_2 == c)]) / (n * (n - 1) / 2) for c, n in sizes.items()}
rows = []
for i, a in enumerate(sorted(sizes.index)):
    for b in sorted(sizes.index)[i + 1:]:
        between = len(edges[(edges.community_1 == a) & (edges.community_2 == b) | (edges.community_1 == b) & (edges.community_2 == a)]) / (sizes[a] * sizes[b])
        low, high = sorted([density[a], density[b]])
        architecture = 'Assortative' if between < low else 'Disassortative' if between > high else 'Core-periphery'
        rows.append([a, b, density[a], density[b], between, architecture])
pd.DataFrame(rows, columns=['community_1', 'community_2', 'within_density_1', 'within_density_2', 'between_density', 'architecture']).to_csv(OUT / 'pairwise_densities.csv', index=False)
FUNCTIONS = ['Nitrifiers', 'Denitrifiers', 'Anammox', 'DNRA', 'PAOs', 'GAOs', 'SRB', 'SOB', 'Filamentous', 'Acetogen', 'Methanogen', 'Fermenter', 'Sugars', 'Lipids', 'Proteins']

def flag(name):
    return pd.to_numeric(nodes[name], errors='coerce').fillna(0).gt(0)
proc = nodes['v_process'].fillna('').astype(str)
roles = {'Heterotrophic organic-matter degradation': proc.str.contains('C_cycle', case=False, regex=False), 'Fermentation and anaerobic carbon conversion': flag('Fermenter') | flag('Acetogen') | flag('Methanogen'), 'Protein/sugar/lipid transformation': flag('Proteins') | flag('Sugars') | flag('Lipids'), 'Nitrogen removal': flag('Nitrifiers') | flag('Denitrifiers') | flag('Anammox') | flag('DNRA') | proc.str.contains('N_', case=False, regex=False), 'Phosphorus removal': flag('PAOs') | proc.str.contains('P_', case=False, regex=False), 'Sulfur transformation': flag('SRB') | flag('SOB'), 'Floc/structural maintenance': flag('Filamentous')}
role = pd.DataFrame(index=sorted(sizes.index))
for name, mask in roles.items():
    role[name] = [nodes.loc[mask & nodes.community.eq(c), 'MRA'].sum() for c in role.index]
role = role.div(role.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
role.to_csv(OUT / 'functional_composition.csv')
