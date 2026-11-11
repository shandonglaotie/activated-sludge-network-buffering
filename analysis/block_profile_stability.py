from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'shared/data'
SRC = ROOT / 'results/block_profile_stability'
SRC.mkdir(parents=True, exist_ok=True)
nodes = pd.read_csv(DATA / '05_Network' / '05A_network_node_attributes_456.csv')
edges = pd.read_csv(DATA / '05_Network' / '05B_positive_weighted_edge_list_13875.csv')
comm_ids = sorted(nodes['community_id'].astype(int).unique())
comm_to_zero = {c: i for i, c in enumerate(comm_ids)}
comm = dict(zip(nodes['node_id'].astype(str), nodes['community_id'].astype(int).map(comm_to_zero)))
sizes_by = nodes.assign(_community_zero=nodes['community_id'].astype(int).map(comm_to_zero)).groupby('_community_zero').size().sort_index().to_numpy(float)
u = edges['source_node'].astype(str).map(comm).to_numpy(int)
v = edges['target_node'].astype(str).map(comm).to_numpy(int)
w = edges['abs_weight'].to_numpy(float)

def weighted_profile(indices: np.ndarray) -> np.ndarray:
    mat = np.zeros((8, 8), float)
    for idx in indices:
        a, b, ww = (u[idx], v[idx], w[idx])
        mat[a, b] += ww
        mat[b, a] += ww
    return mat
full = weighted_profile(np.arange(len(edges)))
resample_engine = np.random.Generator(np.random.PCG64(0))
vals = []
keep = 12487
for _ in range(2320):
    cur = weighted_profile(resample_engine.choice(len(edges), size=keep, replace=False))
    vals.append(float(np.mean([np.corrcoef(full[i], cur[i])[0, 1] for i in range(8)])))
vals = np.asarray(vals)
pd.DataFrame({'resample': np.arange(1, 2321), 'mean_weighted_block_profile_r': vals}).to_csv(SRC / 'Fig_S1d_edge_resampling_2320_values.csv', index=False)
