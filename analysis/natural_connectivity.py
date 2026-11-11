from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/natural_connectivity'
OUT.mkdir(parents=True, exist_ok=True)
nodes = pd.read_csv(ROOT / 'shared/data/05_Network/05A_network_node_attributes_456.csv')
edges = pd.read_csv(ROOT / 'shared/data/05_Network/05B_positive_weighted_edge_list_13875.csv')
roles = pd.read_csv(ROOT / 'results/roles_and_resampling/node_roles.csv').set_index('Id').loc[nodes.node_id]
index = {node: i for i, node in enumerate(nodes.node_id)}
u = edges.source_node.map(index).to_numpy()
v = edges.target_node.map(index).to_numpy()
adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
adjacency[u, v] = adjacency[v, u] = 1.0
provincial = np.flatnonzero(roles.hub_role.eq('Provincial hubs'))
connector = np.flatnonzero(roles.hub_role.eq('Connector hubs'))
maximum = min(len(provincial), len(connector))
steps = np.unique(np.r_[np.arange(0, maximum + 1, 3), maximum])
rows = []
order_rows = []
for replicate in range(1, 21):
    rng = np.random.default_rng(20260726 + replicate)
    orders = {
        'Random nodes': rng.choice(len(nodes), size=maximum, replace=False),
        'Provincial hubs': rng.permutation(provincial)[:maximum],
        'Connector hubs': rng.choice(connector, size=maximum, replace=False),
    }
    for group, order in orders.items():
        for rank, node in enumerate(order, 1):
            order_rows.append([replicate, group, rank, nodes.node_id.iloc[node]])
        for removed in steps:
            retained = np.ones(len(nodes), dtype=bool)
            retained[order[:removed]] = False
            spectrum = np.linalg.eigvalsh(adjacency[np.ix_(retained, retained)])
            connectivity = float(logsumexp(spectrum) - np.log(retained.sum()))
            rows.append([replicate, group, removed, connectivity])
    print(f'Replicate {replicate}/20', flush=True)
raw = pd.DataFrame(rows, columns=['replicate', 'group', 'nodes_removed', 'natural_connectivity'])
raw.to_csv(OUT / 'node_removal_replicates.csv', index=False)
pd.DataFrame(order_rows, columns=['replicate', 'group', 'removal_rank', 'node_id']).to_csv(OUT / 'node_removal_orders.csv', index=False)
summary = raw.groupby(['group', 'nodes_removed']).natural_connectivity.agg(
    mean='mean', sd='std', low95=lambda x: x.quantile(.025), high95=lambda x: x.quantile(.975)
).reset_index()
summary.to_csv(OUT / 'natural_connectivity_summary.csv', index=False)
