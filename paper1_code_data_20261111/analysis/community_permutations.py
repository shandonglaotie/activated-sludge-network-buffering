import sys
sys.modules['bottleneck'] = None
from pathlib import Path
import sys
sys.modules.setdefault('bottleneck', None)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / 'shared/data'
OUT = PROJECT_ROOT / 'results/permutations'
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = PROJECT_ROOT / 'shared/section_metrics/community_structural_core_node_audit.csv'
EDGE = DATA / '10_Raw_source_data/RMT_ASV_0.00001_0.01_edge.csv'
COMM = [f'C{i:02d}' for i in range(1, 9)]
K = 8
N_PERM = 5000
SEED = 20260716
EPS = 1e-12
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
nodes = pd.read_csv(AUDIT, low_memory=False)
nodes = nodes[nodes['structural_core'].astype(str).str.lower().eq('true')].copy()
nodes['Id'] = nodes['Id'].astype(str)
nodes['community'] = nodes['community'].astype(str)
expected = {'C01': 161, 'C02': 108, 'C03': 49, 'C04': 48, 'C05': 42, 'C06': 22, 'C07': 15, 'C08': 11}
assert nodes.groupby('community').size().to_dict() == expected
node_index = {node_id: i for i, node_id in enumerate(nodes['Id'])}
community_index = {community: i for i, community in enumerate(COMM)}
labels = nodes['community'].map(community_index).to_numpy(np.int16)
sizes = np.bincount(labels, minlength=K)
edges = pd.read_csv(EDGE, low_memory=False)
edges['Source'] = edges['Source'].astype(str)
edges['Target'] = edges['Target'].astype(str)
edges = edges[edges['Source'].isin(node_index) & edges['Target'].isin(node_index) & edges['Source'].ne(edges['Target'])].copy()
u = edges['Source'].map(node_index).to_numpy(np.int32)
v = edges['Target'].map(node_index).to_numpy(np.int32)
w = np.abs(pd.to_numeric(edges['Weight'], errors='coerce').fillna(0).to_numpy(float))
edge_count = len(edges)
mean_edge_weight = float(np.mean(w))

def continuous_participation(node_labels):
    a = node_labels[u]
    b = node_labels[v]
    same = a == b
    within = np.zeros(K, dtype=float)
    between = np.zeros(K, dtype=float)
    incident = np.zeros(K, dtype=float)
    np.add.at(within, a[same], 1.0)
    np.add.at(between, a[~same], 1.0)
    np.add.at(between, b[~same], 1.0)
    np.add.at(incident, a, 1.0)
    np.add.at(incident, b, 1.0)
    core = 100.0 * within / max(edge_count, 1)
    node_strength = np.zeros(len(node_labels), dtype=float)
    np.add.at(node_strength, u, w)
    np.add.at(node_strength, v, w)
    global_mean_strength = node_strength.mean()
    community_mean_strength = np.array([node_strength[node_labels == i].mean() for i in range(K)])
    periphery = 100.0 * (global_mean_strength - community_mean_strength) / global_mean_strength
    within_weight = np.zeros(K, dtype=float)
    between_weight = np.zeros((K, K), dtype=float)
    np.add.at(within_weight, a[same], w[same])
    np.add.at(between_weight, (a[~same], b[~same]), w[~same])
    np.add.at(between_weight, (b[~same], a[~same]), w[~same])
    sample_sizes = np.bincount(node_labels, minlength=K).astype(float)
    possible_within = sample_sizes * (sample_sizes - 1.0) / 2.0
    correction = 0.5 * mean_edge_weight
    d_in = np.divide(within_weight + correction, possible_within + 0.5, out=np.zeros(K), where=possible_within > 0)
    possible_between = np.outer(sample_sizes, sample_sizes)
    d_between = np.divide(between_weight + correction, possible_between + 0.5, out=np.zeros_like(between_weight), where=possible_between > 0)
    np.fill_diagonal(d_between, 0.0)
    other_sizes = sample_sizes.sum() - sample_sizes
    d_out = np.divide((d_between * sample_sizes[None, :]).sum(axis=1), other_sizes, out=np.zeros(K), where=other_sizes > 0)
    assortative = np.log10((d_in + EPS) / (d_out + EPS))
    return (core, periphery, assortative)
observed = continuous_participation(labels)
rng = np.random.default_rng(SEED)
base_labels = np.repeat(np.arange(K, dtype=np.int16), sizes)
null = [np.zeros((N_PERM, K), dtype=float) for _ in range(3)]
for permutation in range(N_PERM):
    values = continuous_participation(rng.permutation(base_labels))
    for metric_index in range(3):
        null[metric_index][permutation] = values[metric_index]
np.savez_compressed(OUT / 'community_permutations.npz', null_core=null[0], null_periphery=null[1], null_assortative=null[2], observed_core=observed[0], observed_periphery=observed[1], observed_assortative=observed[2])
