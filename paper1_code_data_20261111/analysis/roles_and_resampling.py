import sys
sys.modules['bottleneck'] = None
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/roles_and_resampling'
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = ROOT / 'results/partition/tables/community_structural_core_node_audit.csv'
EDGE = ROOT / 'shared/data/10_Raw_source_data/RMT_ASV_0.00001_0.01_edge.csv'
COMM = [f'C{i:02d}' for i in range(1, 9)]
K = 8
RNG = np.random.default_rng(20260717)
COLORS = {'Provincial hubs': '#5A8E3E', 'Connector hubs': '#405774', 'Dominant hubs': '#AA1E22', 'Non-hubs': '#BDBDBD'}
ROLE_ORDER = ['Provincial hubs', 'Connector hubs', 'Dominant hubs', 'Non-hubs']
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
nodes = pd.read_csv(AUDIT, low_memory=False)
nodes = nodes[nodes['structural_core'].astype(str).str.lower().eq('true')].copy()
nodes['Id'] = nodes['Id'].astype(str)
nodes['community'] = nodes['community'].astype(str)
assert nodes.groupby('community').size().to_dict() == {'C01': 161, 'C02': 108, 'C03': 49, 'C04': 48, 'C05': 42, 'C06': 22, 'C07': 15, 'C08': 11}
ci = {c: i for i, c in enumerate(COMM)}
labels = nodes['community'].map(ci).to_numpy(np.int16)
node_index = {node: i for i, node in enumerate(nodes['Id'])}
N = len(nodes)
edges = pd.read_csv(EDGE, low_memory=False)
edges['Source'] = edges['Source'].astype(str)
edges['Target'] = edges['Target'].astype(str)
edges = edges[edges['Source'].isin(node_index) & edges['Target'].isin(node_index) & edges['Source'].ne(edges['Target'])].copy()
u = edges['Source'].map(node_index).to_numpy(np.int32)
v = edges['Target'].map(node_index).to_numpy(np.int32)
w = np.abs(pd.to_numeric(edges['Weight'], errors='coerce').fillna(0).to_numpy(float))
M = len(edges)
degree_by_comm = np.zeros((N, K), dtype=float)
np.add.at(degree_by_comm, (u, labels[v]), 1.0)
np.add.at(degree_by_comm, (v, labels[u]), 1.0)
total_degree = degree_by_comm.sum(1)
within_degree = degree_by_comm[np.arange(N), labels]
zi = np.zeros(N)
for k in range(K):
    hit = labels == k
    sd = within_degree[hit].std(ddof=1) if hit.sum() > 1 else 0
    zi[hit] = (within_degree[hit] - within_degree[hit].mean()) / sd if sd > 0 else 0
pc = np.zeros(N)
valid = total_degree > 0
pc[valid] = 1 - np.sum((degree_by_comm[valid] / total_degree[valid, None]) ** 2, axis=1)
nodes['Zi'] = zi
nodes['PC'] = pc
pre_role = np.full(N, 'Non-hubs', dtype=object)
pre_role[(zi > 0) & (pc < 0.3)] = 'Provincial hubs'
pre_role[(zi > 0) & (pc >= 0.3) & (pc < 0.75)] = 'Connector hubs'
nodes['pre_role'] = pre_role

def clean_taxon(row):
    bad = {'', 'unknown', 'unclassified', 'nan', 'none'}
    for col in ('v_genus', 'v_family', 'v_order'):
        value = str(row.get(col, '')).strip()
        if value.lower() not in bad and (not value.lower().startswith('midas_')):
            return value
    return 'Unknown'
nodes['Taxonomic Classification'] = nodes.apply(clean_taxon, axis=1)
nodes['MRA'] = pd.to_numeric(nodes['MRA'], errors='coerce').fillna(0)
taxon_mra = nodes.groupby('Taxonomic Classification')['MRA'].sum()
nodes['taxon_total_MRA'] = nodes['Taxonomic Classification'].map(taxon_mra)
hub_taxa = nodes.loc[nodes['pre_role'].ne('Non-hubs'), 'Taxonomic Classification'].unique()
threshold = 0.004
nodes['hub_role'] = nodes['pre_role']
dominant = nodes['pre_role'].ne('Non-hubs') & (nodes['taxon_total_MRA'] >= threshold)
nodes.loc[dominant, 'hub_role'] = 'Dominant hubs'
topology = np.full(N, 'Peripheral nodes', dtype=object)
topology[(zi <= 0) & (pc <= 0.05)] = 'Ultra-peripheral nodes'
topology[(zi <= 0) & (pc > 0.05) & (pc < 0.625)] = 'Peripheral nodes'
topology[(zi <= 0) & (pc >= 0.625) & (pc < 0.8)] = 'Non-hub connectors'
topology[(zi <= 0) & (pc >= 0.8)] = 'Non-hub kinless nodes'
topology[(zi > 0) & (pc < 0.3)] = 'Provincial hubs'
topology[(zi > 0) & (pc >= 0.3) & (pc < 0.75)] = 'Connector hubs'
topology[(zi > 0) & (pc >= 0.75)] = 'Kinless hubs'
nodes['topology_role'] = topology

def block_density(indices):
    a = labels[u[indices]]
    b = labels[v[indices]]
    counts = np.zeros((K, K))
    np.add.at(counts, (a, b), 1)
    diff = a != b
    np.add.at(counts, (b[diff], a[diff]), 1)
    sizes = np.bincount(labels, minlength=K).astype(float)
    poss = np.outer(sizes, sizes)
    np.fill_diagonal(poss, sizes * (sizes - 1) / 2)
    return np.divide(counts, poss, out=np.zeros_like(counts), where=poss > 0)
tri = np.triu_indices(K)
obs_vec = block_density(np.arange(M))[tri]
runs = 58
repeats = 40
keep = int(0.9 * M)
raw = np.zeros((runs, repeats))
for run in range(runs):
    for rep in range(repeats):
        take = RNG.choice(M, size=keep, replace=False)
        vec = block_density(take)[tri]
        r = np.corrcoef(obs_vec, vec)[0, 1]
        raw[run, rep] = max(0, 1 - r)
relative = raw / raw.sum(axis=0, keepdims=True)
contrib = relative.mean(1)
se = relative.std(1, ddof=1) / math.sqrt(repeats)
lower = np.clip(contrib - 1.96 * se, 0, None)
upper = contrib + 1.96 * se
threshold_line = float(upper.max() + 0.003)
scan_pct = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])
scan_taxa = []
scan_hubs = []
for pct0 in scan_pct:
    hit = nodes['pre_role'].ne('Non-hubs') & (nodes['taxon_total_MRA'] >= pct0 / 100)
    scan_hubs.append(int(hit.sum()))
    scan_taxa.append(int(nodes.loc[hit, 'Taxonomic Classification'].nunique()))
threshold_scan = pd.DataFrame({'threshold_percent': scan_pct, 'dominant_hubs': scan_hubs, 'dominant_taxa': scan_taxa})
threshold_scan.to_csv(OUT / 'Dominant_threshold_sensitivity.csv', index=False, encoding='utf-8-sig')
nodes.to_csv(OUT / 'node_roles.csv', index=False)
pd.DataFrame({'run': np.arange(1, runs + 1), 'relative_network_contribution_u': contrib, 'lower_95pct': lower, 'upper_95pct': upper, 'upper_95pct_resampling_envelope': threshold_line}).to_csv(OUT / 'edge_resampling.csv', index=False)
np.savez_compressed(OUT / 'edge_resampling_arrays.npz', raw=raw, relative=relative)
