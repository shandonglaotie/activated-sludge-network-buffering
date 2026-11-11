from __future__ import annotations
import sys
sys.modules['bottleneck'] = None
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'input'
OUTPUT = ROOT / 'output_cp_uniform_random_fixed_order_20260825'
DRAW_RIBBONS = False
PRESSURE_LABEL = {'High TN': 'TN', 'High NH4-N': 'NH$_4$-N', 'High TP': 'TP', 'High COD': 'COD', 'Low C/N ratio': 'Low C/N', 'Low C/P ratio': 'Low C/P', 'Low temperature': 'Low temp.', 'High industrial proportion': 'Industrial wastewater'}
MAIN_PRESSURES = ['High TN', 'High NH4-N', 'High TP', 'High COD']
SUPP_PRESSURES = ['Low C/N ratio', 'Low C/P ratio', 'Low temperature', 'High industrial proportion']
DARK = '#641764'
LIGHT = '#C8A8D4'
LIGHT_FILL = '#E5C4DE'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--x-points', type=int, default=21)
    parser.add_argument('--pressure-set', choices=('all', 'main', 'supplement'), default='all')
    return parser.parse_args()

def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)

def internal_metrics(node_count: int, edges: np.ndarray, active: np.ndarray, community_idx: np.ndarray, base_internal_counts: np.ndarray, base_internal_lcc: np.ndarray) -> tuple[float, float]:
    keep = active[edges[:, 0]] & active[edges[:, 1]]
    kept_edges = edges[keep]
    same = community_idx[kept_edges[:, 0]] == community_idx[kept_edges[:, 1]]
    internal_edges = kept_edges[same]
    current_counts = np.bincount(community_idx[internal_edges[:, 0]], minlength=len(base_internal_counts))
    valid = base_internal_counts > 0
    direct = 100.0 * float(np.mean(current_counts[valid] / base_internal_counts[valid]))
    active_idx = np.flatnonzero(active)
    remap = np.full(node_count, -1, dtype=np.int32)
    remap[active_idx] = np.arange(len(active_idx))
    if len(internal_edges):
        rows = remap[internal_edges[:, 0]]
        cols = remap[internal_edges[:, 1]]
        graph = sparse.csr_matrix((np.ones(len(rows) * 2, dtype=np.int8), (np.r_[rows, cols], np.r_[cols, rows])), shape=(len(active_idx), len(active_idx)))
        _, labels = connected_components(graph, directed=False)
    else:
        labels = np.arange(len(active_idx))
    community_lcc = []
    for c in range(len(base_internal_lcc)):
        members = np.flatnonzero(active[active_idx] & (community_idx[active_idx] == c))
        if len(members) == 0 or base_internal_lcc[c] == 0:
            community_lcc.append(0.0)
            continue
        counts = np.bincount(labels[members])
        community_lcc.append(float(counts.max()) / base_internal_lcc[c])
    return (direct, 100.0 * float(np.mean(community_lcc)))

def core_periphery_support(edges: np.ndarray, active: np.ndarray, is_core: np.ndarray, baseline_cp_edges: int) -> float:
    keep = active[edges[:, 0]] & active[edges[:, 1]]
    kept = edges[keep]
    cp = is_core[kept[:, 0]] ^ is_core[kept[:, 1]]
    return 100.0 * int(cp.sum()) / max(1, baseline_cp_edges)

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ['mechanism', 'pressure', 'pressure_label', 'state', 'damage_intensity']
    grouped = df.groupby(keys).agg(metric1_mean=('metric1', 'mean'), metric1_sd=('metric1', 'std'), metric1_n=('metric1', 'count'), metric2_mean=('metric2', 'mean'), metric2_sd=('metric2', 'std'), metric2_n=('metric2', 'count'), pool_size_mean=('pool_size', 'mean')).reset_index()
    for metric in ('metric1', 'metric2'):
        half = 1.96 * grouped[f'{metric}_sd'].fillna(0) / np.sqrt(grouped[f'{metric}_n'].clip(lower=1))
        grouped[f'{metric}_low95'] = (grouped[f'{metric}_mean'] - half).clip(0, 100)
        grouped[f'{metric}_high95'] = (grouped[f'{metric}_mean'] + half).clip(0, 100)
    return grouped

def plot_grid(summary: pd.DataFrame, mechanism: str, pressures: list[str], filename: str, ylabels: tuple[str, str], xlabel: str, state_labels: tuple[str, str]) -> None:
    sub = summary[(summary['mechanism'] == mechanism) & summary['pressure'].isin(pressures)]
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in', 'font.size': 9})
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.9), sharex=False)
    colors = {state_labels[0]: DARK, state_labels[1]: LIGHT}
    fills = {state_labels[0]: '#BFAAD2', state_labels[1]: LIGHT_FILL}
    for col, pressure in enumerate(pressures):
        for row in range(2):
            ax = axes[row, col]
            pdat = sub[sub['pressure'] == pressure]
            for state in state_labels:
                dat = pdat[pdat['state'] == state].sort_values('damage_intensity')
                x = dat['damage_intensity'].to_numpy()
                mean = dat[f'metric{row + 1}_mean'].to_numpy()
                low = dat[f'metric{row + 1}_low95'].to_numpy()
                high = dat[f'metric{row + 1}_high95'].to_numpy()
                if DRAW_RIBBONS:
                    ax.fill_between(x, low, high, color=fills[state], alpha=0.28, linewidth=0)
                ax.plot(x, mean, color=colors[state], marker='o', markersize=2.7, linewidth=1.8, label=state)
            if row == 0:
                strip = mpl.patches.Rectangle((0.0, 1.015), 1.0, 0.085, transform=ax.transAxes, facecolor='#D9D9D9', edgecolor='black', linewidth=1.0, clip_on=False, zorder=3)
                ax.add_patch(strip)
                ax.text(0.5, 1.057, PRESSURE_LABEL[pressure], transform=ax.transAxes, ha='center', va='center', fontsize=10.5, zorder=4)
            if col == 0:
                ax.set_ylabel(ylabels[row], fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_xticklabels([f'{value:.1f}' for value in np.linspace(0, 1, 6)])
            ax.grid(False)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.tick_params(width=0.8, length=3)
            observed_min = float(pdat[f'metric{row + 1}_low95'].min())
            ymin = max(0.0, np.floor((observed_min - 10.0) / 10.0) * 10.0)
            ax.set_ylim(ymin, 100.0)
            ax.set_yticks(np.arange(ymin, 100.1, 10.0))
            ax.set_xlabel(xlabel, fontsize=9.2)
    fig.text(0.013, 0.965, 'a', fontsize=13, weight='bold')
    fig.text(0.013, 0.49, 'b', fontsize=13, weight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=10, handlelength=2.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.13, wspace=0.16, hspace=0.31)
    fig.savefig(OUTPUT / f'{filename}.svg', bbox_inches='tight')
    fig.savefig(OUTPUT / f'{filename}.png', dpi=600, bbox_inches='tight')
    fig.savefig(OUTPUT / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(OUTPUT / f'{filename}.tiff', dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

def plot_existing_bridge() -> None:
    bridge = pd.read_csv(INPUT / 'bridge_replicates_100.csv')
    bridge = bridge.rename(columns={'A_direct_intercommunity_support_retention_percent': 'metric1', 'B_alternative_intercommunity_path_retention_percent': 'metric2'})
    bridge['mechanism'] = 'bridge'
    bridge['pool_size'] = bridge['pool_size'].astype(float)
    bridge['state'] = bridge['state'].replace({'Structure-preserved': 'Low-priority inter-community edges removed', 'Connector structure damaged': 'High-priority pressure-sensitive edges removed'})
    bridge = summarize(bridge)
    for pressures, suffix in [(MAIN_PRESSURES, 'main'), (SUPP_PRESSURES, 'supplement')]:
        plot_grid(bridge, 'bridge', pressures, f'Figure_bridge_damage_{suffix}', ('Direct inter-community\nsupport retention (%)', 'Alternative inter-community\npath retention (%)'), 'Inter-community edge damage intensity', ('Low-priority inter-community edges removed', 'High-priority pressure-sensitive edges removed'))

def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(INPUT / 'network_nodes.csv')
    edges_df = pd.read_csv(INPUT / 'network_edges.csv')
    annotations = pd.read_csv(INPUT / 'node_annotations.csv').set_index('Id')
    profiles = pd.read_csv(INPUT / 'pressure_sensitive_profiles.csv')
    params = pd.read_csv(INPUT / 'pressure_attack_parameters_extended.csv').set_index('pressure')
    node_ids = sorted(nodes['Id'].tolist())
    node_index = {node: i for i, node in enumerate(node_ids)}
    node_count = len(node_ids)
    edge_array = np.array([(node_index[u], node_index[v]) for u, v in edges_df[['Source', 'Target']].itertuples(index=False, name=None)], dtype=np.int16)
    node_table = nodes.set_index('Id').loc[node_ids]
    annotations = annotations.loc[node_ids]
    communities = sorted(node_table['community'].unique())
    community_map = {c: i for i, c in enumerate(communities)}
    community_idx = node_table['community'].map(community_map).to_numpy(dtype=np.int8)
    is_core = node_table['fixed_core_for_CP_benchmark'].astype(str).str.lower().eq('true').to_numpy()
    role = node_table['hub_role'].fillna('').astype(str)
    degree = node_table['v_degree'].astype(float).to_numpy()
    between = node_table['v_betweenness'].fillna(0).astype(float).to_numpy()
    zi = node_table['Zi'].fillna(0).astype(float).to_numpy()
    degree_scaled = robust_scale(degree)
    between_scaled = robust_scale(between)
    zi_positive = robust_scale(np.maximum(zi, 0))
    internal_degree = np.zeros(node_count, dtype=float)
    same = community_idx[edge_array[:, 0]] == community_idx[edge_array[:, 1]]
    np.add.at(internal_degree, edge_array[same, 0], 1)
    np.add.at(internal_degree, edge_array[same, 1], 1)
    internal_degree_scaled = robust_scale(internal_degree)
    pressure_scores: dict[str, np.ndarray] = {}
    if args.pressure_set == 'main':
        selected_pressures = MAIN_PRESSURES
    elif args.pressure_set == 'supplement':
        selected_pressures = SUPP_PRESSURES
    else:
        selected_pressures = list(PRESSURE_LABEL)
    for pressure in PRESSURE_LABEL:
        score = pd.Series(0.0, index=annotations.index)
        for row in profiles[profiles['pressure'].eq(pressure)].itertuples(index=False):
            if row.feature == 'connector':
                score += role.eq('Connector hubs').astype(float) * float(row.weight)
            elif row.feature == 'provincial':
                score += role.eq('Provincial hubs').astype(float) * float(row.weight)
            elif row.feature == 'dominant':
                score += role.eq('Dominant hubs').astype(float) * float(row.weight)
            elif row.feature == 'betweenness':
                score += between_scaled * float(row.weight)
            else:
                score += annotations[row.feature].fillna(0).astype(float) * float(row.weight)
        pressure_scores[pressure] = robust_scale(score.to_numpy(dtype=float))
    severity_raw = params['random_edge_fraction'].fillna(0) + params['random_node_fraction'].fillna(0) + params['connector_node_fraction'].fillna(0) + params['intercommunity_edge_fraction'].fillna(0)
    severity = pd.Series(robust_scale(severity_raw.to_numpy()), index=severity_raw.index)
    all_node_ids = np.array(node_ids, dtype=object)
    connector_dominant = np.array([node_index[n] for n in node_table[role.isin(['Connector hubs', 'Dominant hubs'])].index], dtype=np.int16)

    def background_edges_and_active(pressure: str, repeat: int) -> tuple[np.ndarray, np.ndarray]:
        p = params.loc[pressure]
        seed = sum((ord(ch) for ch in pressure))
        rng = np.random.default_rng(800000 + seed * 1000 + repeat)
        active = np.ones(node_count, dtype=bool)
        n_random = int(round(float(p['random_node_fraction']) * node_count))
        n_cd = int(round(float(p['connector_node_fraction']) * node_count))
        if n_random:
            active[rng.choice(node_count, size=min(n_random, node_count), replace=False)] = False
        if n_cd:
            pool = connector_dominant[active[connector_dominant]]
            if len(pool):
                active[rng.choice(pool, size=min(n_cd, len(pool)), replace=False)] = False
        keep = active[edge_array[:, 0]] & active[edge_array[:, 1]]
        current = edge_array[keep]
        n_edge = int(round(float(p['random_edge_fraction']) * len(edge_array)))
        if n_edge and len(current):
            erng = np.random.default_rng(10000000 + seed * 10000 + repeat * 10 + 2)
            drop = erng.choice(len(current), size=min(n_edge, len(current)), replace=False)
            mask = np.ones(len(current), dtype=bool)
            mask[drop] = False
            current = current[mask]
        inter_fraction = float(p['intercommunity_edge_fraction'])
        if inter_fraction and len(current):
            inter = np.where(community_idx[current[:, 0]] != community_idx[current[:, 1]])[0]
            n_inter = int(round(inter_fraction * len(inter)))
            if n_inter:
                irng = np.random.default_rng(10000000 + seed * 10000 + repeat * 10 + 5)
                drop = irng.choice(inter, size=min(n_inter, len(inter)), replace=False)
                mask = np.ones(len(current), dtype=bool)
                mask[drop] = False
                current = current[mask]
        return (current, active)
    rows: list[dict] = []
    cp_pool_rows: list[dict] = []
    x_values = np.linspace(0, 1, args.x_points)
    for pressure in selected_pressures:
        pscore = pressure_scores[pressure]
        pressure_seed = sum((ord(ch) for ch in pressure))
        assort_pool_fraction = 0.08 + 0.1 * float(severity.loc[pressure])
        cp_pool_fraction = 0.06 + 0.1 * float(severity.loc[pressure])
        for repeat in range(1, args.repeats + 1):
            base_edges, base_active = background_edges_and_active(pressure, repeat)
            assort_priority = 1.8 * pscore + 1.3 * role.eq('Provincial hubs').to_numpy(dtype=float) + 0.9 * zi_positive + 0.7 * internal_degree_scaled
            candidates = np.flatnonzero(base_active & (internal_degree > 0))
            pool_size = max(3, int(round(assort_pool_fraction * len(candidates))))
            pool_size = min(pool_size, max(1, len(candidates) // 2))
            rng = np.random.default_rng(120000000 + pressure_seed * 1000 + repeat)
            order = candidates[np.argsort(assort_priority[candidates] + rng.normal(0, 1e-07, len(candidates)))]
            targeted_pool = order[-pool_size:][::-1]
            control_candidates = np.setdiff1d(candidates, targeted_pool, assume_unique=False)
            matched_random_pool = rng.choice(control_candidates, size=min(pool_size, len(control_candidates)), replace=False)
            rng.shuffle(matched_random_pool)
            pools = {'Matched random node removal': matched_random_pool, 'Pressure-sensitive assortative structure damaged': targeted_pool}
            base_internal_counts = np.zeros(len(communities), dtype=int)
            base_same = community_idx[base_edges[:, 0]] == community_idx[base_edges[:, 1]]
            np.add.at(base_internal_counts, community_idx[base_edges[base_same, 0]], 1)
            base_internal_lcc = np.zeros(len(communities), dtype=int)
            for c in range(len(communities)):
                members = np.flatnonzero(base_active & (community_idx == c))
                c_edges = base_edges[(community_idx[base_edges[:, 0]] == c) & (community_idx[base_edges[:, 1]] == c)]
                if len(members) == 0:
                    continue
                remap = np.full(node_count, -1, dtype=int)
                remap[members] = np.arange(len(members))
                if len(c_edges):
                    rr = remap[c_edges[:, 0]]
                    cc = remap[c_edges[:, 1]]
                    g = sparse.csr_matrix((np.ones(len(rr) * 2), (np.r_[rr, cc], np.r_[cc, rr])), shape=(len(members), len(members)))
                    _, lab = connected_components(g, directed=False)
                    base_internal_lcc[c] = int(np.bincount(lab).max())
                else:
                    base_internal_lcc[c] = 1
            for state, pool in pools.items():
                for x in x_values:
                    active = base_active.copy()
                    active[pool[:int(round(x * len(pool)))]] = False
                    m1, m2 = internal_metrics(node_count, base_edges, active, community_idx, base_internal_counts, base_internal_lcc)
                    rows.append({'mechanism': 'assortative', 'pressure': pressure, 'pressure_label': PRESSURE_LABEL[pressure], 'replicate': repeat, 'state': state, 'damage_intensity': x, 'pool_size': len(pool), 'metric1': m1, 'metric2': m2})
            candidates = np.flatnonzero(base_active)
            pool_size = max(3, int(round(cp_pool_fraction * len(candidates))))
            pool_size = min(pool_size, max(1, len(candidates) // 2))
            rng = np.random.default_rng(140000000 + pressure_seed * 1000 + repeat)
            cp_incident = np.zeros(node_count, dtype=float)
            cp_mask = is_core[base_edges[:, 0]] ^ is_core[base_edges[:, 1]]
            np.add.at(cp_incident, base_edges[cp_mask, 0], 1)
            np.add.at(cp_incident, base_edges[cp_mask, 1], 1)
            cp_priority = 1.5 * pscore + 1.5 * is_core.astype(float) + 0.6 * degree_scaled + 0.5 * between_scaled + 2.2 * robust_scale(cp_incident)
            order = candidates[np.argsort(cp_priority[candidates] + rng.normal(0, 1e-07, len(candidates)))]
            targeted_pool = order[-pool_size:][::-1]
            control_candidates = np.setdiff1d(candidates, targeted_pool, assume_unique=False)
            matched_random_pool = rng.choice(control_candidates, size=pool_size, replace=False)
            rng.shuffle(matched_random_pool)
            pools = {'Matched random node removal': matched_random_pool, 'Pressure-sensitive core-periphery structure damaged': targeted_pool}
            for state, pool in pools.items():
                for sequence_rank, node_idx in enumerate(pool, start=1):
                    cp_pool_rows.append({'pressure': pressure, 'pressure_label': PRESSURE_LABEL[pressure], 'replicate': repeat, 'state': state, 'pool_size': len(pool), 'sequence_rank': sequence_rank, 'node_id': all_node_ids[node_idx], 'cp_priority': cp_priority[node_idx], 'pressure_sensitivity': pscore[node_idx], 'is_core': bool(is_core[node_idx]), 'degree': degree[node_idx], 'betweenness': between[node_idx], 'cp_incident_edges': cp_incident[node_idx]})
            base_cp = int((is_core[base_edges[:, 0]] ^ is_core[base_edges[:, 1]]).sum())
            for state, pool in pools.items():
                for x in x_values:
                    active = base_active.copy()
                    active[pool[:int(round(x * len(pool)))]] = False
                    m1 = core_periphery_support(base_edges, active, is_core, base_cp)
                    rows.append({'mechanism': 'core_periphery', 'pressure': pressure, 'pressure_label': PRESSURE_LABEL[pressure], 'replicate': repeat, 'state': state, 'damage_intensity': x, 'pool_size': len(pool), 'metric1': m1, 'metric2': np.nan})
        print(f'Finished {pressure}')
    replicates = pd.DataFrame(rows)
    replicates.to_csv(OUTPUT / 'three_mechanism_new_replicates.csv', index=False)
    pd.DataFrame(cp_pool_rows).to_csv(OUTPUT / 'core_periphery_pool_membership_and_fixed_order.csv', index=False)
    summary = summarize(replicates)
    summary.to_csv(OUTPUT / 'three_mechanism_new_summary.csv', index=False)
    summary[np.isclose(summary['damage_intensity'], 1.0)].to_csv(OUTPUT / 'three_mechanism_new_endpoints_x1.csv', index=False)
    plot_sets = []
    if args.pressure_set in ('all', 'main'):
        plot_sets.append((MAIN_PRESSURES, 'main'))
    if args.pressure_set in ('all', 'supplement'):
        plot_sets.append((SUPP_PRESSURES, 'supplement'))
    for pressures, suffix in plot_sets:
        plot_grid(summary, 'assortative', pressures, f'Figure_assortative_damage_{suffix}', ('Within-community direct\nsupport retention (%)', 'Within-community module\nintegrity retention (%)'), 'Assortative-structure node damage intensity', ('Matched random node removal', 'Pressure-sensitive assortative structure damaged'))
    plot_existing_bridge()
    contract = pd.DataFrame([{'mechanism': 'bridge', 'damage_object': 'pressure-sensitive inter-community edges', 'damage_type': 'edge', 'metric_1': 'Direct inter-community support retention', 'metric_2': 'Alternative inter-community path retention'}, {'mechanism': 'assortative', 'damage_object': 'pressure-sensitive provincial/high-Zi internal-support nodes', 'damage_type': 'node', 'metric_1': 'Within-community direct support retention', 'metric_2': 'Within-community module integrity retention'}, {'mechanism': 'core_periphery', 'damage_object': 'pressure-sensitive high-core/high-degree backbone nodes', 'damage_type': 'node', 'metric_1': 'Core-to-periphery direct support retention', 'metric_2': ''}])
    contract.to_csv(OUTPUT / 'figure_method_contract.csv', index=False)
if __name__ == '__main__':
    main()
