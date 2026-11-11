from __future__ import annotations
import sys
sys.modules['bottleneck'] = None
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import networkx as nx

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pressure-specific 100% inter-community structure damage simulation')
    parser.add_argument('--repeats', type=int, default=100, help='Number of paired simulation repeats per pressure')
    parser.add_argument('--x-points', type=int, default=21, help='Number of damage intensity points including 0 and 1')
    parser.add_argument('--input-dir', type=Path, default=Path('input'))
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    return parser.parse_args()

def mean_flow(support: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    k = support.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(k))
    for i, j in pairs:
        capacity = float(support[i, j])
        if capacity > 0:
            graph.add_edge(i, j, capacity=capacity)
    components = list(nx.connected_components(graph))
    component_id = {node: idx for idx, comp in enumerate(components) for node in comp}
    trees: dict[int, nx.Graph] = {}
    for idx, comp in enumerate(components):
        if len(comp) >= 2:
            trees[idx] = nx.gomory_hu_tree(graph.subgraph(comp).copy(), capacity='capacity')
    values: list[float] = []
    for i, j in pairs:
        if component_id.get(i) != component_id.get(j):
            values.append(0.0)
            continue
        tree = trees[component_id[i]]
        path = nx.shortest_path(tree, i, j)
        values.append(min((float(tree[a][b]['weight']) for a, b in zip(path[:-1], path[1:]))))
    return float(np.mean(values))

def cumulative_pair_removal_counts(pool_pair_ids: np.ndarray, pair_count: int) -> np.ndarray:
    m = len(pool_pair_ids)
    out = np.zeros((m + 1, pair_count), dtype=np.int32)
    if m == 0:
        return out
    rows = np.arange(1, m + 1)
    out[rows, pool_pair_ids] = 1
    np.cumsum(out, axis=0, out=out)
    return out

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = output_dir / 'reps'
    reps_dir.mkdir(exist_ok=True)
    annotations = pd.read_csv(input_dir / 'node_annotations.csv').set_index('Id')
    network_nodes = pd.read_csv(input_dir / 'network_nodes.csv')
    network_edges = pd.read_csv(input_dir / 'network_edges.csv')
    pressure_params = pd.read_csv(input_dir / 'pressure_attack_parameters.csv')
    pressure_params_ext = pd.read_csv(input_dir / 'pressure_attack_parameters_extended.csv')
    profiles_df = pd.read_csv(input_dir / 'pressure_sensitive_profiles.csv')
    industrial = pressure_params_ext[pressure_params_ext['pressure'].eq('High industrial proportion')][pressure_params.columns]
    pressure_params = pd.concat([pressure_params, industrial], ignore_index=True).set_index('pressure')
    annotations = annotations.join(network_nodes.set_index('Id')[['community', 'hub_role', 'v_betweenness']], how='inner', rsuffix='_network')
    annotations['betweenness_scaled'] = annotations['v_betweenness'].fillna(0).astype(float)
    max_bet = float(annotations['betweenness_scaled'].max())
    if max_bet > 0:
        annotations['betweenness_scaled'] /= max_bet
    role = annotations['hub_role'].fillna('')
    node_ids = sorted(network_nodes['Id'].tolist())
    node_count = len(node_ids)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    all_node_ids = np.array(node_ids, dtype=object)
    connector_dominant_pool = np.array(sorted(network_nodes[network_nodes['hub_role'].isin(['Connector hubs', 'Dominant hubs'])]['Id'].tolist()), dtype=object)
    communities = sorted(network_nodes['community'].unique().tolist())
    community_to_index = {community: idx for idx, community in enumerate(communities)}
    community_index = np.array([community_to_index[c] for c in network_nodes.set_index('Id').loc[node_ids, 'community']], dtype=np.int8)
    community_count = len(communities)
    community_pairs = [(i, j) for i in range(community_count) for j in range(i + 1, community_count)]
    pair_to_id = {pair: idx for idx, pair in enumerate(community_pairs)}
    pair_count = len(community_pairs)
    edge_array = np.array([(node_index[u], node_index[v]) for u, v in network_edges[['Source', 'Target']].itertuples(index=False, name=None)], dtype=np.int16)
    pressure_label = {'High TN': 'TN', 'High NH4-N': 'NH$_4$-N', 'High TP': 'TP', 'High COD': 'COD', 'Low C/N ratio': 'Low C/N', 'Low C/P ratio': 'Low C/P', 'Low temperature': 'Low temp.', 'High industrial proportion': 'Industrial wastewater'}
    pressure_order = list(pressure_label)
    node_score_maps: dict[str, np.ndarray] = {}
    for pressure in pressure_order:
        score = pd.Series(0.0, index=annotations.index)
        for row in profiles_df[profiles_df['pressure'].eq(pressure)].itertuples(index=False):
            feature = row.feature
            weight = float(row.weight)
            if feature == 'connector':
                score += role.eq('Connector hubs').astype(float) * weight
            elif feature == 'provincial':
                score += role.eq('Provincial hubs').astype(float) * weight
            elif feature == 'dominant':
                score += role.eq('Dominant hubs').astype(float) * weight
            elif feature == 'betweenness':
                score += annotations['betweenness_scaled'] * weight
            else:
                if feature not in annotations.columns:
                    raise KeyError(f'Missing annotation column required by profile: {feature}')
                score += annotations[feature].fillna(0).astype(float) * weight
        node_score_maps[pressure] = score.reindex(node_ids).to_numpy(dtype=float)
    severity_raw = pressure_params['random_edge_fraction'].fillna(0).astype(float) + pressure_params['random_node_fraction'].fillna(0).astype(float) + pressure_params['connector_node_fraction'].fillna(0).astype(float) + pressure_params['intercommunity_edge_fraction'].fillna(0).astype(float)
    severity_norm = (severity_raw - severity_raw.min()) / (severity_raw.max() - severity_raw.min() + 1e-12)
    target_pool_fraction = (0.18 + 0.17 * severity_norm).to_dict()

    def pressure_background_network(pressure: str, repeat_id: int) -> np.ndarray:
        params = pressure_params.loc[pressure]
        pressure_seed = sum((ord(ch) for ch in pressure))
        rng = np.random.default_rng(800000 + pressure_seed * 1000 + repeat_id)
        removed_nodes = np.zeros(node_count, dtype=bool)
        random_node_n = int(round(float(params['random_node_fraction']) * node_count))
        connector_node_n = int(round(float(params['connector_node_fraction']) * node_count))
        if random_node_n > 0:
            chosen = rng.choice(all_node_ids, size=min(random_node_n, node_count), replace=False)
            removed_nodes[[node_index[node_id] for node_id in chosen]] = True
        if connector_node_n > 0:
            remaining_pool = np.array([node_id for node_id in connector_dominant_pool if not removed_nodes[node_index[node_id]]], dtype=object)
            if len(remaining_pool):
                chosen = rng.choice(remaining_pool, size=min(connector_node_n, len(remaining_pool)), replace=False)
                removed_nodes[[node_index[node_id] for node_id in chosen]] = True
        keep_edges = np.ones(len(edge_array), dtype=bool)
        random_edge_n = int(round(float(params['random_edge_fraction']) * len(edge_array)))
        if random_edge_n > 0:
            edge_rng = np.random.default_rng(10000000 + pressure_seed * 10000 + repeat_id * 10 + 2)
            keep_edges[edge_rng.choice(len(edge_array), size=min(random_edge_n, len(edge_array)), replace=False)] = False
        edges = edge_array[keep_edges]
        if removed_nodes.any():
            edges = edges[~removed_nodes[edges[:, 0]] & ~removed_nodes[edges[:, 1]]]
        inter_fraction = float(params['intercommunity_edge_fraction'])
        if inter_fraction > 0 and len(edges) > 0:
            inter_indices = np.where(community_index[edges[:, 0]] != community_index[edges[:, 1]])[0]
            inter_n = int(round(inter_fraction * len(inter_indices)))
            if inter_n > 0:
                inter_rng = np.random.default_rng(10000000 + pressure_seed * 10000 + repeat_id * 10 + 5)
                chosen = inter_rng.choice(inter_indices, size=min(inter_n, len(inter_indices)), replace=False)
                keep = np.ones(len(edges), dtype=bool)
                keep[chosen] = False
                edges = edges[keep]
        return edges

    def support_and_pair_ids(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source_community = community_index[edges[:, 0]]
        target_community = community_index[edges[:, 1]]
        inter_mask = source_community != target_community
        pair_ids = np.full(len(edges), -1, dtype=np.int16)
        if inter_mask.any():
            a = np.minimum(source_community[inter_mask], target_community[inter_mask])
            b = np.maximum(source_community[inter_mask], target_community[inter_mask])
            pair_ids[inter_mask] = np.fromiter((pair_to_id[int(i), int(j)] for i, j in zip(a, b)), dtype=np.int16, count=int(inter_mask.sum()))
        counts = np.bincount(pair_ids[inter_mask], minlength=pair_count) if inter_mask.any() else np.zeros(pair_count, dtype=int)
        support = np.zeros((community_count, community_count), dtype=float)
        for pair_id, (i, j) in enumerate(community_pairs):
            support[i, j] = counts[pair_id]
            support[j, i] = counts[pair_id]
        return (support, pair_ids, inter_mask, counts)
    x_values = np.linspace(0.0, 1.0, args.x_points)
    all_pressure_results: list[pd.DataFrame] = []
    for pressure in pressure_order:
        pressure_rows: list[dict] = []
        node_scores = node_score_maps[pressure]
        sensitive_pool_fraction = float(target_pool_fraction[pressure])
        pressure_seed = sum((ord(ch) for ch in pressure))
        for repeat_id in range(1, args.repeats + 1):
            base_edges = pressure_background_network(pressure, repeat_id)
            base_support, pair_ids, inter_mask, base_pair_counts = support_and_pair_ids(base_edges)
            base_mean_flow = mean_flow(base_support, community_pairs)
            inter_indices = np.where(inter_mask)[0]
            if len(inter_indices) == 0:
                continue
            endpoint_sensitivity = node_scores[base_edges[:, 0]] + node_scores[base_edges[:, 1]]
            max_pair_count = max(1, int(base_pair_counts.max()))
            rarity = np.ones(len(base_edges), dtype=float)
            rarity[inter_mask] = np.array([max_pair_count / max(1, base_pair_counts[pair_id]) for pair_id in pair_ids[inter_mask]], dtype=float)
            damage_priority = endpoint_sensitivity * rarity
            pool_size = max(1, int(round(sensitive_pool_fraction * len(inter_indices))))
            pool_size = min(pool_size, len(inter_indices) // 2 if len(inter_indices) >= 2 else 1)
            tie_rng = np.random.default_rng(91000000 + pressure_seed * 1000 + repeat_id)
            ranked = np.argsort(damage_priority[inter_indices] + tie_rng.normal(0.0, 1e-06, size=len(inter_indices)))
            preserve_pool = inter_indices[ranked[:pool_size]]
            damage_pool = inter_indices[ranked[-pool_size:]][::-1]
            preserve_cumulative = cumulative_pair_removal_counts(pair_ids[preserve_pool], pair_count)
            damage_cumulative = cumulative_pair_removal_counts(pair_ids[damage_pool], pair_count)
            for state, cumulative, state_pool_size in [('Structure-preserved', preserve_cumulative, len(preserve_pool)), ('Connector structure damaged', damage_cumulative, len(damage_pool))]:
                for x in x_values:
                    removed_n = int(round(float(x) * state_pool_size))
                    removed_by_pair = cumulative[removed_n]
                    current_support = base_support.copy()
                    for pair_id, (i, j) in enumerate(community_pairs):
                        remaining = max(0.0, current_support[i, j] - removed_by_pair[pair_id])
                        current_support[i, j] = remaining
                        current_support[j, i] = remaining
                    direct_retention = 100.0 * np.mean([current_support[i, j] / base_support[i, j] for i, j in community_pairs if base_support[i, j] > 0])
                    alternative_retention = 100.0 * mean_flow(current_support, community_pairs) / base_mean_flow if base_mean_flow > 0 else 0.0
                    pressure_rows.append({'pressure': pressure, 'pressure_label': pressure_label[pressure], 'replicate': repeat_id, 'state': state, 'damage_intensity': float(x), 'pool_size': state_pool_size, 'target_frac_of_intercommunity_edges': sensitive_pool_fraction, 'edges_removed': removed_n, 'A_direct_intercommunity_support_retention_percent': direct_retention, 'B_alternative_intercommunity_path_retention_percent': alternative_retention})
        pressure_df = pd.DataFrame(pressure_rows)
        safe_name = pressure.replace('/', '_').replace(' ', '_')
        pressure_df.to_csv(reps_dir / f'{safe_name}.csv', index=False)
        all_pressure_results.append(pressure_df)
        print(f'Finished {pressure}: {len(pressure_df):,} rows')
    all_results = pd.concat(all_pressure_results, ignore_index=True)
    all_results.to_csv(output_dir / 'buffering_response_replicates_100_real_pressure_specific.csv', index=False)
    summary = all_results.groupby(['pressure', 'pressure_label', 'state', 'damage_intensity']).agg(A_mean=('A_direct_intercommunity_support_retention_percent', 'mean'), A_low95=('A_direct_intercommunity_support_retention_percent', lambda s: np.quantile(s, 0.025)), A_high95=('A_direct_intercommunity_support_retention_percent', lambda s: np.quantile(s, 0.975)), B_mean=('B_alternative_intercommunity_path_retention_percent', 'mean'), B_low95=('B_alternative_intercommunity_path_retention_percent', lambda s: np.quantile(s, 0.025)), B_high95=('B_alternative_intercommunity_path_retention_percent', lambda s: np.quantile(s, 0.975)), pool_size_mean=('pool_size', 'mean'), target_frac_mean=('target_frac_of_intercommunity_edges', 'mean')).reset_index()
    summary.to_csv(output_dir / 'buffering_response_summary_100_real_pressure_specific.csv', index=False)
    summary[np.isclose(summary['damage_intensity'], 1.0)].to_csv(output_dir / 'endpoint_summary_x1.0.csv', index=False)
    metadata = pd.DataFrame({'item': ['node_count', 'edge_count', 'community_count', 'repeats', 'x_points', 'damage_x1_definition'], 'value': [node_count, len(edge_array), community_count, args.repeats, args.x_points, '100% of the pressure-specific sensitive inter-community edge pool, not 100% of all network edges']})
    metadata.to_csv(output_dir / 'run_metadata.csv', index=False)
    print(f'All outputs written to: {output_dir}')
if __name__ == '__main__':
    main()
