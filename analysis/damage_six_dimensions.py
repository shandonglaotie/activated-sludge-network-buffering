from __future__ import annotations
import sys
sys.modules['bottleneck'] = None
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'shared'
DATA = PKG / 'data'
SIM_INPUT = PKG / 'six_dimension_inputs'
OUT = HERE.parent / 'results/damage_standard/source_data/sixdim_r30'
PRESSURES = {'Low C/N ratio': ('low_CN', 'low_CN_flag'), 'Low C/P ratio': ('low_CP', 'low_CP_flag'), 'Low temperature': ('low_temp', 'low_temp_flag'), 'High industrial proportion': ('industrial', 'industrial_flag')}
MECHANISMS = ('bridge', 'assortative', 'core_periphery', 'abundance')
DIMS = ('E', 'F', 'C', 'S', 'V', 'D')

def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--x-points', type=int, default=21)
    p.add_argument('--bridge-no-global', action='store_true')
    p.add_argument('--expanded-target-pool', action='store_true', help='Sensitivity analysis: double bridge/node target-pool sizes; abundance redistribution is unchanged.')
    return p.parse_args()

def norm(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, float), 0.0, None)
    s = v.sum()
    return v / s if s > 0 else np.zeros_like(v)

def robust01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    lo, hi = (np.nanmin(v), np.nanmax(v))
    return np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)

def lcc_fraction(n: int, edges: np.ndarray) -> float:
    if n <= 0:
        return 0.0
    if len(edges) == 0:
        return 1.0 / n
    graph = sparse.csr_matrix((np.ones(len(edges) * 2, np.int8), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n))
    _, lab = connected_components(graph, directed=False)
    return float(np.bincount(lab).max() / n)

def evenness_and_balance(q: np.ndarray, community: np.ndarray, n_comm: int) -> tuple[float, float]:
    ca = np.bincount(community, weights=q, minlength=n_comm)
    ca = norm(ca)
    pos = ca[ca > 0]
    sh = -float(np.sum(pos * np.log(pos))) if len(pos) else 0.0
    even = float(np.exp(sh) / n_comm)
    dominance = float(ca.max()) if len(ca) else 1.0
    balance = float((1.0 - dominance) / (1.0 - 1.0 / n_comm))
    return (even, max(0.0, balance))

def raw_metrics(q: np.ndarray, active: np.ndarray, edges: np.ndarray, weights: np.ndarray, community: np.ndarray, connector: np.ndarray, n_full_edges: int, n_connectors: int, n_comm: int) -> dict[str, float]:
    keep = active[edges[:, 0]] & active[edges[:, 1]] if len(edges) else np.array([], bool)
    ee = edges[keep]
    ww = weights[keep]
    n_active = int(active.sum())
    e_rate = len(ee) / max(n_full_edges, 1)
    if n_active:
        idx = np.flatnonzero(active)
        remap = np.full(len(active), -1, int)
        remap[idx] = np.arange(len(idx))
        mapped = np.column_stack((remap[ee[:, 0]], remap[ee[:, 1]])).astype(int) if len(ee) else np.empty((0, 2), int)
        lcc = lcc_fraction(n_active, mapped)
    else:
        lcc = 0.0
    cfrac = float(np.sum(active & connector) / max(n_connectors, 1))
    if len(ee):
        support = np.sqrt(q[ee[:, 0]] * q[ee[:, 1]]) * ww
        inter = community[ee[:, 0]] != community[ee[:, 1]]
        total = float(support.sum())
        sfrac = float(support[inter].sum() / total) if total > 0 else 0.0
    else:
        sfrac = 0.0
    even, balance = evenness_and_balance(q, community, n_comm)
    return {'E': e_rate, 'F': lcc, 'C': cfrac, 'S': sfrac, 'V': even, 'D': balance}

def loss(cur: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    out = {}
    for d in DIMS:
        b = base[d]
        out[d] = 100.0 * (1.0 - cur[d] / b) if b > 1e-12 else 0.0
    return out

def matched_pool(target: pd.DataFrame, eligible: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    picks: list[str] = []
    used: set[str] = set()
    for stratum, block in target.groupby('matched_control_stratum', dropna=False):
        cand = eligible[eligible['matched_control_stratum'].eq(stratum)]
        cand = cand[~cand.edge_id.isin(used)]
        if len(cand) < len(block):
            cand = eligible[~eligible.edge_id.isin(used)]
        if len(cand) == 0:
            continue
        chosen = rng.choice(cand.edge_id.to_numpy(), size=min(len(block), len(cand)), replace=False)
        picks.extend(chosen.tolist())
        used.update(chosen.tolist())
    if len(picks) < len(target):
        cand = eligible[~eligible.edge_id.isin(used)]
        if len(cand):
            chosen = rng.choice(cand.edge_id.to_numpy(), size=min(len(target) - len(picks), len(cand)), replace=False)
            picks.extend(chosen.tolist())
    return np.asarray(picks, object)

def matched_realized_edge_pool(target: pd.DataFrame, eligible: pd.DataFrame, rng: np.random.Generator, allow_global: bool=True) -> tuple[np.ndarray, pd.DataFrame]:
    picks: list[str] = []
    audit: list[dict] = []
    used: set[str] = set()
    tiers = [('exact', ['community_pair', 'edge_weight_bin', 'degree_sum_bin']), ('pair_weight', ['community_pair', 'edge_weight_bin']), ('pair', ['community_pair'])]
    if allow_global:
        tiers.append(('global', []))
    for row in target.itertuples(index=False):
        chosen = None
        tier_used = 'unmatched'
        for tier, cols in tiers:
            cand = eligible[~eligible.edge_id.isin(used)]
            for col in cols:
                cand = cand[cand[col].eq(getattr(row, col))]
            if len(cand):
                chosen = str(rng.choice(cand.edge_id.to_numpy(object)))
                tier_used = tier
                break
        if chosen is not None:
            picks.append(chosen)
            used.add(chosen)
        audit.append({'target_id': row.edge_id, 'control_id': chosen, 'match_tier': tier_used})
    return (np.asarray(picks, object), pd.DataFrame(audit))

def quantile_bins(v: np.ndarray, mask: np.ndarray, n_bins: int=4) -> np.ndarray:
    out = np.full(len(v), -1, int)
    idx = np.flatnonzero(mask)
    if len(idx):
        ranks = pd.Series(v[idx]).rank(method='average', pct=True).to_numpy()
        out[idx] = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    return out

def matched_node_pool(target: np.ndarray, candidates: np.ndarray, community: np.ndarray, internal_degree_bin: np.ndarray, abundance_bin: np.ndarray, role_class: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    available = set(np.setdiff1d(candidates, target).tolist())
    picks: list[int] = []
    audit: list[dict] = []
    tiers = [('exact', (community, internal_degree_bin, abundance_bin, role_class)), ('community_degree_abundance', (community, internal_degree_bin, abundance_bin)), ('community_degree', (community, internal_degree_bin)), ('community', (community,)), ('global', tuple())]
    for t in target:
        chosen = None
        tier_used = 'unmatched'
        for tier, arrays in tiers:
            cand = np.asarray(sorted(available), int)
            for arr in arrays:
                cand = cand[arr[cand] == arr[t]]
            if len(cand):
                chosen = int(rng.choice(cand))
                tier_used = tier
                break
        if chosen is not None:
            picks.append(chosen)
            available.remove(chosen)
        audit.append({'target_node_index': int(t), 'control_node_index': chosen, 'match_tier': tier_used})
    return (np.asarray(picks, int), pd.DataFrame(audit))

def common_support_assortative_pairs(candidates: np.ndarray, priority: np.ndarray, community: np.ndarray, internal_degree: np.ndarray, q0: np.ndarray, provincial: np.ndarray, is_core: np.ndarray, rng: np.random.Generator, requested_k: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ranked = candidates[np.argsort(priority[candidates])[::-1]]
    n_top = max(requested_k, int(np.ceil(0.5 * len(ranked))))
    target_candidates = ranked[:n_top]
    control_candidates = ranked[n_top:]
    ideg_bin = quantile_bins(np.log1p(internal_degree), np.isin(np.arange(len(priority)), candidates))
    abundance_bin = quantile_bins(np.log10(q0 + 1e-12), np.isin(np.arange(len(priority)), candidates))
    role = provincial.astype(int) + 2 * is_core.astype(int)
    ideg_z = robust01(np.log1p(internal_degree))
    abundance_z = robust01(np.log10(q0 + 1e-12))
    available = set(control_candidates.tolist())
    targets: list[int] = []
    controls: list[int] = []
    audit: list[dict] = []
    for t in target_candidates:
        cand = np.asarray(sorted(available), int)
        if len(cand):
            cand = cand[community[cand] == community[t]]
        if len(cand):
            distance = np.abs(ideg_z[cand] - ideg_z[t]) + np.abs(abundance_z[cand] - abundance_z[t]) + 0.25 * (role[cand] != role[t]).astype(float)
            best = cand[np.flatnonzero(np.isclose(distance, distance.min()))]
            c = int(rng.choice(best))
            targets.append(int(t))
            controls.append(c)
            available.remove(c)
            audit.append({'target_node_index': int(t), 'control_node_index': c, 'match_tier': 'common_support_nearest', 'distance': float(distance[cand == c][0])})
        else:
            audit.append({'target_node_index': int(t), 'control_node_index': None, 'match_tier': 'trimmed_outside_common_support', 'distance': None})
        if len(targets) >= requested_k:
            break
    return (np.asarray(targets, int), np.asarray(controls, int), pd.DataFrame(audit))

def redistribute_community(ca, delta, x, targeted):
    ca = norm(ca)
    sink = int(np.argmax(ca))
    ranked = np.asarray([i for i in np.argsort(-np.abs(delta)) if i != sink and abs(delta[i]) > 0], int)
    k = max(1, min(3, len(ranked) // 2))
    donors = ranked[:k] if targeted else ranked[-k:]
    sensitivity = np.abs(delta[donors]) / max(float(np.abs(delta).max()), 1e-12)
    q = ca.copy()
    moved = x * q[donors] * sensitivity
    q[donors] -= moved
    q[sink] += moved.sum()
    return norm(q)

def scale_asvs_to_communities(q: np.ndarray, community: np.ndarray, desired: np.ndarray) -> np.ndarray:
    current = np.bincount(community, weights=q, minlength=len(desired))
    out = q.copy()
    for c in range(len(desired)):
        members = community == c
        if current[c] > 0:
            out[members] *= desired[c] / current[c]
    return norm(out)

def main() -> None:
    global OUT
    cfg = args()
    if cfg.bridge_no_global:
        OUT = HERE / 'unified_six_dimension_r30_v4_bridge_no_global_sensitivity'
    if cfg.expanded_target_pool:
        OUT = HERE / 'unified_six_dimension_r10_v5_expanded_target_pool'
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    nodes = pd.read_csv(DATA / '05_Network' / '05A_network_node_attributes_456.csv')
    edges = pd.read_csv(DATA / '05_Network' / '05B_positive_weighted_edge_list_13875.csv')
    rel = pd.read_csv(DATA / '03_Processed_ASV_abundance' / '03B_ASV_relative_abundance_matrix_456_by_1186_samples.csv', index_col=0)
    groups = pd.read_csv(HERE.parent / 'shared/damage_standard/latest_sample_pressure_groups.csv')
    relevance = pd.read_csv(SIM_INPUT / '03_pressure_node_relevance.csv')
    edge_pool = pd.read_csv(SIM_INPUT / '04_pressure_target_edge_pool.csv')
    nodes = nodes.sort_values('node_id').reset_index(drop=True)
    node_ix = {v: i for i, v in enumerate(nodes.node_id)}
    asv_ix = {v: i for i, v in enumerate(nodes.asv_id)}
    rel = rel.loc[nodes.asv_id]
    common_samples = [c for c in rel.columns if c in set(groups.sample_id)]
    rel = rel[common_samples]
    groups = groups.set_index('sample_id').loc[common_samples].reset_index()
    earr = np.asarray([(node_ix[a], node_ix[b]) for a, b in edges[['source_node', 'target_node']].itertuples(index=False, name=None)], np.int32)
    ew = edges.abs_weight.to_numpy(float)
    edge_id_to_pos = {v: i for i, v in enumerate(edges.edge_id)}
    communities = sorted(nodes.community_name.unique())
    comm_map = {v: i for i, v in enumerate(communities)}
    comm = nodes.community_name.map(comm_map).to_numpy(int)
    connector = nodes.is_connector.astype(int).eq(1).to_numpy()
    provincial = nodes.is_provincial.astype(int).eq(1).to_numpy()
    is_core = nodes.is_core.astype(int).eq(1).to_numpy()
    zi = nodes.zi.fillna(0).to_numpy(float)
    degree = nodes.degree.fillna(0).to_numpy(float)
    internal_degree = np.zeros(len(nodes), float)
    same = comm[earr[:, 0]] == comm[earr[:, 1]]
    np.add.at(internal_degree, earr[same, 0], 1)
    np.add.at(internal_degree, earr[same, 1], 1)
    xgrid = np.linspace(0, 1, cfg.x_points)
    rows: list[dict] = []
    zero_notes: list[dict] = []
    match_audit_rows: list[dict] = []
    for p_idx, (pressure, (pcode, flag)) in enumerate(PRESSURES.items()):
        sample_ids = groups.loc[groups[flag].eq(1), 'sample_id'].tolist()
        ref_map = {'Low temperature': 'latest_low_temp_reference_flag', 'Low C/N ratio': 'latest_low_CN_reference_flag', 'Low C/P ratio': 'latest_low_CP_reference_flag', 'High industrial proportion': 'latest_industrial_reference_flag'}
        ref_ids = groups.loc[groups[ref_map[pressure]].eq(1), 'sample_id'].tolist()
        pmatrix = rel[sample_ids].to_numpy(float)
        pmean = norm(pmatrix.mean(axis=1))
        rmean = norm(rel[ref_ids].to_numpy(float).mean(axis=1))
        base_ca_mean = np.bincount(comm, weights=pmean, minlength=len(communities))
        ref_ca_mean = np.bincount(comm, weights=rmean, minlength=len(communities))
        delta_ca = base_ca_mean - ref_ca_mean
        pr = relevance[relevance.pressure.eq(pcode)].set_index('node_id')
        pscore = nodes.node_id.map(pr.sensitivity_score).fillna(0).to_numpy(float)
        pscore = robust01(pscore)
        ep = edge_pool[edge_pool.pressure.eq(pcode) & edge_pool.dimension.eq('S')].copy()
        ep = ep[ep.intercommunity_flag.astype(int).eq(1)]
        target_edges = ep[ep.target_flag.astype(int).eq(1)].sort_values('priority_score', ascending=False)
        eligible_edges = ep[ep.control_eligible_flag.astype(int).eq(1) & ~ep.edge_id.isin(target_edges.edge_id)]
        if False and cfg.expanded_target_pool:
            n_add = min(len(target_edges), max(0, (len(eligible_edges) - len(target_edges)) // 2))
            added = eligible_edges.nlargest(n_add, 'priority_score')
            target_edges = pd.concat([target_edges, added], ignore_index=True).sort_values('priority_score', ascending=False)
            eligible_edges = eligible_edges[~eligible_edges.edge_id.isin(added.edge_id)]
        for repeat in range(1, cfg.repeats + 1):
            rng = np.random.default_rng(2026082800 + p_idx * 10000 + repeat)
            draw = rng.integers(0, pmatrix.shape[1], size=pmatrix.shape[1])
            q0 = norm(pmatrix[:, draw].mean(axis=1))
            active0 = q0 > 0
            base = raw_metrics(q0, active0, earr, ew, comm, connector, len(earr), int(connector.sum()), len(communities))
            realized_ids = set(edges.loc[active0[earr[:, 0]] & active0[earr[:, 1]], 'edge_id'])
            bt_frame = target_edges[target_edges.edge_id.isin(realized_ids)].copy()
            be_frame = eligible_edges[eligible_edges.edge_id.isin(realized_ids)].copy()
            bc, baudit = matched_realized_edge_pool(bt_frame, be_frame, rng, allow_global=not cfg.bridge_no_global)
            matched_target_ids = baudit.loc[baudit.control_id.notna(), 'target_id'].to_numpy(object)
            if not cfg.bridge_no_global and len(bc) != len(bt_frame):
                raise RuntimeError(f'Incomplete bridge matching for {pressure}, repeat {repeat}')
            bt = matched_target_ids
            baudit.insert(0, 'pressure', pressure)
            baudit.insert(1, 'replicate', repeat)
            baudit.insert(2, 'mechanism', 'bridge')
            match_audit_rows.extend(baudit.to_dict('records'))
            for state, order in (('targeted', bt), ('matched_control', bc)):
                positions = np.asarray([edge_id_to_pos[e] for e in order if e in edge_id_to_pos], int)
                for x in xgrid:
                    n_drop = int(round(x * len(positions)))
                    keep = np.ones(len(earr), bool)
                    keep[positions[:n_drop]] = False
                    cur = raw_metrics(q0, active0, earr[keep], ew[keep], comm, connector, len(earr), int(connector.sum()), len(communities))
                    ls = loss(cur, base)
                    ls['V'] = 0.0
                    ls['D'] = 0.0
                    for d in DIMS:
                        rows.append({'pressure': pressure, 'mechanism': 'bridge', 'replicate': repeat, 'state': state, 'intensity': x, 'dimension': d, 'loss_percent': ls[d], 'value_origin': 'zero_by_design' if d in ('V', 'D') else 'calculated'})
            mechanisms = {'assortative': 1.8 * pscore * (0.35 + 0.65 * robust01(internal_degree)) + 1.0 * provincial + 0.7 * robust01(np.maximum(zi, 0)) + 0.5 * robust01(internal_degree), 'core_periphery': 1.8 * pscore + 1.5 * is_core + 0.9 * robust01(degree)}
            for mech, priority in mechanisms.items():
                candidates = np.flatnonzero(active0 & (internal_degree > 0 if mech == 'assortative' else np.ones(len(nodes), bool)))
                frac = (0.24 if mech == 'assortative' else 0.2) if cfg.expanded_target_pool else 0.12 if mech == 'assortative' else 0.1
                k = max(3, min(int(round(frac * len(candidates))), max(1, len(candidates) // 2)))
                jitter = rng.normal(0, 1e-09, len(candidates))
                if mech == 'assortative':
                    candidates = candidates[~connector[candidates]]
                    k = max(3, min(int(round(frac * len(candidates))), max(1, len(candidates) // 2)))
                    target, control, naudit = common_support_assortative_pairs(candidates, priority, comm, internal_degree, q0, provincial, is_core, rng, k)
                    if len(target) < 3 or len(control) != len(target):
                        raise RuntimeError(f'Insufficient assortative common support for {pressure}, repeat {repeat}')
                    naudit.insert(0, 'pressure', pressure)
                    naudit.insert(1, 'replicate', repeat)
                    naudit.insert(2, 'mechanism', 'assortative')
                    match_audit_rows.extend(naudit.to_dict('records'))
                else:
                    target = candidates[np.argsort(priority[candidates] + jitter)][-k:][::-1]
                    remaining = np.setdiff1d(candidates, target)
                    control = rng.choice(remaining, size=min(k, len(remaining)), replace=False)
                for state, order in (('targeted', target), ('matched_control', control)):
                    for x in xgrid:
                        deleted = order[:int(round(x * len(order)))]
                        active = active0.copy()
                        active[deleted] = False
                        q = q0.copy()
                        q[deleted] = 0.0
                        q = norm(q)
                        cur = raw_metrics(q, active, earr, ew, comm, connector, len(earr), int(connector.sum()), len(communities))
                        ls = loss(cur, base)
                        for d in DIMS:
                            rows.append({'pressure': pressure, 'mechanism': mech, 'replicate': repeat, 'state': state, 'intensity': x, 'dimension': d, 'loss_percent': ls[d], 'value_origin': 'calculated'})
            ca0 = np.bincount(comm, weights=q0, minlength=len(communities))
            abundance_targeted_active_x1 = None
            for state, targeted in (('targeted', True), ('matched_control', False)):
                for x in xgrid:
                    desired = redistribute_community(ca0, delta_ca, float(x), targeted)
                    q = scale_asvs_to_communities(q0, comm, desired)
                    active = q > 0
                    cur = raw_metrics(q, active, earr, ew, comm, connector, len(earr), int(connector.sum()), len(communities))
                    ls = loss(cur, base)
                    for d in DIMS:
                        rows.append({'pressure': pressure, 'mechanism': 'abundance', 'replicate': repeat, 'state': state, 'intensity': x, 'dimension': d, 'loss_percent': ls[d], 'value_origin': 'calculated'})
                    if state == 'targeted' and np.isclose(x, 1.0):
                        abundance_targeted_active_x1 = int(active.sum())
            zero_notes.append({'pressure': pressure, 'replicate': repeat, 'abundance_active_asvs_x0': int(active0.sum()), 'abundance_active_asvs_x1_targeted': abundance_targeted_active_x1})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / '01_pilot_replicate_curves.csv', index=False)
    pd.DataFrame(zero_notes).to_csv(OUT / '02_activity_threshold_audit.csv', index=False)
    pd.DataFrame(match_audit_rows).to_csv(OUT / '02B_matching_audit.csv', index=False)
    effects = []
    for keys, block in raw.groupby(['pressure', 'mechanism', 'replicate', 'dimension']):
        p, m, r, d = keys
        wide = block.pivot(index='intensity', columns='state', values='loss_percent').sort_index()
        diff = wide.targeted - wide.matched_control
        effects.append({'pressure': p, 'mechanism': m, 'replicate': r, 'dimension': d, 'auc_effect': float(np.trapezoid(diff.to_numpy(), wide.index.to_numpy())), 'effect_x50': float(diff.iloc[np.argmin(np.abs(wide.index.to_numpy() - 0.5))]), 'effect_x100': float(diff.iloc[-1]), 'negative_auc': bool(np.trapezoid(diff.to_numpy(), wide.index.to_numpy()) < 0), 'near_zero_auc': bool(abs(np.trapezoid(diff.to_numpy(), wide.index.to_numpy())) < 1e-09)})
    eff = pd.DataFrame(effects)
    eff.to_csv(OUT / '03_pilot_six_dimension_effect_vectors_by_replicate.csv', index=False)
    summary = eff.groupby(['pressure', 'mechanism', 'dimension'], as_index=False).agg(auc_mean=('auc_effect', 'mean'), auc_sd=('auc_effect', 'std'), x50_mean=('effect_x50', 'mean'), x100_mean=('effect_x100', 'mean'), negative_replicates=('negative_auc', 'sum'), zero_replicates=('near_zero_auc', 'sum'), n=('replicate', 'count'))
    summary.to_csv(OUT / '04_pilot_effect_summary.csv', index=False)
    manifest = {'status': 'simulation_complete_stop_after_requested_repeats', 'repeats': cfg.repeats, 'x_points': cfg.x_points, 'pressures': list(PRESSURES), 'mechanisms': list(MECHANISMS), 'dimensions': list(DIMS), 'bridge_V_D_policy': 'numeric zero, explicitly tagged zero_by_design', 'bridge_matching': 'repeat-specific realized edges; community-pair x weight-bin x degree-sum-bin with audited fixed relaxation tiers', 'bridge_global_fallback_allowed': not cfg.bridge_no_global, 'assortative_matching': 'common-support trimming; same community hard constraint; continuous nearest match on internal degree and baseline abundance plus fixed 0.25 broad-role penalty; without replacement; no cross-community/global fallback', 'assortative_priority': 'pressure sensitivity jointly weighted by within-community structural contribution; no pressure-specific post-hoc tuning', 'node_abundance_policy': 'deleted node abundance set to zero; remaining abundance renormalized', 'abundance_activity_threshold': '>0 after community-level redistribution and ASV proportional scaling', 'primary_effect': 'trapezoidal AUC of targeted loss minus matched-control loss over x=0..1', 'expanded_target_pool': bool(cfg.expanded_target_pool), 'sensitivity': ['x=0.5', 'x=1.0'], 'elapsed_seconds': time.perf_counter() - start, 'raw_rows': len(raw), 'effect_rows': len(eff)}
    (OUT / '00_run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    main()
