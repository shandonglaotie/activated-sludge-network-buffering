import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from . import config as C
from .utils import read_csv_smart, safe_col, save_csv, safe_print

def _load_keystones():
    if not C.KEYSTONE_FILE.exists():
        return set()
    df = read_csv_smart(C.KEYSTONE_FILE)
    for c in ['Label', 'ASV_ID', 'Unnamed: 0']:
        if c in df.columns:
            return set(df[c].astype(str))
    return set(df.iloc[:, 0].astype(str))

def load_network():
    safe_print('Loading the RMT network')
    node0 = read_csv_smart(C.NODE_FILE)
    edge0 = read_csv_smart(C.EDGE_FILE)
    cols = {}
    cols['node_id_col'] = safe_col(node0, ['Id', 'id', 'ID'], desc='node_id')
    cols['asv_col'] = safe_col(node0, ['Label', 'label', 'ASV_ID', 'v_name'], desc='ASV_id')
    cols['phylum_col'] = safe_col(node0, ['v_phylum', 'Phylum', 'phylum'], required=False, desc='phylum')
    cols['function_col'] = safe_col(node0, ['v_function', 'Function', 'function'], required=False, desc='function')
    cols['mra_col'] = safe_col(node0, ['MRA', 'mra'], required=False, desc='MRA')
    source_col = safe_col(edge0, ['Source', 'source'], desc='edge_source')
    target_col = safe_col(edge0, ['Target', 'target'], desc='edge_target')
    if 'e_correlation' in edge0.columns:
        corr_col = 'e_correlation'
    elif len(edge0.columns) >= 8:
        corr_col = edge0.columns[7]
    else:
        raise KeyError('e_correlation  8 /H 。')
    cols.update({'source_col': source_col, 'target_col': target_col, 'corr_col': corr_col})
    node_ids0 = node0[cols['node_id_col']].astype(str).tolist()
    asv_ids0 = node0[cols['asv_col']].astype(str).tolist()
    id_to_old = {x: i for i, x in enumerate(node_ids0)}
    n0 = len(node_ids0)
    e = edge0[[source_col, target_col, corr_col]].copy()
    e[source_col] = e[source_col].astype(str)
    e[target_col] = e[target_col].astype(str)
    e[corr_col] = pd.to_numeric(e[corr_col], errors='coerce')
    e = e.dropna(subset=[corr_col])
    e = e[e[source_col].isin(id_to_old) & e[target_col].isin(id_to_old)].copy()
    e['i'] = e[source_col].map(id_to_old).astype(int)
    e['j'] = e[target_col].map(id_to_old).astype(int)
    e = e[e['i'] != e['j']].copy()
    e['u'] = np.minimum(e['i'], e['j'])
    e['v'] = np.maximum(e['i'], e['j'])
    e = e.groupby(['u', 'v'], as_index=False).agg(w=(corr_col, 'mean'), raw_n=(corr_col, 'size'))
    e = e[e['w'] != 0].copy()
    strong_mask = (e['w'] >= C.STRUCTURE_POS_CORR_CUTOFF) | (e['w'] < 0) & (np.abs(e['w']) >= C.STRUCTURE_NEG_ABS_CUTOFF)
    e_strong0 = e[strong_mask].copy().reset_index(drop=True)
    src0 = e_strong0['u'].to_numpy(dtype=int)
    dst0 = e_strong0['v'].to_numpy(dtype=int)
    w0 = e_strong0['w'].to_numpy(dtype=float)
    deg = np.bincount(np.r_[src0, dst0], minlength=n0).astype(int)
    strength = np.bincount(np.r_[src0, dst0], weights=np.r_[np.abs(w0), np.abs(w0)], minlength=n0)
    posdeg = np.bincount(np.r_[src0[w0 > 0], dst0[w0 > 0]], minlength=n0).astype(int)
    negdeg = np.bincount(np.r_[src0[w0 < 0], dst0[w0 < 0]], minlength=n0).astype(int)
    keystones = _load_keystones()
    is_key = np.array([a in keystones for a in asv_ids0], dtype=bool)
    q = float(np.quantile(strength[strength > 0], C.STRENGTH_KEEP_QUANTILE)) if np.any(strength > 0) else 0.0
    keep = (deg >= C.MIN_STRONG_DEGREE_KEEP) & (strength >= q)
    if C.KEEP_KEYSTONE_IF_CONNECTED:
        keep = keep | is_key & (deg >= 1)
    active0 = np.where(keep)[0]
    active_set = set(active0.tolist())
    e_tmp = e_strong0[e_strong0['u'].isin(active_set) & e_strong0['v'].isin(active_set)].copy()
    if len(e_tmp) > 0:
        old_to_tmp = {old: i for i, old in enumerate(active0)}
        rows = e_tmp['u'].map(old_to_tmp).to_numpy(dtype=int)
        cols2 = e_tmp['v'].map(old_to_tmp).to_numpy(dtype=int)
        A = sp.csr_matrix((np.ones(len(e_tmp) * 2), (np.r_[rows, cols2], np.r_[cols2, rows])), shape=(len(active0), len(active0)))
        n_comp, comp_labels = connected_components(A, directed=False)
        comp_sizes = np.bincount(comp_labels)
        comp_keep = comp_sizes[comp_labels] >= C.MIN_STRONG_COMPONENT_SIZE
        active0 = active0[comp_keep]
        keep = np.zeros(n0, dtype=bool)
        keep[active0] = True
    else:
        keep = np.zeros(n0, dtype=bool)
    filter_table = pd.DataFrame({'old_node_index': np.arange(n0), 'ASV_ID': asv_ids0, 'strong_degree': deg, 'strong_pos_degree': posdeg, 'strong_neg_degree': negdeg, 'strong_abs_strength': strength, 'is_original_keystone': is_key, 'keep_for_partition': keep})
    reasons = []
    for i in range(n0):
        if keep[i]:
            reasons.append('kept_structural_node')
        elif deg[i] == 0:
            reasons.append('isolated_in_strong_network_removed')
        elif deg[i] < C.MIN_STRONG_DEGREE_KEEP:
            reasons.append('near_isolated_or_weak_degree_removed')
        elif strength[i] < q:
            reasons.append('low_strength_background_removed')
        else:
            reasons.append('small_component_or_noise_removed')
    filter_table['filter_reason'] = reasons
    save_csv(filter_table, C.TAB_DIR / '00_node_filter_audit.csv')
    kept_old = np.where(keep)[0]
    kept_set = set(kept_old.tolist())
    e_strong = e_strong0[e_strong0['u'].isin(kept_set) & e_strong0['v'].isin(kept_set)].copy()
    active_old = sorted(set(e_strong['u'].astype(int)).union(set(e_strong['v'].astype(int))))
    old_to_new = {old: i for i, old in enumerate(active_old)}
    node = node0.iloc[active_old].copy().reset_index(drop=True)
    asv_ids = node[cols['asv_col']].astype(str).tolist()
    e_strong = e_strong[e_strong['u'].isin(old_to_new) & e_strong['v'].isin(old_to_new)].copy()
    e_strong['u_new'] = e_strong['u'].map(old_to_new).astype(int)
    e_strong['v_new'] = e_strong['v'].map(old_to_new).astype(int)
    e_part = e_strong[['u_new', 'v_new', 'w', 'raw_n']].rename(columns={'u_new': 'u', 'v_new': 'v'}).reset_index(drop=True)
    src = e_part['u'].to_numpy(dtype=int)
    dst = e_part['v'].to_numpy(dtype=int)
    w = e_part['w'].to_numpy(dtype=float)
    n = len(node)
    diag = pd.DataFrame({'item': ['raw_nodes', 'raw_edges', 'strong_edges_before_node_filter', 'partition_nodes', 'partition_edges', 'removed_nodes', 'positive_corr_cutoff', 'negative_abs_cutoff', 'min_strong_degree_keep', 'min_component_size', 'strength_keep_quantile', 'run_tag'], 'value': [n0, len(e), len(e_strong0), n, len(e_part), n0 - n, C.STRUCTURE_POS_CORR_CUTOFF, C.STRUCTURE_NEG_ABS_CUTOFF, C.MIN_STRONG_DEGREE_KEEP, C.MIN_STRONG_COMPONENT_SIZE, C.STRENGTH_KEEP_QUANTILE, C.RUN_TAG]})
    save_csv(diag, C.TAB_DIR / '00_network_filter_summary.csv')
    safe_print(f'Raw nodes: {n0}; raw edges: {len(e)}')
    safe_print(f'Positive-edge cutoff: {C.STRUCTURE_POS_CORR_CUTOFF}; filtered nodes: {n}; filtered edges: {len(e_part)}')
    return {'node_df': node, 'edge_clean': e_part, 'src': src, 'dst': dst, 'w_signed': w, 'n_nodes': n, 'asv_ids': asv_ids, 'idx_to_asv': {i: asv_ids[i] for i in range(n)}, 'keystone_asvs': keystones, 'cols': cols}
