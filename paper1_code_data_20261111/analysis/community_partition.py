import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
from cp_code import config as C
from cp_code.data import load_network
from cp_code.partition import build_embedding, partition_for_k, block_model_score
TAIL_MERGE_REQUESTED_K = int(getattr(C, 'TAIL_MERGE_REQUESTED_K', 10))
TAIL_MERGE_TARGET_EFFECTIVE_K = int(getattr(C, 'TAIL_MERGE_TARGET_EFFECTIVE_K', 8))
TAIL_MICRO_MAX_SIZE = int(getattr(C, 'TAIL_MICRO_MAX_SIZE', 3))
TAIL_RECEIVER_MIN_SIZE = int(getattr(C, 'TAIL_RECEIVER_MIN_SIZE', 5))
TAIL_RECEIVER_COUNT = int(getattr(C, 'TAIL_RECEIVER_COUNT', 2))
TAIL_FORCE_ONE_TO_EACH = bool(getattr(C, 'TAIL_FORCE_ONE_TO_EACH', True))
TAIL_BALANCE_WEIGHT = float(getattr(C, 'TAIL_BALANCE_WEIGHT', 0.1))

def _label_name(i):
    return f'C{int(i) + 1:02d}'

def _compact_relabel_by_size(labels):
    labels = np.asarray(labels, dtype=int).copy()
    uniq, cnt = np.unique(labels, return_counts=True)
    order = sorted(zip(uniq, cnt), key=lambda x: (-int(x[1]), int(x[0])))
    mapping = {old: new for new, (old, _) in enumerate(order)}
    out = np.array([mapping[x] for x in labels], dtype=int)
    map_df = pd.DataFrame([{'old_label': int(old), 'new_label': int(new), 'new_label_name': _label_name(new), 'size': int(cnt)} for new, (old, cnt) in enumerate(order)])
    return (out, map_df)

def _community_centroids(labels, X):
    labels = np.asarray(labels, dtype=int)
    X = np.asarray(X, dtype=float)
    centroids = {}
    for lab in sorted(np.unique(labels)):
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        c = np.nanmean(X[idx, :], axis=0)
        norm = np.linalg.norm(c)
        if np.isfinite(norm) and norm > 0:
            c = c / norm
        centroids[int(lab)] = c
    return centroids

def _centroid_distance(a, b):
    if a is None or b is None:
        return 1000000000.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    return float(np.linalg.norm(a - b))

def merge_k10_tail_micro_communities_to_8(labels_raw, X, seed=0):
    rng = np.random.default_rng(seed)
    labels0, size_map0 = _compact_relabel_by_size(labels_raw)
    sizes0 = pd.Series(labels0).value_counts().sort_index()
    current_k = int(len(sizes0))
    log_rows = []
    for lab, sz in sizes0.items():
        log_rows.append({'step': 'raw_after_size_relabel', 'old_label': _label_name(lab), 'old_size': int(sz), 'receiver_label': '', 'receiver_size_before': '', 'receiver_size_after': '', 'reason': 'raw requested-k partition, relabeled by descending size'})
    if current_k <= TAIL_MERGE_TARGET_EFFECTIVE_K or current_k != TAIL_MERGE_REQUESTED_K:
        labels_final, final_map = _compact_relabel_by_size(labels0)
        log_df = pd.DataFrame(log_rows)
        log_df['cleanup_mode'] = 'no_tail_merge_needed'
        return (labels_final, log_df)
    need_merge = current_k - TAIL_MERGE_TARGET_EFFECTIVE_K
    micro_candidates = [int(l) for l, s in sizes0.sort_values(ascending=True).items() if int(s) <= TAIL_MICRO_MAX_SIZE]
    if len(micro_candidates) < need_merge:
        for l, s in sizes0.sort_values(ascending=True).items():
            l = int(l)
            if l not in micro_candidates:
                micro_candidates.append(l)
            if len(micro_candidates) >= need_merge:
                break
    micro_labels = micro_candidates[:need_merge]
    receiver_candidates = [int(l) for l, s in sizes0.sort_values(ascending=True).items() if int(l) not in set(micro_labels) and int(s) >= TAIL_RECEIVER_MIN_SIZE]
    if len(receiver_candidates) < TAIL_RECEIVER_COUNT:
        receiver_candidates = [int(l) for l, s in sizes0.sort_values(ascending=True).items() if int(l) not in set(micro_labels)]
    receivers = receiver_candidates[:max(1, min(TAIL_RECEIVER_COUNT, len(receiver_candidates)))]
    labels_work = labels0.copy()
    centroids = _community_centroids(labels0, X)
    size_now = {int(k): int(v) for k, v in sizes0.items()}
    if len(receivers) == 0 or len(micro_labels) == 0:
        labels_final, final_map = _compact_relabel_by_size(labels_work)
        log_df = pd.DataFrame(log_rows)
        log_df['cleanup_mode'] = 'tail_merge_failed_no_receiver_or_micro'
        return (labels_final, log_df)
    assignments = []
    if TAIL_FORCE_ONE_TO_EACH and len(micro_labels) <= len(receivers):
        import itertools
        best = None
        for perm in itertools.permutations(receivers, len(micro_labels)):
            final_sizes = size_now.copy()
            total_dist = 0.0
            for m, r in zip(micro_labels, perm):
                total_dist += _centroid_distance(centroids.get(m), centroids.get(r))
                final_sizes[r] = final_sizes.get(r, 0) + size_now.get(m, 0)
            tail_sizes = [final_sizes.get(r, 0) for r in receivers]
            balance_penalty = float(np.std(tail_sizes)) / max(1.0, float(np.mean(tail_sizes)))
            score = total_dist + TAIL_BALANCE_WEIGHT * balance_penalty
            if best is None or score < best[0]:
                best = (score, list(zip(micro_labels, perm)), total_dist, balance_penalty)
        assignments = best[1] if best is not None else []
    else:
        for m in micro_labels:
            best_r = None
            best_score = None
            for r in receivers:
                dist = _centroid_distance(centroids.get(m), centroids.get(r))
                size_penalty = size_now.get(r, 0) / max(1.0, np.mean([size_now.get(x, 0) for x in receivers]))
                score = dist + TAIL_BALANCE_WEIGHT * size_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    best_r = r
            assignments.append((m, best_r))
            size_now[best_r] = size_now.get(best_r, 0) + size_now.get(m, 0)
    for m, r in assignments:
        m_size = int(np.sum(labels_work == m))
        r_before = int(np.sum(labels_work == r))
        labels_work[labels_work == m] = r
        r_after = int(np.sum(labels_work == r))
        log_rows.append({'step': 'tail_micro_merge', 'old_label': _label_name(m), 'old_size': m_size, 'receiver_label': _label_name(r), 'receiver_size_before': r_before, 'receiver_size_after': r_after, 'reason': f'micro community size <= {TAIL_MICRO_MAX_SIZE}; merged into one of tail receivers by embedding/connection-profile similarity; force_one_to_each={TAIL_FORCE_ONE_TO_EACH}'})
    labels_final, final_map = _compact_relabel_by_size(labels_work)
    final_sizes = pd.Series(labels_final).value_counts().sort_index()
    for lab, sz in final_sizes.items():
        log_rows.append({'step': 'final_after_tail_merge', 'old_label': '', 'old_size': '', 'receiver_label': _label_name(lab), 'receiver_size_before': '', 'receiver_size_after': int(sz), 'reason': 'final labels after compact relabeling by descending size'})
    log_df = pd.DataFrame(log_rows)
    log_df['cleanup_mode'] = f'k10_tail_micro_merge_to_effective_{len(final_sizes)}'
    log_df['target_effective_k'] = TAIL_MERGE_TARGET_EFFECTIVE_K
    log_df['micro_max_size'] = TAIL_MICRO_MAX_SIZE
    log_df['receiver_min_size'] = TAIL_RECEIVER_MIN_SIZE
    return (labels_final, log_df)

def main():
    for folder in [C.TAB_DIR, C.MID_DIR, C.FIG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    net = load_network()
    X = build_embedding(net)
    rows = []
    labels = []
    for repeat in range(1, 71):
        lab = partition_for_k(X, net, 10, C.RANDOM_SEED + 10000 + repeat)
        rows.append(dict(repeat=repeat, **block_model_score(lab, net)))
        labels.append(lab)
    scan = pd.DataFrame(rows)
    scan.to_csv(C.TAB_DIR / 'partition_70_repeats.csv', index=False)
    best = int(scan.representative_score.argmax())
    final, merge_log = merge_k10_tail_micro_communities_to_8(labels[best], X, C.RANDOM_SEED + 10)
    assignment = net['node_df'].copy()
    assignment['community_id_zero_based'] = final
    assignment['node_index'] = np.arange(len(final))
    assignment['community'] = [f'C{x + 1:02d}' for x in final]
    assignment.to_csv(C.TAB_DIR / 'node_communities.csv', index=False)
    assignment['structural_core']=False
    assignment['MRA']=pd.to_numeric(assignment.MRA,errors='coerce').fillna(0)
    for community,sub in assignment.groupby('community'):
        count=len(sub) if len(sub)<=25 else int(np.ceil(0.4*len(sub)))
        chosen=sub.sort_values(['v_degree','MRA','Id'],ascending=[False,False,True],kind='mergesort').head(count).index
        assignment.loc[chosen,'structural_core']=True
    assignment.to_csv(C.TAB_DIR/'community_structural_core_node_audit.csv',index=False)

    merge_log.to_csv(C.TAB_DIR / 'tail_merge_audit.csv', index=False)
    np.savez_compressed(C.MID_DIR / 'partition_repeats.npz', labels=np.stack(labels), embedding=X, final_labels=final)
    print(assignment.community.value_counts().sort_index().to_string())
if __name__ == '__main__':
    main()
