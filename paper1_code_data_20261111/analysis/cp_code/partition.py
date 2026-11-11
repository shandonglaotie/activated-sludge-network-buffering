import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from . import config as C
from .utils import relabel_by_size, relabel_contiguous, gini, entropy_norm, pairwise_nmi, save_csv, safe_print

def _cluster_model(n_clusters, seed, n_init=None, max_iter=500):
    if n_init is None:
        n_init = C.KMEANS_N_INIT
    if getattr(C, 'USE_MINIBATCH_KMEANS', True):
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=max(3, n_init), max_iter=max_iter, batch_size=getattr(C, 'MINIBATCH_SIZE', 1024), reassignment_ratio=0.01)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init, max_iter=max_iter, algorithm='lloyd')

def _sym_norm(A):
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv[mask] = 1.0 / np.sqrt(deg[mask])
    D = sp.diags(inv)
    return (D @ A @ D, deg)

def build_embedding(net, save_svd=True, tag='main'):
    n = net['n_nodes']
    src, dst, w = (net['src'], net['dst'], net['w_signed'])
    rows = np.r_[src, dst]
    cols = np.r_[dst, src]
    data = np.r_[w, w]
    Apos = sp.csr_matrix((np.where(data > 0, data, 0.0), (rows, cols)), shape=(n, n))
    Aneg = sp.csr_matrix((np.where(data < 0, -data, 0.0), (rows, cols)), shape=(n, n))
    Aabs = sp.csr_matrix((np.abs(data), (rows, cols)), shape=(n, n))
    Apos_n, deg_pos = _sym_norm(Apos)
    Aneg_n, deg_neg = _sym_norm(Aneg)
    Aabs_n, deg_abs = _sym_norm(Aabs)
    Xsp = sp.hstack([Apos_n, Aneg_n, Aabs_n], format='csr')
    nc = min(C.EMBED_DIMS, max(2, Xsp.shape[1] - 1), max(2, n - 2))
    svd = TruncatedSVD(n_components=nc, random_state=C.RANDOM_SEED, algorithm='randomized', n_iter=2, n_oversamples=5)
    X0 = svd.fit_transform(Xsp)
    deg_all = deg_pos + deg_neg
    degree_feats = np.c_[np.log1p(deg_abs), np.log1p(deg_pos), np.log1p(deg_neg), deg_pos / np.maximum(1e-09, deg_all), deg_neg / np.maximum(1e-09, deg_all)]
    X = np.c_[X0, degree_feats]
    X = StandardScaler().fit_transform(X)
    if save_svd:
        save_csv(pd.DataFrame({'component': np.arange(1, nc + 1), 'explained_variance_ratio': svd.explained_variance_ratio_}), C.MID_DIR / f'01_svd_explained_variance_{tag}.csv')
    return X

def connection_profiles(labels, net):
    labels = relabel_contiguous(labels)
    n = net['n_nodes']
    k = labels.max() + 1
    src, dst, w = (net['src'], net['dst'], net['w_signed'])
    pos = np.zeros((n, k))
    neg = np.zeros((n, k))
    cnt = np.zeros((n, k))
    ab = np.zeros((n, k))
    for u, v, ww in zip(src, dst, w):
        cu, cv = (labels[u], labels[v])
        a = abs(float(ww))
        if ww >= 0:
            pos[u, cv] += a
            pos[v, cu] += a
        else:
            neg[u, cv] += a
            neg[v, cu] += a
        ab[u, cv] += a
        ab[v, cu] += a
        cnt[u, cv] += 1
        cnt[v, cu] += 1
    sizes = np.bincount(labels, minlength=k).astype(float)
    norm = np.maximum(1.0, np.sqrt(sizes))[None, :]
    norm_size = np.maximum(1.0, sizes)[None, :]
    prof = np.c_[pos / norm, neg / norm, ab / norm, cnt / norm_size, pos[np.arange(n), labels], neg[np.arange(n), labels], ab[np.arange(n), labels]]
    return StandardScaler().fit_transform(prof)

def partition_for_k(X, net, k, seed):
    km = _cluster_model(k, seed, n_init=C.KMEANS_N_INIT, max_iter=300)
    labels = relabel_by_size(km.fit_predict(X))
    for rr in range(C.PROFILE_REFINE_ROUNDS):
        prof = connection_profiles(labels, net)
        Z = StandardScaler().fit_transform(np.c_[X, (1.1 + 0.2 * rr) * prof])
        labels_new = _cluster_model(k, seed + 100 + rr, n_init=max(3, C.KMEANS_N_INIT // 2), max_iter=300).fit_predict(Z)
        labels_new = relabel_by_size(labels_new)
        if np.array_equal(labels, labels_new):
            break
        labels = labels_new
    return relabel_by_size(labels)

def balance_metrics(labels):
    labels = relabel_contiguous(labels)
    sizes = np.bincount(labels)
    n = sizes.sum()
    k = len(sizes)
    second = np.sort(sizes)[-2] if k > 1 else 0
    return {'k_effective': int(k), 'size_max': int(sizes.max()), 'size_second': int(second), 'size_min': int(sizes.min()), 'size_mean': float(np.mean(sizes)), 'size_median': float(np.median(sizes)), 'max_share': float(sizes.max() / n), 'max_over_second': float(sizes.max() / max(1, second)) if k > 1 else np.nan, 'gini_size': gini(sizes), 'entropy_norm': entropy_norm(labels), 'n_tiny_blocks': int(np.sum(sizes < C.REP_TINY_BLOCK_SIZE))}

def block_model_score(labels, net):
    labels = relabel_contiguous(labels)
    src, dst, w = (net['src'], net['dst'], net['w_signed'])
    k = labels.max() + 1
    m = len(w)
    a, b = (labels[src], labels[dst])
    lo, hi = (np.minimum(a, b), np.maximum(a, b))
    bid = lo * k + hi
    minlength = k * k
    edge_count = np.bincount(bid, minlength=minlength).astype(float)
    sum_w = np.bincount(bid, weights=w, minlength=minlength).astype(float)
    mean_w = np.zeros(minlength)
    has = edge_count > 0
    mean_w[has] = sum_w[has] / edge_count[has]
    pred = mean_w[bid]
    sse = float(np.sum((w - pred) ** 2))
    sigma2 = max(sse / max(1, m), 1e-12)
    ll_weights = -0.5 * m * (np.log(2 * np.pi * sigma2) + 1.0)
    sizes = np.bincount(labels, minlength=k).astype(float)
    possible = np.zeros(minlength)
    for i in range(k):
        for j in range(i, k):
            possible[i * k + j] = sizes[i] * (sizes[i] - 1) / 2 if i == j else sizes[i] * sizes[j]
    valid = possible > 0
    M = edge_count[valid]
    T = possible[valid]
    p = np.clip((M + 0.5) / (T + 1.0), 1e-12, 1 - 1e-12)
    ll_edges = float(np.sum(M * np.log(p) + (T - M) * np.log(1 - p)))
    score_logpr = float(ll_edges + ll_weights)
    bm = balance_metrics(labels)
    giant_pen = C.REP_SCORE_GIANT_WEIGHT * max(0.0, bm['max_share'] - C.REP_MAX_SHARE_SOFT_TARGET) / C.REP_MAX_SHARE_SOFT_TARGET
    ratio_pen = C.REP_SCORE_RATIO_WEIGHT * max(0.0, bm['max_over_second'] - C.REP_MAX_RATIO_SOFT_TARGET) / C.REP_MAX_RATIO_SOFT_TARGET
    tiny_pen = C.REP_SCORE_TINY_WEIGHT * bm['n_tiny_blocks']
    representative_score = score_logpr - abs(score_logpr) * 0.01 * (giant_pen + ratio_pen + tiny_pen)
    return {'score_logpr': score_logpr, 'representative_score': float(representative_score), 'll_edges': float(ll_edges), 'll_weights': float(ll_weights), 'sigma2_weight': float(sigma2), **bm}
