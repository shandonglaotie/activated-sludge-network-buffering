import numpy as np
import pandas as pd

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except OSError:
        pass

def read_csv_smart(path):
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']
    last = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise RuntimeError(f'：{path}：{last}')

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')

def safe_col(df, candidates, required=True, desc=''):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'：{desc}； ={candidates}； ={list(df.columns)}')
    return None

def relabel_contiguous(labels):
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(labels)
    mp = {old: new for new, old in enumerate(uniq)}
    return np.array([mp[x] for x in labels], dtype=int)

def relabel_by_size(labels):
    labels = np.asarray(labels, dtype=int)
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]
    mp = {old: new for new, old in enumerate(order)}
    return np.array([mp[x] for x in labels], dtype=int)

def gini(values):
    x = np.asarray(values, dtype=float)
    x = x[x >= 0]
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x)) - (n + 1) / n)

def entropy_norm(labels):
    labels = np.asarray(labels, dtype=int)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p + 1e-12))
    return float(h / np.log(len(counts))) if len(counts) > 1 else 0.0

def pairwise_nmi(labels_list):
    from sklearn.metrics import normalized_mutual_info_score
    vals = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            vals.append(normalized_mutual_info_score(labels_list[i], labels_list[j]))
    if not vals:
        return (np.nan, np.nan, [])
    return (float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), vals)

def community_names(labels):
    return np.array([f'C{x + 1:02d}' for x in labels], dtype=object)

def rounded_ymax(v):
    v = float(v)
    if v <= 100:
        base = 10
    elif v <= 1000:
        base = 50
    elif v <= 3000:
        base = 250
    else:
        base = 500
    return int(base * np.ceil(v * 1.14 / base))

def make_palette(names):
    import matplotlib.pyplot as plt
    base = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
    return {name: base[i % len(base)] for i, name in enumerate(names)}

def phylum_palette(names):
    fixed = {'Proteobacteria': '#1f77b4', 'Bacteroidota': '#aec7e8', 'Bacteroidetes': '#aec7e8', 'Others': '#ff9896', 'Myxococcota': '#ff7f0e', 'Chloroflexi': '#2ca02c', 'Planctomycetota': '#b2df8a', 'Acidobacteriota': '#d62728', 'Patescibacteria': '#fb9a99', 'Verrucomicrobiota': '#9467bd', 'Actinobacteriota': '#c5b0d5', 'Bdellovibrionota': '#8c564b', 'Desulfobacterota': '#c49c94', 'Unknown': '#bdbdbd'}
    pal = make_palette(names)
    for n in names:
        if n in fixed:
            pal[n] = fixed[n]
    return pal
