from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/partition_stability'
OUT.mkdir(parents=True, exist_ok=True)
a = pd.read_csv(ROOT / 'results/partition/tables/node_communities.csv').sort_values('node_index')
reference = a.community_id_zero_based.to_numpy(int)
rng = np.random.default_rng(20260716)
values = []
for _ in range(1000):
    idx = rng.choice(len(reference), int(round(0.5 * len(reference))), replace=False)
    perturbed = reference.copy()
    perturbed[idx] = rng.permutation(reference[idx])
    values.append(normalized_mutual_info_score(reference, perturbed))
pd.DataFrame({'label_perturbation_null_NMI': values}).to_csv(OUT / 'label_perturbation.csv', index=False)
labels = np.load(ROOT / 'results/partition/intermediate/partition_repeats.npz')['labels']
rows = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        rows.append([i + 1, j + 1, normalized_mutual_info_score(labels[i], labels[j])])
pd.DataFrame(rows, columns=['repeat_1', 'repeat_2', 'NMI']).to_csv(OUT / 'partition_pairwise_NMI.csv', index=False)
