from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/observed_fingerprints'
OUT.mkdir(parents=True, exist_ok=True)
plant = pd.read_csv(ROOT / 'shared/data/04_Metadata/04C_plant_metadata_252.csv').merge(pd.read_csv(ROOT / 'results/plant_metrics/plant_pca_scores.csv'), on=['Site_ID', 'strict_223_subset'])
Z = ['robust_z_' + d for d in 'EFCSVD']
rng = np.random.default_rng(20260720)
coherent = plant.AtInfTN.notna() & plant.AtInfNH4N.notna() & plant.AtInfNH4N.ge(0) & plant.AtInfNH4N.le(plant.AtInfTN)
nitrogen = plant[coherent].copy()
nitrogen['Joint_N_rank'] = (nitrogen.AtInfTN.rank(pct=True) + nitrogen.AtInfNH4N.rank(pct=True)) / 2
pos = plant.loc[plant.Industrial_pct > 0, 'Industrial_pct']
industrial_cut = pos.quantile(0.8)
specs = [('Low temperature', plant, plant.MIT < 18, (plant.MIT >= 18) & (plant.MIT < 22)), ('High temperature', plant, plant.MIT >= 28, (plant.MIT >= 18) & (plant.MIT < 22)), ('High COD', plant, plant.AtInfCOD >= plant.AtInfCOD.quantile(0.8), plant.AtInfCOD < plant.AtInfCOD.quantile(0.8)), ('High NH4-N', nitrogen, nitrogen.AtInfNH4N >= nitrogen.AtInfNH4N.quantile(0.8), nitrogen.AtInfNH4N < nitrogen.AtInfNH4N.quantile(0.8)), ('High TN', nitrogen, nitrogen.Joint_N_rank >= nitrogen.Joint_N_rank.quantile(0.8), nitrogen.Joint_N_rank < nitrogen.Joint_N_rank.quantile(0.8)), ('High TP', plant, plant.AtInfTP >= plant.AtInfTP.quantile(0.8), plant.AtInfTP < plant.AtInfTP.quantile(0.8)), ('High industrial proportion', plant, plant.Industrial_pct >= industrial_cut, plant.Industrial_pct.eq(0)), ('Low C/N ratio', plant, plant.C_N_ratio <= plant.C_N_ratio.quantile(0.2), plant.C_N_ratio > plant.C_N_ratio.quantile(0.2)), ('Low C/P ratio', plant, plant.C_P_ratio <= plant.C_P_ratio.quantile(0.2), plant.C_P_ratio > plant.C_P_ratio.quantile(0.2))]
rows = []
for name, data, exposed, reference in specs:
    for dimension, column in zip('EFCSVD', Z):
        a = data.loc[exposed, column].dropna().to_numpy()
        b = data.loc[reference, column].dropna().to_numpy()
        boot = np.empty(3000)
        for k in range(3000):
            boot[k] = a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        rows.append([name, dimension, a.mean() - b.mean(), *np.quantile(boot, [0.025, 0.975]), len(a), len(b)])
pd.DataFrame(rows, columns=['pressure', 'code', 'difference', 'low95', 'high95', 'pressure_n', 'reference_n']).to_csv(OUT / 'all_observed_fingerprints.csv', index=False)
