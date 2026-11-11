from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'damage_standard' / 'source_data'
OUT = ROOT / 'results/magnitude_profile'
SVG = OUT / 'SVG'
PREVIEW = OUT / 'preview'
SVG.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)
DIMS = list('EFCSVD')
CONDITIONS = ['Low temperature', 'Low C/N ratio', 'Low C/P ratio', 'High industrial proportion']
CONDITION_LABELS = ['Low temperature', 'Low C/N', 'Low C/P', 'Industrial wastewater']
MECHANISMS = ['bridge', 'assortative', 'core_periphery', 'abundance']
MECHANISM_LABELS = ['Bridge', 'Assortative', 'Core–periphery', 'Abundance']
EXPOSED_FLAGS = {CONDITIONS[0]: 'low_temp_flag', CONDITIONS[1]: 'low_CN_flag', CONDITIONS[2]: 'low_CP_flag', CONDITIONS[3]: 'industrial_flag'}
REFERENCE_FLAGS = {CONDITIONS[0]: 'latest_low_temp_reference_flag', CONDITIONS[1]: 'latest_low_CN_reference_flag', CONDITIONS[2]: 'latest_low_CP_reference_flag', CONDITIONS[3]: 'latest_industrial_reference_flag'}
mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
meta = pd.read_csv(ROOT / 'shared/data/04_Metadata/04C_plant_metadata_252.csv')
scores = pd.read_csv(ROOT / 'shared/data/08_PCA_source_data/08A_PCA_input_and_scores_252_plants.csv')
plants = meta.merge(scores, on=['Site_ID', 'strict_223_subset'])
groups = pd.read_csv(ROOT / 'shared/damage_standard/latest_sample_pressure_groups.csv')
effects = pd.read_csv(SOURCE / 'sixdim_r30/03_pilot_six_dimension_effect_vectors_by_replicate.csv')
zcols = [f'robust_z_{d}' for d in DIMS]

def hedges_g(a, b):
    n, m = (len(a), len(b))
    pooled = np.sqrt(((n - 1) * a.var(0, ddof=1) + (m - 1) * b.var(0, ddof=1)) / (n + m - 2))
    correction = 1 - 3 / (4 * (n + m) - 9)
    return correction * np.divide(a.mean(0) - b.mean(0), pooled, out=np.zeros(6), where=pooled > 1e-12)

def one_sample_dz(x):
    sd = x.std(0, ddof=1)
    correction = 1 - 3 / (4 * len(x) - 5)
    return correction * np.divide(x.mean(0), sd, out=np.zeros(6), where=sd > 1e-12)

def magnitude_profile_similarity(observed, simulated):
    observed = np.abs(observed)
    simulated = np.abs(simulated)
    denominator = np.linalg.norm(observed) * np.linalg.norm(simulated)
    return float(observed @ simulated / denominator) if denominator else np.nan
rng = np.random.default_rng(20260921)
rows = []
for condition in CONDITIONS:
    exposed_ids = set(groups.loc[groups[EXPOSED_FLAGS[condition]].eq(1), 'wwtp_id'])
    reference_ids = set(groups.loc[groups[REFERENCE_FLAGS[condition]].eq(1), 'wwtp_id'])
    exposed = plants[plants.Site_ID.isin(exposed_ids)][zcols].to_numpy()
    reference = plants[plants.Site_ID.isin(reference_ids)][zcols].to_numpy()
    observed = hedges_g(exposed, reference)
    for mechanism in MECHANISMS:
        simulated_replicates = effects[effects.pressure.eq(condition) & effects.mechanism.eq(mechanism)].pivot(index='replicate', columns='dimension', values='auc_effect').reindex(columns=DIMS).to_numpy()
        simulated = one_sample_dz(simulated_replicates)
        boot = np.empty(3000)
        for k in range(len(boot)):
            observed_boot = hedges_g(exposed[rng.integers(0, len(exposed), len(exposed))], reference[rng.integers(0, len(reference), len(reference))])
            simulated_boot = one_sample_dz(simulated_replicates[rng.integers(0, len(simulated_replicates), len(simulated_replicates))])
            boot[k] = magnitude_profile_similarity(observed_boot, simulated_boot)
        low, high = np.nanquantile(boot, [0.025, 0.975])
        rows.append([condition, mechanism, magnitude_profile_similarity(observed, simulated), low, high])
result = pd.DataFrame(rows, columns=['condition', 'mechanism', 'similarity', 'low95', 'high95'])
result.to_csv(OUT / 'Fig4c_magnitude_profile_similarity.csv', index=False)
matrix = result.pivot(index='condition', columns='mechanism', values='similarity').reindex(index=CONDITIONS, columns=MECHANISMS)
cmap = mpl.colors.LinearSegmentedColormap.from_list('profile_similarity', ['#F7F6F7', '#E8CED1', '#D89CA3', '#C96F78'])
fig, ax = plt.subplots(figsize=(5.043, 2.964))
image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(4), MECHANISM_LABELS)
ax.set_yticks(range(4), CONDITION_LABELS)
ax.tick_params(length=0)
for i, condition in enumerate(CONDITIONS):
    for j, mechanism in enumerate(MECHANISMS):
        row = result[result.condition.eq(condition) & result.mechanism.eq(mechanism)].iloc[0]
        ax.text(j, i, f'{row.similarity:.2f}\n[{row.low95:.2f}, {row.high95:.2f}]', ha='center', va='center', fontsize=6.1, color='white' if row.similarity >= 0.68 else '#222')
colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, ticks=np.linspace(0, 1, 6))
colorbar.set_label('Magnitude-profile similarity')
ax.set_xlabel('Simulated damage response')
ax.set_ylabel('Observed six-dimensional structural fingerprint')
fig.subplots_adjust(left=0.28, right=0.89, bottom=0.2, top=0.96)
stem = 'Main_Fig4c_magnitude_profile_similarity'
fig.savefig(SVG / f'{stem}.svg', bbox_inches=None)
fig.savefig(OUT / f'{stem}.pdf', bbox_inches=None)
fig.savefig(OUT / f'{stem}.tiff', dpi=600, bbox_inches=None)
fig.savefig(PREVIEW / f'{stem}.png', dpi=300, bbox_inches=None)
plt.close(fig)
print(result.to_string(index=False))
