from __future__ import annotations
import sys
sys.modules['bottleneck'] = None
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(__file__).resolve().parents[3] / 'shared' / 'community_abundance'
OUTPUT = ROOT / 'output'
OUTPUT.mkdir(parents=True, exist_ok=True)
ABUNDANCE_FILE = DATA_ROOT / '06_community_abundance.csv'
GROUP_FILE = DATA_ROOT / '05_sample_pressure_groups.csv'
PRESSURES = {'TN': ('high_TN_flag', 'TN'), 'NH4-N': ('high_NH4N_flag', 'NH$_4$-N'), 'TP': ('high_TP_flag', 'TP'), 'COD': ('high_COD_flag', 'COD'), 'Low C/N': ('low_CN_flag', 'Low C/N'), 'Low C/P': ('low_CP_flag', 'Low C/P'), 'Low temp.': ('low_temp_flag', 'Low temp.'), 'Industrial wastewater': ('industrial_flag', 'Industrial wastewater')}
MAIN = ['TN', 'NH4-N', 'TP', 'COD']
SUPPLEMENT = ['Low C/N', 'Low C/P', 'Low temp.', 'Industrial wastewater']
DEEP = '#64136B'
LIGHT = '#C7A5D5'
XGRID = np.linspace(0.0, 1.0, 21)
N_REPEAT = 100
SEED = 20260730
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.9, 'xtick.direction': 'in', 'ytick.direction': 'in', 'svg.fonttype': 'none', 'pdf.fonttype': 42})

def pielou(p: np.ndarray) -> float:
    positive = p[p > 0]
    if positive.size <= 1:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / np.log(len(p)))

def dominance_balance(p: np.ndarray) -> float:
    return float((1.0 - p.max()) / (1.0 - 1.0 / len(p)))

def normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    total = p.sum()
    return p / total if total > 0 else np.repeat(1.0 / len(p), len(p))

def pressure_profile(abundance: pd.DataFrame, groups: pd.DataFrame, pressure: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flag, _ = PRESSURES[pressure]
    wide = abundance.pivot(index='sample_id', columns='community_name', values='community_relative_abundance').reindex(columns=[f'C{i:02d}' for i in range(1, 9)])
    pressure_ids = groups.loc[groups[flag] == 1, 'sample_id']
    reference_ids = groups.loc[groups['baseline_reference_flag'] == 1, 'sample_id']
    pressure_matrix = wide.loc[wide.index.intersection(pressure_ids)].to_numpy(float)
    reference_mean = wide.loc[wide.index.intersection(reference_ids)].mean().to_numpy(float)
    pressure_mean = pressure_matrix.mean(axis=0)
    delta = pressure_mean - reference_mean
    return (pressure_matrix, pressure_mean, delta)

def donor_and_sink(delta: np.ndarray, p: np.ndarray, metric: str) -> tuple[np.ndarray, int]:
    sink = int(np.argmax(p))
    donors = np.asarray([i for i in np.argsort(-np.abs(delta)) if i != sink and abs(delta[i]) > 0], dtype=int)
    return (donors, sink)

def redistribute_targeted(p: np.ndarray, delta: np.ndarray, x: float, metric: str) -> np.ndarray:
    q = p.copy()
    donors, sink = donor_and_sink(delta, p, metric)
    if donors.size == 0:
        return q
    sensitivity = np.abs(delta[donors])
    sensitivity /= max(sensitivity.max(), 1e-12)
    removed = x * q[donors] * sensitivity
    q[donors] -= removed
    q[sink] += removed.sum()
    return normalize(q)

def redistribute_preserved(p: np.ndarray, delta: np.ndarray, x: float, metric: str, rng: np.random.Generator) -> np.ndarray:
    q = p.copy()
    target_donors, _ = donor_and_sink(delta, p, metric)
    k = max(1, len(target_donors))
    order = np.argsort(np.abs(delta))
    low_priority = [i for i in order if i != int(np.argmax(p))][:k]
    donors = np.asarray(low_priority, dtype=int)
    if donors.size == 0:
        return q
    sensitivity = np.abs(delta[donors])
    sensitivity /= max(np.abs(delta).max(), 1e-12)
    removed = x * q[donors] * sensitivity
    q[donors] -= removed
    mass = float(removed.sum())
    sink = int(np.argmax(q))
    q[sink] += mass
    return normalize(q)

def simulate() -> pd.DataFrame:
    abundance = pd.read_csv(ABUNDANCE_FILE)
    groups = pd.read_csv(GROUP_FILE)
    rows: list[dict] = []
    for p_idx, pressure in enumerate(PRESSURES):
        matrix, _, delta = pressure_profile(abundance, groups, pressure)
        for repeat in range(N_REPEAT):
            rng = np.random.default_rng(SEED + p_idx * 10000 + repeat)
            sample_idx = rng.integers(0, len(matrix), size=len(matrix))
            p0 = normalize(matrix[sample_idx].mean(axis=0))
            baseline = {'V': pielou(p0), 'D': dominance_balance(p0)}
            for metric in ('V', 'D'):
                for x in XGRID:
                    for state in ('preserved', 'damaged'):
                        if state == 'preserved':
                            q = redistribute_preserved(p0, delta, float(x), metric, rng)
                        else:
                            q = redistribute_targeted(p0, delta, float(x), metric)
                        value = pielou(q) if metric == 'V' else dominance_balance(q)
                        retention = 100.0 * value / baseline[metric] if baseline[metric] > 0 else 0.0
                        rows.append({'pressure': pressure, 'repeat': repeat + 1, 'metric': metric, 'state': state, 'intensity': float(x), 'retention': float(np.clip(retention, 0.0, 100.0))})
    return pd.DataFrame(rows)

def plot_grid(summary: pd.DataFrame, pressures: list[str], suffix: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.8), squeeze=False)
    ylabels = ('Community abundance\nevenness retention (%)', 'Dominance-balance\nretention (%)')
    for row, metric in enumerate(('V', 'D')):
        for col, pressure in enumerate(pressures):
            ax = axes[row, col]
            dat = summary[(summary.pressure == pressure) & (summary.metric == metric)]
            for state, color, label in (('preserved', DEEP, 'Abundance organization preserved'), ('damaged', LIGHT, 'Pressure-sensitive abundance organization damaged')):
                line = dat[dat.state == state].sort_values('intensity')
                ax.plot(line['intensity'], line['retention'], color=color, marker='o', markersize=2.7, linewidth=1.8, label=label)
            if row == 0:
                strip = mpl.patches.Rectangle((0.0, 1.015), 1.0, 0.085, transform=ax.transAxes, facecolor='#D9D9D9', edgecolor='black', linewidth=1.0, clip_on=False)
                ax.add_patch(strip)
                ax.text(0.5, 1.057, PRESSURES[pressure][1], transform=ax.transAxes, ha='center', va='center', fontsize=10.5)
            if col == 0:
                ax.set_ylabel(ylabels[row], fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(0, 1, 6)])
            observed_min = float(dat['retention'].min())
            ymin = max(0.0, np.floor((observed_min - 8.0) / 10.0) * 10.0)
            ax.set_ylim(ymin, 100.0)
            ax.set_yticks(np.arange(ymin, 100.1, 10.0))
            ax.set_xlabel('Community-abundance redistribution intensity', fontsize=9.2)
            ax.grid(False)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.tick_params(width=0.8, length=3)
    fig.text(0.013, 0.965, 'a', fontsize=13, weight='bold')
    fig.text(0.013, 0.49, 'b', fontsize=13, weight='bold')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=10, handlelength=2.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.13, wspace=0.16, hspace=0.31)
    prefix = OUTPUT / f'Figure_abundance_organization_damage_{suffix}'
    fig.savefig(prefix.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.png'), dpi=600, bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.tiff'), dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

def main() -> None:
    raw = simulate()
    raw.to_csv(OUTPUT / 'abundance_organization_damage_replicates.csv', index=False)
    summary = raw.groupby(['pressure', 'metric', 'state', 'intensity'], as_index=False)['retention'].mean().rename(columns={'retention': 'mean_retention'})
    summary.to_csv(OUTPUT / 'abundance_organization_damage_summary.csv', index=False)
    plot_grid(summary.rename(columns={'mean_retention': 'retention'}), MAIN, 'main')
    plot_grid(summary.rename(columns={'mean_retention': 'retention'}), SUPPLEMENT, 'supplement')
    print(OUTPUT)
if __name__ == '__main__':
    main()
