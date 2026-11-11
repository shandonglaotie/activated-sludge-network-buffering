from __future__ import annotations
import sys
sys.modules['bottleneck'] = None
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output_cp_uniform_random_fixed_order_20260825'
SUMMARY = OUTPUT / 'three_mechanism_new_summary.csv'
TARGET = 'Pressure-sensitive core-periphery structure damaged'
CONTROL = 'Matched random node removal'
DARK = '#641764'
LIGHT = '#C8A8D4'
PRESSURE_LABEL = {'High TN': 'High TN', 'High NH4-N': 'High NH$_4$-N', 'High TP': 'High TP', 'High COD': 'High COD', 'Low C/N ratio': 'Low C/N ratio', 'Low C/P ratio': 'Low C/P ratio', 'Low temperature': 'Low temperature', 'High industrial proportion': 'Industrial wastewater'}
MAIN_CONDITIONS = ['Low C/N ratio', 'Low C/P ratio', 'Low temperature', 'High industrial proportion']
HIGH_LOAD_CONDITIONS = ['High TN', 'High NH4-N', 'High TP', 'High COD']

def configure() -> None:
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out', 'font.size': 8, 'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False})

def plot_one_row(summary: pd.DataFrame, pressures: list[str], filename: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.45), sharex=True, sharey=True)
    for ax, pressure in zip(axes, pressures):
        pdat = summary[(summary['mechanism'] == 'core_periphery') & (summary['pressure'] == pressure)]
        for state, color in [(CONTROL, DARK), (TARGET, LIGHT)]:
            dat = pdat[pdat['state'] == state].sort_values('damage_intensity')
            ax.plot(dat['damage_intensity'], dat['metric1_mean'], color=color, marker='o', markersize=2.2, linewidth=1.45, label=state)
        ax.set_title(PRESSURE_LABEL[pressure], fontsize=8.5, pad=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlabel('Core-periphery node damage intensity', fontsize=7.5)
        ax.tick_params(width=0.7, length=2.5, labelsize=7)
    axes[0].set_ylabel('Core-periphery direct\nsupport retention (%)', fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.025), ncol=2, fontsize=7.5, handlelength=2.4)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.28, wspace=0.16)
    stem = OUTPUT / Path(filename).stem
    fig.savefig(stem.with_suffix('.svg'), format='svg', bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    configure()
    summary = pd.read_csv(SUMMARY)
    plot_one_row(summary, MAIN_CONDITIONS, 'Fig4e_core_periphery_direct_support_rerun.svg')
    plot_one_row(summary, HIGH_LOAD_CONDITIONS, 'FigS6_core_periphery_direct_support_rerun.svg')
if __name__ == '__main__':
    main()
