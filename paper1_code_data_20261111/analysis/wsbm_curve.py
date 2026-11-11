from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'figures/supplementary/figure_s01/data/01_community_number_selection.csv'
OUT = ROOT / 'results/wsbm_curve'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260826
N_BOOT = 2000

def saturation(k, plateau, amplitude, rate):
    return plateau - amplitude * np.exp(-rate * (k - 5.0))

def fit_curve(k, y):
    p0 = (float(max(y)), float(max(y) - min(y)), 0.35)
    pars, _ = curve_fit(saturation, k, y, p0=p0, bounds=([min(y), 0.0, 1e-05], [max(y) * 1.1, max(y), 5.0]), maxfev=20000)
    return pars

def main():
    raw = pd.read_csv(SOURCE)
    raw = raw.loc[raw['status'].eq('OK'), ['k', 'logpr']].copy()
    groups = {int(k): g['logpr'].to_numpy(float) for k, g in raw.groupby('k')}
    if any((len(v) != 5 for v in groups.values())):
        raise ValueError('Expected five successful WSBM fits for every k.')
    k = np.array(sorted(groups), dtype=float)
    mean = np.array([groups[int(v)].mean() for v in k])
    x = np.linspace(k.min(), k.max(), 500)
    fitted = saturation(x, *fit_curve(k, mean))
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        yb = np.array([rng.choice(groups[int(v)], size=5, replace=True).mean() for v in k])
        try:
            boot.append(saturation(x, *fit_curve(k, yb)))
        except RuntimeError:
            continue
    boot = np.asarray(boot)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    pd.DataFrame({'k': x, 'fitted_logpr': fitted, 'bootstrap_95_low': lo, 'bootstrap_95_high': hi}).to_csv(OUT / 'saturation_curve.csv', index=False)
    pd.DataFrame({'k': k.astype(int), 'observed_mean_logpr': mean, 'fitted_logpr': saturation(k, *fit_curve(k, mean))}).to_csv(OUT / 'saturation_markers.csv', index=False)
if __name__ == '__main__':
    main()
