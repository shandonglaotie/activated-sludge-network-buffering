from pathlib import Path
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / 'results/sample_metrics'
OUT = ROOT / 'results/plant_metrics'
OUT.mkdir(parents=True, exist_ok=True)

def robust_z(s):
    s = pd.to_numeric(s, errors='coerce')
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = s.std()
    return ((s - med) / iqr, med, iqr)

def make_site_table():
    d = pd.read_csv(PRE / '02_sample_level_analysis_base.csv', low_memory=False)
    net = ['Active_ASVs_456', 'Edge_realization_rate', 'Intercommunity_edge_fraction', 'Abundance_weighted_intercommunity_support', 'Community_abundance_evenness', 'Community_abundance_dominance', 'Largest_component_fraction', 'Fragmentation_1_minus_LCC', 'Connector_presence_fraction', 'Connector_relative_abundance']
    first = ['Lat', 'Lon', 'match_method', 'strict_223_subset', 'MIT', 'AtInfCOD', 'AtInfNH4N', 'AtInfTN', 'AtInfTP', 'DC', 'HRT', 'SRT', 'DO', 'Ph', 'Industry wastewater source Contained in Inf?', 'EffCOD', 'EffNH4N', 'EffTN', 'EffTP', 'EffCOD_KNN', 'EffNH4N_KNN', 'EffTN_KNN', 'EffTP_KNN']
    agg = {c: 'median' for c in net}
    agg.update({c: 'first' for c in first})
    s = d.groupby('Site_ID', as_index=False).agg(agg)
    lr = np.log1p(s.Active_ASVs_456.to_numpy(float))
    R = np.c_[np.ones(len(s)), lr, lr * lr]
    components = [('Edge_realization_rate', -1), ('Abundance_weighted_intercommunity_support', -1), ('Community_abundance_evenness', -1), ('Community_abundance_dominance', 1), ('Largest_component_fraction', -1), ('Connector_presence_fraction', -1)]
    risk_cols = []
    comp_rows = []
    for metric, direction in components:
        y = s[metric].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(lr)
        b = np.linalg.lstsq(R[ok], y[ok], rcond=None)[0]
        resid = y - R @ b
        z, med, iqr = robust_z(pd.Series(resid))
        rc = metric + '_richness_adjusted_risk_z'
        s[rc] = direction * z
        risk_cols.append(rc)
        comp_rows.append([metric, direction, float(b[0]), float(b[1]), float(b[2]), med, iqr])
    s['Network_Structural_Risk_Index'] = s[risk_cols].mean(axis=1)
    s['Network_Structural_Risk_Index_scaled_0_100'] = 100 * (s.Network_Structural_Risk_Index - s.Network_Structural_Risk_Index.min()) / (s.Network_Structural_Risk_Index.max() - s.Network_Structural_Risk_Index.min())
    Z = s[risk_cols].to_numpy(float)
    k = Z.shape[1]
    alpha = k / (k - 1) * (1 - np.nansum(np.nanvar(Z, axis=0, ddof=1)) / np.nanvar(np.nansum(Z, axis=1), ddof=1))
    temp_ext = (s.MIT - s.MIT.median()).abs().rank(pct=True)
    stress = pd.concat([temp_ext.rename('T')] + [s[c].rank(pct=True).rename(c) for c in ['AtInfCOD', 'AtInfNH4N', 'AtInfTN', 'AtInfTP']], axis=1).mean(axis=1)
    s['Composite_pressure_rank'] = stress
    ref = (s['Industry wastewater source Contained in Inf?'] == 'No') & (stress <= stress.median())
    dev = []
    for c in risk_cols:
        med = s.loc[ref, c].median()
        iqr = s.loc[ref, c].quantile(0.75) - s.loc[ref, c].quantile(0.25)
        if not np.isfinite(iqr) or iqr <= 0:
            iqr = s.loc[ref, c].std()
        dev.append((s[c] - med) / iqr)
    D = np.vstack([v.to_numpy(float) for v in dev]).T
    s['Network_Structural_Deviation_Index'] = np.sqrt(np.nanmean(D * D, axis=1))
    s['Reference_cohort_for_NSDI'] = ref
    comp = pd.DataFrame(comp_rows, columns=['component_metric', 'risk_direction', 'richness_intercept', 'richness_log_linear', 'richness_log_quadratic', 'residual_median', 'residual_IQR'])
    return (s, comp, risk_cols, float(alpha))
if __name__ == '__main__':
    site, parameters, risk, alpha = make_site_table()
    site.to_csv(OUT / 'plant_structural_metrics.csv', index=False)
    parameters.to_csv(OUT / 'richness_adjustment_parameters.csv', index=False)
    risk_order = ['Edge_realization_rate', 'Largest_component_fraction', 'Connector_presence_fraction', 'Abundance_weighted_intercommunity_support', 'Community_abundance_evenness', 'Community_abundance_dominance']
    X = site[[m + '_richness_adjusted_risk_z' for m in risk_order]]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    Z = (X - X.median()) / (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1)
    centered = Z - Z.mean()
    u, s, vt = np.linalg.svd(centered.to_numpy(), full_matrices=False)
    scores = centered.to_numpy() @ vt[:2].T
    output = site[['Site_ID', 'strict_223_subset']].copy()
    for i, d in enumerate('EFCSVD'):
        output['robust_z_' + d] = X.iloc[:, i]
    output['PC1_score'] = scores[:, 0]
    output['PC2_score'] = scores[:, 1]
    output.to_csv(OUT / 'plant_pca_scores.csv', index=False)
    pd.DataFrame({'component': np.arange(1, 7), 'explained_variance_percent': s * s / (s * s).sum() * 100}).to_csv(OUT / 'pca_variance.csv', index=False)
