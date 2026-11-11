from __future__ import annotations
import sys
sys.modules['bottleneck'] = None
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / 'shared/analysis_inputs'
OUT = ROOT / 'results/sample_metrics'
OUT.mkdir(exist_ok=True)

def mode_or_nan(s: pd.Series):
    q = s.dropna()
    if q.empty:
        return np.nan
    return q.astype(str).mode().iloc[0]

def parse_gexf():
    root = ET.parse(INP / 'triangle456.gexf').getroot()
    attr = {}
    for block in root.iter():
        if block.tag.endswith('attributes') and block.attrib.get('class') == 'node':
            for a in list(block):
                attr[a.attrib['id']] = a.attrib.get('title', a.attrib['id'])
    nodes = []
    for n in root.iter():
        if not n.tag.endswith('node'):
            continue
        d = {'Id': n.attrib['id']}
        for v in n.iter():
            if v.tag.endswith('attvalue'):
                d[attr.get(v.attrib['for'], v.attrib['for'])] = v.attrib.get('value')
        nodes.append(d)
    edges = []
    for e in root.iter():
        if e.tag.endswith('edge'):
            edges.append((e.attrib['source'], e.attrib['target'], float(e.attrib.get('weight', 1.0))))
    return (pd.DataFrame(nodes), edges)

def haversine_matrix(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0088
    la1 = np.radians(np.asarray(a_lat))[:, None]
    lo1 = np.radians(np.asarray(a_lon))[:, None]
    la2 = np.radians(np.asarray(b_lat))[None, :]
    lo2 = np.radians(np.asarray(b_lon))[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def match_sites(meta, water):
    sites = meta.groupby(['Lat', 'Lon'], as_index=False).agg(Site_ID=('Sample_ID', lambda s: str(s.iloc[0])[:-1]), Sample_count=('Sample_ID', 'size'))
    valid = water[water.longitude.notna()].copy()
    D = haversine_matrix(sites.Lat, sites.Lon, valid.latitude, valid.longitude)
    nearest = np.argmin(D, axis=1)
    d1 = D[np.arange(len(sites)), nearest]
    reverse = np.argmin(D, axis=0)
    reciprocal = np.array([reverse[j] == i for i, j in enumerate(nearest)])
    matched = {}
    records = []
    valid_indices = valid.index.to_numpy()
    for i in range(len(sites)):
        if reciprocal[i] and d1[i] <= 0.01:
            xr = int(valid_indices[nearest[i]])
            matched[i] = xr
            records.append([i, xr, 'coordinate_bidirectional', d1[i], True])
    for xr in water.index[water.longitude.isna()]:
        delta = np.abs(sites.Lat.to_numpy() - float(water.loc[xr, 'latitude']))
        order = np.argsort(delta)
        i = int(order[0])
        unique = len(order) == 1 or delta[order[1]] - delta[order[0]] > 0.0001
        if delta[i] <= 0.0001 and unique and (i not in matched):
            matched[i] = int(xr)
            records.append([i, int(xr), 'latitude_unique_rescue', np.nan, False])
    audit = pd.DataFrame(records, columns=['site_index', 'water_row', 'match_method', 'distance_km', 'strict_223_subset'])
    assert len(matched) == 252 and len(set(matched.values())) == 252
    site_rows = []
    for i, s in sites.iterrows():
        w = water.loc[matched[i]].to_dict()
        r = s.to_dict()
        r.update({'WWTP_Lat': s.Lat, 'WWTP_Lon': s.Lon, 'water_row': matched[i]})
        r.update(w)
        ar = audit[audit.site_index == i].iloc[0]
        r.update({'match_method': ar.match_method, 'match_distance_km': ar.distance_km, 'strict_223_subset': bool(ar.strict_223_subset)})
        site_rows.append(r)
    return (pd.DataFrame(site_rows), sites, audit)

def aggregate_meta(meta):
    cat = {'Nitri', 'Denitri', 'IndConInf'}
    aggs = {}
    for c in meta.columns:
        if c in {'Sample_ID', 'Lat', 'Lon'}:
            continue
        aggs[c] = mode_or_nan if c in cat or meta[c].dtype == object else 'median'
    return meta.groupby(['Lat', 'Lon'], as_index=False).agg(aggs)

def gower_distance(train, query, continuous, categorical):
    pieces = []
    for c in continuous:
        tr = pd.to_numeric(train[c], errors='coerce').to_numpy(float)
        q = float(query[c]) if pd.notna(query[c]) else np.nan
        if not np.isfinite(q):
            continue
        med = np.nanmedian(tr)
        q75, q25 = np.nanpercentile(tr, [75, 25])
        scale = q75 - q25
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.nanstd(tr)
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        diff = np.abs(tr - q) / scale
        diff[~np.isfinite(tr)] = np.nan
        pieces.append(np.clip(diff, 0, 4) / 4)
    for c in categorical:
        tr = train[c].astype(object).to_numpy()
        q = query[c]
        if pd.isna(q):
            continue
        d = (tr.astype(str) != str(q)).astype(float)
        d[pd.isna(tr)] = np.nan
        pieces.append(d)
    if not pieces:
        return np.full(len(train), np.inf)
    P = np.vstack(pieces)
    valid = np.isfinite(P)
    denom = valid.sum(axis=0)
    out = np.nansum(P, axis=0) / np.maximum(denom, 1)
    out[denom < max(3, int(0.25 * len(pieces)))] = np.inf
    return out

def weighted_knn_predict(train, query, target, k, continuous, categorical):
    donor = train[train[target].notna()].copy()
    if donor.empty:
        return (np.nan, np.nan, np.nan, 0)
    d = gower_distance(donor, query, continuous, categorical)
    keep = np.isfinite(d)
    donor = donor.loc[keep]
    d = d[keep]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan, 0)
    o = np.argsort(d)[:min(k, len(d))]
    vals = donor[target].to_numpy(float)[o]
    ds = d[o]
    if np.any(ds < 1e-12):
        pred = float(np.mean(vals[ds < 1e-12]))
    else:
        wt = 1 / (ds + 1e-06)
        pred = float(np.sum(wt * vals) / np.sum(wt))
    return (pred, float(np.min(vals)), float(np.max(vals)), len(vals))

def knn_cv_and_impute(site, targets, continuous, categorical):
    ks = [3, 5, 7, 9, 11, 15]
    cv_rows, choices, imp_rows = ([], {}, [])
    for target in targets:
        obs_idx = site.index[site[target].notna()].to_numpy()
        y = site.loc[obs_idx, target].to_numpy(float)
        baseline = float(np.median(y))
        for k in ks:
            pred = []
            for idx in obs_idx:
                train = site.drop(index=idx)
                p, _, _, _ = weighted_knn_predict(train, site.loc[idx], target, k, continuous, categorical)
                pred.append(p)
            pred = np.asarray(pred)
            ok = np.isfinite(pred)
            yy, pp = (y[ok], pred[ok])
            mae = float(np.mean(np.abs(yy - pp)))
            rmse = float(np.sqrt(np.mean((yy - pp) ** 2)))
            medae = float(np.median(np.abs(yy - pp)))
            denom = np.sum((yy - np.mean(yy)) ** 2)
            r2 = float(1 - np.sum((yy - pp) ** 2) / denom) if denom > 0 else np.nan
            rho = float(pd.Series(yy).rank().corr(pd.Series(pp).rank()))
            iqr = float(np.percentile(yy, 75) - np.percentile(yy, 25))
            cv_rows.append([target, k, len(yy), mae, rmse, medae, r2, rho, rmse / iqr if iqr else np.nan, float(np.mean(np.abs(yy - baseline)))])
        block = pd.DataFrame([r for r in cv_rows if r[0] == target], columns=['target', 'k', 'n', 'MAE', 'RMSE', 'Median_AE', 'R2', 'Spearman', 'NRMSE_IQR', 'Median_baseline_MAE'])
        best = int(block.sort_values(['RMSE', 'MAE']).iloc[0].k)
        choices[target] = best
        for idx in site.index[site[target].isna()]:
            p, lo, hi, n = weighted_knn_predict(site, site.loc[idx], target, best, continuous, categorical)
            imp_rows.append([idx, target, best, p, lo, hi, n])
    cv = pd.DataFrame(cv_rows, columns=['target', 'k', 'n', 'MAE', 'RMSE', 'Median_AE', 'R2', 'Spearman', 'NRMSE_IQR', 'Median_baseline_MAE'])
    imp = pd.DataFrame(imp_rows, columns=['site_row', 'target', 'selected_k', 'KNN_estimate', 'neighbor_min', 'neighbor_max', 'neighbors_used'])
    return (cv, choices, imp)

def dsu_lcc(n, edges):
    if n == 0:
        return (0, 0)
    parent = np.arange(n)
    size = np.ones(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = (find(a), find(b))
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = (rb, ra)
            parent[rb] = ra
            size[ra] += size[rb]
    roots = [find(i) for i in range(n)]
    comp = len(set(roots))
    return (int(max((size[find(i)] for i in range(n)))), comp)

def build_network_metrics(meta, node_df, edge_list):
    node_map = pd.read_csv(INP / 'nodes.csv', usecols=['Label', 'Id']).drop_duplicates('Id')
    node_df = node_df.merge(node_map, on='Id', how='left')
    assert node_df.Label.notna().all()
    community = dict(zip(node_df.Label, node_df.community_raw))
    asvs = node_df.Label.tolist()
    matrix = pd.read_csv(INP / 'asv_filtered.csv', index_col=0)
    missing = sorted(set(asvs) - set(matrix.index))
    assert not missing, missing[:5]
    matrix = matrix.loc[asvs, meta.Sample_ID].astype(float)
    id_to_asv = dict(zip(node_df.Id, node_df.Label))
    edge = [(id_to_asv[s], id_to_asv[t], w) for s, t, w in edge_list]
    ix = {a: i for i, a in enumerate(asvs)}
    eu = np.array([ix[a] for a, _, _ in edge], dtype=int)
    ev = np.array([ix[b] for _, b, _ in edge], dtype=int)
    ew = np.array([abs(w) for *_, w in edge], dtype=float)
    comms = sorted(node_df.community_raw.unique())
    ci = np.array([comms.index(community[a]) for a in asvs], dtype=int)
    inter = ci[eu] != ci[ev]
    degree = np.bincount(np.r_[eu, ev], minlength=len(asvs))
    ks = np.zeros((len(asvs), len(comms)), dtype=float)
    for a, b in zip(eu, ev):
        ks[a, ci[b]] += 1
        ks[b, ci[a]] += 1
    pc = np.ones(len(asvs))
    nz = degree > 0
    pc[nz] = 1 - np.sum((ks[nz] / degree[nz, None]) ** 2, axis=1)
    pc[~nz] = 0
    connector = pc > 0.62
    node_metrics = node_df[['Label', 'Id', 'community_raw', 'community', 'phylum']].copy()
    node_metrics['full_degree'] = degree
    node_metrics['participation_coefficient'] = pc
    node_metrics['connector_candidate_PC_gt_0.62'] = connector
    out = []
    for sid in meta.Sample_ID:
        counts = matrix[sid].to_numpy(float)
        active = counts > 0
        nact = int(active.sum())
        eactive = active[eu] & active[ev]
        ne = int(eactive.sum())
        rel = counts / counts.sum() if counts.sum() > 0 else counts
        ca = np.bincount(ci, weights=rel, minlength=len(comms))
        pos = ca[ca > 0]
        shannon = -float(np.sum(pos * np.log(pos))) if len(pos) else 0.0
        even = float(np.exp(shannon) / len(comms)) if len(comms) else np.nan
        dominance = float(ca.max()) if len(ca) else np.nan
        ie = int(np.sum(eactive & inter))
        support = np.sqrt(rel[eu] * rel[ev]) * ew
        total_support = float(support[eactive].sum())
        inter_support = float(support[eactive & inter].sum())
        active_ids = np.where(active)[0]
        remap = {old: i for i, old in enumerate(active_ids)}
        pairs = [(remap[a], remap[b]) for a, b, ok in zip(eu, ev, eactive) if ok]
        lcc, comps = dsu_lcc(nact, pairs)
        con_present = int(np.sum(active & connector))
        con_ab = float(rel[connector].sum())
        out.append({'Sample_ID': sid, 'Active_ASVs_456': nact, 'Active_ASV_fraction': nact / len(asvs), 'Realized_edges': ne, 'Edge_realization_rate': ne / len(edge), 'Mean_realized_edge_weight': float(ew[eactive].mean()) if ne else np.nan, 'Intercommunity_edges': ie, 'Intercommunity_edge_fraction': ie / ne if ne else np.nan, 'Abundance_weighted_intercommunity_support': inter_support / total_support if total_support else np.nan, 'Community_coverage': int(np.sum(ca > 0)), 'Community_abundance_evenness': even, 'Community_abundance_dominance': dominance, 'Largest_component_ASVs': lcc, 'Largest_component_fraction': lcc / nact if nact else np.nan, 'Connected_components': comps, 'Fragmentation_1_minus_LCC': 1 - lcc / nact if nact else np.nan, 'Connector_candidates_present': con_present, 'Connector_presence_fraction': con_present / max(int(connector.sum()), 1), 'Connector_relative_abundance': con_ab, **{f'{c}_relative_abundance': float(ca[j]) for j, c in enumerate(comms)}})
    return (pd.DataFrame(out), node_metrics)

def pressure_groups(site):
    variables = ['MIT', 'AtInfCOD', 'AtInfNH4N', 'AtInfTN', 'AtInfTP']
    thresholds = []
    for v in variables:
        q20, q80 = site[v].quantile([0.2, 0.8])
        thresholds.append([v, q20, q80, 'site-level empirical 20th/80th percentiles; continuous variable remains primary'])
        if v == 'MIT':
            site[v + '_pressure_group'] = np.where(site[v] <= q20, 'low-temperature', np.where(site[v] >= q80, 'high-temperature', 'reference-middle'))
        else:
            site[v + '_pressure_group'] = np.where(site[v] >= q80, 'high-pressure', 'reference-not-high')
    return (site, pd.DataFrame(thresholds, columns=['variable', 'Q20', 'Q80', 'rule']))

def main():
    meta = pd.read_csv(INP / 'meta.csv', encoding='utf-8-sig')
    water = pd.read_excel(INP / 'water_quality_geo.xlsx')
    site_water, sites, audit = match_sites(meta, water)
    meta_site = aggregate_meta(meta)
    site = site_water.merge(meta_site, left_on=['WWTP_Lat', 'WWTP_Lon'], right_on=['Lat', 'Lon'], how='left', suffixes=('_water', ''))
    site.insert(0, 'site_row', np.arange(len(site)))
    site, thresholds = pressure_groups(site)
    targets = {'EffCOD': 'effluent COD', 'EffNH4N': 'effluent NH4N', 'EffTN': 'effluent TN', 'EffTP': 'effluent TP'}
    for new, old in targets.items():
        site[new] = pd.to_numeric(site[old], errors='coerce')
    continuous = [c for c in ['MAT', 'MMT', 'SMT', 'AP', 'SMP', 'GDP', 'Pop', 'DC', 'InfR', 'Vol', 'HRT', 'AtHRT', 'SRT', 'InfCOD', 'AtInfCOD', 'InfNH4N', 'AtInfNH4N', 'InfTN', 'AtInfTN', 'InfTP', 'AtInfTP', 'MLSS', 'DO', 'Ph', 'MIT', 'Con', 'NH4N', 'NO3N', 'SVI'] if c in site]
    categorical = [c for c in ['Nitri', 'Denitri', 'Industry wastewater source Contained in Inf?', 'General type ', 'Detailed type'] if c in site]
    cv, choices, imputed = knn_cv_and_impute(site, list(targets), continuous, categorical)
    for t in targets:
        site[t + '_KNN'] = site[t]
        site[t + '_value_source'] = np.where(site[t].notna(), 'observed', 'KNN_sensitivity_only')
        block = imputed[imputed.target == t].set_index('site_row')
        site.loc[site[t].isna(), t + '_KNN'] = block.loc[site.index[site[t].isna()], 'KNN_estimate'].to_numpy()
    node_df, edge_list = parse_gexf()
    network, node_metrics = build_network_metrics(meta, node_df, edge_list)
    sample = meta.merge(site, on=['Lat', 'Lon'], how='left', suffixes=('', '_site'), validate='many_to_one')
    sample = sample.merge(network, on='Sample_ID', how='left', validate='one_to_one')
    assert len(sample) == 1186 and sample.match_method.notna().all()
    for p in ['COD', 'NH4N', 'TN', 'TP']:
        inflow = pd.to_numeric(site[f'inflow {p}'], errors='coerce')
        eff = site[f'Eff{p}']
        valid = inflow.gt(0) & eff.ge(0)
        site[f'Plant_apparent_removal_{p}_pct'] = np.where(valid, 100 * (inflow - eff) / inflow, np.nan)
    site.to_csv(OUT / '01_WWTP_site_level_matched_and_imputed.csv', index=False, encoding='utf-8-sig')
    sample.to_csv(OUT / '02_sample_level_analysis_base.csv', index=False, encoding='utf-8-sig')
    network.to_csv(OUT / '03_sample_network_stability_metrics.csv', index=False, encoding='utf-8-sig')
    node_metrics.to_csv(OUT / '04_456ASV_node_structural_metrics.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(OUT / '05_coordinate_match_audit.csv', index=False, encoding='utf-8-sig')
    cv.to_csv(OUT / '06_KNN_leave_one_WWTP_out_CV.csv', index=False, encoding='utf-8-sig')
    imputed.to_csv(OUT / '07_KNN_missing_value_estimates.csv', index=False, encoding='utf-8-sig')
    thresholds.to_csv(OUT / '08_pressure_thresholds_for_visualization_only.csv', index=False, encoding='utf-8-sig')
    missing_rows = []
    for t in targets:
        missing_rows.append([t, int(site[t].notna().sum()), int(site[t].isna().sum()), float(site[t].isna().mean()), choices[t]])
    missingness = pd.DataFrame(missing_rows, columns=['target', 'observed_WWTPs', 'missing_WWTPs', 'missing_fraction', 'selected_KNN_k'])
    missingness.to_csv(OUT / '09_outcome_missingness_summary.csv', index=False, encoding='utf-8-sig')
    miss_audit = []
    for t in targets:
        observed = site[t].notna()
        for c in continuous:
            a = pd.to_numeric(site.loc[observed, c], errors='coerce').dropna()
            b = pd.to_numeric(site.loc[~observed, c], errors='coerce').dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
            smd = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
            miss_audit.append([t, c, len(a), len(b), a.mean(), b.mean(), smd])
    miss_audit = pd.DataFrame(miss_audit, columns=['target', 'predictor', 'observed_n', 'missing_n', 'observed_mean', 'missing_mean', 'standardized_mean_difference'])
    miss_audit['abs_SMD'] = miss_audit.standardized_mean_difference.abs()
    miss_audit.sort_values(['target', 'abs_SMD'], ascending=[True, False]).to_csv(OUT / '10_missingness_predictor_audit.csv', index=False, encoding='utf-8-sig')
    removal_qc = []
    for p in ['COD', 'NH4N', 'TN', 'TP']:
        c = f'Plant_apparent_removal_{p}_pct'
        s = site[c]
        removal_qc.append([p, int(s.notna().sum()), int((s < 0).sum()), int((s > 100).sum()), float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))])
    pd.DataFrame(removal_qc, columns=['pollutant', 'paired_WWTPs', 'negative_removal', 'above_100pct', 'median_pct', 'Q25_pct', 'Q75_pct']).to_csv(OUT / '11_apparent_removal_QC.csv', index=False, encoding='utf-8-sig')
    metric_dictionary = pd.DataFrame([['Active_ASVs_456', 'Number of the 456 structural-core ASVs detected with count > 0 in the activated-sludge sample', 'realized diversity/context; not whole-plant richness'], ['Edge_realization_rate', 'Fraction of the 13,894 full-network edges whose two endpoint ASVs are both detected in the sample', 'sample-specific realization of the global structural network'], ['Mean_realized_edge_weight', 'Mean absolute RMT association weight among realized edges', 'strength of realized associations; descriptive'], ['Intercommunity_edge_fraction', 'Realized edges connecting different C01-C08 communities divided by all realized edges', 'structural integration between communities'], ['Abundance_weighted_intercommunity_support', 'Intercommunity edge support divided by total support; each edge weighted by absolute RMT weight and geometric mean endpoint abundance', 'abundance-aware structural integration'], ['Community_coverage', 'Number of C01-C08 communities represented by at least one detected ASV', 'breadth of structural community representation'], ['Community_abundance_evenness', 'Effective Shannon number of community abundance divided by eight', 'balance among structural communities'], ['Community_abundance_dominance', 'Largest community relative abundance in the sample', 'concentration/dependence on one community'], ['Largest_component_fraction', 'ASVs in the largest connected component divided by detected structural ASVs', 'global connectivity of the realized network'], ['Fragmentation_1_minus_LCC', 'One minus largest-component fraction', 'realized network fragmentation'], ['Connector_presence_fraction', 'Detected full-network connector candidates divided by all connector candidates; candidates use participation coefficient > 0.62', 'retention of potential cross-community bridge ASVs; candidate definition only'], ['Connector_relative_abundance', 'Total relative abundance of connector-candidate ASVs', 'abundance support for potential bridges']], columns=['metric', 'calculation_without_formula', 'interpretation'])
    metric_dictionary.to_csv(OUT / '12_network_metric_dictionary.csv', index=False, encoding='utf-8-sig')
    summary = {'sites': len(site), 'samples': len(sample), 'strict_coordinate_sites': int(site.strict_223_subset.sum()), 'latitude_rescue_sites': int((site.match_method == 'latitude_unique_rescue').sum()), 'network_nodes': len(node_df), 'network_edges': len(edge_list), 'connector_candidates_PC_gt_0.62': int(node_metrics['connector_candidate_PC_gt_0.62'].sum()), 'selected_knn_k': choices, 'observed_effluent_sites': {t: int(site[t].notna().sum()) for t in targets}, 'analysis_scope': 'activated-sludge microbial network stability; no community-level removal-function claims', 'primary_outcome_policy': 'observed effluent values; KNN values sensitivity-only', 'removal_policy': 'paired water-table inflow/effluent only; never AtInf-to-Eff removal'}
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    workbook_payload = {'summary': summary, 'match_audit': site[['Site_ID', 'WWTP_Lat', 'WWTP_Lon', 'latitude', 'longitude', 'match_method', 'match_distance_km', 'strict_223_subset', 'Sample_count']].replace({np.nan: None}).to_dict('records'), 'missingness': missingness.replace({np.nan: None}).to_dict('records'), 'knn_cv': cv.replace({np.nan: None}).to_dict('records'), 'pressure_thresholds': thresholds.replace({np.nan: None}).to_dict('records'), 'metric_dictionary': metric_dictionary.replace({np.nan: None}).to_dict('records'), 'missingness_audit_top': miss_audit.sort_values(['target', 'abs_SMD'], ascending=[True, False]).groupby('target').head(10).replace({np.nan: None}).to_dict('records'), 'removal_qc': pd.DataFrame(removal_qc, columns=['pollutant', 'paired_WWTPs', 'negative_removal', 'above_100pct', 'median_pct', 'Q25_pct', 'Q75_pct']).replace({np.nan: None}).to_dict('records'), 'site_analysis_key': site[['site_row', 'Site_ID', 'Sample_count', 'match_method', 'strict_223_subset', 'MIT', 'AtInfCOD', 'AtInfNH4N', 'AtInfTN', 'AtInfTP', 'Industry wastewater source Contained in Inf?', 'EffCOD', 'EffCOD_KNN', 'EffCOD_value_source', 'EffNH4N', 'EffNH4N_KNN', 'EffNH4N_value_source', 'EffTN', 'EffTN_KNN', 'EffTN_value_source', 'EffTP', 'EffTP_KNN', 'EffTP_value_source']].replace({np.nan: None}).to_dict('records'), 'sample_network_metrics': sample[['Sample_ID', 'Site_ID', 'match_method', 'strict_223_subset', 'Active_ASVs_456', 'Edge_realization_rate', 'Intercommunity_edge_fraction', 'Abundance_weighted_intercommunity_support', 'Community_coverage', 'Community_abundance_evenness', 'Community_abundance_dominance', 'Largest_component_fraction', 'Fragmentation_1_minus_LCC', 'Connector_presence_fraction', 'Connector_relative_abundance', 'MIT', 'AtInfCOD', 'AtInfNH4N', 'AtInfTN', 'AtInfTP', 'Industry wastewater source Contained in Inf?']].replace({np.nan: None}).to_dict('records')}
    (OUT / 'workbook_payload.json').write_text(json.dumps(workbook_payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
