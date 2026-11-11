from pathlib import Path
import json
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def folder(name):
    return ROOT / 'figures' / ('supplementary' if name.startswith('figure_s') else 'main') / name

def read(name):
    return pd.read_csv(ROOT / name)

def write(name, index, frame, columns=None):
    target = folder(name)
    listing = json.loads((target / 'sources.json').read_text())
    if columns is not None:
        frame = frame.copy()
        frame.columns = columns
    frame.to_csv(target / listing[index]['file'], index=False)
    listing[index]['rows'] = len(frame)
    (target / 'sources.json').write_text(json.dumps(listing, indent=2), encoding='utf-8')

def refresh():
    pair = read('results/community_metrics/pairwise_densities.csv')
    write('figure_02', 3, pair, ['community 1','community 2','within density 1','within density 2','between density','architecture'])
    nodes = read('results/roles_and_resampling/node_roles.csv')
    columns = ['Id','v_name','community','Zi','PC','pre_role','hub_role','MRA','Taxonomic Classification']
    write('figure_03', 1, nodes[columns], [c.replace('_',' ') for c in columns])
    roles = ['Provincial hubs','Connector hubs','Dominant hubs','Non-hubs']
    for i, key in [(2,'community'),(3,'Taxonomic Classification')]:
        table = pd.crosstab(nodes[key],nodes.hub_role).reindex(columns=roles,fill_value=0)
        write('figure_03', i, table.div(table.sum(axis=1),axis=0).mul(100).reset_index())
    selected = nodes[nodes.v_phylum.eq('Proteobacteria') | nodes.v_genus.eq('Nitrospira') | nodes.v_genus.str.contains('Accumulibacter',na=False)]
    write('figure_03', 4, selected[['Id','v_name','v_phylum','v_genus','community','pre_role','hub_role','Zi','PC']])
    summary = read('results/natural_connectivity/natural_connectivity_summary.csv')
    write('figure_03',5,summary,['Node-removal group','Nodes removed','Mean natural connectivity','Standard deviation','2.5th percentile','97.5th percentile'])
    raw = read('results/natural_connectivity/node_removal_replicates.csv')
    write('figure_03',6,raw,['Replicate','Node-removal group','Nodes removed','Natural connectivity'])
    conditions = ['Low temperature','Low C/N ratio','Low C/P ratio','High industrial proportion']
    observed = read('results/observed_fingerprints/all_observed_fingerprints.csv')
    write('figure_04',0,observed[observed.pressure.isin(conditions)])
    effect = read('results/damage_standard/source_data/sixdim_r30/03_pilot_six_dimension_effect_vectors_by_replicate.csv')
    means = effect.groupby(['pressure','mechanism','dimension'],as_index=False).auc_effect.mean()
    write('figure_04',2,means,['pressure','mechanism','dimension','auc effect'])
    write('figure_04',3,read('results/damage_standard/source_data/Fig3b_final_summary.csv'))
    curves = read('results/damage_standard/source_data/seven_metrics_final_curves.csv').groupby(['pressure','metric','intensity','series']).retention.mean().unstack('series').reset_index()
    curves['Additional loss (percentage points)'] = curves.Control-curves.Targeted
    write('figure_04',4,curves)
    write('figure_s02',0,read('results/roles_and_resampling/edge_resampling.csv'))
    null = np.load(ROOT/'results/permutations/community_permutations.npz')
    labels = [f'C{i:02d}' for i in range(1,9)]
    for i,key in enumerate(['core','periphery','assortative']):
        values = null['null_'+key];observed = null['observed_'+key]
        if key=='periphery':values=100/(2-values/100);observed=100/(2-observed/100)
        table=pd.DataFrame(values,columns=labels);table.insert(0,'Permutation',np.arange(1,len(table)+1))
        write('figure_s03',i*2,table);write('figure_s03',i*2+1,pd.DataFrame([observed],columns=labels))
    write('figure_s04',0,read('results/partition_stability/label_perturbation.csv'))
    nmi=read('results/partition_stability/partition_pairwise_NMI.csv')
    write('figure_s04',1,pd.DataFrame({'observed median NMI':[nmi.NMI.median()]}))
    write('figure_s04',2,read('results/block_profile_stability/Fig_S1d_edge_resampling_2320_values.csv'))
    write('figure_s04',3,read('results/roles_and_resampling/Dominant_threshold_sensitivity.csv'))
    write('figure_s05',0,read('results/community_metrics/functional_composition.csv'))
    high=read('results/high_load_fingerprints/high_load_fingerprints.csv')
    high['name']=high.name.str.replace('NH4','NH3',regex=False)
    high=high.set_index(['name','dimension']).reindex(pd.MultiIndex.from_product([['High TN','High NH3-N','High TP','High COD'],list('EFCSVD')],names=['name','dimension'])).reset_index()
    write('figure_s06',0,high[['name','dimension','difference','n_exposed','n_reference','low95','high95']],['Condition','Dimension','Exposure-reference difference','Exposed n','Reference n','low95','high95'])
    write('figure_s07',0,read('results/damage_high_load/corrected_56_mean_curves.csv'))
    write('figure_s08',0,read('results/magnitude_profile/Fig4c_magnitude_profile_similarity.csv'))
    write('figure_s09',0,means,['pressure','mechanism','dimension','auc effect'])
    profiles=means.groupby(['mechanism','dimension'],as_index=False).auc_effect.mean()
    profiles['displayed_link']=profiles.auc_effect.abs().ge(profiles.auc_effect.abs().quantile(.60))
    write('figure_s09',1,profiles)
    write('figure_s10',0,read('results/swi_cross_validation/10_directional_percentile_summary.csv'))
    for index,filename in [(1,'saturation_curve.csv'),(2,'saturation_markers.csv')]:write('figure_s01',index,read('results/wsbm_curve/'+filename))
    weights=read('results/swi/source_data/percentile_SWI_formulas.csv')
    write('figure_05',0,weights[['condition',*list('EFCSVD'),'exposed_n','reference_n']])
    scores=read('results/swi/source_data/Supplementary_Data_6_percentile_SWI.csv')
    scores=scores[scores.strict_223_subset].copy();names=[c[:-5] for c in scores if c.endswith('__SWI')]
    for c in scores.filter(like='__state'):scores[c]=scores[c].replace({'Reference range':'Preserved'})
    values=[];counts=[]
    for name in names:
        for row in scores[scores[name+'__exposed']].to_dict('records'):values.append([name,row['Site_ID'],row[name+'__SWI'],row[name+'__state']])
        count=scores[name+'__state'].value_counts();counts.append([name,*[int(count.get(state,0)) for state in ['Not exposed','Preserved','Concern','Warning']]])
    write('figure_05',1,pd.DataFrame(values,columns=['Condition','Site ID','SWI','State']))
    count=pd.DataFrame(counts,columns=['Condition','Not exposed','Preserved','Concern','Warning'])
    write('figure_05',2,count)
    percent=pd.concat([count[['Condition']],count.iloc[:,1:].astype(float)/223*100],axis=1)
    write('figure_05',3,percent)
    metadata=read('shared/data/04_Metadata/04C_plant_metadata_252.csv').set_index('Site_ID')
    priority={'Not exposed':0,'Preserved':1,'Concern':2,'Warning':3};rows=[]
    for row in scores.to_dict('records'):
        states=[row[name+'__state'] for name in names];site=metadata.loc[row['Site_ID']]
        rows.append([row['Site_ID'],site.Lat,site.Lon,*states,max(states,key=priority.get)])
    write('figure_s11',0,pd.DataFrame(rows,columns=['Site ID','Latitude','Longitude',*names,'Overall state']))

if __name__=='__main__':
    refresh()
