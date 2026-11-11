from pathlib import Path
import json
import sys
sys.modules['bottleneck'] = None
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def compare_csv(left, right, keys=None, columns=None):
    a=pd.read_csv(ROOT/left);b=pd.read_csv(ROOT/right)
    if columns is not None:a=a[columns];b=b[columns]
    if keys:a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,atol=1e-9,rtol=1e-9)
    return {'rows':len(a),'status':'passed'}

def verify():
    results={}
    results['standard_damage']=compare_csv('results/damage_standard/source_data/seven_metrics_final_curves.csv','shared/damage_standard/seven_metrics_final_curves.csv',['pressure','metric','replicate','intensity','series'])
    results['standard_damage_summary']=compare_csv('results/damage_standard/source_data/Fig3b_final_summary.csv','shared/damage_standard/Fig3b_final_summary.csv')
    results['high_load_damage']=compare_csv('results/damage_high_load/corrected_all_117600_replicate_values.csv','shared/damage_high_load/corrected_all_117600_replicate_values.csv',['pressure','metric','replicate','intensity','series'])
    for name in ['01_pilot_replicate_curves.csv','03_pilot_six_dimension_effect_vectors_by_replicate.csv','04_pilot_effect_summary.csv']:
        results[name]=compare_csv('results/damage_standard/source_data/sixdim_r30/'+name,'shared/damage_standard/sixdim_r30/'+name)
    results['swi_weights']=compare_csv('results/swi/source_data/percentile_SWI_formulas.csv','shared/swi/percentile_SWI_formulas.csv')
    results['swi_plant_scores']=compare_csv('results/swi/source_data/Supplementary_Data_6_percentile_SWI.csv','shared/swi/Supplementary_Data_6_percentile_SWI.csv')
    for name in ['09_directional_percentile_cv_runs.csv','10_directional_percentile_summary.csv']:
        results[name]=compare_csv('results/swi_cross_validation/'+name,'shared/swi_cross_validation/'+name)
    results['plant_metrics_and_pca']=compare_csv('results/plant_metrics/plant_pca_scores.csv','shared/data/08_PCA_source_data/08A_PCA_input_and_scores_252_plants.csv')
    base=np.load(ROOT/'shared/section_metrics/Supplemental_Figure_5_selected_latest_null_arrays.npz')
    regenerated=np.load(ROOT/'results/permutations/community_permutations.npz')
    for key in regenerated.files:np.testing.assert_allclose(regenerated[key],base[key],rtol=1e-12,atol=1e-12)
    results['community_permutations']={'permutations':5000,'communities':8,'statistics':3,'status':'passed'}
    nodes=pd.read_csv(ROOT/'results/partition/tables/community_structural_core_node_audit.csv')
    frozen=pd.read_csv(ROOT/'shared/section_metrics/community_structural_core_node_audit.csv')
    pd.testing.assert_frame_equal(nodes[['Id','community','structural_core']].sort_values('Id').reset_index(drop=True),frozen[['Id','community','structural_core']].sort_values('Id').reset_index(drop=True),check_dtype=False)
    results['community_partition']={'nodes':len(nodes),'core_nodes':int(nodes.structural_core.sum()),'status':'passed'}
    order=pd.read_csv(ROOT/'results/natural_connectivity/node_removal_orders.csv')
    assert order.groupby(['replicate','group']).node_id.nunique().eq(73).all()
    natural=pd.read_csv(ROOT/'results/natural_connectivity/node_removal_replicates.csv')
    baseline=pd.read_csv(ROOT/'shared/natural_connectivity/node_removal_replicates.csv')
    pd.testing.assert_frame_equal(natural,baseline,check_exact=False,check_dtype=False,atol=1e-9,rtol=1e-10)
    results['natural_connectivity']={'rows':len(natural),'max_absolute_error':float(np.abs(natural.natural_connectivity-baseline.natural_connectivity).max()),'status':'passed'}
    original=pd.read_csv(ROOT/'shared/data/07_Six_dimensional_structural_metrics/07C_pressure_specific_six_dimension_effects.csv')
    observed=pd.read_csv(ROOT/'results/observed_fingerprints/all_observed_fingerprints.csv')
    for column in ['difference','low95','high95']:np.testing.assert_allclose(original[column],observed[column],rtol=1e-10,atol=1e-10)
    results['observed_fingerprints']={'rows':len(observed),'status':'passed'}
    mapping=pd.read_csv(next((ROOT/'figures/supplementary/figure_s11/data').glob('01*')))
    assert mapping['Overall state'].value_counts().to_dict()=={'Not exposed':116,'Preserved':52,'Concern':50,'Warning':5}
    results['global_structural_states']={'plants':223,'status':'passed'}
    report={'checks':results,'wsbm_model_input':{'nodes':281,'identifier_type':'sequencing_run','archived_fit_rows':80,'all_80_model_refits_verified':False},'network_edge_counts':{'sample_metric_graph':13894,'figure_graph':13875,'negative_correlations_in_sample_metric_graph':19,'self_loops':0}}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
    return report

if __name__=='__main__':
    verify()
