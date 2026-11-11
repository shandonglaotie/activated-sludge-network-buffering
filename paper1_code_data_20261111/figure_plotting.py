from pathlib import Path
import os
import json
import re
import numpy as np
import sys
sys.modules['bottleneck'] = None
import sys
sys.modules['bottleneck'] = None
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
COLORS = ['#607FAE', '#7651A5', '#8D91B9', '#CF607E']
COMMUNITY_COLORS = ['#607FAE', '#7651A5', '#60B6CA', '#D86A9D', '#EE5B73', '#AB8AD1', '#A92E55', '#EBA5C1']
ROLE_COLORS = {'Provincial hubs': '#6C80B9', 'Connector hubs': '#7651A5', 'Dominant hubs': '#C52C48', 'Non-hubs': '#D2C9DF'}
CONDITIONS = ['Low temperature', 'Low C/N ratio', 'Low C/P ratio', 'High industrial proportion']
CONDITION_LABELS = ['Low temperature', 'Low C/N', 'Low C/P', 'Industrial wastewater']
DIMS = list('EFCSVD')
MECHANISMS = ['bridge', 'assortative', 'core_periphery', 'abundance']
METRICS = ['bridge_direct', 'bridge_paths', 'within_direct', 'within_integrity', 'cp_direct', 'evenness', 'dominance']
plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'], 'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8, 'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':7, 'axes.linewidth':0.75, 'lines.linewidth':0.75, 'xtick.major.width':0.75, 'ytick.major.width':0.75, 'xtick.direction':'out', 'ytick.direction':'out', 'svg.fonttype':'none', 'pdf.fonttype':42, 'legend.frameon':False})

def column_name(value):
    return re.sub(r'[^a-z0-9]+','_',str(value).lower()).strip('_')

def tables(folder):
    result=[]
    for item in json.loads((folder/'sources.json').read_text(encoding='utf-8')):
        data=pd.read_csv(folder/item['file'])
        data.columns=[column_name(c) for c in data.columns]
        result.append(data)
    return result

def save(fig,folder,panel):
    output=folder/'output';output.mkdir(exist_ok=True)
    fig.savefig(output/f'{panel}.svg',bbox_inches='tight',pad_inches=0.04)
    fig.savefig(output/f'{panel}.pdf',bbox_inches='tight',pad_inches=0.04)
    fig.savefig(output/f'{panel}.png',dpi=300,bbox_inches='tight',pad_inches=0.04)
    plt.close(fig)

def label_pairs(data):
    return data.community_1.astype(str)+'–'+data.community_2.astype(str)

def stacked(data,key,folder,panel):
    values=data.set_index(key)
    fig,ax=plt.subplots(figsize=(7.2,2.8 if len(values)<12 else 3.5))
    bottom=np.zeros(len(values))
    for role in values.columns:
        ax.bar(np.arange(len(values)),values[role],bottom=bottom,color=ROLE_COLORS.get(role.replace('_',' ').capitalize(), '#A895BF'),label=role.replace('_',' ').title(),width=0.85)
        bottom+=values[role].to_numpy()
    ax.set_xticks(np.arange(len(values)),values.index,rotation=60 if len(values)>12 else 0,ha='right' if len(values)>12 else 'center');ax.set_ylabel('Nodes (%)');ax.set_ylim(0,100)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.12),ncol=min(len(values.columns),4))
    save(fig,folder,panel)

def ellipse(ax,points,color):
    if len(points)<3:return
    values,vectors=np.linalg.eigh(np.cov(points.T));order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    angle=np.degrees(np.arctan2(vectors[1,0],vectors[0,0]))
    ax.add_patch(Ellipse(points.mean(axis=0),*(2*1.96*np.sqrt(np.maximum(values,0))),angle=angle,fill=False,edgecolor=color,linestyle='--',linewidth=0.75))

def pca_panels(groups,folder,panel):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6),layout='constrained')
    for i,(name,data) in enumerate(groups):
        ax=axes.flat[i]
        x='pc1' if 'pc1' in data else 'pc1_score';y='pc2' if 'pc2' in data else 'pc2_score'
        for group,color in [('Reference','#B9BEC8'),('Exposed',COLORS[i])]:
            sub=data[data.group.eq(group)]
            ax.scatter(sub[x],sub[y],s=10,color=color,alpha=0.7,edgecolors='none',label=group)
            ellipse(ax,sub[[x,y]].to_numpy(),color)
        ax.set(title=name,xlabel='PC1 (45.8%)',ylabel='PC2 (32.7%)')
    axes.flat[0].legend();save(fig,folder,panel)

def curves(data,folder,panel,rows=METRICS):
    pressures=list(dict.fromkeys(data.pressure))
    fig,axes=plt.subplots(len(rows),len(pressures),figsize=(7.4,1.55*len(rows)),squeeze=False,layout='constrained')
    for i,metric in enumerate(rows):
        for j,pressure in enumerate(pressures):
            ax=axes[i,j];sub=data[data.pressure.eq(pressure)&data.metric.eq(metric)]
            for series,color in [('Control','#502880'),('Targeted','#B093CC')]:
                d=sub[sub.series.eq(series)].sort_values('intensity')
                ax.plot(d.intensity,d.retention,color=color,marker='o',markersize=1.5,label=series)
            ax.set(ylim=(0,103),xlim=(0,1))
            if i==0:ax.set_title(pressure.replace('High NH4-N','High NH3-N'))
            if j==0:ax.set_ylabel(metric.replace('_',' ')+'\nRetention (%)')
            if i==len(rows)-1:ax.set_xlabel('Damage intensity')
    axes[0,0].legend();save(fig,folder,panel)

def heatmap_effects(data,folder,panel):
    fig,axes=plt.subplots(1,4,figsize=(10,2.5),layout='constrained')
    for i,p in enumerate(CONDITIONS):
        values=data[data.pressure.eq(p)].pivot(index='mechanism',columns='dimension',values='auc_effect').reindex(index=MECHANISMS,columns=DIMS)
        ax=axes[i];im=ax.imshow(values,cmap='PuOr_r',vmin=-60,vmax=60,aspect='auto')
        ax.set_xticks(range(6),DIMS);ax.set_yticks(range(4),[v.replace('_',' ') for v in MECHANISMS]);ax.set_title(CONDITION_LABELS[i])
        for y in range(4):
            for x in range(6):ax.text(x,y,f'{values.iloc[y,x]:.1f}',ha='center',va='center',fontsize=7)
    fig.colorbar(im,ax=axes,label='Targeted − matched-control integrated loss (percentage points)',shrink=0.8)
    save(fig,folder,panel)

def render(folder):
    folder=Path(folder).resolve();identifier=folder.name;data=tables(folder)
    if identifier=='figure_01':
        fig,ax=plt.subplots(figsize=(7.4,3.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        ax.text(0.5,0.95,'Influent fluctuations',ha='center',weight='bold')
        for i,row in data[0].iterrows():
            x=0.125+i*0.25
            ax.text(x,0.65,row.structural_component.replace(' ','\n',1),ha='center',va='center',bbox={'boxstyle':'round,pad=0.5','facecolor':'#F4F1F8','edgecolor':COLORS[i]})
            ax.annotate('',xy=(x,0.77),xytext=(0.5,0.9),arrowprops={'arrowstyle':'->','color':COLORS[i]})
            ax.text(x,0.3,row.support_function.replace(' ','\n',1),ha='center',va='center')
            ax.annotate('',xy=(x,0.4),xytext=(x,0.53),arrowprops={'arrowstyle':'->','color':COLORS[i]})
        ax.text(0.5,0.05,'Distributed structural buffering',ha='center',weight='bold')
        save(fig,folder,'figure_01')
    elif identifier=='figure_02':
        from matplotlib.collections import LineCollection
        nodes,edges,_,pairs=data;positions=nodes.set_index('id')[['x','y']].to_dict('index')
        segments=[[(positions[s]['x'],positions[s]['y']),(positions[t]['x'],positions[t]['y'])] for s,t in zip(edges.source_node,edges.target_node)]
        fig,ax=plt.subplots(figsize=(5,5));ax.add_collection(LineCollection(segments,colors='#9F91BF',linewidths=0.15,alpha=0.15))
        for i,(name,sub) in enumerate(nodes.groupby('community')):ax.scatter(sub.x,sub.y,s=3,color=COMMUNITY_COLORS[i],label=name)
        ax.autoscale();ax.set_aspect('equal');ax.axis('off');ax.legend(markerscale=2);save(fig,folder,'panel_a')
        fig,axes=plt.subplots(1,4,figsize=(7.2,2));rng=np.random.default_rng(17)
        for ax,(name,pin,pout) in zip(axes,[('Assortative',0.9,0.1),('Core–periphery',0.8,0.6),('Disassortative',0.1,0.9),('Mixed',0.5,0.4)]):
            angles=np.linspace(0,2*np.pi,5,endpoint=False);xy=np.vstack([np.c_[np.cos(angles)*0.2-0.4,np.sin(angles)*0.2],np.c_[np.cos(angles)*0.2+0.4,np.sin(angles)*0.2]])
            for u in range(10):
                for v in range(u+1,10):
                    same=(u<5)==(v<5)
                    probability=pin if same else pout
                    if name=='Core–periphery':probability=(0.95 if u<5 else 0.02) if same else 0.45
                    if rng.random()<probability:ax.plot(xy[[u,v],0],xy[[u,v],1],color='#A28BBB',alpha=0.6,linewidth=0.5)
            ax.scatter(xy[:,0],xy[:,1],s=10,color=['#C95F78']*5+['#7651A5']*5);ax.set_title(name);ax.axis('off');ax.set_aspect('equal')
        save(fig,folder,'panel_b')
        low=pairs[['within_density_1','within_density_2']].min(axis=1);high=pairs[['within_density_1','within_density_2']].max(axis=1)
        fig,ax=plt.subplots(figsize=(9,2.8));x=np.arange(len(pairs));ax.vlines(x,low,high,color='#A38BC3',linewidth=0.75);ax.scatter(x,high,s=9,color='#7651A5',label='Higher internal density');ax.scatter(x,low,s=9,facecolors='white',edgecolors='#7651A5',linewidths=0.75,label='Lower internal density');ax.scatter(x,pairs.between_density,marker='D',s=9,color='#D76083',label='Between-community density');ax.set_xticks(x,label_pairs(pairs),rotation=90);ax.set_ylabel('Connection density');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.3));save(fig,folder,'panel_c')
    elif identifier=='figure_03':
        roles=data[1]
        fig,ax=plt.subplots(figsize=(5,2));ax.axis('off')
        for i,row in data[0].iterrows():ax.text(0.02,0.85-i*0.3,f'{row.role}: {row.within_community_criterion}; {row.additional_criterion}',va='top')
        save(fig,folder,'panel_a')
        fig,ax=plt.subplots(figsize=(4,3))
        for name,sub in roles.groupby('hub_role'):ax.scatter(sub.pc,sub.zi,s=12,color=ROLE_COLORS.get(name,'#999'),label=name,alpha=0.75)
        ax.set(xlabel='Participation coefficient',ylabel='Within-community z-score');ax.legend();save(fig,folder,'panel_b')
        stacked(data[2],'community',folder,'panel_c')
        counts=roles.groupby('taxonomic_classification').hub_role.agg(total='size',hubs=lambda v:v.ne('Non-hubs').sum())
        eligible=counts[(counts.hubs>=1)&(counts.total>=2)].sort_values(['hubs','total'],ascending=False).head(45).index.tolist()
        if 'Unknown' in eligible:eligible=[v for v in eligible if v!='Unknown']+['Unknown']
        selected_taxa=data[3].set_index('taxonomic_classification').reindex(eligible).reset_index()
        stacked(selected_taxa,'taxonomic_classification',folder,'panel_d')
        selected=data[4];fig,axes=plt.subplots(1,3,figsize=(8,2.8),layout='constrained')
        masks=[selected.v_phylum.eq('Proteobacteria'),selected.v_genus.eq('Nitrospira'),selected.v_genus.str.contains('Accumulibacter',na=False)]
        for ax,mask,name in zip(axes,masks,['Proteobacteria','Nitrospira','Ca. Accumulibacter']):
            t=pd.crosstab(selected.loc[mask,'community'],selected.loc[mask,'pre_role']).reindex(index=[f'C{i:02d}' for i in range(1,9)],fill_value=0);t=t.div(t.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)*100;bottom=np.zeros(8)
            for role in t:ax.bar(np.arange(8),t[role],bottom=bottom,color=ROLE_COLORS.get(role,'#AAA'),label=role);bottom+=t[role]
            ax.set_xticks(range(8),t.index,rotation=90);ax.set_title(name,fontstyle='italic');ax.set_ylim(0,100)
        axes[0].set_ylabel('Nodes (%)');axes[0].legend(fontsize=6);save(fig,folder,'panel_e')
        fig,ax=plt.subplots(figsize=(4.5,3))
        for role,sub in data[5].groupby('node_removal_group'):
            sub=sub.sort_values('nodes_removed');color={'Random nodes':'#8C909A','Provincial hubs':'#B6A1D1','Connector hubs':'#502880'}.get(role,'#777');ax.plot(sub.nodes_removed,sub.mean_natural_connectivity,label=role,color=color);ax.fill_between(sub.nodes_removed,sub['2_5th_percentile'],sub['97_5th_percentile'],color=color,alpha=0.15)
        ax.set(xlabel='Nodes removed',ylabel='Natural connectivity');ax.legend();save(fig,folder,'panel_f')
    elif identifier=='figure_04':
        plant=pd.read_csv(ROOT/'shared/data/04_Metadata/04C_plant_metadata_252.csv').merge(pd.read_csv(ROOT/'shared/data/08_PCA_source_data/08A_PCA_input_and_scores_252_plants.csv'),on=['Site_ID','strict_223_subset'])
        members=pd.read_csv(ROOT/'shared/damage_standard/latest_sample_pressure_groups.csv')
        fig,axes=plt.subplots(1,4,figsize=(8.5,2.6),sharey=True,layout='constrained')
        for i,condition in enumerate(CONDITIONS):
            q=data[0][data[0].pressure.eq(condition)].set_index('code').reindex(DIMS)
            axes[i].barh(range(6),q.difference,color=np.where(q.difference>=0,'#CF607E','#7651A5'))
            axes[i].errorbar(q.difference,range(6),xerr=np.vstack([q.difference-q.low95,q.high95-q.difference]),fmt='none',ecolor='#333',capsize=2)
            axes[i].axvline(0,color='#888');axes[i].set_yticks(range(6),DIMS);axes[i].set_title(CONDITION_LABELS[i]);axes[i].set_xlabel('Exposure − reference (risk z)')
        axes[0].invert_yaxis();save(fig,folder,'panel_a')
        pca_panels([(CONDITION_LABELS[i],data[1][data[1].condition.eq(c)]) for i,c in enumerate(CONDITIONS)],folder,'panel_b');heatmap_effects(data[2],folder,'panel_c')
        fig,axes=plt.subplots(1,4,figsize=(8.8,2.6),layout='constrained')
        selected_metrics=['bridge_direct','within_direct','cp_direct','dominance']
        titles=['Bridge edges','Assortative-community nodes','Core–periphery nodes','Abundance redistribution']
        for ax,metric,title in zip(axes,selected_metrics,titles):
            for i,condition in enumerate(CONDITIONS):
                q=data[4][data[4].pressure.eq(condition)&data[4].metric.eq(metric)].sort_values('intensity')
                ax.plot(q.intensity,q.additional_loss_percentage_points,color=COLORS[i],marker=['^','o','s','D'][i],markersize=3,markerfacecolor='white',label=CONDITION_LABELS[i])
            ax.set(title=title,xlabel='Damage intensity',xlim=(0,1));ax.set_ylabel('Additional loss (percentage points)')
        axes[0].legend(fontsize=6);save(fig,folder,'panel_d')
    elif identifier=='figure_05':
        weights,sw,counts,_=data
        fig,ax=plt.subplots(figsize=(6,2));matrix=weights[[v.lower() for v in DIMS]].to_numpy();im=ax.imshow(matrix,cmap='PuOr_r',vmin=-1,vmax=1,aspect='auto');ax.set_xticks(range(6),DIMS);ax.set_yticks(range(4),CONDITION_LABELS)
        for i in range(4):
            for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center')
        fig.colorbar(im,ax=ax,label='Direction coefficient');save(fig,folder,'panel_a')
        fig,axes=plt.subplots(2,2,figsize=(6.6,4.8),layout='constrained')
        for i,p in enumerate(CONDITIONS):
            ax=axes.flat[i];values=sw.loc[sw.condition.eq(p),'swi'];ax.hist(values,bins=np.linspace(0,100,21),color=COLORS[i],edgecolor='white',linewidth=0.3);ax.axvline(75,color='#ECA3B1',linestyle='--');ax.axvline(95,color='#C52C48',linestyle='--');ax.set(title=CONDITION_LABELS[i],xlabel='SWI',ylabel='Number of plants')
        save(fig,folder,'panel_b')
        fig,ax=plt.subplots(figsize=(5.5,3));values=[sw.loc[sw.condition.eq(c),'swi'].to_numpy() for c in CONDITIONS];bp=ax.boxplot(values,patch_artist=True,widths=0.55)
        rng=np.random.default_rng(20261111)
        for i,(points,color) in enumerate(zip(values,COLORS)):
            bp['boxes'][i].set(facecolor=matplotlib.colors.to_rgba(color,0.15),edgecolor=color);ax.scatter(rng.normal(i+1,0.05,len(points)),points,s=4,color=color,alpha=0.4)
        ax.axhline(75,color='#ECA3B1',linestyle='--');ax.axhline(95,color='#C52C48',linestyle='--');ax.set(ylim=(0,103),ylabel='SWI');ax.set_xticks(range(1,5),['Low\ntemperature','Low C/N','Low C/P','Industrial\nwastewater']);save(fig,folder,'panel_c')
        fig,ax=plt.subplots(figsize=(5.5,3));bottom=np.zeros(4)
        for state,color in [('not_exposed','#F4F0F3'),('preserved','#E4DCEB'),('concern','#ECA3B1'),('warning','#C52C48')]:
            val=counts[state].to_numpy()/223*100;ax.bar(range(4),val,bottom=bottom,color=color,label=state.replace('_',' ').title(),edgecolor='white');bottom+=val
        ax.set_xticks(range(4),['Low\ntemperature','Low C/N','Low C/P','Industrial\nwastewater']);ax.set(ylabel='All WWTPs (%)',ylim=(0,100));ax.legend(ncol=2);save(fig,folder,'panel_d')
    elif identifier=='figure_s01':
        curve,markers=data[1:3];fig,ax=plt.subplots(figsize=(4.4,3))
        ax.fill_between(curve.k,curve.bootstrap_95_low,curve.bootstrap_95_high,color='#C8C7E8',alpha=0.7)
        ax.plot(curve.k,curve.fitted_logpr,color='#222');ax.scatter(markers.k,markers.fitted_logpr,color='#222',s=12)
        for k in [10,11]:ax.axvline(k,color='#7651A5',linestyle='--')
        ax.set(xlabel='Number of communities',ylabel='Marginal log-likelihood (LogPr)');save(fig,folder,'figure_s01')
    elif identifier=='figure_s02':
        t=data[0];fig,ax=plt.subplots(figsize=(5,2.8));ax.plot(t.run,t.relative_network_contribution_u,color=COLORS[1]);ax.fill_between(t.run,t.lower_95pct,t.upper_95pct,color=COLORS[1],alpha=0.15);ax.plot(t.run,t.upper_95pct_resampling_envelope,color='#C52C48',linestyle='--');ax.set(xlabel='Resampling run',ylabel='Relative inter-community contribution');save(fig,folder,'figure_s02')
    elif identifier=='figure_s03':
        fig=plt.figure(figsize=(7.2,12.5),layout='constrained');grid=fig.add_gridspec(8,3)
        for j in range(3):
            null,observed=data[j*2:j*2+2]
            for i in range(8):
                name=f'C{i+1:02d}';values=null[name.lower()].to_numpy();point=float(observed[name.lower()].iloc[0]);counts,edges=np.histogram(values,bins=22);span=edges[-1]-edges[0];low=edges[0]-.045*span;high=edges[-1]+.045*span;broken=point<low-.35*span or point>high+.35*span
                if broken:
                    right=point>high;inner=grid[i,j].subgridspec(1,2,width_ratios=[3.5,1] if right else [1,3.5],wspace=.06);axes=[fig.add_subplot(inner[0,k]) for k in range(2)];histax=axes[0] if right else axes[1];obsax=axes[1] if right else axes[0];padding=max(span*.09,abs(point)*.035,.0001);histax.set_xlim(low,high);obsax.set_xlim(point-padding,point+padding);obsax.set_xticks([point]);obsax.tick_params(labelleft=False,left=False);histax.spines['right' if right else 'left'].set_visible(False);obsax.spines['left' if right else 'right'].set_visible(False)
                    for ax,x in [(axes[0],1),(axes[1],0)]:ax.plot([x-.015,x+.015],[-.015,.015],transform=ax.transAxes,color='black',clip_on=False,linewidth=.6)
                else:
                    histax=fig.add_subplot(grid[i,j]);obsax=histax;axes=[histax];histax.set_xlim(min(low,point-.03*span),max(high,point+.03*span))
                histax.bar(edges[:-1],counts,width=np.diff(edges),align='edge',color='#86B7DF',edgecolor='#637FBC',linewidth=.25);obsax.axvline(point,color='#EF4E5B',linestyle='--',linewidth=.75)
                for ax in axes:ax.set_ylim(0,max(100,int(np.ceil(counts.max()/100)*100)));ax.tick_params(labelsize=5.5,length=2)
                histax.set_title(name,loc='left',fontsize=7)
                if j==0:histax.set_ylabel('Permutations')
                if i==7:histax.set_xlabel(['Core participation (%)','Peripheral position index (0–100)','Community assortativity'][j],fontsize=7)
        save(fig,folder,'figure_s03')
    elif identifier=='figure_s04':
        fig,axes=plt.subplots(1,3,figsize=(8.5,2.8),layout='constrained');axes[0].hist(data[0].iloc[:,0],bins=35,color='#B6A1CC');axes[0].axvline(data[1].iloc[0,0],color='#C52C48',linestyle='--');axes[0].set(xlabel='NMI',ylabel='Permutations');axes[1].hist(data[2].mean_weighted_block_profile_r,bins=35,color='#A1B6D1');axes[1].set(xlabel='Weighted block-profile correlation',ylabel='Resamples');axes[2].plot(data[3].threshold_percent,data[3].dominant_hubs,label='Dominant hubs');axes[2].plot(data[3].threshold_percent,data[3].dominant_taxa,label='Dominant taxa');axes[2].set(xlabel='Abundance threshold (%)',ylabel='Count');axes[2].legend();save(fig,folder,'figure_s04')
    elif identifier=='figure_s05':
        t=data[0];fig,ax=plt.subplots(figsize=(7,3));bottom=np.zeros(len(t));palette=plt.cm.tab20(np.linspace(0,1,len(t.columns)-1))
        for col,color in zip(t.columns[1:],palette):ax.bar(t.iloc[:,0],t[col],bottom=bottom,label=col.replace('_',' '),color=color);bottom+=t[col]
        ax.set_ylabel('Relative contribution (%)');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=6);save(fig,folder,'figure_s05')
    elif identifier=='figure_s06':
        t=data[0];fig,axes=plt.subplots(1,4,figsize=(9,2.6),sharey=True,layout='constrained')
        for i,(condition,sub) in enumerate(t.groupby('condition',sort=False)):
            q=sub.set_index('dimension').reindex(DIMS)
            axes[i].barh(range(6),q.exposure_reference_difference,color=np.where(q.exposure_reference_difference>=0,'#CF607E','#7651A5'))
            axes[i].errorbar(q.exposure_reference_difference,range(6),xerr=np.vstack([q.exposure_reference_difference-q.low95,q.high95-q.exposure_reference_difference]),fmt='none',ecolor='#333',capsize=2)
            axes[i].set_yticks(range(6),DIMS);axes[i].set_title(condition);axes[i].axvline(0,color='#888');axes[i].set_xlabel('High-load − reference (risk z)')
        axes[0].invert_yaxis();save(fig,folder,'panel_a')
        groups=[(f'High {name}',d) for name,d in zip(['TN','NH3-N','TP','COD'],data[2:])];pca_panels(groups,folder,'panel_b')
    elif identifier=='figure_s07':curves(data[0],folder,'figure_s07')
    elif identifier=='figure_s08':
        t=data[0].pivot(index='condition',columns='mechanism',values='similarity').reindex(index=CONDITIONS,columns=MECHANISMS);fig,ax=plt.subplots(figsize=(5.5,3));im=ax.imshow(t,vmin=0,vmax=1,cmap='Purples',aspect='auto');ax.set_xticks(range(4),['Bridge','Assortative','Core–periphery','Abundance']);ax.set_yticks(range(4),CONDITION_LABELS)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{t.iloc[i,j]:.2f}\n[{data[0].iloc[i*4+j].low95:.2f}, {data[0].iloc[i*4+j].high95:.2f}]',ha='center',va='center',color='white' if t.iloc[i,j]>0.65 else 'black')
        fig.colorbar(im,ax=ax,label='Cosine similarity');save(fig,folder,'figure_s08')
    elif identifier=='figure_s09':
        from matplotlib.patches import FancyBboxPatch
        t=data[1];fig,ax=plt.subplots(figsize=(8,4.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        left=np.linspace(.85,.20,4);right=np.linspace(.90,.18,6);scale=t.auc_effect.abs().max()
        for i,mechanism in enumerate(MECHANISMS):
            q=t[t.mechanism.eq(mechanism)].set_index('dimension').reindex(DIMS)
            for j,dimension in enumerate(DIMS):
                value=q.loc[dimension,'auc_effect']
                if q.loc[dimension,'displayed_link']:
                    ax.add_patch(FancyArrowPatch((.34,left[i]),(.73,right[j]),connectionstyle=f'arc3,rad={(j-i)*.025}',arrowstyle='-',color='#CF7182' if value>=0 else '#9882BE',linewidth=.7+3*abs(value)/scale,alpha=.7))
            ax.add_patch(FancyBboxPatch((.015,left[i]-.06),.30,.12,boxstyle='round,pad=.01',facecolor='#F8F8FA',edgecolor=COLORS[i]))
            ax.text(.025,left[i]+.035,['Bridge','Assortative','Core–periphery','Abundance'][i],color=COLORS[i],weight='bold')
            baseline=left[i]-.006;ax.plot([.035,.29],[baseline,baseline],color='#AAA',linewidth=.4)
            for j,value in enumerate(q.auc_effect):
                height=.035*abs(value)/scale
                ax.add_patch(plt.Rectangle((.04+j*.041,baseline if value>=0 else baseline-height),.03,height,facecolor='#CF7182' if value>=0 else '#9882BE'))
                ax.text(.055+j*.041,left[i]-.05,DIMS[j],ha='center',fontsize=7)
        for j,(dimension,y) in enumerate(zip(DIMS,right)):
            ax.add_patch(Ellipse((.80,y),.065,.052,facecolor=['#507EB4','#70529C','#8094C0','#D8A2AA','#CF7182','#9882BE'][j]));ax.text(.80,y,dimension,ha='center',va='center')
            ax.text(.85,y,['Edge realization','Network fragmentation','Connector loss','Intercommunity support','Evenness disruption','Dominance imbalance'][j],va='center',fontsize=7)
        ax.plot([],[],color='#CF7182',label='Positive net effect');ax.plot([],[],color='#9882BE',label='Negative net effect');ax.legend(loc='lower center',ncol=2)
        save(fig,folder,'figure_s09')
    elif identifier=='figure_s10':
        t=data[0];fig,ax=plt.subplots(figsize=(4.8,2.8));x=t.cv_auc_median;ax.errorbar(x,range(4),xerr=np.vstack([x-t.cv_auc_low2_5,t.cv_auc_high97_5-x]),fmt='none',ecolor='#80669E',capsize=2);ax.scatter(x,range(4),c=COLORS,s=18);ax.set_yticks(range(4),t.condition);ax.axvline(0.5,color='#999',linestyle='--');ax.set_xlabel('Cross-validated AUC (95% interval)');save(fig,folder,'figure_s10')
    elif identifier=='figure_s11':
        os.environ['BASEMAPDATA']=str(ROOT/'shared/geography')
        from mpl_toolkits.basemap import Basemap
        fig,ax=plt.subplots(figsize=(7.5,3.7));m=Basemap(projection='robin',lon_0=0,resolution='c',ax=ax);m.drawmapboundary(fill_color='#F5F7FA',linewidth=0.75);m.fillcontinents(color='#E2E5E9',lake_color='#F5F7FA');m.drawcoastlines(color='#A7ADB5',linewidth=0.35);m.drawcountries(color='#C2C6CC',linewidth=0.2)
        for state,color in [('Warning','#C52C48'),('Concern','#7651A5'),('Preserved','#607FAE'),('Not exposed','#FCE6EC')]:
            sub=data[0][data[0].overall_state.eq(state)];x,y=m(sub.longitude.to_numpy(),sub.latitude.to_numpy());ax.scatter(x,y,color=color,s=20,edgecolors='white',linewidths=0.5,label=state,zorder=5)
        ax.legend(loc='lower left',fontsize=7);save(fig,folder,'figure_s11')
    else:raise ValueError(identifier)

if __name__=='__main__':
    import sys
    render(Path(sys.argv[1]))
