import plumbum as pb
import matplotlib
import itertools
if not 'DISPLAY' in pb.local.env:
    matplotlib.use('Agg')  # use this backend when no X server
import matplotlib.pyplot as plt
import logging
import numpy as np
import seaborn as sns
import pandas as pd
import os
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.pyplot import cm
from scipy import stats,interpolate,signal
from io import StringIO
from Bio import Phylo

def node_averages(df):
    # note that this returns a df with fewer rows, i.e. one for every
    # node in the gene family trees.
    return df.groupby(["family", "node"])["dS"].mean()

def node_weights(df):
    # note that this returns a df with the same number of rows
    return 1 / df.groupby(["family", "node"])["dS"].transform('count')

def parse_filter(s):
    x = [x.strip() for x in s.split("<")]
    if len(x) != 3:
        raise(ValueError("invalid 'x < field < y' filter string"))
    return (x[1], float(x[0]), float(x[2]))

def parse_filters(filterstring):
    return [parse_filter(s) for s in filterstring.split(",")]

def apply_filters(df, filters):
    for key, lower, upper in filters:
        df = df[df[key] > lower]
        df = df[df[key] < upper]
    return df

_labels = {
        "dS" : "$K_\mathrm{S}$",
        "dN" : "$K_\mathrm{A}$",
        "dN/dS": "$\omega$"}

def getspair_ks(spair,df,spgenemap,reweight,onlyrootout,sptree=None):
    df_perspair = {}
    allspair = []
    paralog_pair = []
    for i in spair:
        pair = '__'.join(sorted([j.strip() for j in i.split(';')]))
        if i.split(';')[0].strip() == i.split(';')[1].strip(): paralog_pair.append(pair)
        if pair not in allspair: allspair.append(pair)
    #If users provide various paralogous pair, we still only consider the first one in correction
    df['spair'] = ['__'.join(sorted([spgenemap[g1],spgenemap[g2]])) for g1,g2 in zip(df['gene1'],df['gene2'])]
    #If users provide no paralogous pair, we don't do the correction
    if sptree != None and len(paralog_pair) !=0 : corrected_ks_spair,Outgroup_spnames = correctks(df,sptree,paralog_pair[0],reweight,onlyrootout)
    else: corrected_ks_spair,Outgroup_spnames = None,None
    for p in allspair: df_perspair[p] = df[df['spair']==p]
    return df_perspair,allspair,paralog_pair,corrected_ks_spair,Outgroup_spnames

def findoutgroup(focusp,first_children_of_root):
    for clade in first_children_of_root:
        clade.name = clade.name + "_Outgroup"
        for tip in clade.get_terminals():
            if tip.name == focusp:
                clade.name = clade.name.replace("_Outgroup","_Ingroup")
                break

def gettrios(focusp,Ingroup_spnames,Outgroup_spnames):
    all_spairs,spairs,trios,Trios_dict = [],[],[],{}
    for sister in Ingroup_spnames:
        if sister == focusp:
            continue
        sppair = "{}".format("__".join(sorted([sister,focusp])))
        spairs.append(sppair)
        all_spairs.append(sppair)
        Trios_dict[sppair] = []
        for outgroup in Outgroup_spnames:
            # The order in (focus,sister,outgroup)
            all_spairs.append("{}".format("__".join(sorted([sister,outgroup]))))
            all_spairs.append("{}".format("__".join(sorted([focusp,outgroup]))))
            trios.append((focusp,sister,outgroup))
            Trios_dict[sppair].append((focusp,sister,outgroup))
    return all_spairs,spairs,trios,Trios_dict

def getoutinin(mrca,focusp,Ingroup_spnames):
    mrca_species_pool = [i.name for i in mrca.get_terminals()]
    outgroup_in_ingroup = set(Ingroup_spnames) - set(mrca_species_pool)
    return [i for i in outgroup_in_ingroup]

def gettrios_overall(focusp,Ingroup_spnames,Outgroup_spnames,Ingroup_clade):
    all_spairs,spairs,trios,Trios_dict = [],[],[],{}
    for sister in Ingroup_spnames:
        if sister == focusp:
            continue
        sppair = "{}".format("__".join(sorted([sister,focusp])))
        spairs.append(sppair)
        all_spairs.append(sppair)
        Trios_dict[sppair] = []
        mrca = Ingroup_clade.common_ancestor({"name": sister}, {"name": focusp})
        outgroup_in_ingroup = getoutinin(mrca,focusp,Ingroup_spnames)
        working_Outgroup = outgroup_in_ingroup + Outgroup_spnames
        for outgroup in working_Outgroup:
            # The order in (focus,sister,outgroup)
            all_spairs.append("{}".format("__".join(sorted([sister,outgroup]))))
            all_spairs.append("{}".format("__".join(sorted([focusp,outgroup]))))
            trios.append((focusp,sister,outgroup))
            Trios_dict[sppair].append((focusp,sister,outgroup))
    return all_spairs,spairs,trios,Trios_dict

def getspairks(spairs,df,reweight,method='mode'):
    bins = 50
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    spairs_ks = {}
    for spair in spairs:
        df_tmp = df[df['spair']==spair]
        x = df_tmp['dS']
        y = x[np.isfinite(x)]
        w = reweighted(df_tmp) if reweight else df_tmp['weightoutlierexcluded']
        if method == 'mode':
            kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
            kde_y = kde(kde_x)
            mode, maxim = kde_mode(kde_x, kde_y)
            spairs_ks[spair] = mode
        elif method == 'median':
            median = np.median(kde_y)
            spairs_ks[spair] = median
    return spairs_ks

def ksadjustment(Trios_dict,ks_spair):
    corrected_ks_spair = {}
    for spair,trios in Trios_dict.items():
        Ks_adjusted_all = []
        for trio in trios:
            Ks_focus_outgroup = ks_spair["{}".format("__".join(sorted([trio[0],trio[2]])))]
            Ks_sister_outgroup = ks_spair["{}".format("__".join(sorted([trio[1],trio[2]])))]
            Ks_focus_sister = ks_spair["{}".format("__".join(sorted([trio[0],trio[1]])))]
            Ks_focus_specific = ((Ks_focus_outgroup - Ks_sister_outgroup) + Ks_focus_sister)/2
            Ks_adjusted = Ks_focus_specific * 2
            Ks_adjusted_all.append(Ks_adjusted)
        final_adjusted_Ks = np.array(Ks_adjusted_all).mean()
        corrected_ks_spair[spair] = final_adjusted_Ks
    return corrected_ks_spair

def correctks(df,sptree,focus,reweight,onlyrootout):
    focusp = focus.split('__')[0]
    tree = Phylo.read(sptree, "newick")
    for i,clade in enumerate(tree.get_nonterminals()): clade.name = "internal_node_{}".format(i)
    tree.root.name = 'assumed_root'
    focussp_clade = next(tree.find_clades({"name": focusp}))
    Depths = tree.root.depths(unit_branch_lengths=True)
    first_children_of_root = []
    for clade,depth in Depths.items():
        if depth == 1: first_children_of_root.append(clade)
    findoutgroup(focusp,first_children_of_root)
    Outgroup_clade = first_children_of_root[0] if first_children_of_root[0].name.endswith('_Outgroup') else first_children_of_root[1]
    Ingroup_clade = first_children_of_root[0] if first_children_of_root[0].name.endswith('_Ingroup') else first_children_of_root[1]
    Outgroup_spnames = [i.name.replace('_Outgroup','').replace('_Ingroup','') for i in Outgroup_clade.get_terminals()]
    Ingroup_spnames = [i.name.replace('_Outgroup','').replace('_Ingroup','') for i in Ingroup_clade.get_terminals()]
    if onlyrootout: all_spairs,spairs,Trios,Trios_dict = gettrios(focusp,Ingroup_spnames,Outgroup_spnames)
    else: all_spairs,spairs,Trios,Trios_dict = gettrios_overall(focusp,Ingroup_spnames,Outgroup_spnames,Ingroup_clade)
    ks_spair = getspairks(all_spairs,df,reweight,method='mode')
    corrected_ks_spair = ksadjustment(Trios_dict,ks_spair)
    return corrected_ks_spair,Outgroup_spnames

def get_totalH(Hs):
    CHF = 0
    for i in Hs: CHF = CHF + i
    return CHF

def writespgenemap(spgenemap,outdir):
    fname = os.path.join(outdir,'gene_species.map')
    with open(fname,'w') as f:
        for gene,sp in spgenemap.items(): f.write('{0} {1}\n'.format(gene,sp))

def getgsmap(gsmap):
    spgenemap = {}
    with open(gsmap,'r') as f:
        lines = f.readlines()
        for line in lines:
           gs = [i.strip() for i in line.split(' ')]
           spgenemap[gs[0]] = gs[1]
    return spgenemap

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

def reweighted(df_per):
    return 1 / df_per.groupby(["family", "node"])["dS"].transform('count')

def multi_sp_plot(df,spair,gsmap,outdir,onlyrootout,title='',ylabel='',viz=False,plotkde=False,reweight=True,sptree=None,ksd=False,ap=None):
    if not ksd: spgenemap = getgsmap(gsmap)
    else: spgenemap = gsmap
    if not viz: writespgenemap(spgenemap,outdir)
    df_perspair,allspair,paralog_pair,corrected_ks_spair,Outgroup_spnames = getspair_ks(spair,df,spgenemap,reweight,onlyrootout,sptree=sptree)
    if len(paralog_pair) != 0:
        fnames = os.path.join(outdir,'{}_Corrected.ksd.svg'.format(paralog_pair[0].split('__')[0]))
        fnamep = os.path.join(outdir,'{}_Corrected.ksd.pdf'.format(paralog_pair[0].split('__')[0]))
    else:
        fnames = os.path.join(outdir,'Raw_Orthologues.ksd.svg')
        fnamep = os.path.join(outdir,'Raw_Orthologues.ksd.pdf')
    cs = cm.viridis(np.linspace(0, 1, len(allspair)))
    keys = ["dS", "dS", "dN", "dN/dS"]
    np.seterr(divide='ignore')
    funs = [lambda x: x, np.log10, np.log10, np.log10]
    #fig, axs = plt.subplots(2, 2)
    fig, ax = plt.subplots()
    #df_pers = [df_perspair[i] for i in allspair]
    bins = 50
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    if reweight: logging.info('Recalculating the weights per species pair')
    else: logging.info('De-redundancy via the weights from overall species')
    if plotkde: logging.info('Plotting kde curve over histogram')
    else: logging.info('Plotting histogram without kde curve')
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    for i,item in enumerate(df_perspair.items()):
        pair,df_per = item[0],item[1]
        #for ax, k, f in zip(axs.flatten(), keys, funs):
        w = reweighted(df_per) if reweight else df_per['weightoutlierexcluded']
        x = df_per['dS']
        y = x[np.isfinite(x)]
        w = w[np.isfinite(x)]
        if pair in paralog_pair:
            Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color=cs[i], alpha=0.8, rwidth=0.8,label=pair,edgecolor='black',linewidth=0.8)
        else:
            Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color=cs[i], alpha=0.5, rwidth=0.8,label=pair)
        if corrected_ks_spair != None:
            if pair in corrected_ks_spair.keys():
                #since paralogous genes and orthologous genes between focus and outgroup speices have no corrected Ks data
                ax.axvline(x = corrected_ks_spair[pair], color = cs[i], alpha = 0.8, ls = '-.', lw = 1,label = 'Corrected mode {:.2f} of {}'.format(corrected_ks_spair[pair],pair))
                logging.info('The corrected mode of species pair {} is {:.2f}'.format(pair,corrected_ks_spair[pair]))
        if pair not in paralog_pair:
            kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
            kde_y = kde(kde_x)
            mode, maxim = kde_mode(kde_x, kde_y)
            logging.info('The mode of species pair {} is {:.2f}'.format(pair,mode))
            CHF = get_totalH(Hs)
            scale = CHF*0.1
            if plotkde: ax.plot(kde_x, kde_y*scale, color=cs[i],alpha=0.4, ls = '--')
            ax.axvline(x = mode, color = cs[i], alpha = 0.8, ls = ':', lw = 1,label = 'Original mode {:.2f} of {}'.format(mode,pair))
            if corrected_ks_spair != None:
                if pair in corrected_ks_spair.keys():
                    ax.quiver(mode,maxim*scale, corrected_ks_spair[pair]-mode, 0, angles='xy', scale_units='xy', scale=1,color=cs[i],width=0.005,headwidth=2,headlength=2,headaxislength=2)
    if ap != None:
        df_ap = pd.read_csv(ap,header=0,index_col=0,sep='\t')
        df_ap.loc[:,"pair"] = df_ap[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
        df_working = df_ap.set_index('pair').join(df).dropna(subset=['dS','weightoutlierexcluded'])
        w = reweighted(df_working) if reweight else df_working['weightoutlierexcluded']
        x = df_working['dS']
        y = x[np.isfinite(x)]
        w = w[np.isfinite(x)]
        ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, fill=False, rwidth=0.8,label='Anchor pairs',linewidth=0,hatch = '////////',edgecolor='black')
    ax.set_xlabel(_labels["dS"])
    #ax.legend(loc=1,fontsize=5,bbox_to_anchor=(0.95, 0.95),frameon=False)
    ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5),frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    if len(paralog_pair) !=0: title = 'Corrected $K_\mathrm{S}$ ' + 'distribution of {}'.format(paralog_pair[0].split('__')[0])
    else: title = 'Orthologous $K_\mathrm{S}$ distribution'
    ax.set_title(title)
    #fig.tight_layout()
    #plt.subplots_adjust(top=0.85)
    fig.savefig(fnames,bbox_inches='tight')
    fig.savefig(fnamep,bbox_inches='tight')
    plt.close()

def reflect_logks(ks,w):
    cutoff = 1
    right = []
    max_ks = max(ks)
    right_w = []
    for i in range(len(ks)):
        if ks[i] >= max_ks - cutoff and  ks[i] != max_ks:
            right.append(max_ks + (max_ks-ks[i]))
            right_w.append(w[i])
    ks_refed,w_refed = np.hstack([ks,np.array(right)]),np.hstack([w,np.array(right_w)])
    return ks_refed,cutoff,w_refed

def elmm_plot(df,sp,outdir,max_EM_iterations=200,num_EM_initializations=200,peak_threshold=0.1,na=False,rel_height=0.4):
    if na:
        df = df.drop_duplicates(subset=['family','node'])
        df = df.loc[:,['node_averaged_dS_outlierexcluded']].copy().rename(columns={'node_averaged_dS_outlierexcluded':'dS'})
        df['weightoutlierexcluded'] = 1
    df = df.dropna(subset=['dS','weightoutlierexcluded'])
    df = df.loc[(df['dS']>0) & (df['dS']<=5),:]
    ks_or = np.array(df['dS'])
    w = np.array(df['weightoutlierexcluded'])
    if na: deconvoluted_data = ks_or.copy()
    else: deconvoluted_data = get_deconvoluted_data(ks_or,w)
    hist_property = np.histogram(ks_or, weights=w, bins=50, density=True)
    init_lambd = hist_property[0][0]
    ks = np.log(ks_or)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5,2)
    ax.set_title('KDE and spline of log-transformed $K_\mathrm{S}$ of species '+ '{}'.format(sp))
    ax.set_xlabel("ln $K_\mathrm{S}$")
    ax.set_ylabel("Density of retained duplicates")
    max_ks,min_ks = ks.max(),ks.min()
    ks_refed,cutoff,w_refed = reflect_logks(ks,w)
    kde = stats.gaussian_kde(ks_refed, bw_method="scott", weights=w_refed)
    bw_modifier = 0.4
    kde.set_bandwidth(kde.factor * bw_modifier)
    kde_x = np.linspace(min_ks-cutoff, max_ks+cutoff,num=500)
    kde_y = kde(kde_x)
    ax.plot(kde_x, kde_y, color="k", lw=1, label="KDE")
    spl = interpolate.UnivariateSpline(kde_x, kde_y)
    spl.set_smoothing_factor(0.01)
    #here the round is equal to floor + 1
    spl_x = np.linspace(min_ks, max_ks+0.1, num=int(round((abs(min_ks) + (max_ks+0.1)) *100)))
    spl_y = spl(spl_x)
    ax.plot(kde_x, spl(kde_x), 'b', lw=1, label="Spline on KDE")
    ax.hist(ks_refed, weights=w_refed, bins=np.arange(-5, 2.1, 0.1), color="gray", alpha=0.2, density=True,rwidth=0.8)
    ax.axvline(x=max_ks, color="r", linestyle="--", lw=1, label=f"Reflection boundary")
    ax.legend(loc=2,fontsize='large',frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.tight_layout()
    if na:
        fig.savefig(os.path.join(outdir, "{}.spline_node_averaged.svg".format(sp)))
        fig.savefig(os.path.join(outdir, "{}.spline_node_averaged.pdf".format(sp)))
    else:
        fig.savefig(os.path.join(outdir, "{}.spline_weighted.svg".format(sp)))
        fig.savefig(os.path.join(outdir, "{}.spline_weighted.pdf".format(sp)))
    plt.close(fig)
    logging.info('Initiative detection of likely peaks')
    init_means, init_stdevs, good_prominences = find_peak_init_parameters(spl_x,spl_y,sp,outdir,peak_threshold=peak_threshold,na=na,rel_height=rel_height)
    logging.info('Found {} likely peak signals'.format(len(init_means)))
    reduced_gaussians = False
    # here I select the peak via prominence
    if len(init_stdevs) > 4:
        sor_by_prom = [(m,s) for m,s,_ in sorted(zip(init_means,init_stdevs,good_prominences), key=lambda y: y[2])]
        init_means,init_stdevs = [i for i,j in sor_by_prom[:4]],[j for i,j in sor_by_prom[:4]]
        reduced_gaussians = True
        logging.info('Too many peak signals detected among which only 4 with highest prominences are retained')
    buffer_maxks,buffer_std = 5,0.3
    logging.info('An extra buffer lognormal component with mean {:.2f} and std 0.30 is appended'.format(buffer_maxks))
    init_means.append(np.log(buffer_maxks))
    init_stdevs.append(buffer_std)
    for m,s in zip(init_means,init_stdevs): logging.info('The initiative means and stds is {:.2f} {:.2f}'.format(np.exp(m),s))
    num_comp = len(init_means)+1
    init_weights = [1/num_comp] * num_comp
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16), sharey="row")
    fig.suptitle("Exponential-Lognormal mixture model on {} ".format(sp)+"$K_\mathregular{S}$ " + "paranome\n\nInitialized from data",fontsize='x-large')
    axes[0,0].set_title("Model 1")
    axes[1,0].set_xlabel("$K_\mathregular{S}$")
    axes[1,1].set_xlabel("ln $K_\mathregular{S}$")
    for i in [0,1]: axes[i,0].set_xlim(0, 5)
    for i in [0,1]: axes[i,1].set_xlim(-5, 2)
    for i in [0,1]:
        if na: axes[i,0].hist(ks_or, bins=np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='gray',rwidth=0.8, label='Whole-paranome (node averaged)',density=True)
        else: axes[i,0].hist(ks_or, bins=np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='gray',rwidth=0.8, label='Whole-paranome (weighted)',density=True)
        if na: axes[i,0].set_ylabel("Density of node-averaged retained paralogs",fontsize='x-large')
        else: axes[i,0].set_ylabel("Density of weighted retained paralogs",fontsize='x-large')
    for i in [0,1]: axes[i,1].hist(ks,weights=w,bins=np.arange(-10, 10 + 0.1, 0.1),density=True,color='gray',alpha=0.5,label='Log-transformed paranome',rwidth=0.8)
    all_models_init_parameters,bic_dict,all_models_fitted_parameters = {},{},{}
    all_models_init_parameters['Model1'] = [init_means, init_stdevs, init_lambd, init_weights]
    plot_init_model(axes[0,0], axes[0,1],init_means, init_stdevs, init_lambd, init_weights)
    num_comp = len(init_means) + 1
    logging.info("Performing EM algorithm from initializated data (Model1)")
    bic, new_means, new_stdevs, new_lambd, new_weights, convergence = EM_step(num_comp,deconvoluted_data,init_means, init_stdevs, init_lambd, init_weights,max_EM_iterations=max_EM_iterations,max_num_comp = 5, reduced_gaussians_flag=reduced_gaussians)
    #for m,s in zip(new_means,new_stdevs): logging.info('The optimized means and stds is {:.2f} {:.2f}'.format(np.exp(m),s))
    if convergence: logging.info('The EM algorithm has reached convergence')
    else: logging.info("The EM algorithm hasn't reached convergence")
    all_models_fitted_parameters['Model1'] = [new_means, new_stdevs, new_lambd, new_weights]
    bic_dict['Model1'] = bic
    logging.info('BIC of Model1: {:.2f}'.format(bic))
    plot_fitted_model(axes[0,0], axes[0,1],new_means, new_stdevs, new_lambd, new_weights)
    logging.info("Performing EM algorithm from initializated data plus a random lognormal component (Model2)")
    axes[1,0].set_title("Model 2")
    bic_from_same_num_comp,start_parameters,final_parameters = [],[],[]
    for i in range(num_EM_initializations):
        if len(init_means) > 4:
            updated_means,updated_stdevs = init_means[:4]+[init_means[-1]],init_stdevs[:4]+init_stdevs[-1]
            reduced_gaussians = True
        else:
            updated_means,updated_stdevs = init_means.copy(), init_stdevs.copy()
            reduced_gaussians = False
        updated_means.append(round(np.random.choice(np.arange(-0.5, 1, 0.1)), 1))
        updated_stdevs.append(round(np.random.choice(np.arange(0.3, 0.9, 0.1)), 1))
        num_comp = len(updated_means) + 1
        updated_weights = [1/num_comp] * num_comp
        start_parameters.append([updated_means, updated_stdevs, init_lambd, updated_weights])
        bic, new_means, new_stdevs, new_lambd, new_weights, convergence = EM_step(num_comp,deconvoluted_data,updated_means, updated_stdevs, init_lambd, updated_weights,max_EM_iterations=max_EM_iterations,max_num_comp = 5, reduced_gaussians_flag=reduced_gaussians)
        #if convergence: logging.info('The EM algorithm has reached convergence at iteration {}'.format(i+1))
        #else: logging.info("The EM algorithm hasn't reached convergence at iteration {}".format(i+1))
        bic_from_same_num_comp.append(bic)
        final_parameters.append([new_means, new_stdevs, new_lambd, new_weights])
    updated_means, updated_stdevs, init_lambd, updated_weights = start_parameters[np.argmin(bic_from_same_num_comp)]
    all_models_init_parameters['Model2'] = [updated_means, updated_stdevs, init_lambd, updated_weights]
    plot_init_model(axes[1,0], axes[1,1],updated_means, updated_stdevs, init_lambd, updated_weights)
    final_means, final_stdevs, final_lambd, final_weights = final_parameters[np.argmin(bic_from_same_num_comp)]
    all_models_fitted_parameters['Model2'] = [final_means, final_stdevs, final_lambd, final_weights]
    plot_fitted_model(axes[1,0], axes[1,1],final_means, final_stdevs, final_lambd, final_weights)
    bic_dict['Model2'] = min(bic_from_same_num_comp)
    #for m,s in zip(final_means,final_stdevs): logging.info('The optimized means and stds is {:.2f} {:.2f}'.format(np.exp(m),s))
    logging.info('BIC of Model2 : {:.2f}'.format(min(bic_from_same_num_comp)))
    if na:
        fig.savefig(os.path.join(outdir, "elmm_{}_models_data_driven_node_averaged.pdf".format(sp)),bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "elmm_{}_models_data_driven_node_averaged.svg".format(sp)),bbox_inches="tight")
    else:
        fig.savefig(os.path.join(outdir, "elmm_{}_models_data_driven_weighted.pdf".format(sp)),bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "elmm_{}_models_data_driven_weighted.svg".format(sp)),bbox_inches="tight")
    plt.close()
    elmm_random(ks_or,w,ks,num_EM_initializations,deconvoluted_data,max_EM_iterations,outdir,sp,all_models_init_parameters,all_models_fitted_parameters,bic_dict,na=na)
    logging.info("Models are evaluated according to BIC scores")
    model_bic = [(k,v) for k,v in bic_dict.items()]
    modelist, bic_list = [k for k,v in model_bic],[v for k,v in model_bic]
    best_model_id = modelist[np.argmin(bic_list)]
    logging.info("The best fitted model via BIC is {}".format(best_model_id))
    bic_info(modelist, bic_list)
    model_bic_ordered = [(k,v) for k,v in sorted(bic_dict.items(), key=lambda y:y[0])]
    model_ordered, bic_ordered = [k for k,v in model_bic_ordered],[v for k,v in model_bic_ordered]
    plot_bic(model_ordered,bic_ordered,outdir,sp,na=na)
    plot_final(ks_or,deconvoluted_data,w,sp,outdir,best_model_id,all_models_fitted_parameters,na=na)

def plot_final(ks_or,deconvoluted_data,w,sp,outdir,best_model_id,all_models_fitted_parameters,na=False):
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 7.0))
    fig.suptitle("$K_\mathregular{S}$" + " distribution for {}".format(sp))
    if na: hist = ax.hist(ks_or,weights=w,bins=np.linspace(0, 50, num=51,dtype=int)/10,color='gray',label="Whole-paranome (node averaged)",rwidth=0.8)
    else: hist = ax.hist(ks_or,weights=w,bins=np.linspace(0, 50, num=51,dtype=int)/10,color='gray',label="Whole-paranome (weighted)",rwidth=0.8)
    ax.set_ylim(0, max(hist[0]) * 1.25)
    final_means, final_stdevs, final_lambd, final_weights = all_models_fitted_parameters[best_model_id]
    bin_width = 0.1
    scaling = 0.1 * len(deconvoluted_data[deconvoluted_data <= 5])
    plot_fitted_model(ax,None,final_means,final_stdevs,final_lambd,final_weights,scaling=scaling,final=True)
    if na: ax.set_ylabel("Number of retained duplicates (node averaged)",fontsize='x-large')
    else: ax.set_ylabel("Number of retained duplicates (weighted)",fontsize='x-large')
    ax.set_xlabel("$K_\mathregular{S}$",fontsize='x-large')
    sns.despine(offset=10)
    ax.set_xlim(0, 5)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=90, verticalalignment='center')
    plt.tight_layout()
    if na:
        fig.savefig(os.path.join(outdir, "elmm_{}_best_models_node_averaged.svg".format(sp)))
        fig.savefig(os.path.join(outdir, "elmm_{}_best_models_node_averaged.pdf".format(sp)))
    else:
        fig.savefig(os.path.join(outdir, "elmm_{}_best_models_weighted.svg".format(sp)))
        fig.savefig(os.path.join(outdir, "elmm_{}_best_models_weighted.pdf".format(sp)))
    plt.close()

def bic_info(modelist, bic_list):
    b_min = min(bic_list)
    logging.info("Rules-of-thumb (Kass & Raftery, 1995) evaluates the outperformance of the BIC-best model over remaining:")
    for m, b in zip(modelist, bic_list):
        if b != b_min:
            ABS = abs(b_min - b)
            if ABS < 2: logging.info("Such outperformance is not worth more than a bare mention for {}".format(m))
            elif 2<=ABS<6: logging.info("Such outperformance is positively evidenced for {}".format(m))
            elif 6<=ABS<=10: logging.info("Such outperformance is strongly evidenced for {}".format(m))
            else: logging.info("Such outperformance is very strongly evidenced for {}".format(m))

def plot_bic(model,bic,outdir,sp,na=False):
    fig, axes = plt.subplots(figsize=(6, 3))
    axes.plot(np.arange(1, len(bic)+1), bic, color='k', marker='o')
    axes.set_xticks(list(range(1, len(bic) + 1)))
    axes.set_xticklabels(model)
    axes.grid(ls=":")
    axes.set_ylabel("BIC")
    axes.set_xlabel("Model")
    fig.tight_layout()
    if na:
        fig.savefig(os.path.join(outdir, "elmm_BIC_{}_node_averaged.svg".format(sp)))
        fig.savefig(os.path.join(outdir, "elmm_BIC_{}_node_averaged.pdf".format(sp)))
    else:
        fig.savefig(os.path.join(outdir, "elmm_BIC_{}_weighted.svg".format(sp)))
        fig.savefig(os.path.join(outdir, "elmm_BIC_{}_weighted.pdf".format(sp)))
    plt.close()

def elmm_random(ks_or,w,ks,num_EM_initializations,deconvoluted_data,max_EM_iterations,outdir,sp,all_models_init_parameters,all_models_fitted_parameters,bic_dict,na):
    min_num_comp,max_num_comp = 2,5
    fig, axes = plt.subplots(nrows=((5-2+1)), ncols=2, figsize=(20, 8*(5-2+1)), sharey="row")
    fig.suptitle("Exponential-Lognormal mixture model on {} ".format(sp)+"$K_\mathregular{S}$ " + "paranome\n\nInitialized randomly",fontsize='x-large')
    num_comp_list = np.arange(min_num_comp, max_num_comp + 1)
    axes_ids,model_ids = num_comp_list-min_num_comp,num_comp_list+1
    for num_comp, ax_id, model_id in zip(num_comp_list, axes_ids, model_ids):
        logging.info("Performing EM algorithm from random initialization with {0} components (Model{1})".format(num_comp,model_id))
        ax0, ax1 = axes[ax_id][0], axes[ax_id][1]
        if na: ax0.hist(ks_or, bins=np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='gray',rwidth=0.8, label='Whole-paranome (node averaged)',density=True)
        else: ax0.hist(ks_or, bins=np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='gray',rwidth=0.8, label='Whole-paranome (weighted)',density=True)
        ax1.hist(ks,weights=w,bins=np.arange(-10, 10 + 0.1, 0.1),density=True,color='gray',alpha=0.5,label='Log-transformed paranome',rwidth=0.8)
        ax0.set_title("Model {}".format(model_id))
        ax0.set_xlim(0, 5)
        ax1.set_xlim(-5, 2)
        if model_id == model_ids[-1]: ax0.set_xlabel("$K_\mathregular{S}$")
        if model_id == model_ids[-1]: ax1.set_xlabel("ln $K_\mathregular{S}$")
        bic_from_same_num_comp = []
        start_parameters, final_parameters = [], []
        for i in range(num_EM_initializations):
            init_means, init_stdevs, init_weights = [], [], [1/num_comp] * num_comp
            init_lambd = round(np.random.choice(np.arange(0.2, 1, 0.1)), 2)
            for j in range(num_comp-2):
                #init_means.append(round(np.random.choice(np.arange(-0.5, 1, 0.1)),1))
                init_means.append(round(np.random.choice(np.arange(-0.5, 1, 0.01)),1))
                #init_stdevs.append(round(np.random.choice(np.arange(0.3, 0.9, 0.1)),1))
                init_stdevs.append(round(np.random.choice(np.arange(0.3, 0.9, 0.01)),1))
            init_means.append(np.log(5))
            init_stdevs.append(0.3)
            start_parameters.append([init_means, init_stdevs, init_lambd, init_weights])
            bic, new_means, new_stdevs, new_lambd, new_weights, convergence = EM_step(num_comp,deconvoluted_data,init_means, init_stdevs, init_lambd, init_weights,max_EM_iterations=max_EM_iterations,max_num_comp = 5)
            #if convergence: logging.info('The EM algorithm has reached convergence')
            #else: logging.info("The EM algorithm hasn't reached convergence")
            bic_from_same_num_comp.append(bic)
            final_parameters.append([new_means, new_stdevs, new_lambd, new_weights])
        init_means, init_stdevs, init_lambd, init_weights = start_parameters[np.argmin(bic_from_same_num_comp)]
        all_models_init_parameters["Model{}".format(model_id)] = [init_means, init_stdevs, init_lambd, init_weights]
        plot_init_model(ax0, ax1, init_means, init_stdevs, init_lambd, init_weights)
        final_means, final_stdevs, final_lambd, final_weights = final_parameters[np.argmin(bic_from_same_num_comp)]
        all_models_fitted_parameters["Model{}".format(model_id)] = [final_means, final_stdevs, final_lambd, final_weights]
        plot_fitted_model(ax0, ax1, final_means, final_stdevs, final_lambd, final_weights)
        bic_dict["Model{}".format(model_id)] = min(bic_from_same_num_comp)
        #for m,s in zip(final_means,final_stdevs): logging.info('The optimized means and stds is {:.2f} {:.2f}'.format(np.exp(m),s))
        logging.info('BIC of Model{} : {:.2f}'.format(model_id,min(bic_from_same_num_comp)))
    if na:
        fig.savefig(os.path.join(outdir, "elmm_{}_models_random_node_averaged.pdf".format(sp)),bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "elmm_{}_models_random_node_averaged.svg".format(sp)),bbox_inches="tight")
    else:
        fig.savefig(os.path.join(outdir, "elmm_{}_models_random_weighted.pdf".format(sp)),bbox_inches="tight")
        fig.savefig(os.path.join(outdir, "elmm_{}_models_random_weighted.svg".format(sp)),bbox_inches="tight")
    plt.close()

def plot_fitted_model(ax1,ax2,means,stds,lambd,weights,scaling=1, final=False):
    x_points = np.linspace(-5, 5, int((5 + 5) *100))
    x_points_strictly_positive = np.linspace(0, 5, int(5 * 100))
    total_pdf_log = 0
    total_pdf = weights[0] * stats.expon.pdf(x_points_strictly_positive, scale=1/lambd)
    #colors = cm.rainbow(np.linspace(0, 1, len(stds)))
    ax1.plot(x_points_strictly_positive,scaling*weights[0]*stats.expon.pdf(x_points_strictly_positive, scale=1/lambd), c='g', ls='-', lw=1.5, alpha=0.8, label='Exponential optimized')
    lognormal_peaks = {i:round(np.exp(means[i] - pow(stds[i], 2)), 2) for i in range(len(stds))}
    #The formula of mode of the log-normal distribution
    lognormals_sorted_by_peak = [k for k,v in sorted(lognormal_peaks.items(), key=lambda y:y[1])]
    letter_dict = dict(zip(lognormals_sorted_by_peak, [ "a", "b", "c", "d", "e", "f", "g"][:len(stds)]))
    colors = ["b", "r", "c", "m", "k"][:len(stds)-1] + ["y"]
    for comp, color in zip(lognormals_sorted_by_peak, colors):
        ax1.plot(x_points_strictly_positive,scaling*weights[comp+1]*stats.lognorm.pdf(x_points_strictly_positive, scale=np.exp(means[comp]),s=stds[comp]), c=color, ls='-', lw=1.5, alpha=0.8, label=f'Lognormal {letter_dict[comp]} optimized (mode {lognormal_peaks[comp]})')
        if ax2!=None: ax2.plot(x_points,weights[comp+1]*stats.norm.pdf(x_points,means[comp],stds[comp]),c=color,ls='-',lw=1.5,alpha=0.8, label=f'Norm {letter_dict[comp]} optimized')
        total_pdf_log = total_pdf_log + weights[comp+1]*stats.norm.pdf(x_points,means[comp],stds[comp])
        total_pdf = total_pdf + weights[comp+1]*stats.lognorm.pdf(x_points_strictly_positive,scale=np.exp(means[comp]),s=stds[comp])
    ax1.plot(x_points_strictly_positive, scaling*total_pdf, "k-", lw=1.5, label=f'Exp-lognormal mixture model')
    ax1.legend(loc=1,frameon=False)
    if final: ax1.legend(loc=1,bbox_to_anchor=(0.95, 0.90),frameon=False)
    if ax2!=None: ax2.plot(x_points, total_pdf_log, "k-", lw=1.5, label=f'Total PDF')
    if ax2!=None: ax2.legend(loc=2,frameon=False)

def get_deconvoluted_data(ks,w):
    tail_length = 0.5
    bin_list_for_deconvoluted_data = np.arange(0, 5 + 0.01, 0.01)
    hist_data = np.histogram(ks, bins=bin_list_for_deconvoluted_data, weights=w)
    d = np.array([])
    for i in range(len(hist_data[0])):
        midpoint = round((hist_data[1][i+1] - 0.01 / 2), 3)
        d = np.append(d,np.repeat(midpoint,round(hist_data[0][i])))
    max_ks_tail = 5 + tail_length
    for i in np.arange(5+0.01, max_ks_tail + 0.01, 0.01):
        midpoint = round((i - 0.01 / 2), 3)
        d = np.append(d,np.repeat(midpoint,round(np.mean(hist_data[0][-51:]))))
    return d

def e_step(num_comp, ks, means, stdevs, weights, lambd):
    products = []
    #print(ks.shape)
    for k in range(num_comp):
        # at the very beginning, the weight are randomly given as equal
        #print((k,lambd,np.exp(means[k-1]),stdevs[k-1],weights[k]))
        if k == 0: prod = weights[k] * stats.expon.pdf(ks, scale=1/lambd)
        else:
            #print(len(ks))
            #print(type(ks))
            #print(stats.lognorm.pdf(ks, scale=np.exp(means[k-1]), s=stdevs[k-1]))
            prod = weights[k] * stats.lognorm.pdf(ks, scale=np.exp(means[k-1]), s=stdevs[k-1])
        # The addition of all pdf with weight
        products.append(prod)
    sum_comp_perpoint = sum(products)
    log_sum_comp_perpoint = np.log(sum_comp_perpoint)
    fit_loglikelihood = sum(log_sum_comp_perpoint)
    posteriors = [products[i] / sum_comp_perpoint for i in range(num_comp)]
    return fit_loglikelihood, posteriors

def m_step(num_comp, ks, posteriors):
    new_lambda = sum(posteriors[0]) / sum(posteriors[0] * ks)
    points_per_k = [sum(posteriors[i]) for i in range(num_comp)]
    # here the new weights is overall weight instead of weight per datapoint
    new_weights = [points_per_k[i]/len(ks) for i in range(num_comp)]
    new_means = [sum(posteriors[i+1] * np.log(ks)) / points_per_k[i+1] for i in range(num_comp-1)]
    new_stdevs = [np.sqrt(sum(posteriors[i+1]*pow(np.log(ks)-new_means[i],2))/points_per_k[i+1]) for i in range(num_comp-1)]
    return new_means, new_stdevs, new_weights, new_lambda

def EM_step(num_comp,data,means,stds,lambd,weights,max_EM_iterations=200,max_num_comp = 5, reduced_gaussians_flag=None):
    #data = np.array(deconvoluted_data).reshape(-1, 1)
    convergence,ks = False,data[data>0]
    curr_loglik, posteriors = e_step(num_comp, ks, means, stds, weights, lambd)
    new_means, new_stdevs, new_weights, new_lambd = m_step(num_comp, ks, posteriors)
    for i in range(max_EM_iterations-1):
        new_loglik, posteriors = e_step(num_comp, ks, new_means, new_stdevs, new_weights, new_lambd)
        new_means, new_stdevs, new_weights, new_lambd = m_step(num_comp, ks, posteriors)
        if abs(curr_loglik - new_loglik) <= 1e-6:
            convergence = True
            break
        else: curr_loglik = new_loglik
    bic = -2 * new_loglik + num_comp * np.log(len(ks))
    #if convergence: logging.info('The EM algorithm has reached convergence')
    #else: logging.info("The EM algorithm hasn't reached convergence")
    #logging.info("The Log-likelihood and BIC of Model 1 are {:.3f} and {:.3f}".format(new_loglik,bic))
    return bic, new_means, new_stdevs, new_lambd, new_weights, convergence

def plot_init_model(ax1,ax2,means,stds,lambd,weights):
    x = np.linspace(-5, 10, 15*100)
    ax1.plot(x, weights[0] * stats.expon.pdf(x, scale=1/lambd),'g:', lw=2, alpha=0.5,label='Exponential initiated')
    styles = ["b:", "r:", "c:", "m:", "k:"][:len(means)-1] + ["y:"]
    lognormal_peaks = {i:round(np.exp(means[i] - pow(stds[i], 2)), 2) for i in range(len(stds))}
    lognormals_sorted_by_peak = [k for k,v in sorted(lognormal_peaks.items(), key=lambda y:y[1])]
    letter_dict = dict(zip(lognormals_sorted_by_peak, [ "a", "b", "c", "d", "e", "f", "g"][:len(stds)]))
    colors = ["b", "r", "c", "m", "k"][:len(stds)-1] + ["y"]
    #for i, st in zip(range(len(stds)), styles):
    for i, st in zip(lognormals_sorted_by_peak,styles):
        ax1.plot(x, weights[i+1] * stats.lognorm.pdf(x, scale=np.exp(means[i]), s=stds[i]), st, lw=1.5, alpha=0.4,label=f'Lognormal {letter_dict[i]} initiated (mode {lognormal_peaks[i]})')
        ax2.plot(x, weights[i+1] * stats.norm.pdf(x, means[i], stds[i]),st,lw=1.5, alpha=0.4,label=f'Norm {letter_dict[i]} initiated')

def find_peak_init_parameters(spl_x,spl_y,sp,outdir,peak_threshold=0.1,na=False, guide=None,rel_height=0.4):
    peaks, properties = signal.find_peaks(spl_y)
    #prominences = properties["prominences"]
    prominences = signal.peak_prominences(spl_y, peaks)[0]
    fig, axes = plt.subplots(nrows=len(peaks)+1, ncols=2, figsize=(14, 7*(len(peaks)+1)), sharey=True)
    fig.suptitle("Peak detection in log-transformed $K_\mathrm{S}$ distribution of " + "{}".format(sp), y = 0.92,fontsize=20)
    #plt.title("Peak detection in log-transformed $K_\mathrm{S}$ distribution of " + "{}".format(sp))
    for w in range(len(peaks)+1): axes[w][0].set_ylabel("Density of retained duplicates")
    for w in [0,1]: axes[len(peaks)][w].set_xlabel("ln $K_\mathrm{S}$")
    for ax in axes:
        ax[0].plot(spl_x, spl_y, color="gray", linewidth=1)
        ax[1].plot(spl_x, spl_y, color="gray", linewidth=1)
    for i in [0,1]: axes[0,i].scatter(spl_x[peaks], spl_y[peaks], marker="x", c="b", label="all potential peaks")
    axes[0,0].vlines(x=spl_x[peaks], ymin=spl_y[peaks]-prominences, ymax=spl_y[peaks], color="b", label="prominences")
    axes[0,1].vlines(x=spl_x[peaks], ymin=spl_y[peaks]-prominences, ymax=spl_y[peaks], color="b", label="prominences")
    prominences_refed_R1,width_refed_R1,prominences_refed_L1,width_refed_L1 = [],[],[],[]
    for i in range(len(peaks)):
        peak_index = peaks[i]
        #Note here the y value is exactly reflected without changing
        spl_peak_refl_y = np.concatenate((np.flip(spl_y[peak_index+1:]), spl_y[peak_index:]))
        spl_peak_refl_x = np.concatenate((np.flip(spl_x[peak_index+1:] * -1 + 2 * spl_x[peak_index]), spl_x[peak_index:]))
        axes[i+1,0].plot(spl_peak_refl_x, spl_peak_refl_y,color='g',label='peak {} right-reflected'.format(i+1),linewidth=1)
        current_peak_index = int((len(spl_peak_refl_x)-1)/2)
        #current_peak_index = np.floor(len(spl_peak_refl_x)/2)
        new_prominences = signal.peak_prominences(spl_peak_refl_y,[current_peak_index])[0][0]
        new_width,new_height,left_ips,right_ips = signal.peak_widths(spl_peak_refl_y, [current_peak_index], rel_height=rel_height)
        if new_width[0] > 150: new_width[0] = 150
        prominences_refed_R1.append(new_prominences)
        width_refed_R1.append(new_width[0])
        c = "r" if new_prominences >= peak_threshold else 'gray'
        axes[i+1,0].vlines(x=spl_x[peak_index], ymin=spl_y[peak_index] - new_prominences, ymax = spl_y[peak_index], color=c, label='prominence {:.2f}'.format(new_prominences))
        w = new_width[0]/2/100
        if new_prominences >= peak_threshold: axes[i+1,0].hlines(y=new_height[0], xmin=spl_x[peak_index], xmax=spl_x[peak_index]+w, linestyles="-", color="darkred", lw=1, label='width {:.2f}'.format(w))
        axes[i+1,0].legend(frameon=False)
        spl_peak_refl_y_L = np.concatenate((spl_y[:peak_index+1], np.flip(spl_y[:peak_index])))
        spl_peak_refl_x_L = np.concatenate((spl_x[:peak_index+1], np.flip(spl_x[:peak_index]) * -1 + 2*spl_x[peak_index]))
        axes[i+1,1].plot(spl_peak_refl_x_L, spl_peak_refl_y_L,color='g',label='peak {} left-reflected'.format(i+1),linewidth=1)
        #current_peak_index = np.floor(len(spl_peak_refl_x_L)/2)
        current_peak_index = int((len(spl_peak_refl_x_L)-1)/2)
        new_prominences = signal.peak_prominences(spl_peak_refl_y_L,[current_peak_index])[0][0]
        new_width,new_height,left_ips,right_ips = signal.peak_widths(spl_peak_refl_y_L, [current_peak_index], rel_height=rel_height)
        if new_width[0] > 150: new_width[0] = 150
        prominences_refed_L1.append(new_prominences)
        width_refed_L1.append(new_width[0])
        c = "r" if new_prominences >= peak_threshold else 'gray'
        axes[i+1,1].vlines(x=spl_x[peak_index], ymin=spl_y[peak_index] - new_prominences, ymax = spl_y[peak_index], color=c, label='prominence {:.2f}'.format(new_prominences))
        w = new_width[0]/2/100
        if new_prominences >= peak_threshold: axes[i+1,1].hlines(y=new_height[0], xmin=spl_x[peak_index], xmax=spl_x[peak_index]+w, linestyles="-", color="darkred", lw=1, label='width {:.2f}'.format(w))
        axes[i+1,1].legend(frameon=False)
    good_peaks_R1,good_peaks_L1 = [i>=peak_threshold for i in prominences_refed_R1],[i>=peak_threshold for i in prominences_refed_L1]
    axes[0,0].set_title('Reflection L <-- R', fontsize=17)
    axes[0,0].legend(frameon=False)
    axes[0,1].set_title('Reflection L --> R', fontsize=17)
    axes[0,1].legend(frameon=False)
    original_y_lim = axes[0][0].get_ylim()[1]
    for ax in axes.flatten(): ax.set_ylim(0, original_y_lim * 1.2)
    #fig.tight_layout()
    if guide == None:
        if na:
            fig.savefig(os.path.join(outdir, "{}_peak_detection_node_averaged.pdf".format(sp)))
            fig.savefig(os.path.join(outdir, "{}_peak_detection_node_averaged.svg".format(sp)))
        else:
            fig.savefig(os.path.join(outdir, "{}_peak_detection_weighted.pdf".format(sp)))
            fig.savefig(os.path.join(outdir, "{}_peak_detection_weighted.svg".format(sp)))
    else:
        fig.savefig(os.path.join(outdir, "{}_peak_detection_guided_by_{}.pdf".format(sp,guide)))
        fig.savefig(os.path.join(outdir, "{}_peak_detection_guided_by_{}.svg".format(sp,guide)))
    plt.close()
    good_prominences,init_means,init_stdevs = [],[],[]
    for i in range(len(peaks)):
        if good_peaks_R1[i] or good_peaks_L1[i]:
            init_means.append(spl_x[peaks[i]])
            best = np.argmax((prominences_refed_R1[i], prominences_refed_L1[i]))
            good_prominences.append(max([prominences_refed_R1[i], prominences_refed_L1[i]]))
            width = [width_refed_R1[i], width_refed_L1[i]][best]
            init_stdevs.append(width/2/100)
    return init_means, init_stdevs,good_prominences

def default_plot(
        *args, 
        alphas=None,
        colors=None,
        weighted=True, 
        title="",
        ylabel="duplication events",
        **kwargs):
    """
    Make a figure of node-weighted histograms for multiple distributions and
    variables. Returns the figure object.
    
    !!! note: Assumes the data frames are filtered as desired. 
    """
    ndists = len(args)
    alphas = alphas or list(np.linspace(0.2, 1, ndists))
    colors = colors or ['black'] * ndists 
    # assemble panels
    keys = ["dS", "dS", "dN", "dN/dS"]
    np.seterr(divide='ignore')
    funs = [lambda x: x, np.log10, np.log10, np.log10]
    fig, axs = plt.subplots(2, 2)
    for (c, a, dist) in zip(colors, alphas, args):
        for ax, k, f in zip(axs.flatten(), keys, funs):
            w = node_weights(dist)
            x = f(dist[k])
            y = x[np.isfinite(x)]
            w = w[np.isfinite(x)]
            if funs[0] == f: ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color=c, alpha=a, rwidth=0.8)
            else: ax.hist(y, weights=w, color=c, alpha=a, rwidth=0.8,**kwargs)
            xlabel = _labels[k]
            if f == np.log10:
                xlabel = "$\log_{10}" + xlabel[1:-1] + "$"
            ax.set_xlabel(xlabel)
    axs[0,0].set_ylabel(ylabel)
    axs[1,0].set_ylabel(ylabel)
    axs[0,0].set_xticks([0,1,2,3,4,5])
    # finalize plot
    sns.despine(offset=1)
    fig.suptitle(title, x=0.125, y=0.9, ha="left", va="top")
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)  # prevent suptitle from overlapping
    return fig

def syntenic_depth_plot(segprofile):
    cols = segprofile.columns
    n = len(cols)
    fig, axs = plt.subplots(1, int(n + n*(n-1)/2))
    if n == 1:
        axs = [axs]  # HACK
    k = 0
    for i in range(n):
        for j in range(i, n):
            pairs, counts = dupratios(segprofile[cols[i]], segprofile[cols[j]])
            ax = axs[k]
            ax.barh(np.arange(len(pairs)), counts, color="k", alpha=0.2)
            ax.set_yticks(np.arange(len(pairs)))
            ax.set_yticklabels(["{}:{}".format(int(x[0]), int(x[1])) for x in pairs])
            #ax.set_title("${}$:${}$".format(cols[i], cols[j]), fontsize=9)
            ax.set_ylabel("{} : {}".format(cols[i], cols[j]))
            k += 1
    for ax in axs:
        ymn, ymx = ax.get_ylim()
        ax.set_ylim(-0.5, ymx)
        #ax.set_xlabel("# segments")
    #axs[0].set_ylabel("A:B ratio")
    fig.suptitle('Collinear ratio', x=0.5, y=1.02, ha='center', va='top')
    plt.figtext(0.5, 0.02, 'Number of segments', ha='center', va='top')
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig


def dupratios(col1, col2, by="first"):
    d = {}
    for pair in zip(col1,col2):
        if pair not in d:
            d[pair] = 0
        d[pair] += 1
    if by == "first":
        keyfun = lambda x: x
    elif by == "ratio":
        lambda x: x[0]/(1+x[1])
    elif by == "second":
        keyfun = lambda x: x[1]
    kys = sorted(d, key=keyfun)
    return kys, [d[k] for k in kys]

def sankey_plot_self(sp, df, minlen,outdir, seg, multi):
    lens = df.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1 = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1.len.max() * 0.1
    df1 = df1.loc[df1.len > minlen]
    seg = seg.loc[seg['genome']==sp].copy()
    segs = list(seg.groupby('list'))
    scaflabels = list(map(lambda x: x[0],segs))
    patchescoordif = list(map(lambda x: list(x[1].loc[:,'first']),segs))
    patchescoordil = list(map(lambda x: list(x[1].loc[:,'last']),segs))
    patchessegid = list(map(lambda x: list(x[1].index),segs))
    gene_start = {gene:start for gene,start in zip(df.index,list(df['start']))}
    multi = multi.loc[:,['id','level']].copy()
    seg_with_level = seg.merge(multi,left_on='multiplicon', right_on='id').drop(columns='id')
    segs_levels = {seglabel:level for seglabel,level in zip(list(seg_with_level.index+1),list(seg_with_level['level']))}
    highest_level = max(segs_levels.values())
    plothlines(highest_level,segs_levels,sp,gene_start,df1.len,df1.index,outdir,scaflabels,patchescoordif,patchescoordil,patchessegid)

def AK_plot(spx,dfx,ancestor,backbone=False,colortable=None,seg=None,maxsize=0,minlen=0,outdir=None):
    lens = dfx.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1x = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1x.len.max() * 0.1
    df1x = df1x.loc[df1x.len > minlen]
    if backbone:
        color_scaff = plot_ancestor(spx,df1x.len,df1x.index,outdir)
        return color_scaff
    elif spx != ancestor:
        seg.loc[:,"segment"] = seg.index
        #seg_unfilterded = seg.loc[seg['genome']==spx].copy()
        segs_info = seg.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
        profile = segs_info.unstack(level=-1).fillna(0)
        profile_good = profile.loc[(profile[spx]>0) & (profile[ancestor]>0)]
        if len(profile_good) == 0: logging.info('No multiplicon contained both genome of {0} and {1}'.format(spx,spy))
        else:
            seg_good = seg.merge(profile_good.reset_index(),on='multiplicon')
            gl_tocolor = seg_good.loc[seg_good['genome']==spx,'list']
            for gl in gl_tocolor: print(gl)
        plot_descendant(spx,df1x.len,df1x.index,outdir,colortable,gl_tocolor,seg_good.loc[:,['genome','list','first','last']])

def plot_descendant(sp,scafflength,scafflabel,outdir,colortable,gl_tocolor,segs_tocolor):
    scafflength_normalized = [i/max(scafflength) for i in scafflength]
    fname = os.path.join(outdir, "{}_descendant_karyotype.png".format(sp))
    fnamep = os.path.join(outdir, "{}_descendant_karyotype.pdf".format(sp))
    fnames = os.path.join(outdir, "{}_descendant_karyotype.svg".format(sp))
    fig, ax = plt.subplots(1, 1, figsize=(10,20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, (3*len(scafflength))+1)
    yticks = []
    yticklabels = []
    for i,le,la in zip(range(len(scafflength)),scafflength_normalized,scafflabel):
        yticks.append((3*i)+1.25)
        yticklabels.append(la)
        ax.add_patch(Rectangle((0, (3*i)+1),le,0.5,fc ='gray',ec ='none',lw = 1, zorder=0, alpha=0.3))
        #verts = [(0,(3*i)+1),(le,(3*i)+1+0.5)]
        #codes = [Path.MOVETO,Path.LINETO]
        #path = Path(verts, codes)
        #ax.add_patch(patches.PathPatch(path,fc='none',ec ='black',lw = 1,zorder=1))
        for f,l,segid in zip(patchescoordif[idc],patchescoordil[idc],patchessegid[idc]):
            left = gene_start[f]/max(scafflength)
            right = gene_start[l]/max(scafflength)
            ple = right - left
    y = lambda x : ["{:.2f}".format(i) for i in x]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(y(ax.get_xticks()*max(scafflength)/1e6))
    ax.xaxis.label.set_fontsize(18)
    ax.set_xlabel("{} (Mb)".format(sp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fnamep)
    fig.savefig(fnames)

def plot_ancestor(sp,scafflength,scafflabel,outdir):
    scafflength_normalized = [i/max(scafflength) for i in scafflength]
    fname = os.path.join(outdir, "{}_ancestor_karyotype.png".format(sp))
    fnamep = os.path.join(outdir, "{}_ancestor_karyotype.pdf".format(sp))
    fnames = os.path.join(outdir, "{}_ancestor_karyotype.svg".format(sp))
    fig, ax = plt.subplots(1, 1, figsize=(10,20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, (3*len(scafflength))+1)
    yticks = []
    yticklabels = []
    colors = cm.viridis(np.linspace(0, 1, len(scafflength)))
    color_scaff = {}
    for i,le,la in zip(range(len(scafflength)),scafflength_normalized,scafflabel):
        yticks.append((3*i)+1.25)
        yticklabels.append(la)
        ax.add_patch(Rectangle((0, (3*i)+1),le,0.5,fc =colors[i],ec ='none',lw = 1, zorder=0, alpha=0.3))
        color_scaff[la]=colors[i]
        verts = [(0,(3*i)+1),(le,(3*i)+1+0.5)]
        codes = [Path.MOVETO,Path.LINETO]
        path = Path(verts, codes)
        ax.add_patch(patches.PathPatch(path,fc='none',ec ='black',lw = 1,zorder=1))
    y = lambda x : ["{:.2f}".format(i) for i in x]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(y(ax.get_xticks()*max(scafflength)/1e6))
    ax.xaxis.label.set_fontsize(18)
    ax.set_xlabel("{} (Mb)".format(sp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fnamep)
    fig.savefig(fnames)
    return color_scaff

def filter_by_dfy(seg,dfy,minlen,spy):
    lens = dfy.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1y = pd.DataFrame(lens).sort_values("len", ascending=False)
    scfa_len = {i:df1y.loc[i,'len'] for i in df1y.index}
    if minlen < 0: minlen = df1y.len.max() * 0.1
    #df1y = df1y.loc[df1y.len > minlen]
    #df1y['genome']=spy
    #df1y = df1y.reset_index().rename(columns={"scaffold": "list"})
    #seg = seg.merge(df1y,on=['genome'])
    rm_indices = []
    for i in seg.index:
        if seg.loc[i,'genome'] == spy:
            scfa = seg.loc[i,'list']
            if scfa_len[scfa] <= minlen: rm_indices.append(i)
    seg = seg.drop(rm_indices)
    return seg

def sankey_plot(spx, dfx, spy, dfy, minseglen, minlen, outdir, seg):
    lens = dfx.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1x = pd.DataFrame(lens).sort_values("len", ascending=False)
    seg.loc[:,"segment"] = seg.index
    #seg = filter_by_dfy(seg,dfy,minlen,spy) #This filter step makes singon segments on level plot
    seg_unfilterded = seg.loc[seg['genome']==spx].copy()
    segs_info = seg.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    profile = segs_info.unstack(level=-1).fillna(0)
    if spy not in profile.columns:
        logging.info('No collinear segments were found involving genome of {}'.format(spy))
    elif spx not in profile.columns:
        logging.info('No collinear segments were found involving genome of {}'.format(spx))
    else:
        #if spx != spy: multi_goodinuse = profile.loc[profile[spy]>0,[spy,spx]].copy()
        if spx != spy: multi_goodinuse = profile.loc[:,[spy,spx]].copy() # here I didn't require the level of spy > 0
        else: multi_goodinuse = profile.loc[profile[spy]>0,[spy]].copy()
        seg_filterded = seg_unfilterded.set_index('multiplicon').merge(multi_goodinuse,left_index=True, right_index=True).drop(columns=spy)
        if len(seg_filterded) == 0:
            logging.info('No multiplicon contained both genome of {0} and {1}'.format(spx,spy))
        else:
            segs_levels_spx = None
            spy_multl_level = {m:int(l) for m,l in zip(multi_goodinuse.index,list(multi_goodinuse[spy]))}
            segs_multi_good = {s:m for s,m in zip(list(seg_filterded['segment']),list(seg_filterded.index))}
            segs_levels = {s:spy_multl_level[m] for s,m in segs_multi_good.items()}
            segs = list(seg_filterded.groupby('list'))
            scaflabels = list(map(lambda x: x[0],segs))
            patchescoordif = list(map(lambda x: list(x[1].loc[:,'first']),segs))
            patchescoordil = list(map(lambda x: list(x[1].loc[:,'last']),segs))
            patchessegid = list(map(lambda x: list(x[1].loc[:,'segment']),segs))
            gene_start = {gene:start for gene,start in zip(dfx.index,list(dfx['start']))}
            highest_level = max(segs_levels.values())
            if spx != spy:
                spx_multl_level = {m:int(l) for m,l in zip(multi_goodinuse.index,list(multi_goodinuse[spx]))}
                segs_levels_spx = {s:spx_multl_level[m] for s,m in segs_multi_good.items()}
                highest_level = max([ly+segs_levels_spx[seg] for seg,ly in segs_levels.items()])
            plothlines(minseglen,highest_level,segs_levels,spx,gene_start,df1x.len,df1x.index,outdir,scaflabels,patchescoordif,patchescoordil,patchessegid,spy = spy,spx_level = segs_levels_spx)

def plothlines(minseglen,highest_level,segs_levels,sp,gene_start,scafflength,scafflabel,outdir,patchedscaflabels,patchescoordif,patchescoordil,patchessegid,spy = None, spx_level = None):
    scafflength_normalized = [i/max(scafflength) for i in scafflength]
    fname = os.path.join(outdir, "{}_multiplicons_level.png".format(sp))
    if spy != None:
        fname = os.path.join(outdir, "{0}_{1}_multiplicons_level.png".format(sp,spy))
        fnamep = os.path.join(outdir, "{0}_{1}_multiplicons_level.pdf".format(sp,spy))
        fnames = os.path.join(outdir, "{0}_{1}_multiplicons_level.svg".format(sp,spy))
    fig, ax = plt.subplots(1, 1, figsize=(10,20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, (3*len(scafflength))+1)
    common = list(set(scafflabel) & set(patchedscaflabels))
    yticks = []
    yticklabels = []
    for i,le,la in zip(range(len(scafflength)),scafflength_normalized,scafflabel):
        lower = (3*i)+1+0.5+0.1
        #upper = (3*i)+3-0.75-0.1
        upper = (3*i)+1+3-0.1
        height_increment = (upper-lower)/highest_level
        yticks.append((3*i)+1.25)
        yticklabels.append(la)
        ax.add_patch(Rectangle((0, (3*i)+1),le,0.5,fc ='black',ec ='none',lw = 1, zorder=0, alpha=0.3))
        #ax.text(0, (3*i)+0.25,la)
        if la in common:
            idc = patchedscaflabels.index(la)
            for f,l,segid in zip(patchescoordif[idc],patchescoordil[idc],patchessegid[idc]):
                left = gene_start[f]/max(scafflength)
                right = gene_start[l]/max(scafflength)
                ple = right - left
                # Since the previous step has done the filtering already
                #if ple < minseglen*le:
                #    continue
                if spx_level is None: level = segs_levels[segid]
                else: level = segs_levels[segid] + spx_level[segid] - 1
                hscaled = 0.75*height_increment
                iternum = level-1 if sp == spy else level
                color = 'green' if sp == spy else 'blue'
                hatchs = '//////' if color == 'green' else '\\\\\\\\\\\\'
                if ple > 0:
                    ax.add_patch(Rectangle((left, (3*i)+1),ple,0.5,fc ='green',ec ='none',lw = 1, zorder=2,alpha=0.4))
                    if spx_level is None:
                        for lev in range(iternum):
                            ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.4, hatch=hatchs))
                    else:
                        spx_times = 0
                        level_x = spx_level[segid] - 1
                        if level_x == 0:
                            for lev in range(iternum):
                                ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.4, hatch=hatchs))
                        else:
                            for lev in range(iternum):
                                # Here first pile the green (self) segments, above which piled the blue (other sp) segments
                                if spx_times < level_x:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc ='green',ec ='none',lw = 1, zorder=1,alpha=0.4,hatch='//////'))
                                else:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.4,hatch=hatchs))
                                spx_times = spx_times + 1
                else:
                    ax.add_patch(Rectangle((right, (3*i)+1),-ple,0.5,fc = 'green',ec ='none',lw = 1, zorder=2,alpha=0.4))
                    if spx_level is None:
                        for lev in range(iternum):
                            ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.4,hatch=hatchs))
                    else:
                        spx_times = 0
                        level_x = spx_level[segid] - 1
                        if level_x == 0:
                            for lev in range(iternum):
                                ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.4,hatch=hatchs))
                        else:
                            for lev in range(iternum):
                                if spx_times < level_x:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc ='green',ec ='none',lw = 1, zorder=1,alpha=0.4,hatch='//////'))
                                else:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.4,hatch=hatchs))
                                spx_times = spx_times + 1
    y = lambda x : ["{:.2f}".format(i) for i in x]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(y(ax.get_xticks()*max(scafflength)/1e6))
    ax.xaxis.label.set_fontsize(18)
    ax.set_xlabel("{} (Mb)".format(sp))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    if spy != None:
        fig.savefig(fnamep)
        fig.savefig(fnames)
    plt.close()

def get_marco_whole(dfs,seg, multi, minseglen, maxsize=None, minlen=None, outdir=None):
    #species = [i.loc[:,'species'][0] for i in dfs]
    sp_scaffla_scaffle = {}
    gene_start = {}
    for df in dfs:
        gene_start.update({gene:start for gene,start in zip(df.index,list(df['start']))})
        species = df.loc[:,'species'][0]
        lens = df.groupby("scaffold")["start"].agg(max)
        lens.name = "len"
        df_tmp = pd.DataFrame(lens).sort_values("len", ascending=False)
        sp_scaffla_scaffle[species] = [list(df_tmp.index),list(df_tmp['len'])]
    seg.loc[:,"segment"] = seg.index
    segs_info = seg.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    profile = segs_info.unstack(level=-1).fillna(0)
    if len(profile.columns) == 0: logging.info('No collinear segments were found')
    elif len(profile.columns) == 1 and len(dfs) > 1:
        logging.info('No inter-specific collinear segments were found')
    else:
        y_filter = lambda x: sum([i>0 for i in x])
        profile.loc[:,'num_species'] = [y_filter(profile.loc[i,:]) for i in profile.index]
        profile = profile.loc[profile['num_species'] == len(dfs),['num_species']]
        if len(profile) == 0:
            logging.info('No collinear segments contained all species')
        else:
            seg_filtered = seg.set_index('multiplicon').merge(profile,left_index=True, right_index=True).drop(columns='num_species')
            #seg_filtered.loc[:,'multiplicon'] = seg_filtered.index
            plot_marco_whole(sp_scaffla_scaffle,seg_filtered,gene_start,outdir,minseglen)

def get_vertices(dic,order):
    sps = list(dic.keys())
    vertices = []
    sp_levels = {sp:len(dic[sp]) for sp in sps}
    if max(sp_levels.values())==1: color = 'gray'
    elif max(sp_levels.values())==2: color = 'green'
    elif max(sp_levels.values())==3: color = 'blue'
    elif max(sp_levels.values())==4: color = 'red'
    else: color = 'yellow'
    for i in range(len(sps)):
        for j in range(i+1,len(sps)):
            spi,spj = sps[i],sps[j]
            spi_indice,spj_indice = order.index(spi),order.index(spj)
            if spi_indice-spj_indice==1 or spj_indice-spi_indice==1:
                for coordi in dic[spi]:
                    for coordj in dic[spj]:
                        f1,l1,f2,l2 = coordi[0],coordi[1],coordj[0],coordj[1]
                        if spi_indice-spj_indice==1: f2,l2 = (f2[0],f2[1]+0.75), (l2[0],l2[1]+0.75)
                        else: f1,l1 = (f1[0],f1[1]+0.75), (l1[0],l1[1]+0.75)
                        vertices.append([f1,l1,l2,f2])
    return vertices,color


def plot_marco_whole(scaf_info,seg_f,gene_start,outdir,minseglen):
    fig, ax = plt.subplots(1, 1, figsize=(100,100))
    fname = os.path.join(outdir, "All_species_marcosynteny.png")
    fnamep = os.path.join(outdir, "All_species_marcosynteny.pdf")
    fnames = os.path.join(outdir, "All_species_marcosynteny.svg")
    num_sp = len(scaf_info)
    colors = cm.viridis(np.linspace(0, 1, num_sp))
    sp_wholelengths = {sp:sum(info[1]) for sp,info in scaf_info.items()}
    sp_segs_starts = {sp:{} for sp in scaf_info.keys()}
    sp_segs_y = {}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1+(num_sp-1)*10+1)
    sp_bottom_up = []
    yticks = []
    yticklabels = []
    for indice,sp_info in enumerate(scaf_info.items()):
        sp,info = sp_info[0],sp_info[1]
        sp_bottom_up.append(sp)
        scaled_le = [i/sp_wholelengths[sp] for i in info[1]]
        color = colors[indice]
        leng_done = 0
        for lb,le in zip(info[0],scaled_le):
            le_scaled = le * 0.75
            ax.add_patch(Rectangle((leng_done, 1+indice*10),le_scaled,0.75,fc = color,ec ='none',lw = 1, zorder=0, alpha=1))
            ax.text(leng_done+(le_scaled/2),1+indice*10+0.75/2,lb,size=60,zorder=1,color="w",ha="center",va="center")
            yticks.append(1+indice*10+0.75/2)
            yticklabels.append(sp)
            sp_segs_y[sp] = 1+indice*10
            sp_segs_starts[sp].update({lb:leng_done})
            leng_done = leng_done + le
    multis = list(seg_f.reset_index().groupby('multiplicon'))
    multi_indices = list(map(lambda x: x[0],multis))
    coordi = list(map(lambda x: x[1].loc[:,['genome','first','last','list']],multis))
    #coordi = {sp:list(map(lambda x: x[1].loc[x[1]['genome']==sp,['first','last','list']],multis)) for sp in scaf_info.keys()}
    #coordi_x = list(map(lambda x: x[1].loc[x[1]['genome']==sp1,['first','last','list']],multis))
    #coordi_y = list(map(lambda x: x[1].loc[x[1]['genome']==sp2,['first','last','list']],multis))
    lb_le_persp = {sp:{} for sp in scaf_info.keys()}
    for sp in scaf_info.keys(): lb_le_persp[sp]={lb:le for lb,le in zip(scaf_info[sp][0],scaf_info[sp][1])}
    for coord in coordi:
        sp_occur = set(coord['genome'])
        coord_sp = {sp:[] for sp in sp_occur}
        for sp in sp_occur:
            df = coord.loc[coord['genome']==sp,['first','last','list']]
            for i in df.index:
                f,l,li = df.loc[i,'first'],df.loc[i,'last'],df.loc[i,'list']
                if not li in lb_le_persp[sp].keys():
                    continue
                # Since previous step has already filtered
                #ratio = (gene_start[l]-gene_start[f])/lb_le_persp[sp][li]
                #if ratio < minseglen:
                #    continue
                f,l = 0.75*gene_start[f]/sp_wholelengths[sp]+sp_segs_starts[sp][li],0.75*gene_start[l]/sp_wholelengths[sp]+sp_segs_starts[sp][li]
                coord_sp[sp].append([(f,sp_segs_y[sp]),(l,sp_segs_y[sp])])
        vertices,color = get_vertices(coord_sp,sp_bottom_up)
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
        for vertice in vertices:
            pp = patches.PathPatch(Path(vertice,codes),fc=color,alpha=0.5,zorder=1,lw=0.1)
            ax.add_patch(pp)
                #coord_sp[sp].append((f,sp_segs_y[sp]))
                #coord_sp[sp].append((l,sp_segs_y[sp]))
    #for cix,ciy in zip(coordi_x,coordi_y):
    #    for i in cix.index:
    #        f,l,li1 = cix.loc[i,'first'],cix.loc[i,'last'],cix.loc[i,'list']
    #        f1,l1 = 0.75*genex_start[f]/sum(sp1_scafflength)+segs1_starts[li1],0.75*genex_start[l]/sum(sp1_scafflength)+segs1_starts[li1]
    #        for j in ciy.index:
    #            f2,l2,li2 = ciy.loc[j,'first'],ciy.loc[j,'last'],ciy.loc[j,'list']
    #            if f==f2 and l==l2:
    #                continue
    #            ratio = (geney_start[l2]-geney_start[f2])/sp2_scafflabel_length[li2]
    fig.tight_layout()
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize=150, labelbottom = False, bottom = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fnamep)
    fig.savefig(fnames)

def get_marco(dfx, dfy, seg, multi, minseglen, maxsize=None, minlen=None, outdir=None):
    spx=dfx.loc[:,'species'][0]
    spy=dfy.loc[:,'species'][0]
    lens = dfx.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1x = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1x.len.max() * 0.1
    df1x = df1x.loc[df1x.len > minlen]
    dfx.join(df1x,on="scaffold").dropna()
    lens = dfy.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1y = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1y.len.max() * 0.1
    df1y = df1y.loc[df1y.len > minlen]
    dfy.join(df1y,on="scaffold").dropna()
    seg.loc[:,"segment"] = seg.index
    seg_unfilterded = seg.loc[(seg['genome']==spx) | (seg['genome']==spy)].copy()
    segs_info = seg.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    profile = segs_info.unstack(level=-1).fillna(0)
    if spy not in profile.columns: logging.info('No collinear segments were found involving genome of {}'.format(spy))
    elif spx not in profile.columns: logging.info('No collinear segments were found involving genome of {}'.format(spx))
    else:
        if spx == spy: multi_goodinuse = profile.loc[profile[spx]>0,[spy]].copy()
        else: multi_goodinuse = profile.loc[(profile[spy]>0) & (profile[spx]>0),[spx,spy]].copy()
        seg_filterded = seg_unfilterded.set_index('multiplicon').merge(multi_goodinuse,left_index=True, right_index=True).drop(columns=spy)
        if len(seg_filterded) == 0: logging.info('No collinear segments contained both genome of {0} and {1}'.format(spx,spy))
        else:
            seg_filterded.loc[:,'multiplicon'] = seg_filterded.index
            genex_start = {gene:start for gene,start in zip(dfx.index,list(dfx['start']))}
            geney_start = {gene:start for gene,start in zip(dfy.index,list(dfy['start']))}
            plot_marco(spx,spy,df1x.index,df1x.len,df1y.index,df1y.len,outdir,genex_start,geney_start,seg_filterded,minseglen)

def plot_marco(sp1,sp2,sp1_scafflabel,sp1_scafflength,sp2_scafflabel,sp2_scafflength,outdir,genex_start,geney_start,seg_f,minseglen):
    ## Only consider inter-specific links
    fname = os.path.join(outdir, "{0}_{1}_marcosynteny.png".format(sp1,sp2))
    fnamep = os.path.join(outdir, "{0}_{1}_marcosynteny.pdf".format(sp1,sp2))
    fnames = os.path.join(outdir, "{0}_{1}_marcosynteny.svg".format(sp1,sp2))
    fig, ax = plt.subplots(1, 1, figsize=(100,100))
    sp1_wholelength = sum(sp1_scafflength)
    sp2_wholelength = sum(sp2_scafflength)
    sp1_length_scaled = [i/sp1_wholelength for i in sp1_scafflength]
    sp2_length_scaled = [i/sp2_wholelength for i in sp2_scafflength]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    colors = cm.viridis(np.linspace(0, 1, 2))
    sp1_leng_done = 0
    segs1_starts = {}
    yticks = [1+0.75/2,1+10+0.75/2]
    yticklabels = [sp1,sp2]
    sp1_scafflabel_length = {lb:le for lb,le in zip(sp1_scafflabel,sp1_scafflength)}
    sp2_scafflabel_length = {lb:le for lb,le in zip(sp2_scafflabel,sp2_scafflength)}
    for i,le,lb in zip(range(len(sp1_length_scaled)),sp1_length_scaled,sp1_scafflabel):
        le_scaled = le * 0.75
        ax.add_patch(Rectangle((sp1_leng_done, 1),le_scaled,0.75,fc =colors[0],ec ='none',lw = 1, zorder=0, alpha=1))
        ax.text(sp1_leng_done+(le_scaled/2),1+0.75/2,lb,size=60,zorder=1,color="w",ha="center",va="center")
        segs1_starts[lb]= sp1_leng_done
        sp1_leng_done = sp1_leng_done + le
    sp2_leng_done = 0
    segs2_starts = {}
    for i,le,lb in zip(range(len(sp2_length_scaled)),sp2_length_scaled,sp2_scafflabel):
        le_scaled = le * 0.75
        ax.add_patch(Rectangle((sp2_leng_done, 1 + 10),le_scaled,0.75,fc =colors[1],ec ='none',lw = 1, zorder=0, alpha=1))
        ax.text(sp2_leng_done+le_scaled/2,1+10+0.75/2,lb,size=60,zorder=1,color="w",ha="center",va="center")
        segs2_starts[lb] = sp2_leng_done
        sp2_leng_done = sp2_leng_done + le
    multis = list(seg_f.reset_index(drop=True).groupby('multiplicon'))
    multi_indices = list(map(lambda x: x[0],multis))
    coordi_x = list(map(lambda x: x[1].loc[x[1]['genome']==sp1,['first','last','list']],multis))
    coordi_y = list(map(lambda x: x[1].loc[x[1]['genome']==sp2,['first','last','list']],multis))
    for cix,ciy in zip(coordi_x,coordi_y):
        for i in cix.index:
            f,l,li1 = cix.loc[i,'first'],cix.loc[i,'last'],cix.loc[i,'list']
            if not li1 in sp1_scafflabel:
                continue
            f1,l1 = 0.75*genex_start[f]/sum(sp1_scafflength)+segs1_starts[li1],0.75*genex_start[l]/sum(sp1_scafflength)+segs1_starts[li1]
            for j in ciy.index:
                f2,l2,li2 = ciy.loc[j,'first'],ciy.loc[j,'last'],ciy.loc[j,'list']
                if not li2 in sp2_scafflabel:
                    continue
                if f==f2 and l==l2:
                    continue
                # Since the filtering has been done in previous step
                #ratioy = (geney_start[l2]-geney_start[f2])/sp2_scafflabel_length[li2]
                #ratiox = (genex_start[l]-genex_start[f])/sp1_scafflabel_length[li1]
                #here only the long segments are plotted (>5% of the residing chromosome)
                #if ratiox >= 0.05 or ratioy >= 0.05:
                #if ratiox < minseglen or ratioy < minseglen:
                #    continue
                #if ratiox >= 0.05 and ratioy >= 0.05:
                if len(cix) == 1 and len(ciy) == 1: color = 'gray'
                    #elif len(cix) == 2 or len(ciy) == 2: color = 'green'
                    #elif len(cix) == 3 or len(ciy) == 3: color = 'blue'
                    #elif len(cix) == 4 or len(ciy) == 4: color = 'red'
                elif max([len(cix),len(ciy)]) ==2: color = 'green'
                elif max([len(cix),len(ciy)]) ==3: color = 'blue'
                elif max([len(cix),len(ciy)]) ==4: color = 'red'
                else: color = 'yellow'
                #else: color = 'white'
                f2,l2 = 0.75*geney_start[f2]/sum(sp2_scafflength)+segs2_starts[li2],0.75*geney_start[l2]/sum(sp2_scafflength)+segs2_starts[li2]
                vertices = [(f1,1+0.75),(l1,1+0.75),(l2,11),(f2,11)]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
                pp = patches.PathPatch(Path(vertices,codes),fc=color,alpha=0.5,zorder=1,lw=0.1)
                ax.add_patch(pp)
    # Second the links
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize=150, labelbottom = False, bottom = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fnamep)
    fig.savefig(fnames)
    plt.close()

# dot plot stuff
def all_dotplots(df, segs, multi, minseglen, anchors=None, ancestor=None, **kwargs):
    """
    Generate dot plots for all pairs of species in `df`, coloring anchor pairs.
    """
    # Note that the Chr ID in gff3 file should not be alleen int or float
    gdf = list(df.groupby("species"))
    n = len(gdf)
    figs = {}
    if ancestor != None:
        logging.info("Making ancestral karyotype plot")
        gdf_ances = df[df['species']==ancestor]
        color_scaff = AK_plot(ancestor,gdf_ances,ancestor,backbone=True,**kwargs)
        for i in range(n):
            spx, dfx = gdf[i]
            AK_plot(spx,dfx,ancestor,backbone=False,colortable=color_scaff,seg=segs,**kwargs)
    logging.info("Making dupStack plot")
    for i in range(n):
        for j in range(n):
            spx, dfx = gdf[i]
            spy, dfy = gdf[j]
            logging.info("{} vs. {}".format(spx, spy))
            get_dots(dfx, dfy, segs, multi, minseglen, dupStack = True, **kwargs)
    logging.info("Making dotplots and marco-synteny plots")
    getscafflength(n,gdf,**kwargs)
    if n > 1: get_marco_whole(list(map(lambda x:x[1],gdf)),segs, multi, minseglen,**kwargs)
    for i in range(n):
        for j in range(i, n):
            fig, ax = plt.subplots(1, 1, figsize=(10,10))
            ax2 = ax.twinx()
            ax3 = ax.twiny()
            spx, dfx = gdf[i]
            spy, dfy = gdf[j]
            logging.info("{} vs. {}".format(spx, spy))
            get_marco(dfx, dfy, segs, multi, minseglen, **kwargs)
            df, xs, ys, scaffxlabels, scaffylabels, scaffxtick, scaffytick = get_dots(dfx, dfy, segs, multi, minseglen, dupStack = False, **kwargs)
            #print(xs)
            #print(ys)
            #for i,j in zip(df.x,df.y): print((i,j))
            if df is None:  # HACK, in case we're dealing with RBH orthologs...
                continue
            #for x,y in zip(df.x,df.y): ax.scatter(x, y, s=1, color="k", alpha=0.1)
            #print((len(list(itertools.chain(df.x))),len(list(itertools.chain(df.y)))))
            ax.scatter(list(itertools.chain(df.x)), list(itertools.chain(df.y)), s=1, color="k", alpha=0.1)
            #ax.scatter(np.array(df.x,dtype=float), np.array(df.y,dtype=float), s=1, color="k", alpha=0.1)
            if not (anchors is None):
                andf = df.join(anchors, how="inner")
                #for x,y in zip(andf.x,andf.y): ax.scatter(x, y, s=1, color="r", alpha=0.9)
                ax.scatter(list(itertools.chain(andf.x)), list(itertools.chain(andf.y)), s=1, color="r", alpha=0.9)
                #ax.scatter(np.array(andf.x,dtype=float), np.array(andf.y,dtype=float), s=1, color="red", alpha=0.9)
            xlim = max(scaffxtick)
            ylim = max(scaffytick)
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            #ax.vlines(xs, ymin=0, ymax=ys[-1], alpha=0.8, color="k")
            #ax.vlines(xs, ymin=0, ymax=ylim, alpha=0.8, color="k")
            ax.vlines(xs+[xmax], ymin=0, ymax=ylim, alpha=0.8, color="k")
            #ax.vlines(scaffxtick, ymin=0, ymax=ylim, alpha=0.8, color="k")
            #ax.hlines(ys, xmin=0, xmax=xs[-1], alpha=0.8, color="k")
            #ax.hlines(ys, xmin=0, xmax=xlim, alpha=0.8, color="k")
            ax.hlines(ys+[ymax], xmin=0, xmax=xlim, alpha=0.8, color="k")
            #ax.hlines(scaffytick, xmin=0, xmax=xlim, alpha=0.8, color="k")
            #xlim = max(scaffxtick)
            #ax.set_xlim(0, xs[-1])
            #ax.set_xlim(0, xlim)
            #ylim = max(scaffytick)
            #ax.set_ylim(0, ys[-1])
            #ax.set_ylim(0, ylim)
            #ax.set_xlabel("${}$ (Mb)".format(spx))
            #ax.set_ylabel("${}$ (Mb)".format(spy))
            ax.set_xlabel("{}".format(spx))
            ax.set_ylabel("{}".format(spy))
            ax2.set_ylabel("{} (Mb)".format(spy))
            ax2.yaxis.label.set_fontsize(18)
            ax2.set_yticklabels(ax.get_yticks() / 1e6)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            ax3.set_xlabel("{} (Mb)".format(spx))
            ax3.xaxis.label.set_fontsize(18)
            ax3.set_xticklabels(ax.get_xticks() / 1e6)
            ax3.tick_params(axis='both', which='major', labelsize=16)
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xticks(scaffxtick)
            ax.set_xticklabels(scaffxlabels,rotation=45)
            #ax.set_xticklabels(ax.get_xticks() / 1e6)  # in Mb
            #ax.set_yticklabels(ax.get_yticks() / 1e6)  # in Mb
            ax.set_yticks(scaffytick)
            ax.set_yticklabels(scaffylabels,rotation=45)
            fig.tight_layout()
            figs[spx + "-vs-" + spy] = fig
    return figs

def filter_by_minlength(genetable,segs,minlen,multi,keepredun,outdir,minseglen):
    gdf,Index_torm,I2,I3 = list(genetable.copy().groupby("species")),[],[],[]
    Sf_len_lab_persp = {sp:{} for sp,df in gdf}
    for sp,df in gdf:
        lens = df.groupby("scaffold")["start"].agg(max)
        lens.name = "len"
        lens = pd.DataFrame(lens).sort_values("len", ascending=False)
        noriginal = len(lens.index)
        if minlen < 0:
            Minlen = lens.len.max() * 0.1
            logging.info("`minlen` not set, taking 10% of longest scaffold ({}) for {}".format(Minlen,sp))
            lens = lens.loc[lens.len > Minlen]
            logging.info("Dropped {} scaffolds in {} because they are on scaffolds shorter than {}".format(noriginal-len(lens.index),sp,Minlen))
        else:
            lens = lens.loc[lens.len > minlen]
            logging.info("Dropped {} scaffolds in {} because they are on scaffolds shorter than {}".format(noriginal-len(lens.index),sp,minlen))
        segs_tmp = segs.loc[(segs['genome']==sp) & (~segs['list'].isin(lens.index)),:]
        gt_tmp = genetable.loc[(genetable['species']==sp) & (~genetable['scaffold'].isin(lens.index)),:]
        for i in segs_tmp.index: Index_torm.append(i)
        for j in gt_tmp.index: I2.append(j)
        for lab in lens.index: Sf_len_lab_persp[sp][lab] = lens.loc[lab,'len']
    segs = segs.drop(Index_torm)
    genetable = genetable.drop(I2)
    # Here I removed the redundant multiplicons
    if not keepredun:
        Mul_to_rm = list(multi[multi['is_redundant']==-1].loc[:,'id'])
        Segs_to_rm = segs.loc[segs['multiplicon'].isin(Mul_to_rm),:]
        for i in Segs_to_rm.index: I3.append(i)
        segs = segs.drop(I3)
    segs = Filter_miniseglen(segs,Sf_len_lab_persp,minseglen,genetable)
    counted = segs.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    profile = counted.unstack(level=-1).fillna(0)
    fig = syntenic_depth_plot(profile)
    fig.savefig(os.path.join(outdir, "Syndepth.svg"),bbox_inches='tight')
    fig.savefig(os.path.join(outdir, "Syndepth.pdf"),bbox_inches='tight')
    profile.to_csv(os.path.join(outdir, "Segprofile.csv"))
    return segs,genetable

def Filter_miniseglen(segs,scaf_info,minseglen,genetable):
    rm_indice = []
    gene_start_info = {sp:{} for sp in scaf_info.keys()}
    work1 = genetable.reset_index().loc[:,['gene','species','start']]
    work2 = work1.groupby('species')
    for i,j,k in map(lambda x:(x[0],x[1]['gene'],x[1]['start']) ,work2):
        for gene,start in zip(j,k): gene_start_info[i][gene] = start
    for indice,sp,f,l,sf_la in zip(segs.index,segs['genome'],segs['first'],segs['last'],segs['list']):
        seg_len = abs(gene_start_info[sp][f] - gene_start_info[sp][l])
        if minseglen <= 1:
            if seg_len < scaf_info[sp][sf_la] * minseglen: rm_indice.append(indice)
        else:
            if seg_len < minseglen: rm_indice.append(indice)
    segs = segs.drop(rm_indice)
    return segs

def Ks_dotplots(segs,dff, df, ks, an, minseglen, anchors=None, color_map='Spectral',min_ks=0.05, max_ks=5, minlen=250, maxsize=25, outdir='', **kwargs):
    """
    Generate Ks colored dot plots for all pairs of species in `df`.
    """
    cmap = plt.get_cmap(color_map)
    if len(an["gene_x"]) == 0:
        logging.warning("No multiplicons found!")
        return
    an["pair"] = an.apply(lambda x: '__'.join(
            sorted([x["gene_x"], x["gene_y"]])), axis=1)
    genomic_elements_ = {
        x: 0 for x in list(set(dff['list_x']) | set(dff['list_y']))
        if type(x) == str
    }

    ks_multiplicons = {}
    all_ks = []
    for i in range(len(dff)):
        row = dff.iloc[i]
        pairs = an[an['multiplicon'] == row['id']]['pair']
        med_ks = np.median(ks.loc[ks.index.intersection(pairs)]['dS'])
        ks_multiplicons[row['id']] = med_ks
        all_ks.append(med_ks)

    z = [[0, 0], [0, 0]]
    levels = range(0, 101, 1)
    tmp = plt.contourf(z, levels, cmap=cmap)
    plt.clf()
    
    for key in sorted(genomic_elements_.keys()):
        length = max(list(dff[dff['list_x'] == key]['end_x']) + list(
                dff[dff['list_y'] == key]['end_y']))
        if length >= minlen:
            genomic_elements_[key] = length

    previous = 0
    genomic_elements = {}
    sorted_ge = sorted(genomic_elements_.items(), key=lambda x: x[1],
                       reverse=True)
    labels = [kv[0] for kv in sorted_ge if kv[1] >= minlen]

    for kv in sorted_ge:
        genomic_elements[kv[0]] = previous
        previous += kv[1]

    gdf = list(df.groupby("species"))
    n = len(gdf)
    figs = {}
    for i in range(n):
        for j in range(i, n):
            fig, ax = plt.subplots(1, 1, figsize=(10,10))
            spx, dfx = gdf[i]
            spy, dfy = gdf[j]
            logging.info("{} vs. {}".format(spx, spy))
            df, xs, ys, scafflabels, scaffylabels, scaffxtick, scaffytick = get_dots(dfx, dfy, segs, dff, minseglen, minlen=minlen, maxsize=maxsize, outdir = outdir)
            if df is None:  # HACK, in case we're dealing with RBH orthologs...
                continue
            ax.scatter(df.x, df.y, s=0.1, color="k", alpha=0.5)
            if not (anchors is None):
                andf = df.join(anchors, how="inner")
                for k in range(len(dff)):
                    row = dff.iloc[k]
                    list_x, list_y = row['list_x'], row['list_y']
                    if type(list_x) != float:
                        curr_list_x = list_x
                    x = [genomic_elements[curr_list_x] + x for x in [row['begin_x'], row['end_x']]]
                    y = [genomic_elements[list_y] + x for x in [row['begin_y'], row['end_y']]]                     
                    med_ks = ks_multiplicons[row['id']]
                    if min_ks < med_ks <= max_ks:
                        ax.scatter(andf.x, andf.y, s=0.2, color=cmap(ks_multiplicons[row['id']] / 5), alpha=0.9)
            ax.vlines(xs, ymin=0, ymax=ys[-1], alpha=0.1, color="k")
            ax.hlines(ys, xmin=0, xmax=xs[-1], alpha=0.1, color="k")
            ax.set_xlim(0, xs[-1])
            ax.set_ylim(0, ys[-1])
            ax.set_xlabel("${}$ (Mb)".format(spx))
            ax.set_ylabel("${}$ (Mb)".format(spy))
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xticklabels(ax.get_xticks() / 1e6)  # in Mb
            ax.set_yticklabels(ax.get_yticks() / 1e6)  # in Mb
            figs[spx + "-vs-" + spy] = fig

    # colorbar
    cbar = plt.colorbar(tmp, fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.linspace(0, 5, 11)])

    return figs

def get_dots(dfx, dfy, seg, multi, minseglen, minlen=-1, maxsize=200, outdir = '', dupStack = False):
    spx=dfx.loc[:,'species'][0]
    spy=dfy.loc[:,'species'][0]
    if dupStack: sankey_plot(spx, dfx, spy, dfy, minseglen, minlen, outdir, seg)
    else:
        dfx,scaffxtick = filter_data_dotplot(dfx, minlen)
        dfy,scaffytick = filter_data_dotplot(dfy, minlen)
        dx = {k: list(v.index) for k, v in dfx.groupby("family")}
        dy = {k: list(v.index) for k, v in dfy.groupby("family")}
        xs = []
        for family in dx.keys():
            if not family in dy:
                continue
            if len(dx[family]) > maxsize or len(dy[family]) > maxsize:  
                # large TE families for instance...
                continue
            for (x, y) in itertools.product(dx[family], dy[family]):
                if x == y:
                    continue
                pair = "__".join(sorted([x,y]))
                xs.append({"pair":pair, "x": dfx.loc[x]["x"], "y": dfy.loc[y]["x"]})
        #ax.scatter(xs, ys)
        if len(xs) == 0:  # HACK
            return None, None, None, None, None, None, None
        df = pd.DataFrame.from_dict(xs).set_index("pair")
        scaffxlabels = list(dfx['scaffold'].drop_duplicates())
        scaffylabels = list(dfy['scaffold'].drop_duplicates())
        #xl = list(np.unique(dfx["scaffstart"])) + [max(df.x)]
        #xl = list(dfx["scaffstart"].drop_duplicates()) + [max(df.x)]
        #xl = list(dfx["scaffstart"].drop_duplicates()) + [max(list(df['x']))]
        #xl = list(dfx["scaffstart"].drop_duplicates()) + [df['x'].max()]
        xl = list(dfx.drop_duplicates(subset=['scaffstart']).loc[:,'scaffstart'])
        #yl = list(np.unique(dfy["scaffstart"])) + [max(df.y)]
        #yl = list(dfy["scaffstart"].drop_duplicates()) + [max(df.y)]
        #yl = list(dfy["scaffstart"].drop_duplicates()) + [max(list(df['y']))]
        #yl = list(dfy["scaffstart"].drop_duplicates()) + [df['y'].max()]
        yl = list(dfy.drop_duplicates(subset=['scaffstart']).loc[:,'scaffstart'])
        return df, xl, yl, scaffxlabels, scaffylabels, scaffxtick, scaffytick 

def getscafflength(n,gdf,outdir='',maxsize='',minlen='',ancestor=''):
    Lens = []
    for i in range(n):
        sp, df = gdf[i]
        lens = df.groupby("scaffold")["start"].agg(max)
        lens.name = "length"
        lens = pd.DataFrame(lens).sort_values("length", ascending=False)
        lens.loc[:,'species'] = [sp for j in range(lens.shape[0])]
        Lens.append(lens)
    Df = pd.concat(Lens,ignore_index=False)
    Df.to_csv("{}".format(os.path.join(outdir,'scaffold_length.tsv')),sep='\t',header=True,index=True)

def filter_data_dotplot(df, minlen):
    lens = df.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    lens = pd.DataFrame(lens).sort_values("len", ascending=False)
    scaffstart = [0] + list(np.cumsum(lens.len))[0:-1]
    scafftick = list(np.cumsum(lens.len))
    lens["scaffstart"] = scaffstart
    df = df.join(lens, on="scaffold").sort_values("len", ascending=False).dropna()
    #print(set(df['scaffold']))
    # df now contains scaffold lengths
    #if minlen < 0:  # find a reasonable threshold, 5% of longest scaffold?
    #    minlen = df.len.max() * 0.1
    #    logging.info("`minlen` not set, taking 10% of longest scaffold ({})".format(minlen))
    #noriginal = len(df.index)
    #df = df.loc[df.len > minlen]
    #logging.info("Dropped {} genes because they are on scaffolds shorter "
    #        "than {}".format(noriginal - len(df.index), minlen))
    df["x"] = df["scaffstart"] + df["start"]
    return df,scafftick


def syntenic_dotplot_ks_colored(
        df, an, ks, min_length=50, color_map='Spectral', min_ks=0.05, max_ks=5,
        output_file=None
):
    """
    Syntenic dotplot with segment colored by mean Ks value
    :param df: multiplicons pandas data frame
    :param an: anchorpoints pandas data frame
    :param ks: Ks distribution data frame
    :param min_length: minimum length of a genomic element
    :param color_map: color map string
    :param min_ks: minimum median Ks value
    :param max_ks: maximum median Ks value
    :param output_file: output file name
    :return: figure
    """
    cmap = plt.get_cmap(color_map)
    if len(an["gene_x"]) == 0:
        logging.warning("No multiplicons found!")
        return
    an["pair"] = an.apply(lambda x: '__'.join(
            sorted([x["gene_x"], x["gene_y"]])), axis=1)
    genomic_elements_ = {
        x: 0 for x in list(set(df['list_x']) | set(df['list_y']))
        if type(x) == str
    }

    ks_multiplicons = {}
    all_ks = []
    for i in range(len(df)):
        row = df.iloc[i]
        pairs = an[an['multiplicon'] == row['id']]['pair']
        med_ks = np.median(ks.loc[ks.index.intersection(pairs)]['dS'])
        ks_multiplicons[row['id']] = med_ks
        all_ks.append(med_ks)

    z = [[0, 0], [0, 0]]
    levels = range(0, 101, 1)
    tmp = plt.contourf(z, levels, cmap=cmap)
    plt.clf()

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111)

    for key in sorted(genomic_elements_.keys()):
        length = max(list(df[df['list_x'] == key]['end_x']) + list(
                df[df['list_y'] == key]['end_y']))
        if length >= min_length:
            genomic_elements_[key] = length

    previous = 0
    genomic_elements = {}
    sorted_ge = sorted(genomic_elements_.items(), key=lambda x: x[1],
                       reverse=True)
    labels = [kv[0] for kv in sorted_ge if kv[1] >= min_length]

    for kv in sorted_ge:
        genomic_elements[kv[0]] = previous
        previous += kv[1]

    # plot layout
    x = [genomic_elements[key] for key in sorted(genomic_elements.keys())] + \
        [previous]
    x = sorted(list(set(x)))
    ax.vlines(ymin=0, ymax=previous, x=x, linestyles='dotted', alpha=0.2)
    ax.hlines(xmin=0, xmax=previous, y=x, linestyles='dotted', alpha=0.2)
    ax.plot(x, x, color='k', alpha=0.2)
    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(x))
    ax.set_xticks([(x[i] + x[i - 1]) / 2 for i in range(1, len(x))], minor=True)
    ax.set_xticklabels(labels, minor=True, rotation=45)
    ax.set_yticks([(x[i] + x[i - 1]) / 2 for i in range(1, len(x))], minor=True)
    ax.set_yticklabels(labels, minor=True, rotation=45)
    for tick in ax.xaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_horizontalalignment('center')
    for tick in ax.yaxis.get_minor_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        # tick.label1.set_horizontalalignment('center')

    # the actual dots (or better, line segments)
    for i in range(len(df)):
        row = df.iloc[i]
        list_x, list_y = row['list_x'], row['list_y']
        if type(list_x) != float:
            curr_list_x = list_x
        x = [genomic_elements[curr_list_x] + x for x in
             [row['begin_x'], row['end_x']]]
        y = [genomic_elements[list_y] + x for x in
             [row['begin_y'], row['end_y']]]
        med_ks = ks_multiplicons[row['id']]
        if min_ks < med_ks <= max_ks:
            ax.plot(x, y, alpha=0.9, linewidth=1.5,
                    color=cmap(ks_multiplicons[row['id']] / 5)),
            # path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
            ax.plot(y, x, alpha=0.9, linewidth=1.5,
                    color=cmap(ks_multiplicons[row['id']] / 5))
            # path_effects=[pe.Stroke(linewidth=4, foreground='k'),
            # pe.Normal()])

    # colorbar
    cbar = plt.colorbar(tmp, fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.linspace(0, 5, 11)])

    # saving
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()

    else:
        return fig
    
