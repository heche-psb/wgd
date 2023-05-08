import numpy as np
import logging
from sklearn import mixture
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from scipy import stats,interpolate,signal
import random
from matplotlib.pyplot import cm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import warnings
from KDEpy import NaiveKDE,TreeKDE,FFTKDE
from sklearn.utils import resample
import itertools
import os
from wgd.core import _mkdir
from sklearn_extra.cluster import KMedoids
from wgd.viz import reflect_logks,find_peak_init_parameters,get_deconvoluted_data
from io import StringIO
from sklearn import metrics
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def alnfilter(df,weights_outliers_included, identity, aln_len, coverage, min_ks, max_ks):
    """
    Filter alignment stats and Ks
    """
    df = df.dropna(subset=['dS'])
    df = df[df["alignmentidentity"] >= identity]
    df = df[df["alignmentlength"] >= aln_len]
    df = df[df["alignmentcoverage"] >= coverage]
    if not weights_outliers_included:
        df = df[(df["dS"] > min_ks) & (df["dS"] < max_ks)]
    return df
def group_dS(df):
    mean_df = df.groupby(['family', 'node']).mean()
    weight_col = 1/df.groupby(['family', 'node'])['dS'].transform('count')
    return mean_df, weight_col
def log_trans(df):
    """
    Get an array of log transformed Ks values.
    """
    X = np.array(df["dS"].dropna())
    X = X[X > 0]
    X = np.log(X).reshape(-1, 1)
    return X

def info_components(m,i,info_table):
    means = []
    covariances = []
    weights = []
    precisions = []
    stds = []
    for j in range(i):
        mean = np.exp(m.means_[j][0])
        covariance = m.covariances_[j][0][0]
        std = np.sqrt(covariance)
        weight = m.weights_[j]
        precision = m.precisions_[j][0][0]
        means.append(mean)
        covariances.append(covariance)
        stds.append(std)
        weights.append(weight)
        precisions.append(precision)
        logging.info("Component {0} has mean {1:.3f} ,std {2:.3f} ,weight {3:.3f}, precision {4:.3f}".format(j+1,mean,std,weight,precision))
    info_table['{}component'.format(i)] = {'mean':means,'covariance':covariances,'weight':weights,'precision':precisions,'stds':stds}

def aic_info(aic,n1):
    besta_loc = np.argmin(aic)
    #logging.info("Relative probabilities compared to the AIC-best model (with {} components):".format(besta_loc + 1))
    #for i, aic_i in enumerate(aic):
    #    if i != besta_loc:
    #        p = np.exp((aic[besta_loc] - aic_i) / 2)
    #        logging.info("model with {0} components has a p as {1}".format(i+1, p))
    #logging.info("Significance test between the AIC-best model (with {} components) and remaining:".format(besta_loc + 1))
    logging.info("Rules-of-thumb (Burnham & Anderson, 2002) compares the AIC-best model and remaining:")
    #Refer to course notes for Applied Statistics courses at California State University, Chico
    #Rule-of-thumb
    #https://norcalbiostat.github.io/AppliedStatistics_notes/model-fit-criteria.html
    #https://uncdependlab.github.io/MLM_Tutorial/06_ModelSelection/model_comparison.html
    #https://stats.stackexchange.com/questions/349883/what-is-the-logic-behind-rule-of-thumb-for-meaningful-differences-in-aic
    for i, aic_i in enumerate(aic):
        if i != besta_loc:
            ABS = abs(aic[besta_loc] - aic_i)
            if ABS <= 2:
                logging.info("model with {} components also gets substantial support comparing to the AIC-best model".format(i+n1))
            elif 2<ABS<4:
                logging.info("model with {} components gets not-so-trivial support comparing to the AIC-best model".format(i+n1))
            elif 4<=ABS<=7:
                logging.info("model with {} components gets considerably less support comparing to the AIC-best model".format(i+n1))
            elif 7<ABS<=10:
                logging.info("model with {} components gets few support comparing to the AIC-best model".format(i+n1))
            else:
                logging.info("model with {} components gets essentially no support comparing to the AIC-best model".format(i+n1))

def bic_info(bic,n1):
    bestb_loc = np.argmin(bic)
    logging.info("Rules-of-thumb (Kass & Raftery, 1995) evaluates the outperformance of the BIC-best model over remaining:")
    #https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118856406.app5
    for i, bic_i in enumerate(bic):
        if i != bestb_loc:
            ABS = abs(bic[bestb_loc] - bic_i)
            if ABS < 2:
                logging.info("Such outperformance is not worth more than a bare mention for model with {} components".format(i+n1))
            elif 2<=ABS<6:
                logging.info("Such outperformance is positively evidenced for model with {} components".format(i+n1))
            elif 6<=ABS<=10:
                logging.info("Such outperformance is strongly evidenced for model with {} components".format(i+n1))
            else:
                logging.info("Such outperformance is very strongly evidenced for model with {} components".format(i+n1))

def plot_aic_bic(aic, bic, n1, n2, out_file):
    x_range = list(range(n1, n2 + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].plot(np.arange(1, len(aic) + 1), aic, color='k', marker='o')
    axes[0].set_xticks(list(range(1, len(aic) + 1)))
    axes[0].set_xticklabels(x_range)
    axes[0].grid(ls=":")
    axes[0].set_ylabel("AIC")
    axes[0].set_xlabel("# components")
    axes[1].plot(np.arange(1, len(bic) + 1), bic, color='k', marker='o')
    axes[1].set_xticks(list(range(1, len(bic) + 1)))
    axes[1].set_xticklabels(x_range)
    axes[1].grid(ls=":")
    axes[1].set_ylabel("BIC")
    axes[1].set_xlabel("# components")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close()

def significance_test_cluster(X,n1,n2,labels):
    n_permutations = 100
    for indice,i in enumerate(range(n1, n2 + 1)):
        obs = metrics.silhouette_score(X,labels[indice])
        perm_scores = []
        for j in range(n_permutations):
            perm_labels = np.random.permutation(labels[indice])
            perm_scores.append(metrics.silhouette_score(X, perm_labels))
        p_value = np.mean(np.array(perm_scores) >= obs)
        logging.info("Components {} model: Observed Silhouette Coefficient {:.2f} and P-value {:.3f}".format(i,obs,p_value))

def plot_silhouette_score(X,n1,n2,labels,outdir,prefix,method):
    x_range = list(range(n1, n2 + 1))
    fig, ax = plt.subplots()
    scores = [metrics.silhouette_score(X, label) for label in labels]
    ax.plot(np.arange(1, len(labels) + 1),scores,color='k', marker='o')
    ax.set_xticks(list(range(1, len(labels) + 1)))
    ax.set_xticklabels(x_range)
    ax.grid(ls=":")
    ax.set_ylabel("Silhouette Coefficient")
    ax.set_xlabel("# components")
    fig.tight_layout()
    fname = os.path.join(outdir, "{}_{}_Clustering_Silhouette_Coefficient.pdf".format(method,prefix))
    fig.savefig(fname)
    plt.close()

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

def get_totalH(Hs):
    CHF = 0
    for i in Hs: CHF = CHF + i
    return CHF

def plot_mp_component_lognormal(X,hdr,means,stds,weights,labels,n,bins=50,ylabel="Counts",regime='multiplicon'):
    #labels = labels[(X<5) & (X>0)]
    #X = X[(X<5) & (X>0)]
    colors = cm.viridis(np.linspace(0, 1, n))
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    fig, ax = plt.subplots()
    df = pd.DataFrame.from_dict({'label':labels,'dS':X})
    for i,color in enumerate(colors):
        mean,std,weight = means[i][0],np.sqrt(stds[i][0][0]),weights[i]
        df_comp = df[df['label']==i]
        x = np.array(list(df_comp['dS']))
        y = x[np.isfinite(x)]
        Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, alpha=0.5, rwidth=0.8, label = "component {}".format(i))
        CHF = get_totalH(Hs)
        scaling = CHF*0.1
        #upper_HPD,lower_HPD = calculateHPD(y,hdr)
        #plt.axvline(x = upper_HPD, color = color, alpha = 0.8, ls = '-.', lw = 1,label="{}% HDR CI Upper {:.2f}".format(hdr,upper_HPD))
        #plt.axvline(x = lower_HPD, color = color, alpha = 0.8, ls = '-.', lw = 1,label="{}% HDR CI Lower {:.2f}".format(hdr,lower_HPD))
        ax.plot(kde_x,scaling*weight*stats.lognorm.pdf(kde_x, scale=np.exp(mean),s=std), c=color, ls='-', lw=1.5, alpha=0.8, label='component {} mode {:.2f}'.format(i,np.exp(mean - std**2)))
    ax.legend(loc='upper right', fontsize='small',frameon=False)
    ax.set_xlabel("$K_\mathrm{S}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    if regime== 'multiplicon': plt.title('Multiplicon $K_\mathrm{S}$ GMM modeling')
    elif regime== 'segment': plt.title('Segment $K_\mathrm{S}$ GMM modeling')
    else: plt.title('Basecluster $K_\mathrm{S}$ GMM modeling')
    fig.tight_layout()
    return fig

def plot_mp_component(X,labels,n,bins=50,plot = 'identical',ylabel="Counts",regime='multiplicon'):
    labels = labels[(X<5) & (X>0)]
    X = X[(X<5) & (X>0)]
    colors = cm.viridis(np.linspace(0, 1, n))
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    fig, ax = plt.subplots()
    df = pd.DataFrame.from_dict({'label':labels,'dS':X})
    if plot == 'identical':
        for i,color in enumerate(colors):
            df_comp = df[df['label']==i]
            x = np.array(list(df_comp['dS']))
            y = x[np.isfinite(x)]
            ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, alpha=0.5, rwidth=0.8, label = "component{}".format(i))
    else:
        dist_comps = [df[df['label']==num] for num in range(n)]
        xs = [np.array(list(i['dS'])) for i in dist_comps]
        ys = [x[np.isfinite(x)] for x in xs]
        ax.hist(ys, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color=colors, stacked=True, alpha=0.5, rwidth=0.8, label = ["component {}".format(int(i)) for i in range(n)])
    ax.legend(loc='upper right', fontsize='small',frameon=False)
    ax.set_xlabel("$K_\mathrm{S}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    if regime== 'multiplicon': plt.title('Multiplicon $K_\mathrm{S}$ GMM modeling')
    elif regime== 'segment': plt.title('Segment $K_\mathrm{S}$ GMM modeling')
    else: plt.title('Basecluster $K_\mathrm{S}$ GMM modeling')
    fig.tight_layout()
    return fig

def plot_ak_component(df,nums,bins=50,plot = 'identical',ylabel="Duplication events",weighted=True,regime='multiplicon'):
    colors = cm.viridis(np.linspace(0, 1, nums))
    fig, ax = plt.subplots()
    if plot == 'identical':
        if weighted:
            for num,color in zip(range(nums),colors):
                if nums == 1: df_comp = df.drop_duplicates(subset=['family','node'])
                else: df_comp = df[df['AnchorKs_GMM_Component']==num]
                w = df_comp['weightoutlierexcluded']
                x = np.array(list(df_comp['dS']))
                y = x[np.isfinite(x)]
                w = w[np.isfinite(x)]
                ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, weights=w, alpha=0.5, rwidth=0.8, label = "component {}".format(num))
        else:
            for num,color in zip(range(nums),colors):
                if nums == 1: df_comp = df.drop_duplicates(subset=['family','node'])
                else: df_comp = df[df['AnchorKs_GMM_Component']==num].drop_duplicates(subset=['family','node'])
                x = np.array(list(df_comp['node_averaged_dS_outlierexcluded']))
                y = x[np.isfinite(x)]
                ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, alpha=0.5, rwidth=0.8, label = "component {}".format(num))
    else:
        if weighted:
            if nums == 1:  dist_comps = [df]
            else: dist_comps = [df[df['AnchorKs_GMM_Component']==num] for num in range(nums)]
            ws = [i['weightoutlierexcluded'] for i in dist_comps]
            xs = [np.array(list(i['dS'])) for i in dist_comps]
            ys = [x[np.isfinite(x)] for x in xs]
            ws = [w[np.isfinite(x)] for w,x in zip(ws,xs)]
            ax.hist(ys, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color=colors, stacked=True, weights=ws, alpha=0.5, rwidth=0.8, label = ["component {}".format(int(i)) for i in range(nums)])
        else:
            if nums == 1:  dist_comps = [df]
            else: dist_comps = [df[df['AnchorKs_GMM_Component']==num].drop_duplicates(subset=['family','node']) for num in range(nums)]
            xs = [np.array(list(i['node_averaged_dS_outlierexcluded'])) for i in dist_comps]
            ys = [x[np.isfinite(x)] for x in xs]
            ax.hist(ys, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color=colors, stacked=True, alpha=0.5, rwidth=0.8, label = ["component {}".format(int(i)) for i in range(nums)])
    ax.legend(loc='upper right', fontsize='small',frameon=False)
    ax.set_xlabel("$K_\mathrm{S}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    if regime== 'multiplicon': plt.title('Multilplicon-guided Anchor $K_\mathrm{S}$ GMM modeling')
    elif regime== 'segment': plt.title('Segment-guided Syntelog $K_\mathrm{S}$ GMM modeling')
    elif regime== 'original': plt.title('Original Anchor $K_\mathrm{S}$ GMM modeling')
    else: plt.title('Basecluster-guided Anchor $K_\mathrm{S}$ GMM modeling')
    fig.tight_layout()
    return fig

def plot_ak_component_lognormal(df,means,stds,weights,nums,bins=50,ylabel="Duplication events",weighted=True,regime='multiplicon'):
    colors = cm.viridis(np.linspace(0, 1, nums))
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    fig, ax = plt.subplots()
    if weighted:
        for num,color in zip(range(nums),colors):
            mean,std,weight = means[num][0],np.sqrt(stds[num][0][0]),weights[num]
            if nums == 1: df_comp = df.copy()
            else: df_comp = df[df['AnchorKs_GMM_Component']==num]
            w = df_comp['weightoutlierexcluded']
            x = np.array(list(df_comp['dS']))
            y = x[np.isfinite(x)]
            w = w[np.isfinite(x)]
            if len(y) < 2:
                logging.info("Detected one component with less than 2 elements, will skip it")
                continue
            Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, weights=w, alpha=0.5, rwidth=0.8, label = "component {}".format(num))
            CHF = get_totalH(Hs)
            scaling = CHF*0.1
            ax.plot(kde_x,scaling*weight*stats.lognorm.pdf(kde_x, scale=np.exp(mean),s=std), c=color, ls='-', lw=1.5, alpha=0.8, label='component {} mode {:.2f}'.format(num+1,np.exp(mean - std**2)))
    else:
        for num,color in zip(range(nums),colors):
            mean,std,weight = means[num][0],np.sqrt(stds[num][0][0]),weights[num]
            if nums == 1: df_comp = df.drop_duplicates(subset=['family','node'])
            else: df_comp = df[df['AnchorKs_GMM_Component']==num].drop_duplicates(subset=['family','node'])
            x = np.array(list(df_comp['node_averaged_dS_outlierexcluded']))
            y = x[np.isfinite(x)]
            if len(y) < 2:
                logging.info("Detected one component with less than 2 elements, will skip it")
                continue
            Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, alpha=0.5, rwidth=0.8, label = "component {}".format(num))
            CHF = get_totalH(Hs)
            scaling = CHF*0.1
            ax.plot(kde_x,scaling*weight*stats.lognorm.pdf(kde_x, scale=np.exp(mean),s=std), c=color, ls='-', lw=1.5, alpha=0.8, label='component {} mode {:.2f}'.format(num+1,np.exp(mean - std**2)))
    ax.legend(loc='upper right', fontsize='small',frameon=False)
    ax.set_xlabel("$K_\mathrm{S}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    if regime=='multiplicon': plt.title('Multilplicon-guided Anchor $K_\mathrm{S}$ GMM modeling')
    elif regime=='segment': plt.title('Segment-guided Syntelog $K_\mathrm{S}$ GMM modeling')
    elif regime== 'original': plt.title('Original Anchor $K_\mathrm{S}$ GMM modeling')
    else: plt.title('Basecluster-guided Anchor $K_\mathrm{S}$ GMM modeling')
    fig.tight_layout()
    return fig

def plot_ak_component_kde(df,nums,hdr,bins=50,ylabel="Duplication events",weighted=True,regime='multiplicon'):
    colors = cm.viridis(np.linspace(0, 1, nums))
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    fig, ax = plt.subplots()
    Hs_maxs,y_lim_beforekde_s,HDRs = [],[],{}
    if weighted:
        for num,color in zip(range(nums),colors):
            #if nums == 1: df_comp = df.drop_duplicates(subset=['family','node'])
            if nums == 1: df_comp = df.copy()
            else: df_comp = df[df['AnchorKs_GMM_Component']==num]
            w = df_comp['weightoutlierexcluded']
            x = np.array(list(df_comp['dS']))
            y = x[np.isfinite(x)]
            w = w[np.isfinite(x)]
            if len(y) < 2:
                logging.info("Detected one component with less than 2 elements, will skip it")
                continue
            kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
            kde_y = kde(kde_x)
            mode, maxim = kde_mode(kde_x, kde_y)
            Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, weights=w, alpha=0.5, rwidth=0.8, label = "component {} (mode {:.2f})".format(num,mode))
            Hs_max = max(Hs)
            Hs_maxs.append(Hs_max)
            y_lim_beforekde = ax.get_ylim()[1]
            y_lim_beforekde_s.append(y_lim_beforekde)
            CHF = get_totalH(Hs)
            scaling = CHF*0.1
            ax.plot(kde_x, kde_y*scaling, color=color,alpha=0.4, ls = '--',lw = 1)
            y_lim_afterkde = ax.get_ylim()[1]
            if y_lim_afterkde > y_lim_beforekde: ax.set_ylim(0, y_lim_beforekde)
            safe_max = max([max(y_lim_beforekde_s),max(Hs_maxs)])
            ax.set_ylim(0, safe_max)
            ax.axvline(x = mode, color = color, alpha = 0.8, ls = ':', lw = 1)
            upper_HPD,lower_HPD = calculateHPD(y,hdr)
            ax.axvline(x = upper_HPD, color = color, alpha = 0.8, ls = '-.', lw = 1,label='HDR {:.2f}-{:.2f}'.format(lower_HPD,upper_HPD))
            ax.axvline(x = lower_HPD, color = color, alpha = 0.8, ls = '-.', lw = 1)
            HDRs[num] = (lower_HPD,upper_HPD)
    else:
        for num,color in zip(range(nums),colors):
            if nums == 1: df_comp = df.drop_duplicates(subset=['family','node'])
            else: df_comp = df[df['AnchorKs_GMM_Component']==num].drop_duplicates(subset=['family','node'])
            x = np.array(list(df_comp['node_averaged_dS_outlierexcluded']))
            y = x[np.isfinite(x)]
            if len(y) < 2:
                logging.info("Detected one component with less than 2 elements, will skip it")
                continue
            kde = stats.gaussian_kde(y,bw_method=0.1)
            kde_y = kde(kde_x)
            mode, maxim = kde_mode(kde_x, kde_y)
            Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, alpha=0.5, rwidth=0.8, label = "component {} (mode {:.2f})".format(num,mode))
            Hs_max = max(Hs)
            Hs_maxs.append(Hs_max)
            y_lim_beforekde = ax.get_ylim()[1]
            y_lim_beforekde_s.append(y_lim_beforekde)
            CHF = get_totalH(Hs)
            scaling = CHF*0.1
            ax.plot(kde_x, kde_y*scaling, color=color,alpha=0.4, ls = '--',lw = 1)
            y_lim_afterkde = ax.get_ylim()[1]
            if y_lim_afterkde > y_lim_beforekde: ax.set_ylim(0, y_lim_beforekde)
            safe_max = max([max(y_lim_beforekde_s),max(Hs_maxs)])
            ax.set_ylim(0, safe_max)
            ax.axvline(x = mode, color = color, alpha = 0.8, ls = ':', lw = 1)
            upper_HPD,lower_HPD = calculateHPD(y,hdr)
            ax.axvline(x = upper_HPD, color = color, alpha = 0.8, ls = '-.', lw = 1,label='HDR {:.2f}-{:.2f}'.format(lower_HPD,upper_HPD))
            ax.axvline(x = lower_HPD, color = color, alpha = 0.8, ls = '-.', lw = 1)
            HDRs[num] = (lower_HPD,upper_HPD)
    ax.legend(loc='upper right', fontsize='small',frameon=False)
    ax.set_xlabel("$K_\mathrm{S}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    if regime=='multiplicon': plt.title('Multilplicon-guided Anchor $K_\mathrm{S}$ GMM modeling')
    elif regime=='segment': plt.title('Segment-guided Syntelog $K_\mathrm{S}$ GMM modeling')
    elif regime== 'original': plt.title('Original Anchor $K_\mathrm{S}$ GMM modeling')
    else: plt.title('Basecluster-guided Anchor $K_\mathrm{S}$ GMM modeling')
    fig.tight_layout()
    return fig,HDRs

def default_plot_kde(*args,bins=50,alphas=None,colors=None,weighted=True,title="",ylabel="Duplication events",nums = "", plot = 'identical',**kwargs):
    ndists = len(args)
    alphas = alphas or list(np.linspace(0.2, 1, ndists))
    colors = colors or ['black'] * ndists
    # assemble panels
    keys = ["dS", "dS", "dN", "dN/dS"]
    np.seterr(divide='ignore')
    funs = [lambda x: x, np.log10, np.log10, np.log10]
    fig, axs = plt.subplots(2, 2)
    _labels = {"dS" : "$K_\mathrm{S}$","dN" : "$K_\mathrm{A}$","dN/dS": "$\omega$"}
    bins = 50
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    #color_table = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    for (c, a, dist) in zip(colors, alphas, args):
        dis = dist.dropna(subset=['weightoutlierexcluded'])
        #y_lim = []
        #time = 0
        y_lim_beforekde_s,Hs_maxs = [],[]
        for ax, k, f in zip(axs.flatten(), keys, funs):
            #color_random = cm.rainbow(np.linspace(0, 1, n))
            #time = time + 1
            #comp_time = 0
            for num, color in zip(range(nums),cm.viridis(np.linspace(0, 1, nums))):
                dist_comp = dis[dis['component']==num]
                w = dist_comp['weightoutlierexcluded']
                x = f(dist_comp[k])
                y = x[np.isfinite(x)]
                w = w[np.isfinite(x)]
                if funs[0] == f:
                    #print(y.shape)
                    Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, weights=w, alpha=a, rwidth=0.8, label = "component {}".format(num),**kwargs)
                    #print(Hs[0])
                    Hs_max = max(Hs)
                    Hs_maxs.append(Hs_max)
                    #y_lim.append(ax.get_ylim()[1])
                    #ax.set_ylim(0, max(y_lim) * 1.2)
                    y_lim_beforekde = ax.get_ylim()[1]
                    #print(y_lim_beforekde)
                    y_lim_beforekde_s.append(y_lim_beforekde)
                    kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
                    kde_y = kde(kde_x)
                    mode, maxim = kde_mode(kde_x, kde_y)
                    CHF = get_totalH(Hs)
                    scale = CHF*0.1
                    ax.plot(kde_x, kde_y*scale, color=color,alpha=0.4, ls = '--',lw = 1)
                    y_lim_afterkde = ax.get_ylim()[1]
                    #print(y_lim_afterkde)
                    if y_lim_afterkde > y_lim_beforekde: ax.set_ylim(0, y_lim_beforekde)
                    #y_lim_done = ax.get_ylim()[1]
                    #y_lim_dones.append(y_lim_done)
                    #safe_max = max([max(y_lim_beforekde_s),max(Hs_maxs),max(y_lim_dones)])
                    safe_max = max([max(y_lim_beforekde_s),max(Hs_maxs)])
                    ax.set_ylim(0, safe_max)
                    #yticks = ax.get_yticks()
                    #if y_lim_beforekde < max(y_lim_beforekde_s):
                    #    if max(y_lim_beforekde_s) > max(y_lim_dones): ax.set_ylim(0, max(y_lim_beforekde_s))
                    #    else: ax.set_ylim(0, max(y_lim_dones))
                    ax.axvline(x = mode, color = color, alpha = 0.8, ls = ':', lw = 1)
                    ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1,frameon=False)
                else:
                    #comp_time = comp_time + 1
                    #if comp_time < 2: Hs, Bins, patches = ax.hist(y, bins = 50, color = color, weights=w, alpha=a, rwidth=0.8, label = "component{}".format(num),**kwargs)
                    ax.hist(y, bins = np.linspace(-50, 20, num=70+1,dtype=int)/10, color = color, weights=w, alpha=a, rwidth=0.8, label = "component {}".format(num),**kwargs)
                    ax.legend(loc='upper left', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1,frameon=False)
                    #Hss.append(Hs)
                    #Binss.append(Bins)
                    #patchess.append(patches)
                #kde_x, kde_y = get_kde(train_in,ax)
                #ax.plot(kde_x,kde_y)
            #else:
            #    cs = [color for color in cm.rainbow(np.linspace(0, 1, nums))]
            #    dist_comps = [dis[dis['component']==num] for num in range(nums)]
            #    ws = [i['weightoutlierexcluded'] for i in dist_comps]
            #    xs = [f(i[k]) for i in dist_comps]
            #    ys = [x[np.isfinite(x)] for x in xs]
            #    ws = [w[np.isfinite(x)] for w,x in zip(ws,xs)]
            #    if funs[0] == f:
            #        ax.hist(ys, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color=[cs[i] for i in range(nums)], stacked=True, weights=ws, alpha=a, rwidth=0.8, label = ["component{}".format(int(i)) for i in range(nums)],**kwargs)
            #        ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
            #    else:
                    #maxi = int(max(list(itertools.chain.from_iterable(xs)))*100)
                    #mini = int(min(list(itertools.chain.from_iterable(xs)))*100)
            #        ax.hist(ys, bins = bins, color=[cs[i] for i in range(nums)], weights=ws, alpha=a, rwidth=0.8, label = ["component{}".format(int(i)) for i in range(nums)],**kwargs)
            #        ax.legend(loc='upper left', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
                    #ax.set_xticks(np.round_(np.linspace(mini, maxi, num=5,dtype=int)/100,decimals = 1))
                #kde_x, kde_y = get_kde(train_in,ax)
            #leg = ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
            #for lh in leg.legendHandles: lh.set_alpha(0.1)
        xlabel = _labels[k]
        if f == np.log10:
            xlabel = "$\log_{10}" + xlabel[1:-1] + "$"
        ax.set_xlabel(xlabel)
        #original_y_lim = ax.get_ylim()[1]
        #ax.set_ylim(0, original_y_lim * 1.2)
    yticks = axs[0,0].get_yticks()
    yticklabels = axs[0,0].get_yticklabels()
    #print(yticks)
    #print(safe_max)
    #print(yticklabels)
    axs[0,0].set_ylabel(ylabel)
    axs[0,0].set_yticks(axs[0,0].get_yticks())
    axs[0,0].set_ylim(0, safe_max)
    axs[1,0].set_ylabel(ylabel)
    axs[0,0].set_xticks([0,1,2,3,4,5])
    # finalize plot
    sns.despine(offset=1)
    fig.suptitle(title, x=0.125, y=0.9, ha="left", va="top")
    fig.tight_layout()
    plt.subplots_adjust(top=0.85)  # prevent suptitle from overlapping
    return fig,safe_max,yticks

def default_plot(*args,bins=50,alphas=None,colors=None,weighted=True,title="",ylabel="Duplication events",nums = "", plot = 'identical', ylim=1500,yticks='',**kwargs):
    ndists = len(args)
    alphas = alphas or list(np.linspace(0.2, 1, ndists))
    colors = colors or ['black'] * ndists
    # assemble panels
    keys = ["dS", "dS", "dN", "dN/dS"]
    np.seterr(divide='ignore')
    funs = [lambda x: x, np.log10, np.log10, np.log10]
    fig, axs = plt.subplots(2, 2)
    _labels = {"dS" : "$K_\mathrm{S}$","dN" : "$K_\mathrm{A}$","dN/dS": "$\omega$"}
    #color_table = ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    for (c, a, dist) in zip(colors, alphas, args):
        dis = dist.dropna(subset=['weightoutlierexcluded'])
        #time = 0
        for ax, k, f in zip(axs.flatten(), keys, funs):
            #color_random = cm.rainbow(np.linspace(0, 1, n))
            #time = time + 1
            #comp_time = 0
            if plot == 'identical':
                for num, color in zip(range(nums),cm.viridis(np.linspace(0, 1, nums))):
                    dist_comp = dis[dis['component']==num]
                    w = dist_comp['weightoutlierexcluded']
                    x = f(dist_comp[k])
                    y = x[np.isfinite(x)]
                    w = w[np.isfinite(x)]
                    if funs[0] == f:
                        ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, weights=w, alpha=a, rwidth=0.8, label = "component {}".format(num),**kwargs)
                        #Hs_max = max(Hs)
                        #Hs_maxs.append(Hs_max)
                        #y_lim_beforekde = ax.get_ylim()[1]
                        #y_lim_beforekde_s.append(y_lim_beforekde)
                        #y_lim.append(ax.get_ylim()[1])
                        #ax.set_ylim(0, max(y_lim) * 1.2)
                        #safe_max = max([max(y_lim_beforekde_s),max(Hs_maxs)])
                        ax.set_yticks(yticks)
                        ax.set_ylim(0, ylim)
                        ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
                    else:
                        #comp_time = comp_time + 1
                        #if comp_time < 2: Hs, Bins, patches = ax.hist(y, bins = 50, color = color, weights=w, alpha=a, rwidth=0.8, label = "component{}".format(num),**kwargs)
                        ax.hist(y, bins = np.linspace(-50, 20, num=70+1,dtype=int)/10, color = color, weights=w, alpha=a, rwidth=0.8, label = "component {}".format(num),**kwargs)
                        ax.legend(loc='upper left', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
                        #Hss.append(Hs)
                        #Binss.append(Bins)
                        #patchess.append(patches)
                    #kde_x, kde_y = get_kde(train_in,ax)
                    #ax.plot(kde_x,kde_y)
            else:
                cs = [color for color in cm.viridis(np.linspace(0, 1, nums))]
                dist_comps = [dis[dis['component']==num] for num in range(nums)]
                ws = [i['weightoutlierexcluded'] for i in dist_comps]
                xs = [f(i[k]) for i in dist_comps]
                ys = [x[np.isfinite(x)] for x in xs]
                ws = [w[np.isfinite(x)] for w,x in zip(ws,xs)]
                if funs[0] == f:
                    ax.hist(ys, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color=[cs[i] for i in range(nums)], stacked=True, weights=ws, alpha=a, rwidth=0.8, label = ["component {}".format(int(i)) for i in range(nums)],**kwargs)
                    ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1,frameon=False)
                else:
                    #maxi = int(max(list(itertools.chain.from_iterable(xs)))*100)
                    #mini = int(min(list(itertools.chain.from_iterable(xs)))*100)
                    ax.hist(ys, bins = bins, color=[cs[i] for i in range(nums)], weights=ws, alpha=a, rwidth=0.8, label = ["component {}".format(int(i)) for i in range(nums)],**kwargs)
                    ax.legend(loc='upper left', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1,frameon=False)
                    #ax.set_xticks(np.round_(np.linspace(mini, maxi, num=5,dtype=int)/100,decimals = 1))
                #kde_x, kde_y = get_kde(train_in,ax)
            #leg = ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
            #for lh in leg.legendHandles: lh.set_alpha(0.1)
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

def kde_fill(df,outdir,weighted):
    if weighted:
        df = df.dropna(subset=['weightoutlierexcluded'])

        sns.kdeplot(data=tips, x="total_bill", hue="time", multiple="fill")
    #else:

def add_prediction(ksdf,fn_ksdf,train_in,m):
    data = {'component':m.predict(train_in)}
    predict_column = pd.DataFrame(data,index = fn_ksdf.index).reset_index()
    ksdf_reset = ksdf.reset_index()
    ksdf_predict = ksdf_reset.merge(predict_column, on = ['family','node'])
    return ksdf_predict.set_index('pair')

def fit_gmm(out_file,X, seed, n1, n2, em_iter=100, n_init=1):
    """
    Compute Gaussian mixtures for different numbers of components
    """
    N = np.arange(n1, n2 + 1)
    models = [None for i in N]
    info_table = {}
    for i in N:
        logging.info("Fitting GMM with {} components".format(i))
        models[i-n1] = mixture.GaussianMixture(n_components = i, covariance_type='full', max_iter = em_iter, n_init = n_init, random_state = seed).fit(X)
        if models[i-n1].converged_:
            logging.info("Convergence reached")
        info_components(models[i-n1],i,info_table)
    aic = [m.aic(X) for m in models]
    bic = [m.bic(X) for m in models]
    besta = models[np.argmin(aic)]
    bestb = models[np.argmin(bic)]
    logging.info("The best fitted model via AIC is with {} components".format(np.argmin(aic)+n1))
    aic_info(aic,n1)
    logging.info("The best fitted model via BIC is with {} components".format(np.argmin(bic)+n1))
    bic_info(bic,n1)
    plot_aic_bic(aic, bic, n1, n2, out_file)
    return models, aic, bic, besta, bestb, N

def fit_bgmm(X, seed, n1, n2, em_iter=100, n_init=1):
    """
    Variational Bayesian estimation of a Gaussian mixture
    """
    N = np.arange(n1, n2 + 1)
    models = [None for i in N]
    info_table = {}
    for i in N:
        logging.info("Fitting BGMM with {} components".format(i))
        #'dirichlet_distribution' (can favor more uniform weights) while 'dirichlet_process' (default weight_concentration_prior_type)
        #(using the Stick-breaking representation) seems better
        # default weight_concentration_prior is 1/n_components
        # default mean_precision_prior is 1
        # default mean_prior is the mean of X
        # default degrees_of_freedom_prior is n_features
        # default covariance_prior is the covariance of X
        models[i-n1] = mixture.GaussianMixture(n_components = i, covariance_type='full', max_iter = em_iter, n_init = n_init, random_state = seed).fit(X)
        if models[i-n1].converged_:
            logging.info("Convergence reached")
        info_components(models[i-n1],i,info_table)
    return models, N

def get_gaussian_kde(train_nonan, ks_lower, ks_upper, bin_width, ksdf_filtered, weight_col, weighted=False):
    #default bw_method is 'scott'
    if weighted:
        kde = stats.gaussian_kde(ksdf_filtered['dS'], weights=weight_col)
        kde.set_bandwidth(kde.factor * 0.7)
    else:
        kde = stats.gaussian_kde(train_nonan)
        kde.set_bandwidth(kde.factor * 0.7)
    kde_x = np.linspace(ks_lower, ks_upper, num=500)
    kde_y = kde(kde_x)
    kde_y = kde_y * bin_width * len(train_nonan) #Adaptable
    return kde_x, kde_y

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

#def bootstrap_dates(nd,)
def bootstrap_kde(kdemethod,outdir,train_in, ks_lower, ks_upper, boots, bin_width, ksdf_filtered, weight_col, weighted = False):
    train_nonan = train_in[~np.isnan(train_in)]
    modes = []
    medians = []
    kde_x = np.linspace(ks_lower, ks_upper, num=500)
    for r in range(boots):
        if weighted:
            #bootstrap_sample = random.choices(ksdf_filtered['dS'], weights = weight_col, k=len(weight_col))
            bootstrap_sample = random.choices(ksdf_filtered['dS'], k=len(weight_col))
            #kde_x, kde_y = get_gaussian_kde(bootstrap_sample, ks_lower, ks_upper, bin_width, ksdf_filtered, weight_col, weighted)
            if kdemethod == 'scipy':kde_y=stats.gaussian_kde(bootstrap_sample,bw_method=bin_width,weights=weight_col.tolist()).pdf(kde_x)
            if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(bootstrap_sample,weights=weight_col.tolist()).evaluate(kde_x)
            if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(bootstrap_sample,weights=weight_col.tolist()).evaluate(kde_x)
            if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(bootstrap_sample,weights=weight_col.tolist()).evaluate(kde_x)
        else:
            bootstrap_sample = random.choices(train_nonan, k=len(train_nonan))
            if kdemethod == 'scipy': kde_y=stats.gaussian_kde(bootstrap_sample,bw_method=bin_width).pdf(kde_x)
            if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(bootstrap_sample,weights=None).evaluate(kde_x)
            if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(bootstrap_sample,weights=None).evaluate(kde_x)
            if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(bootstrap_sample,weights=None).evaluate(kde_x)
            #kde_x, kde_y = get_gaussian_kde(bootstrap_sample, ks_lower, ks_upper, bin_width, ksdf_filtered, weight_col, weighted)
        plt.plot(kde_x, kde_y, color = 'black',alpha=0.1)
        mode, maxim = kde_mode(kde_x, kde_y)
        modes.append(mode)
        medians.append(np.median(bootstrap_sample))
    if weighted:
        plt.hist(ksdf_filtered['dS'],bins = np.linspace(0, 50, num=51,dtype=int)/10,density=True,color = 'black', alpha=0.1, rwidth=0.8)
    else:
        plt.hist(train_nonan,bins = np.linspace(0, 50, num=51,dtype=int)/10,density=True,color = 'black', alpha=0.1, rwidth=0.8)
    plt.tight_layout()
    fname = os.path.join(outdir, "ksdf_boots.pdf")
    plt.savefig(fname,format ='pdf',bbox_inches='tight')
    plt.close()
    mean_modes = np.mean(modes, dtype=np.float)
    std_modes = np.std(modes, dtype=np.float)
    mean_medians = np.mean(medians, dtype=np.float)
    std_medians = np.std(medians, dtype=np.float)
    logging.info("The mean of kde modes from {0} bootstrap replicates is {1}".format(boots,mean_modes))
    logging.info("The std of kde modes from {0} bootstrap replicates is {1}".format(boots,std_modes))
    logging.info("The mean of Ks medians from {0} bootstrap replicates is {1}".format(boots,mean_medians))
    logging.info("The std of Ks medians from {0} bootstrap replicates is {1}".format(boots,std_medians))
    return mean_modes, std_modes, mean_medians, std_medians

def kde(train_in,bin_width):
    m = KernelDensity(bandwidth=bin_width).fit(train_in)
    ll = m.score_samples(train_in)
    plt.fill(train_in, np.exp(ll), c='cyan')

def get_empirical_CI(alpha,data):
    p = ((1.0-alpha)/2.0)
    lower = np.quantile(data, p)
    p = (alpha+((1.0-alpha)/2.0))
    upper = np.quantile(data, p)
    return lower, upper

def get_kde(kdemethod,outdir,fn_ksdf,ksdf_filtered,weighted,ks_lower,ks_upper):
    df = ksdf_filtered.dropna(subset=['weightoutlierexcluded'])
    kde_x = np.linspace(ks_lower,ks_upper, num=500)
    if weighted:
        if kdemethod == 'scipy': kde_y=stats.gaussian_kde(df['dS'].tolist(),weights=df['weightoutlierexcluded'].tolist(),bw_method=0.1).pdf(kde_x)
        if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=0.1).fit(df['dS'].tolist(),weights=df['weightoutlierexcluded'].tolist()).evaluate(kde_x)
        if kdemethod == 'treekde': kde_y = TreeKDE(bw=0.1).fit(df['dS'].tolist(),weights=df['weightoutlierexcluded'].tolist()).evaluate(kde_x)
        if kdemethod == 'fftkde': kde_y = FFTKDE(bw=0.1).fit(df['dS'].tolist(),weights=df['weightoutlierexcluded'].tolist()).evaluate(kde_x)
        plt.hist(df['dS'], bins = np.linspace(0, 50, num=51,dtype=int)/10, color = 'black', weights=df['weightoutlierexcluded'], alpha=0.2, rwidth=0.8)
    else:
        X = np.array(fn_ksdf["dS"].dropna())
        if kdemethod == 'scipy': kde_y=stats.gaussian_kde(X,bw_method='silverman').pdf(kde_x)
        if kdemethod == 'naivekde': kde_y = NaiveKDE(bw='silverman').fit(train_in,weights=None).evaluate(kde_x)
        if kdemethod == 'treekde': kde_y = TreeKDE(bw='silverman').fit(train_in,weights=None).evaluate(kde_x)
        if kdemethod == 'fftkde': kde_y = FFTKDE(bw='silverman').fit(train_in,weights=None).evaluate(kde_x)
        plt.hist(X, bins = np.linspace(0, 50, num=51,dtype=int)/10, color = 'black', alpha=0.2, rwidth=0.8)
    plt.plot(kde_x, kde_y, color = 'black',alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(outdir, "ksd_filtered.pdf")
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def Ten_multi(num):
    left=num%10
    return num-left

def draw_kde_CI(kdemethod,outdir,ksdf,boots,bw_method,date_lower = 0,date_upper=4,**kwargs):
    train_in = ksdf['PM']
    maxm = float(train_in.max())
    minm = float(train_in.min())
    kde_x = np.linspace(minm,maxm,num=500)
    modes = []
    f, ax = plt.subplots()
    if bw_method == 'silverman': logging.info("Assmuing the dates is unimodal and close to normal, applying silverman’s rule of thumb")
    else: logging.info("Assmuing the dates is far from normal or multimodal, applying the Improved Sheather Jones (ISJ) algorithm")
    for i,color in zip(range(boots),cm.viridis(np.linspace(0, 1, boots))):
        sample = random.choices(train_in, k=len(train_in))
        #kde = stats.gaussian_kde(sample)
        if kdemethod == 'scipy': kde_y=stats.gaussian_kde(sample,bw_method=bw_method).pdf(kde_x)
        if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bw_method).fit(sample,weights=None).evaluate(kde_x)
        if kdemethod == 'treekde': kde_y = TreeKDE(bw=bw_method).fit(sample,weights=None).evaluate(kde_x)
        if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bw_method).fit(sample,weights=None).evaluate(kde_x)
        #kde_y = NaiveKDE(bw=bw_method).fit(sample,weights=None).evaluate(kde_x)
        mode, maxim = kde_mode(kde_x, kde_y)
        modes.append(mode)
        #kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, color = color,alpha=0.7)
    lower, upper = get_empirical_CI(0.95,modes)
    plt.axvline(x = lower, color = 'green', alpha = 0.8, ls = '--', lw = 1)
    plt.axvline(x = upper, color = 'green', alpha = 0.8, ls = '--', lw = 1)
    #https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    logging.info("The 95% empirical confidence interval (CI) of WGD dates is {0} - {1} billion years".format(lower,upper))
    if kdemethod == 'scipy': kde_y=stats.gaussian_kde(train_in.tolist(),bw_method=bw_method).pdf(kde_x)
    if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bw_method).fit(train_in.tolist(),weights=None).evaluate(kde_x)
    if kdemethod == 'treekde': kde_y = TreeKDE(bw=bw_method).fit(train_in.tolist(),weights=None).evaluate(kde_x)
    if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bw_method).fit(train_in.tolist(),weights=None).evaluate(kde_x)
    mode_orig, maxim_orig = kde_mode(kde_x, kde_y)
    plt.axvline(x = mode_orig, color = 'black', alpha = 0.8, ls = ':', lw = 1)
    if kdemethod == 'scipy': kde_y=stats.gaussian_kde(modes,bw_method=bw_method).pdf(kde_x)
    if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bw_method).fit(modes).evaluate(kde_x)
    if kdemethod == 'treekde': kde_y = TreeKDE(bw=bw_method).fit(modes).evaluate(kde_x)
    if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bw_method).fit(modes).evaluate(kde_x)
    mode_of_modes, maxim_of_modes = kde_mode(kde_x, kde_y)
    plt.axvline(x = mode_of_modes, color = 'red', alpha = 0.8, ls = '-.', lw = 1)
    logging.info("The kde-mode of original WGD dates is {} billion years".format(mode_orig))
    plt.xlabel("Billion years ago", fontsize = 10)
    plt.ylabel("Frequency", fontsize = 10)
    print(Ten_multi(int(maxim_orig)))
    plt.yticks(np.linspace(0,Ten_multi(int(maxim_orig))+10,num=10,dtype=int))
    plt.hist(train_in,bins = np.linspace(minm, maxm, num=50),density=True,color = 'black', alpha=0.15, rwidth=0.8)
    props = dict(boxstyle='round', facecolor='grey', alpha=0.1)
    text = "\n".join(["Raw mode: {:4.4f}".format(mode_of_modes),"Peak: {:4.4f}".format(mode_orig),"CI: {:4.4f}-{:4.4f}".format(lower, upper),"OGs: {}".format(len(train_in))])
    plt.text(0.75,0.95,text,transform=ax.transAxes,fontsize=8,verticalalignment='top',bbox=props)
    fname = os.path.join(outdir, "WGD_peak.pdf")
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def draw_components_kde_bootstrap(kdemethod,outdir,num,ksdf_predict,weighted,boots,bin_width):
    parent = os.getcwd()
    os.chdir(outdir)
    dir_tmp = _mkdir('{}-components_model'.format(num))
    os.chdir(dir_tmp)
    for n in range(num):
        fname = "component{}.pdf".format(n)
        pretrain_in = ksdf_predict[ksdf_predict['component']==n]
        maxm = float(pretrain_in['dS'].max())
        minm = float(pretrain_in['dS'].min())
        kde_x = np.linspace(0,5,num=500)
        modes = []
        f, ax = plt.subplots()
        if weighted:
            train_in = pretrain_in.dropna(subset=['weightoutlierexcluded'])
            for i,color in zip(range(boots),cm.viridis(np.linspace(0, 1, boots))):
                #sample = random.choices(train_in['dS'],weights = train_in['weightoutlierexcluded'], k=len(train_in))
                sample = random.choices(train_in['dS'], k=len(train_in))
                if kdemethod == 'scipy': kde_y=stats.gaussian_kde(sample,weights=train_in['weightoutlierexcluded'].tolist(),bw_method=bin_width).pdf(kde_x)
                if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(sample,weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
                if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(sample,weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
                if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(sample,weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
                mode, maxim = kde_mode(kde_x, kde_y)
                modes.append(mode)
                plt.plot(kde_x, kde_y, color = color,alpha=0.1)
            plt.hist(train_in['dS'],bins = np.linspace(0, 50, num=51,dtype=int)/10,weights=train_in['weightoutlierexcluded'],density=True,color = 'black', alpha=0.15, rwidth=0.8)
        else:
            train_in = pretrain_in.dropna(subset=['node_averaged_dS_outlierexcluded'])
            for i,color in zip(range(boots),cm.viridis(np.linspace(0, 1, boots))):
                sample = random.choices(train_in['node_averaged_dS_outlierexcluded'], k=len(train_in))
                if kdemethod == 'scipy': kde_y=stats.gaussian_kde(sample,bw_method=bin_width).pdf(kde_x)
                if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(sample).evaluate(kde_x)
                if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(sample).evaluate(kde_x)
                if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(sample).evaluate(kde_x)
                mode, maxim = kde_mode(kde_x, kde_y)
                modes.append(mode)
                plt.plot(kde_x, kde_y, color = color,alpha=0.1)
            plt.hist(train_in['node_averaged_dS_outlierexcluded'], bins = np.linspace(0, 50, num=51,dtype=int)/10,density=True, color = 'black', alpha=0.15, rwidth=0.8)
        #ax.set_xticks([0,1,2,3,4,5])
        #plt.xlim(0, 5)
        plt.xlabel("$K_\mathrm{S}$", fontsize = 10)
        plt.ylabel("Frequency", fontsize = 10)
        lower, upper = get_empirical_CI(0.95,modes)
        plt.axvline(x = lower, color = 'green', alpha = 0.8, ls = '--', lw = 1)
        plt.axvline(x = upper, color = 'green', alpha = 0.8, ls = '--', lw = 1)
        if weighted:
            train_in = pretrain_in.dropna(subset=['weightoutlierexcluded'])
            if kdemethod == 'scipy': kde_y=stats.gaussian_kde(train_in['dS'].tolist(),bw_method=bin_width,weights=train_in['weightoutlierexcluded']).pdf(kde_x)
            if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(train_in['dS'].tolist(),weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
            if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(train_in['dS'].tolist(),weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
            if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(train_in['dS'].tolist(),weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
        else:
            train_in = pretrain_in.dropna(subset=['node_averaged_dS_outlierexcluded'])
            if kdemethod == 'scipy': kde_y=stats.gaussian_kde(train_in['node_averaged_dS_outlierexcluded'].tolist(),bw_method=bin_width).pdf(kde_x)
            if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(train_in['node_averaged_dS_outlierexcluded'].tolist()).evaluate(kde_x)
            if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(train_in['node_averaged_dS_outlierexcluded'].tolist()).evaluate(kde_x)
            if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(train_in['node_averaged_dS_outlierexcluded'].tolist()).evaluate(kde_x)
        mode_orig, maxim_orig = kde_mode(kde_x, kde_y)
        #print(modes)
        if all([i==0 for i in modes]):
            #modes = np.array(modes)+1e-10
            modes = [i+np.random.choice(np.arange(1e-10, 1e-8, 1e-10)) for i in modes]
        if kdemethod == 'scipy': kde_y=stats.gaussian_kde(np.array(modes),bw_method=bin_width).pdf(kde_x)
        if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bin_width).fit(modes).evaluate(kde_x)
        if kdemethod == 'treekde': kde_y = TreeKDE(bw=bin_width).fit(modes).evaluate(kde_x)
        if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bin_width).fit(modes).evaluate(kde_x)
        mode_of_modes, maxim_of_modes = kde_mode(kde_x, kde_y)
        plt.axvline(x = mode_orig, color = 'black', alpha = 0.8, ls = ':', lw = 1)
        plt.axvline(x = mode_of_modes, color = 'red', alpha = 0.8, ls = '-.', lw = 1)
        props = dict(boxstyle='round', facecolor='gray', alpha=0.1)
        if weighted: l = len(pretrain_in.dropna(subset=['weightoutlierexcluded']))
        else: l = len(pretrain_in.dropna(subset=['node_averaged_dS_outlierexcluded']))
        text = "\n".join(["Raw mode: {:4.4f}".format(mode_orig),"Peak: {:4.4f}".format(mode_of_modes),"CI: {:4.4f}-{:4.4f}".format(lower, upper),"dS: {}".format(l)])
        plt.text(0.75,0.95,text,transform=ax.transAxes,fontsize=8,verticalalignment='top',bbox=props)
        plt.tight_layout()
        plt.savefig(fname,format ='pdf', bbox_inches='tight')
        plt.close()
    os.chdir(parent)

def info_centers(cluster_centers):
    centers = []
    for i,c in enumerate(cluster_centers):
        c = np.exp(c[0])
        centers.append(c)
        logging.info("cluster {0} centered at {1}".format(i,c))
    return centers

def write_labels(df,df_index,labels,outdir,n,regime='multiplicon'):
    predict_column = pd.DataFrame(labels,index=df_index.index,columns=['KMedoids_Cluster']).reset_index()
    df = df.reset_index().merge(predict_column, on = [regime]).set_index('pair')
    fname = os.path.join(outdir,'{}-guide_Ks_KMedoids_Clustering_{}components_prediction.tsv'.format(regime,n))
    df.to_csv(fname,header=True,index=True,sep='\t')
    return df

def get_CDF_CI(kde_y,alpha,num=50):
    CDFs = []
    CDF = 0
    lower = 0
    upper = 0
    i_low = 0
    i_upp = 0
    p=(1-alpha)/2
    for i in range(num):
        CDF = CDF + kde_y[i]
        CDFs.append(CDF)
        if CDF >= p*kde_y.sum() and lower == 0:
            lower = kde_y[i]
            i_low = i
        if CDF >= (1-p)*kde_y.sum() and upper == 0 :
            upper = kde_y[i]
            i_upp = i+1
    return i_low,i_upp

def get_totalP(kde_y,num=1001):
    """
    The total Proability increases with the num, so the scale of kde_y needs to adapt with the ratio of num:bins
    """
    CDF = 0
    for i in range(num): CDF = CDF + kde_y[i]
    return CDF

def get_totalH(Hs):
    CHF = 0
    for i in Hs: CHF = CHF + i
    return CHF

def kde_method(kdemethod,bw,X,kde_x,w=None):
    if kdemethod == 'scipy':
        kde = stats.gaussian_kde(X,weights=w,bw_method=bw)
        bw_modifier = 1
        kde.set_bandwidth(kde.factor * bw_modifier)
        kde_y = kde(kde_x)
    if kdemethod == 'naivekde': kde_y = NaiveKDE(bw=bw).fit(X,weights=w).evaluate(kde_x)
    if kdemethod == 'treekde': kde_y = TreeKDE(bw=bw).fit(X,weights=w).evaluate(kde_x)
    if kdemethod == 'fftkde': kde_y = FFTKDE(bw=bw).fit(X,weights=w).evaluate(kde_x)
    return kde_y

def plot_kmedoids_kde(boots,kdemethod,dfo,outdir,n,bin_width,bins=50,weighted=True,title="",plot='identical',alpha=0.50,regime='multiplicon'):
    fname = os.path.join(outdir,"{}-guide_AnchorKs_KMedoids_Clustering_{}components_kde.pdf".format(regime,n))
    f, ax = plt.subplots()
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    modes = {i:[] for i in range(n)}
    orig_modes = []
    css = []
    CIs = {}
    if weighted:
        if plot == 'identical':
            for i,c in zip(range(n),cm.viridis(np.linspace(0, 1, n))):
                css.append(c)
                df = dfo[dfo['KMedoids_Cluster']==i]
                df = df.dropna(subset=['weightoutlierexcluded'])
                X = getX(df,'dS')
                w = getX(df,'weightoutlierexcluded')
                kde_y = kde_method(kdemethod,bin_width,X,kde_x,w=w)
                mode, maxim = kde_mode(kde_x, kde_y)
                orig_modes.append(mode)
                Hs, Bins, patches = plt.hist(X,bins = np.linspace(0, 5, num=int(5/bin_width)+1),weights=w,color=c, alpha=0.5, rwidth=0.8,label='component {} (mode {:.2f})'.format(i,mode))
                CHF = get_totalH(Hs)
                scale = CHF*bin_width
                plt.plot(kde_x, kde_y*scale, color = c,alpha=0.4, ls = '--')
                plt.axvline(x = mode, color = c, alpha = 0.8, ls = ':', lw = 1)
                i_low,i_upp = get_CDF_CI(Hs,alpha,num=int(5/bin_width))
                CIs[i]=[Bins[i_low],Bins[i_upp]]
        else:
            dfs = [dfo[dfo['KMedoids_Cluster']==i] for i in range(n)]
            dfs = [df.dropna(subset=['weightoutlierexcluded']) for df in dfs]
            cs = [c for c in cm.viridis(np.linspace(0, 1, n))]
            Xs = [getX(df,'dS') for df in dfs]
            ws = [getX(df,'weightoutlierexcluded') for df in dfs]
            labels = ['component {}'.format(i) for i in range(n)]
            plt.hist(Xs,bins = np.linspace(0, 5, num=int(5/bin_width)+1),weights=ws,color=cs,alpha=0.5,rwidth=0.8,stacked=True,label=labels)
    elif plot == 'identical':
        for i,c in zip(range(n),cm.viridis(np.linspace(0, 1, n))):
            css.append(c)
            dfo = dfo.drop_duplicates(subset=['family', 'node'])
            df = dfo[dfo['KMedoids_Cluster']==i]
            X = getX(df,'node_averaged_dS_outlierexcluded')
            kde_y = kde_method(kdemethod,bin_width,X,kde_x)
            mode, maxim = kde_mode(kde_x, kde_y)
            plt.axvline(x = mode, color = c, alpha = 0.8, ls = ':', lw = 1)
            orig_modes.append(mode)
            Hs, Bins, patches = plt.hist(X,np.linspace(0, 5, num=int(5/bin_width)+1),color = c, alpha=0.5, rwidth=0.8,label='component {} (mode {:.2f})'.format(i,mode))
            CHF = get_totalH(Hs)
            scale = CHF*bin_width
            plt.plot(kde_x, kde_y*scale, color = c,alpha=0.4,ls = '--')
            i_low,i_upp = get_CDF_CI(Hs,alpha,num=int(5/bin_width))
            CIs[i]=[Bins[i_low],Bins[i_upp]]
    else:
        dfo = dfo.drop_duplicates(subset=['family', 'node'])
        dfs = [dfo[dfo['KMedoids_Cluster']==i] for i in range(n)]
        cs = [c for c in cm.viridis(np.linspace(0, 1, n))]
        Xs = [getX(df,'node_averaged_dS_outlierexcluded') for df in dfs]
        labels = ['component {}'.format(i) for i in range(n)]
        plt.hist(Xs,bins = np.linspace(0, 5, num=int(5/bin_width)+1),color=cs,alpha=0.5,rwidth=0.8,stacked=True,label=labels)
    plt.xlabel("$K_\mathrm{S}$", fontsize = 10)
    plt.ylabel("Frequency", fontsize = 10)
    ax.legend(loc=1,fontsize='large',frameon=False)
    sns.despine(offset=1)
    if regime=='multiplicon': plt.title('Multilplicon-guided Anchor $K_\mathrm{S}$ KMedoid modeling')
    elif regime=='segment': plt.title('Segment-guided Anchor $K_\mathrm{S}$ KMedoid modeling')
    else: plt.title('Basecluster-guided Anchor $K_\mathrm{S}$ KMedoid modeling')
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def plot_segment_kmedoids(labels,X,outdir,bin_width,n,regime='segment'):
    fname = os.path.join(outdir,"{}_Ks_KMedoids_Clustering_{}components.pdf".format(regime,n))
    f, ax = plt.subplots()
    df = pd.DataFrame(labels,columns=['KMedoids_Cluster'])
    df.index.name = 'label'
    df.loc[:,['Segment_Ks']] = X
    df = df.reset_index()
    for i,c in zip(range(n),cm.viridis(np.linspace(0, 1, n))):
        if n == 1: dfo = df.copy()
        else: dfo = df[df['KMedoids_Cluster']==i]
        data = getX(dfo,'Segment_Ks')
        ax.hist(data,bins = np.linspace(0, 5, num=int(5/bin_width)+1),color=c,alpha=0.5,rwidth=0.8,label='component {}'.format(i))
    plt.xlabel("$K_\mathrm{S}$", fontsize = 10)
    plt.ylabel("Number of segment pair", fontsize = 10)
    ax.legend(loc=1,fontsize='small',frameon=False)
    sns.despine(offset=1)
    if regime=='multiplicon': plt.title('Multilplicon $K_\mathrm{S}$ KMedoid modeling')
    elif regime=='segment': plt.title('Segment  $K_\mathrm{S}$ KMedoid modeling')
    else: plt.title('Basecluster $K_\mathrm{S}$ KMedoid modeling')
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def plot_kmedoids(boots,kdemethod,dfo,outdir,n,bin_width,bins=50,weighted=True,title="",plot='identical',alpha=0.50,regime='multiplicon'):
    fname = os.path.join(outdir,"{}-guided_AnchorKs_KMedoids_Clustering_{}components.pdf".format(regime,n))
    f, ax = plt.subplots()
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    modes = {i:[] for i in range(n)}
    orig_modes = []
    css = []
    CIs = {}
    if weighted:
        if plot == 'identical':
            for i,c in zip(range(n),cm.viridis(np.linspace(0, 1, n))):
                css.append(c)
                df = dfo[dfo['KMedoids_Cluster']==i]
                df = df.dropna(subset=['weightoutlierexcluded'])
                X = getX(df,'dS')
                w = getX(df,'weightoutlierexcluded')
                Hs, Bins, patches = plt.hist(X,bins = np.linspace(0, 5, num=int(5/bin_width)+1),weights=w,color=c, alpha=0.5, rwidth=0.8,label='component {}'.format(i,mode))
        else:
            dfs = [dfo[dfo['KMedoids_Cluster']==i] for i in range(n)]
            dfs = [df.dropna(subset=['weightoutlierexcluded']) for df in dfs]
            cs = [c for c in cm.viridis(np.linspace(0, 1, n))]
            Xs = [getX(df,'dS') for df in dfs]
            ws = [getX(df,'weightoutlierexcluded') for df in dfs]
            plt.hist(Xs,bins = np.linspace(0, 5, num=int(5/bin_width)+1),weights=ws,color=cs,alpha=0.5,rwidth=0.8,stacked=True)
    elif plot == 'identical':
        for i,c in zip(range(n),cm.viridis(np.linspace(0, 1, n))):
            css.append(c)
            dfo = dfo.drop_duplicates(subset=['family', 'node'])
            df = dfo[dfo['KMedoids_Cluster']==i]
            X = getX(df,'node_averaged_dS_outlierexcluded')
            Hs, Bins, patches = plt.hist(X,np.linspace(0, 5, num=int(5/bin_width)+1),color = c, alpha=0.5, rwidth=0.8,label='component {}'.format(i))
    else:
        dfo = dfo.drop_duplicates(subset=['family', 'node'])
        dfs = [dfo[dfo['KMedoids_Cluster']==i] for i in range(n)]
        cs = [c for c in cm.viridis(np.linspace(0, 1, n))]
        Xs = [getX(df,'node_averaged_dS_outlierexcluded') for df in dfs]
        plt.hist(Xs,bins = np.linspace(0, 5, num=int(5/bin_width)+1),color=cs,alpha=0.5,rwidth=0.8,stacked=True,label='component {}'.format(i))
    plt.xlabel("$K_\mathrm{S}$", fontsize = 10)
    plt.ylabel("Frequency", fontsize = 10)
    ax.legend(loc=1,fontsize='large',frameon=False)
    sns.despine(offset=1)
    if regime=='multiplicon': plt.title('Multilplicon-guided Anchor $K_\mathrm{S}$ KMedoid modeling')
    elif regime=='segment': plt.title('Segment-guided Anchor $K_\mathrm{S}$ KMedoid modeling')
    else: plt.title('Basecluster-guided Anchor $K_\mathrm{S}$ KMedoid modeling')
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def getX(df,column,cutoff=5.0):
    X = np.array(df.loc[:,column].dropna())
    X = X[np.isfinite(X)]
    X = X[X<cutoff]
    return X

def Elbow_lossf(X_log,cluster_centers,labels):
    D = []
    for i,c in enumerate(cluster_centers):
        sum_d = 0
        for x,l in zip(X_log,labels):
            if l == i:
                sum_d = sum_d + (x - c)**2
        D.append(sum_d)
    Loss = sum(D)
    return Loss

def find_mpeak(df,anchor,sp,outdir,guide,peak_threshold=0.1,rel_height=0.4,ci=95,user_low=0,user_upp=1,user=False):
    gs_ks = df.loc[:,['gene1','gene2',guide,'dS']]
    df_withindex,ks_or = bc_group_anchor(df,regime=guide)
    mpKs = pd.DataFrame.from_dict({guide:df_withindex.index,'Median_Ks':ks_or}).set_index(guide)
    df_m = df_withindex.copy()
    df_m['weightoutlierexcluded'] = 1
    w = np.array(df_m['weightoutlierexcluded'])
    ks = np.log(ks_or)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5,2)
    ax.set_title('KDE and spline of log-transformed anchor $K_\mathrm{S}$ of species '+ '{}'.format(sp))
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
    spl_x = np.linspace(min_ks, max_ks+0.1, num=int(round((abs(min_ks) + (max_ks+0.1)) *100)))
    spl_y = spl(spl_x)
    ax.plot(kde_x, spl(kde_x), 'b', lw=1, label="Spline on KDE")
    ax.hist(ks_refed, weights=w_refed, bins=np.arange(-5, 2.1, 0.1), color="gray", alpha=0.8, density=True,rwidth=0.8)
    ax.legend(loc=2,fontsize='large',frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "{}_guided_{}_Ks_spline.pdf".format(guide,sp)))
    plt.close(fig)
    logging.info('Detecting likely peaks from {}-guided Ks data '.format(guide))
    init_means, init_stdevs, good_prominences = find_peak_init_parameters(spl_x,spl_y,sp,outdir,peak_threshold=peak_threshold,guide=guide,rel_height=rel_height)
    lower95CI,upper95CI = plot_95CI_lognorm_hist(init_means, init_stdevs, ks_or, w, outdir, False, sp, ci=ci,guide=guide)
    if user: get95CIap_MP(user_low,user_upp,anchor,gs_ks,outdir,sp,ci,guide,mpKs,user=user)
    else: get95CIap_MP(lower95CI,upper95CI,anchor,gs_ks,outdir,sp,ci,guide,mpKs,user=user)

def find_apeak(df,anchor,sp,outdir,peak_threshold=0.1,na=False,rel_height=0.4,ci=95,user_low=0,user_upp=1,user=False):
    gs_ks = df.loc[:,['gene1','gene2','dS']]
    if na:
        df = df.drop_duplicates(subset=['family','node'])
        df = df.loc[:,['node_averaged_dS_outlierexcluded']].copy().rename(columns={'node_averaged_dS_outlierexcluded':'dS'})
        df['weightoutlierexcluded'] = 1
    df = df.dropna(subset=['dS','weightoutlierexcluded'])
    df = df.loc[(df['dS']>0) & (df['dS']<5),:]
    ks_or = np.array(df['dS'])
    w = np.array(df['weightoutlierexcluded'])
    ks = np.log(ks_or)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-5,2)
    ax.set_title('KDE and spline of log-transformed anchor $K_\mathrm{S}$ of species '+ '{}'.format(sp))
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
    spl_x = np.linspace(min_ks, max_ks+0.1, num=int(round((abs(min_ks) + (max_ks+0.1)) *100)))
    spl_y = spl(spl_x)
    ax.plot(kde_x, spl(kde_x), 'b', lw=1, label="Spline on KDE")
    ax.hist(ks_refed, weights=w_refed, bins=np.arange(-5, 2.1, 0.1), color="gray", alpha=0.8, density=True,rwidth=0.8)
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
    if na: logging.info('Detecting likely peaks from node-averaged data')
    else: logging.info('Detecting likely peaks from node-weighted data')
    init_means, init_stdevs, good_prominences = find_peak_init_parameters(spl_x,spl_y,sp,outdir,peak_threshold=peak_threshold,na=na,rel_height=rel_height)
    #lower95CI,upper95CI = plot_95CI_hist(init_means, init_stdevs, ks_or, w, outdir, na, sp)
    lower95CI,upper95CI = plot_95CI_lognorm_hist(init_means, init_stdevs, ks_or, w, outdir, na, sp, ci=ci)
    if user: get95CIap(user_low,user_upp,anchor,gs_ks,outdir,na,sp,ci,user=user)
    else: get95CIap(lower95CI,upper95CI,anchor,gs_ks,outdir,na,sp,ci,user=user)

def get95CIap(lower,upper,anchor,gs_ks,outdir,na,sp,ci,user=False):
    if type(lower) == float:
        ap_95CI = gs_ks.loc[(gs_ks['dS']<=upper) & (gs_ks['dS']>=lower),:]
        sp_m = '{}'.format(sp)
        if na: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_node_averaged.tsv".format(sp_m))
        else: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_weighted.tsv".format(sp_m))
        ap_95CI.to_csv(fname,header=True,index=True,sep='\t')
        anchors = pd.read_csv(anchor, sep="\t", index_col=0)
        anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
        ap_format = anchors.merge(ap_95CI.reset_index(),on='pair').drop(columns=['gene1', 'gene2','dS','pair'])
        ap_format.index.name = 'id'
        if na: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_node_averaged_format.tsv".format(sp_m))
        else: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_weighted_format.tsv".format(sp_m))
        ap_format.to_csv(fname,header=True,index=True,sep='\t')
    elif len(lower) == 1:
        ap_95CI = gs_ks.loc[(gs_ks['dS']<=upper[0]) & (gs_ks['dS']>=lower[0]),:]
        sp_m = '{}'.format(sp)
        if user:
            if na: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_node_averaged.tsv".format(sp_m))
            else: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_weighted.tsv".format(sp_m))
        else:
            if na: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_node_averaged.tsv".format(sp_m,ci))
            else: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_weighted.tsv".format(sp_m,ci))
        ap_95CI.to_csv(fname,header=True,index=True,sep='\t')
        anchors = pd.read_csv(anchor, sep="\t", index_col=0)
        anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
        ap_format = anchors.merge(ap_95CI.reset_index(),on='pair').drop(columns=['gene1', 'gene2','dS','pair'])
        ap_format.index.name = 'id'
        if user:
            if na: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_node_averaged_format.tsv".format(sp_m))
            else: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_weighted_format.tsv".format(sp_m))
        else:
            if na: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_node_averaged_format.tsv".format(sp_m,ci))
            else: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_weighted_format.tsv".format(sp_m,ci))
        ap_format.to_csv(fname,header=True,index=True,sep='\t')
    else:
        for indice,i in enumerate(zip(lower,upper)):
            lower,upper = i[0],i[1]
            text = 'Peak_{}_'.format(indice+1)
            ap_95CI = gs_ks.loc[(gs_ks['dS']<=upper) & (gs_ks['dS']>=lower),:]
            sp_m = text + '{}'.format(sp)
            if user:
                if na: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_node_averaged.tsv".format(sp_m))
                else: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_weighted.tsv".format(sp_m))
            else:
                if na: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_node_averaged.tsv".format(sp_m,ci))
                else: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_weighted.tsv".format(sp_m,ci))
            ap_95CI.to_csv(fname,header=True,index=True,sep='\t')
            anchors = pd.read_csv(anchor, sep="\t", index_col=0)
            anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
            ap_format = anchors.merge(ap_95CI.reset_index(),on='pair').drop(columns=['gene1', 'gene2','dS','pair'])
            ap_format.index.name = 'id'
            if user:
                if na: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_node_averaged_format.tsv".format(sp_m))
                else: fname = os.path.join(outdir, "{}_Manual_CI_AP_for_dating_weighted_format.tsv".format(sp_m))
            else:
                if na: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_node_averaged_format.tsv".format(sp_m,ci))
                else: fname = os.path.join(outdir, "{}_{}%CI_AP_for_dating_weighted_format.tsv".format(sp_m,ci))
            ap_format.to_csv(fname,header=True,index=True,sep='\t')


def add_mpgmmlabels(df,df_index,labels,outdir,n,regime='multiplicon'):
    predict_column = pd.DataFrame(labels,index=df_index.index,columns=['AnchorKs_GMM_Component']).reset_index()
    df = df.reset_index().merge(predict_column, on = [regime])
    df = df.set_index('pair')
    fname = os.path.join(outdir,'AnchorKs_GMM_{}components_prediction.tsv'.format(n))
    df.to_csv(fname,header=True,index=True,sep='\t')
    return df

def get95CIap_MP(lower,upper,anchor,gs_ks,outdir,sp,ci,guide,mpKs,user=False):
    gs_ks = gs_ks.reset_index().set_index(guide).join(mpKs)
    if type(lower) == float:
        ap_95CI = gs_ks.loc[(gs_ks['Median_Ks']<=upper) & (gs_ks['Median_Ks']>=lower),:]
        sp_m = '{}_guided_{}'.format(guide,sp)
        fname = os.path.join(outdir, "{}_Manual_CI_MP_for_dating.tsv".format(sp_m))
        ap_95CI.to_csv(fname,header=True,index=True,sep='\t')
        anchors = pd.read_csv(anchor, sep="\t", index_col=0)
        anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
        ap_format = anchors.merge(ap_95CI.reset_index(),on='pair').drop(columns=['gene1', 'gene2','dS','pair','Median_Ks'])
        ap_format.index.name = 'id'
        fname = os.path.join(outdir, "{}_Manual_CI_MP_for_dating_format.tsv".format(sp_m))
        ap_format.to_csv(fname,header=True,index=True,sep='\t')
    elif len(lower) == 1:
        ap_95CI = gs_ks.loc[(gs_ks['Median_Ks']<=upper[0]) & (gs_ks['Median_Ks']>=lower[0]),:]
        sp_m = '{}_guided_{}'.format(guide,sp)
        if user: fname = os.path.join(outdir, "{}_Manual_CI_MP_for_dating.tsv".format(sp_m))
        else: fname = os.path.join(outdir, "{}_{}%CI_MP_for_dating.tsv".format(sp_m,ci))
        ap_95CI.to_csv(fname,header=True,index=True,sep='\t')
        anchors = pd.read_csv(anchor, sep="\t", index_col=0)
        anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
        ap_format = anchors.merge(ap_95CI.reset_index(),on='pair').drop(columns=['gene1', 'gene2','dS','pair','Median_Ks'])
        ap_format.index.name = 'id'
        if user: fname = os.path.join(outdir, "{}_Manual_CI_MP_for_dating_format.tsv".format(sp_m))
        else: fname = os.path.join(outdir, "{}_{}%CI_MP_for_dating_format.tsv".format(sp_m,ci))
        ap_format.to_csv(fname,header=True,index=True,sep='\t')
    else:
        for indice,i in enumerate(zip(lower,upper)):
            text = 'Peak_{}_'.format(indice+1)
            lower,upper = i[0],i[1]
            ap_95CI = gs_ks.loc[(gs_ks['Median_Ks']<=upper) & (gs_ks['Median_Ks']>=lower),:]
            sp_m = text + '{}_guided_{}'.format(guide,sp)
            if user: fname = os.path.join(outdir, "{}_Manual_CI_MP_for_dating.tsv".format(sp_m))
            else: fname = os.path.join(outdir, "{}_{}%CI_MP_for_dating.tsv".format(sp_m,ci))
            ap_95CI.to_csv(fname,header=True,index=True,sep='\t')
            anchors = pd.read_csv(anchor, sep="\t", index_col=0)
            anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
            ap_format = anchors.merge(ap_95CI.reset_index(),on='pair').drop(columns=['gene1', 'gene2','dS','pair','Median_Ks'])
            ap_format.index.name = 'id'
            if user: fname = os.path.join(outdir, "{}_Manual_CI_MP_for_dating_format.tsv".format(sp_m))
            else: fname = os.path.join(outdir, "{}_{}%CI_MP_for_dating_format.tsv".format(sp_m,ci))
            ap_format.to_csv(fname,header=True,index=True,sep='\t')


def get_outlierexcluded(df,cutoff = 5):
    df = df[df['dS']<cutoff]
    weight_exc = 1/df.groupby(['family', 'node'])['dS'].transform('count')
    weight_exc = weight_exc.to_frame(name='weightoutlierexcluded')
    return weight_exc

def get_outlierincluded(df):
    weight_inc = 1/df.groupby(['family', 'node'])['dS'].transform('count')
    weight_inc = weight_inc.to_frame(name='weightoutlierincluded')
    return weight_inc

def get_nodeaverged_dS_outlierincluded(df):
    node_averaged_dS_inc = df.groupby(["family", "node"])["dS"].mean()
    node_averaged_dS_inc = node_averaged_dS_inc.to_frame(name='node_averaged_dS_outlierincluded')
    return node_averaged_dS_inc

def get_nodeaverged_dS_outlierexcluded(df,cutoff = 5):
    df = df[df['dS']<cutoff]
    node_averaged_dS_exc = df.groupby(["family", "node"])["dS"].mean()
    node_averaged_dS_exc = node_averaged_dS_exc.to_frame(name='node_averaged_dS_outlierexcluded')
    return node_averaged_dS_exc

def formatv2(ksdf):
    if "Ks" in ksdf.columns: ksdf = ksdf.rename(columns={"Ks":"dS"})
    if "Ka" in ksdf.columns: ksdf = ksdf.rename(columns={"Ka":"dN"})
    if "Omega" in ksdf.columns: ksdf = ksdf.rename(columns={"Omega":"dN/dS"})
    if "Family" in ksdf.columns: ksdf = ksdf.rename(columns={"Family":"family"})
    if "Node" in ksdf.columns: ksdf = ksdf.rename(columns={"Node":"node"})
    if "AlignmentIdentity" in ksdf.columns: ksdf = ksdf.rename(columns={"AlignmentIdentity":"alignmentidentity"})
    if "AlignmentLength" in ksdf.columns: ksdf = ksdf.rename(columns={"AlignmentLength":"alignmentlength"})
    if "AlignmentCoverage" in ksdf.columns: ksdf = ksdf.rename(columns={"AlignmentCoverage":"alignmentcoverage"})
    if "Paralog1" in ksdf.columns: ksdf = ksdf.rename(columns={"Paralog1":"gene1"})
    if "Paralog2" in ksdf.columns: ksdf = ksdf.rename(columns={"Paralog2":"gene2"})
    if "WeightOutliersIncluded" in ksdf.columns: ksdf = ksdf.drop(columns=["WeightOutliersIncluded"])
    if "WeightOutliersExcluded" in ksdf.columns: ksdf = ksdf.drop(columns=["WeightOutliersExcluded"])
    if "weightoutlierexcluded" not in ksdf.columns: weight_inc = get_outlierincluded(ksdf)
    if "weightoutlierincluded" not in ksdf.columns:
        weight_exc = get_outlierexcluded(ksdf,cutoff = 5)
        ksdf = ksdf.join(weight_inc).join(weight_exc)
    if "node_averaged_dS_outlierincluded" not in ksdf.columns:
        node_averaged_dS_inc = get_nodeaverged_dS_outlierincluded(ksdf)
    if "node_averaged_dS_outlierexcluded" not in ksdf.columns:
        node_averaged_dS_exc = get_nodeaverged_dS_outlierexcluded(ksdf,cutoff = 5)
        ksdf = ksdf.reset_index().merge(node_averaged_dS_inc,on = ['family', 'node'])
        ksdf = ksdf.merge(node_averaged_dS_exc,on = ['family', 'node'],how = 'left')
    #ksdf = ksdf.join(weight_inc).join(weight_exc) # here I kept the NaN value
    #ksdf = ksdf.reset_index().merge(node_averaged_dS_inc,on = ['family', 'node'])
    #ksdf = ksdf.merge(node_averaged_dS_exc,on = ['family', 'node'],how = 'left')
    if "index" in ksdf.columns: ksdf = ksdf.set_index('index')
    ksdf.index.name = 'pair'
    return ksdf

def plot_95CI_hist(init_means, init_stdevs, ks_or, w, outdir, na, sp, guide = None):
    text = "AnchorKs_PeakCI_"
    #if guide != None: text = "AnchorKs_PeakCI_{}_guided_".format(guide)
    if guide != None: fname = os.path.join(outdir, "{}{}_guided_{}.pdf".format(text,guide,sp))
    elif na: fname = os.path.join(outdir, "{}{}_node_averaged.pdf".format(text,sp))
    else: fname = os.path.join(outdir, "{}{}_node_weighted.pdf".format(text,sp))
    f, ax = plt.subplots()
    kdesity = 100
    bins = 50
    bin_width = 0.1
    kde_x = np.linspace(0,5,num=bins*kdesity)
    #colors = cm.rainbow(np.linspace(0, 1, len(init_means)))
    alphas = np.linspace(0.3, 0.7, len(init_means))
    for mean,std,i in zip(init_means, init_stdevs,range(len(init_means))):
        Hs, Bins, patches = plt.hist(ks_or,bins = np.linspace(0, 5, num=int(5/bin_width)+1),weights=w,color='gray', alpha=1, rwidth=0.8)
        plt.axvline(x = np.exp(mean), color = 'black', alpha = alphas[i], ls = ':', lw = 1,label='component {} mean {:.2f}'.format(i+1,np.exp(mean)))
        plt.axvline(x = np.exp(mean)+std*2, color = 'black', alpha = alphas[i], ls = '--', lw = 1,label='Peak {} upper 95% CI {:.2f}'.format(i+1,np.exp(mean)+std*2))
        plt.axvline(x = np.exp(mean)-std*2, color = 'black', alpha = alphas[i], ls = '--', lw = 1,label='Peak {} lower 95% CI {:.2f}'.format(i+1,np.exp(mean)-std*2))
    plt.xlabel("$K_\mathrm{S}$", fontsize = 10)
    if guide != None: plt.ylabel("Number of retained duplicates", fontsize = 10)
    elif na: plt.ylabel("Number of retained duplicates (node averaged)", fontsize = 10)
    else: plt.ylabel("Number of retained duplicates (weighted)", fontsize = 10)
    ax.legend(loc=1,fontsize='large',frameon=False)
    sns.despine(offset=1)
    plt.title('Anchor $K_\mathrm{S}$'+' distribution of {}'.format(sp))
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()
    return np.exp(mean)-std*2,np.exp(mean)+std*2

def plot_95CI_lognorm_hist(init_means, init_stdevs, ks_or, w, outdir, na, sp, guide = None, ci=95):
    if guide != None: fname = os.path.join(outdir, "{}Ks_PeakCI_{}.pdf".format(guide,sp))
    elif na: fname = os.path.join(outdir, "AnchorKs_PeakCI_{}_node_averaged.pdf".format(sp))
    else: fname = os.path.join(outdir, "AnchorKs_PeakCI_{}_node_weighted.pdf".format(sp))
    f, ax = plt.subplots()
    x_points_strictly_positive = np.linspace(0, 5, int(5 * 100))
    bin_width = 0.1
    cs = ['b','g','y','r','k']
    alphas = np.linspace(0.3, 0.7, len(init_means))
    ci_l = (1-ci/100)/2
    ci_u = 1-(1-ci/100)/2
    CI_95s = []
    for mean,std,i in zip(init_means, init_stdevs,range(len(init_means))):
        Hs, Bins, patches = ax.hist(ks_or,bins = np.linspace(0, 5, num=int(5/bin_width)+1),weights=w,color='gray', alpha=1, rwidth=0.8)
        CHF = get_totalH(Hs)
        scaling = CHF*0.1
        ax.plot(x_points_strictly_positive,scaling*stats.lognorm.pdf(x_points_strictly_positive, scale=np.exp(mean),s=std), c=cs[i], ls='-', lw=1.5, alpha=0.8, label='Peak {} mode {:.2f}'.format(i+1,np.exp(mean - std**2)))
        CI_95 = stats.lognorm.ppf([ci_l, ci_u], scale=np.exp(mean), s=std)
        CI_95s.append(CI_95)
        plt.axvline(x = CI_95[0], color = cs[i], alpha = alphas[i], ls = ':', lw = 1,label='Peak {} lower {}%CI {:.2f}'.format(i+1,ci,CI_95[0]))
        plt.axvline(x = CI_95[1], color = cs[i], alpha = alphas[i], ls = ':', lw = 1,label='Peak {} upper {}%CI {:.2f}'.format(i+1,ci,CI_95[1]))
    plt.xlabel("$K_\mathrm{S}$", fontsize = 10)
    if guide == 'segment': plt.ylabel("Number of segment pair", fontsize = 10)
    elif guide != None: plt.ylabel("Number of {}".format(guide), fontsize = 10)
    elif na: plt.ylabel("Number of retained duplicates (node averaged)", fontsize = 10)
    else: plt.ylabel("Number of retained duplicates (weighted)", fontsize = 10)
    ax.legend(loc=1,fontsize='large',frameon=False)
    ax.set_xlim(0, 5)
    sns.despine(offset=1)
    plt.title('Anchor $K_\mathrm{S}$'+' distribution of {}'.format(sp))
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()
    return [i[0] for i in CI_95s],[i[1] for i in CI_95s]
    #return CI_95[0],CI_95[1]

def plot_Elbow_loss(Losses,outdir,n1=None,n2=None,method='Medoids',regime=None):
    if regime == 'original': fname = os.path.join(outdir,'{}_Elbow-Loss_Original_Anchor_Ks.pdf'.format(method))
    if regime != None: fname = os.path.join(outdir,'{}_Elbow-Loss_{}_Ks.pdf'.format(method,regime))
    else: fname = os.path.join(outdir,'Elbow-Loss Function.pdf')
    fig, axes = plt.subplots()
    if n1 !=None:
        x_range = list(range(n1, n2 + 1))
        axes.plot(np.arange(n1, n2 + 1), Losses, color='k', marker='o')
        axes.set_xticks(list(range(n1, n2 + 1)))
    else:
        x_range = list(range(1, len(Losses) + 1))
        axes.plot(np.arange(1, len(Losses) + 1), Losses, color='k', marker='o')
        axes.set_xticks(list(range(1, len(Losses) + 1)))
    axes.set_xticklabels(x_range)
    axes.grid(ls=":")
    axes.set_ylabel("Elbow-loss")
    axes.set_xlabel("# {}".format(method))
    fig.tight_layout()
    fig.savefig(fname,format ='pdf')
    plt.close()

def get_anchors(anchor):
    anchors = pd.read_csv(anchor, sep="\t", index_col=0)
    anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
    df = anchors[["pair", "basecluster",'multiplicon']].drop_duplicates("pair").set_index("pair")
    return df

def get_anchor_ksd(ksdf, apdf):
    return ksdf.join(apdf).dropna()
    # here any NA occurred per row is dropped

def bc_group_anchor(df,regime='multiplicon'):
    #median_df = df.groupby(["basecluster"]).median()
    median_df = df.groupby([regime]).median()
    X = getX(median_df,'dS')
    return median_df,X

def add_apgmmlabels(df,df_index,labels,outdir,n,regime='multiplicon'):
    predict_column = pd.DataFrame(labels,index=df_index.index,columns=['AnchorKs_GMM_Component']).reset_index()
    df = df.reset_index().merge(predict_column, on = [regime])
    df = df.set_index('pair')
    fname = os.path.join(outdir,'{}-guided_AnchorKs_GMM_{}components_prediction.tsv'.format(regime,n))
    df.to_csv(fname,header=True,index=True,sep='\t')
    return df

def add_apgmmlabels_pairs(df,df_index,labels,outdir,n):
    predict_column = pd.DataFrame(labels,index=df_index,columns=['AnchorKs_GMM_Component'])
    df = df.join(predict_column)
    fname = os.path.join(outdir,'Original_AnchorKs_GMM_{}components_prediction.tsv'.format(n))
    df.to_csv(fname,header=True,index=True,sep='\t')
    return df

def add_seg(df,listelement,multipliconpairs,segment):
    #Here the df should be Ks 0-5 filter
    #It hasn't to be ap in listelement note that a same pair of syntelogs can be in different multiplicon
    mp = pd.read_csv(multipliconpairs, sep="\t", index_col=0)
    if len(mp.columns) == 5: mp = mp.drop(columns=['gene_y']).rename(columns = {'gene_x':'gene_y'}).rename(columns = {'Unnamed: 2':'gene_x'})
    mp = mp.loc[:,['multiplicon','gene_x','gene_y']]
    mp.loc[:,['pair']] = ["__".join(sorted([x,y])) for x,y in zip(list(mp['gene_x']),list(mp['gene_y']))]
    mp.loc[:,['id']] = [i for i in range(mp.shape[0])]
    #mp= mp.drop_duplicates(subset=['pair']) #Here the same gene pair can be in different multiplicon
    df_seg = pd.read_csv(segment,header=0,index_col=None,sep='\t').rename(columns = {'id':'segment'}).loc[:,['segment','multiplicon']]
    df_le = pd.read_csv(listelement,header=0,index_col=0,sep='\t').merge(df_seg,on='segment').rename(columns = {'multiplicon':'multiplicon_x'}).loc[:,['segment','multiplicon_x','gene']]
    #mp_segment_pools = {mt:list(set(df_seg[df_seg['multiplicon']==mt]['segment'])) for mt in df_seg['multiplicon']}
    #segment_gene_pools = {sg:list(set(df_le[df_le['segment']==sg]['gene'])) for sg in df_le['segment']}
    df_mp_le = mp.merge(df_le,left_on='gene_x',right_on='gene')
    df_mp_le = df_mp_le[df_mp_le['multiplicon_x'] == df_mp_le['multiplicon']]
    df_mp_le = df_mp_le.rename(columns = {'segment':'segment_x'})
    # here the genes belonging to two segments in the same multiplicon were removed because of indeterminacy
    df_mp_le = df_mp_le.drop_duplicates(subset=['id'],keep=False)
    df_mp_le_r = df_mp_le.drop(columns=['multiplicon_x','gene']).merge(df_le,left_on='gene_y',right_on='gene')
    df_mp_le_r = df_mp_le_r[df_mp_le_r['multiplicon_x'] == df_mp_le_r['multiplicon']]
    df_mp_le_r = df_mp_le_r.rename(columns = {'segment':'segment_y'})
    df_mp_le_r = df_mp_le_r.drop_duplicates(subset=['id'],keep=False)
    df_mp_le_r = df_mp_le_r.loc[:,['multiplicon','gene_x','gene_y','segment_x','segment_y','pair']].set_index('pair')
    df_mp_le_r.loc[:,['segment_pair']] = ["__".join(sorted([x,y])) for x,y in zip(list(df_mp_le_r['segment_x'].astype(str)),list(df_mp_le_r['segment_y'].astype(str)))]
    df_cal_medKs = df_mp_le_r.loc[:,['segment_pair','multiplicon']].join(df).rename(columns = {'segment_pair':'segment'}).dropna(subset=['dS']) # here I kept Ks > 5 and duplicated index (pair)
    dS_median = df_cal_medKs.groupby(['segment'])['dS'].median()
    dS_median = dS_median[dS_median<5]
    dS_median.name = 'Segment_dS'
    df_cal_medKs = df_cal_medKs.reset_index().merge(dS_median.reset_index(),on='segment').dropna(subset=['Segment_dS']).set_index('pair')
    #df_cal_medKs = df.join(df_mp_le_r.loc[:,['segment_pair','multiplicon']]).rename(columns = {'segment_pair':'segment'}) # here I did keep the NaN value, which impacted and changed the median Ks values afterwards
    #df_mp_le_r = mp.merge(df_le,left_on='gene_y',right_on='gene')
    #df_mp_le_r = df_mp_le_r[df_mp_le_r['multiplicon_x'] == df_mp_le_r['multiplicon']]
    #df_mp_le_r = df_mp_le_r.rename(columns = {'segment':'segment_y'})
    #df_mp_le_lr = df_mp_le.loc[:,['multiplicon','gene_x','gene_y','segment_x','pair']].merge(df_mp_le_r.loc[:,['pair','segment_y']],on='pair').set_index('pair')
    #df_mp_le_lr.loc[:,['segment_pair']] = ["__".join(sorted([x,y])) for x,y in zip(list(df_mp_le_lr['segment_x'].astype(str)),list(df_mp_le_lr['segment_y'].astype(str)))]
    #df_cal_medKs = df.join(df_mp_le_lr.loc[:,['segment_pair','multiplicon']]).dropna().rename(columns = {'segment_pair':'segment'})
    #df_cal_medKs = df.join(df_mp_le_lr.loc[:,['segment_pair','multiplicon']]).rename(columns = {'segment_pair':'segment'}) # here I did keep the NaN value, which impacted and changed the median Ks values afterwards
    #find_level345(df_cal_medKs)
    #df_le_combined = pd.concat([df_mp_le, df_mp_le_r], ignore_index=True).drop_duplicates(subset=['pair']).set_index('pair')
    #Here the smaller Ks of alternatives are the real age of that genes on that segment
    #df_withKs = df.join(df_le_combined).dropna().sort_values('dS')
    #df_cal_medKs = df_withKs.drop_duplicates(subset=['segment','gene_y']).drop_duplicates(subset=['segment','gene_x'])
    #df_cal_medKs = df_cal_medKs.drop(columns=['gene_y','multiplicon_x','gene_x','gene'])
    #Here we retrive the duplicates information for segment age clustering
    #dup_le = df_withKs[df_withKs.duplicated(subset=['segment','gene_y'], keep=False)]
    #dup_ri = df_withKs[df_withKs.duplicated(subset=['segment','gene_x'], keep=False)]
    #dup_combined = pd.concat([dup_le,dup_ri], ignore_index=True).drop_duplicates()
    #dup_combined_work = dup_combined.loc[:,['segment','gene_x','gene_y','dS']]
    return df_cal_medKs

def getDM(X):
    dm = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            dm[i,j] = dm[j,i] = (abs(X[i][0] - X[j][0]))**2
    return dm

def getnw(X):
    dm = getDM(X)
    linkage_matrix = linkage(dm, method='average')
    tree = to_tree(linkage_matrix)
    newick_str = convert_newick(tree) + ";"
    return newick_str

def convert_newick(node):
    if node.is_leaf():
        return str(node.id)
    else:
        left_branch = convert_newick(node.left)
        right_branch = convert_newick(node.right)
        return "(" + left_branch + ":" + str(node.dist) + "," + right_branch + ":" + str(node.dist) + ")"

def _label_internals(tree):
    for i, c in enumerate(tree.get_nonterminals()):
        c.name = str(i)

def getsegpairs(seg_ids):
    pairs = []
    l = len(seg_ids)
    for i in range(l):
        for j in range(i+1,l):
            pairs.append("__".join(sorted([seg_ids[i],seg_ids[j]])))
    return pairs

def find_level345(df):
    #Here the segment is renamed from segment_pair
    df.loc[:,['weight_segment']] = [1 for i in range(df.shape[0])]
    df_level = df.groupby(['multiplicon'])['segment'].nunique().to_frame().rename(columns = {'segment':'segment_pair'})
    df = df.reset_index().merge(df_level,on='multiplicon').set_index('pair')
    df.loc[:,['segment_x']] = [i.split('__')[0] for i in df['segment']]
    df.loc[:,['segment_y']] = [i.split('__')[1] for i in df['segment']]
    df = df[df['segment_pair']>1]
    df = df.drop_duplicates(subset=['segment'])
    for mp in df.loc[:,['multiplicon']]:
        tmp = df[df['multiplicon']==mp]
        seg_ids = list(tmp.loc[:,['segment']])
        seg_pairs = getsegpairs(seg_ids)
        DM = getDM(list(tmp['segment_x']),list(tmp['segment_y']),list(tmp['Segment_dS']))
        X = [[i] for i in tmp.loc[:,['Segment_dS']]]
        tree = getnw(X)
        tree_object = Phylo.read(StringIO(tree), "newick")
        _label_internals(tree_object)

    #df_level_duplicated = df_level[df_level['segment_pair']>1]
    #df.merge()


#def seg_duplicates_clusering(dup_combined_work):


def calculateHPD(train_in,per):
    sorted_in = np.sort(train_in)
    #The lower_bound might be as zero which abort the HDR
    upper_bound_indice = int(np.floor(per*len(sorted_in)/100))
    lower_bound_indice = int(np.ceil((100-per)*len(sorted_in)/100))
    #upper_bound = np.percentile(train_in, per)
    #lower_bound = np.percentile(train_in, 100-per)
    #upper_bound_indice,lower_bound_indice = 0,0
    cutoff,candidates = int(np.ceil(per*len(sorted_in)/100)),[]
    #for i,v in enumerate(sorted_in):
    #    if v >= upper_bound:
    #        upper_bound_indice = i
    #        break
    #for i,v in enumerate(sorted_in):
    #    if v >= lower_bound:
    #        lower_bound_indice = i
    #        break
    for (x,y) in itertools.product(np.arange(0,lower_bound_indice,1,dtype=int), np.arange(upper_bound_indice,len(sorted_in),1,dtype=int)):
        if (y-x+1) >= cutoff: candidates.append((sorted_in[y] - sorted_in[x],(x,y)))
    lower,upper = sorted(candidates, key=lambda y: y[0])[0][1][0],sorted(candidates, key=lambda y: y[0])[0][1][1]
    return sorted_in[upper],sorted_in[lower]

def fit_apgmm_guide(hdr,guide,anchor,df_nofilter,dfor,seed,components,em_iter,n_init,outdir,method,weighted,plot,segment=None,multipliconpairs=None,listelement=None,cutoff=None):
    if anchor == None:
        logging.error('Please provide anchorpoints.txt file for Anchor Ks GMM Clustering')
        exit(0)
    df_ap = get_anchors(anchor)
    df = get_anchor_ksd(dfor, df_ap)
    df_nofilter = df_nofilter[df_nofilter['dS']>0]
    if segment!= None:
        df = add_seg(df_nofilter,listelement,multipliconpairs,segment)
        df.to_csv(os.path.join(outdir, "Segment_Ks.tsv"),header=True,index=True,sep='\t')
    df_withindex,X = bc_group_anchor(df,regime=guide)
    X_log = np.log(X).reshape(-1, 1)
    out_file = os.path.join(outdir, "{}_Ks_GMM_AIC_BIC.pdf".format(guide))
    if method == 'gmm': models, aic, bic, besta, bestb, N = fit_gmm(out_file, X_log, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    if method == 'bgmm': models, N = fit_bgmm(X_log, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    if components[0] == 1 and components[1] > 1:
        plot_silhouette_score(X_log,components[0]+1,components[1],[m.predict(X_log) for m in models][1:],outdir,guide+'_Ks','GMM')
        significance_test_cluster(X_log,components[0]+1,components[1],[m.predict(X_log) for m in models][1:])
    else:
        plot_silhouette_score(X_log,components[0],components[1],[m.predict(X_log) for m in models],outdir,guide+'_Ks','GMM')
        significance_test_cluster(X_log,components[0],components[1],[m.predict(X_log) for m in models])
    Losses = []
    for n, m in zip(N,models):
        labels = m.predict(X_log)
        df_c = add_apgmmlabels(df,df_withindex,labels,outdir,n,regime=guide)
        means,stds,weights = m.means_,m.covariances_,m.weights_
        Losses.append(Elbow_lossf(X_log,[i[0] for i in means],labels))
        fig = plot_mp_component(X,labels,n,bins=50,plot = plot,ylabel="Number of segment pair",regime=guide)
        fname = os.path.join(outdir, "{}_Ks_Clusters_GMM_Component{}.pdf".format(guide,n))
        fig.savefig(fname)
        plt.close()
        fig = plot_mp_component_lognormal(X,hdr,means,stds,weights,labels,n,bins=50,ylabel="Number of segment pair",regime=guide)
        #hdr_ap(hdr,df,df_withindex)
        fname = os.path.join(outdir, "{}_Ks_Clusters_Lognormal_GMM_Component{}.pdf".format(guide,n))
        fig.savefig(fname)
        plt.close()
    plot_Elbow_loss(Losses,outdir,n1=components[0],n2=components[1],method='GMM',regime=guide)
    for n, m in zip(N,models):
        fig = plot_ak_component(df_c.dropna(),n,bins=50,plot = plot,ylabel="Duplication events",weighted=weighted,regime=guide)
        if weighted: fname = os.path.join(outdir, "{}-guided_AnchorKs_GMM_Component{}_node_weighted.pdf".format(guide,n))
        else: fname = os.path.join(outdir, "{}-guided_AnchorKs_GMM_Component{}_node_averaged.pdf".format(guide,n))
        fig.savefig(fname)
        plt.close()
        fig,HDRs = plot_ak_component_kde(df_c.dropna(),n,hdr,bins=50,ylabel="Duplication events",weighted=weighted,regime=guide)
        getGuided_AP_HDR(HDRs,hdr,n,df_c,outdir,guide,cutoff)
        if weighted: fname = os.path.join(outdir, "{}-guided_AnchorKs_GMM_Component{}_node_weighted_kde.pdf".format(guide,n))
        else: fname = os.path.join(outdir, "{}-guided_AnchorKs_GMM_Component{}_node_averaged_kde.pdf".format(guide,n))
        fig.savefig(fname)
        plt.close()
    return df

def getGuided_AP_HDR(HDRs,hdr,n,df_c,outdir,regime,cutoff):
    for num in HDRs.keys():
        df_tmp = df_c[df_c['AnchorKs_GMM_Component']==num]
        df_tmp = df_tmp.loc[(df_tmp['dS']<=min([cutoff,HDRs[num][1]])) & (df_tmp['dS']>=HDRs[num][0]),:]
        df_tmp = df_tmp.loc[:,['gene1','gene2','dS','Segment_dS','AnchorKs_GMM_Component']].rename(columns={"gene1":"gene_x","gene2":"gene_y"})
        fname = os.path.join(outdir,"{}_guided_{}%HDR_Syntelogs_Component{}_Model{}_WGDating.tsv".format(regime,hdr,num,n))
        df_tmp.to_csv(fname,sep='\t',header=True,index=True)

def fit_apgmm_ap(hdr,anchor,df,seed,components,em_iter,n_init,outdir,method,weighted,plot):
    if anchor == None:
        logging.error('Please provide anchorpoints.txt file for Anchor Ks GMM Clustering')
        exit(0)
    df_ap = get_anchors(anchor)
    df = get_anchor_ksd(df, df_ap)
    df_withindex,X = df.index,getX(df,'dS')
    X_log = np.log(X).reshape(-1, 1)
    out_file = os.path.join(outdir, "Original_AnchorKs_GMM_AIC_BIC.pdf")
    logging.info("GMM modeling on Log-scale original anchor Ks data")
    if method == 'gmm': models, aic, bic, besta, bestb, N = fit_gmm(out_file, X_log, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    if method == 'bgmm': models, N = fit_bgmm(X_log, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    if components[0] == 1 and components[1] > 1:
        plot_silhouette_score(X_log,components[0]+1,components[1],[m.predict(X_log) for m in models][1:],outdir,'Original_AnchorKs','GMM')
        significance_test_cluster(X_log,components[0]+1,components[1],[m.predict(X_log) for m in models][1:])
    else:
        plot_silhouette_score(X_log,components[0],components[1],[m.predict(X_log) for m in models],outdir,'Original_AnchorKs','GMM')
        significance_test_cluster(X_log,components[0],components[1],[m.predict(X_log) for m in models])
    Losses = []
    for n, m in zip(N,models):
        labels = m.predict(X_log)
        means,stds,weights = m.means_,m.covariances_,m.weights_
        Losses.append(Elbow_lossf(X_log,[i[0] for i in means],labels))
        df_c = add_apgmmlabels_pairs(df,df_withindex,labels,outdir,n)
        fig = plot_ak_component(df_c,n,bins=50,plot = plot,ylabel="Duplication events",weighted=weighted,regime='original')
        if weighted: fname = os.path.join(outdir, "Original_AnchorKs_GMM_Component{}_node_weighted.pdf".format(n))
        else: fname = os.path.join(outdir, "Original_AnchorKs_GMM_Component{}_node_averaged.pdf".format(n))
        fig.savefig(fname)
        plt.close()
        #fig = plot_ak_component_kde(df_c,n,bins=50,ylabel="Duplication events",weighted=weighted,regime='original')
        #if weighted: fname = os.path.join(outdir, "Original_AnchorKs_GMM_Component{}_node_weighted_kde.pdf".format(n))
        #else: fname = os.path.join(outdir, "Original_AnchorKs_GMM_Component{}_node_averaged_kde.pdf".format(n))
        #fig.savefig(fname)
        #plt.close()
        fig = plot_ak_component_lognormal(df_c.dropna(),means,stds,weights,n,bins=50,ylabel="Duplication events",weighted=weighted,regime='original')
        if weighted: fname = os.path.join(outdir, "Original_AnchorKs_GMM_Component{}_node_weighted_Lognormal.pdf".format(n))
        else: fname = os.path.join(outdir, "Original_AnchorKs_GMM_Component{}_node_averaged_Lognormal.pdf".format(n))
        fig.savefig(fname)
        plt.close()
    plot_Elbow_loss(Losses,outdir,n1=components[0],n2=components[1],method='GMM',regime='original')
    return df

def fit_kmedoids(guide,anchor, boots, kdemethod, bin_width, weighted, df_nofilter, df, outdir, seed, n, em_iter=100, metric='euclidean', method='pam', init ='k-medoids++', plot = 'identical', n_kmedoids = 5, segment= None, multipliconpairs=None,listelement=None):
    """
    Clustering with KMedoids to delineate different anchor groups from anchor Ks distribution
    """
    if anchor == None:
        logging.error('Please provide anchorpoints.txt file for Anchor Ks KMedoids Clustering')
        exit(0)
    df_ap = get_anchors(anchor)
    df = get_anchor_ksd(df, df_ap)
    #if segment!= None: df = add_seg(df,segment)
    df_nofilter = df_nofilter[df_nofilter['dS']>0]
    if segment!= None:
        df = add_seg(df_nofilter,listelement,multipliconpairs,segment)
        df.to_csv(os.path.join(outdir, "Segment_Ks.tsv"),header=True,index=True,sep='\t')
    df_withindex,X = bc_group_anchor(df,regime=guide)
    #df = df.dropna(subset=['node_averaged_dS_outlierexcluded'])
    #df_rmdup = df.drop_duplicates(subset=['family', 'node'])
    #X = getX(df_rmdup,'node_averaged_dS_outlierexcluded')
    X_log = np.log(X).reshape(-1, 1)
    logging.info("KMedoids clustering with {} component".format(n))
    if n > 1: kmedoids = KMedoids(n_clusters=n,metric=metric,method=method,init=init,max_iter=em_iter,random_state=seed).fit(X_log)
    else: kmedoids = KMedoids(n_clusters=n,metric=metric,method='alternate',init=init,max_iter=em_iter,random_state=seed).fit(X_log)
    cluster_centers = kmedoids.cluster_centers_
    centers = info_centers(cluster_centers)
    labels = kmedoids.labels_
    plot_segment_kmedoids(labels,X,outdir,bin_width,n,regime=guide)
    #labels = kmedoids.predict(X_log)
    Losses,labels_plot = [],[]
    for p in range(1,n_kmedoids+1):
        if p == 1: kmedoids = KMedoids(n_clusters=p,metric=metric,method='alternate',init=init,max_iter=em_iter,random_state=seed).fit(X_log)
        else: kmedoids = KMedoids(n_clusters=p,metric=metric,method=method,init=init,max_iter=em_iter,random_state=seed).fit(X_log)
        Cluster_centers = kmedoids.cluster_centers_
        Labels = kmedoids.labels_
        labels_plot.append(Labels)
        loss = Elbow_lossf(X_log,Cluster_centers,Labels)
        Losses.append(loss)
    if n_kmedoids > 0:
        plot_silhouette_score(X_log,2,n_kmedoids+1,labels_plot[1:],outdir,guide+'_Ks','KMedoids')
        print((len(range(2,n_kmedoids+1)),len(labels_plot[1:])))
        significance_test_cluster(X_log,2,n_kmedoids+1,labels_plot[1:])
    plot_Elbow_loss(Losses,outdir,regime=guide)
    #loss = Elbow_lossf(X_log,cluster_centers,labels)
    #df_labels = pd.DataFrame(labels,columns=['KMedoids_Cluster'])
    df_c = write_labels(df,df_withindex,labels,outdir,n,regime=guide)
    plot_kmedoids(boots,kdemethod,df_c,outdir,n,bin_width,bins=50,weighted=weighted,title="",plot=plot,alpha=0.5,regime=guide)
    plot_kmedoids_kde(boots,kdemethod,df_c,outdir,n,bin_width,bins=50,weighted=weighted,title="",plot=plot,alpha=0.5,regime=guide)
    return df

def retreive95CI(family,ksdf_filtered,outdir,lower,upper):
    df = pd.read_csv(family,header=0,index_col=0,sep='\t')
    cs = list(df.columns)
    focus_ids = [i for i in cs if i.endswith('_ap1') or i.endswith('_ap2')]
    df['ap12'] = ["__".join(sorted([x,y])) for x,y in zip(list(df[focus_ids[0]]),list(df[focus_ids[1]]))]
    ks_tmp = ksdf_filtered.loc[:,['dS','gene1','gene2']].copy()
    ks_tmp['ap12']= ["__".join(sorted([x,y])) for x,y in zip(list(ks_tmp['gene1']),list(ks_tmp['gene2']))]
    ks_tmp = ks_tmp.loc[:,['ap12','dS']]
    #df = df.reset_index()
    df = df.merge(ks_tmp,on='ap12')
    df = df.loc[(df['dS']<=upper) & (df['dS']>=lower),:]
    #df = df.set_index('index')
    df = df.drop(['dS','ap12'],axis=1)
    #here I reindexed the df to output
    df.index = ["GF{:0>8}".format(i+1) for i in range(len(df.index))]
    fname = os.path.join(outdir,os.path.basename(family)+'.95CI')
    df.to_csv(fname,header=True,index=True,sep='\t')

def Getanchor_Ksdf(anchor,ksdf,multiplicon):
    ap = pd.read_csv(anchor,header=0,index_col=0,sep = '\t')
    mp = pd.read_csv(multiplicon,header=0,index_col=0,sep = '\t')
