import numpy as np
import logging
from sklearn import mixture
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import random
from matplotlib.pyplot import cm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import warnings
from KDEpy import NaiveKDE
from sklearn.utils import resample
import itertools
import os
from wgd.core import _mkdir
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def alnfilter(df,weights_outliers_included, identity = 0, aln_len=0, coverage=0, min_ks=0, max_ks=5):
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
    for j in range(i):
        mean = np.exp(m.means_[j][0])
        covariance = m.covariances_[j][0][0]
        weight = m.weights_[j]
        precision = m.precisions_[j][0][0]
        means.append(mean)
        covariances.append(covariance)
        weights.append(weight)
        precisions.append(precision)
        logging.info("Component {0} has mean {1:.3f} ,covariance {2:.3f} ,weight {3:.3f}, precision {4:.3f}".format(j+1,mean,covariance,weight,precision))
    info_table['{}component'.format(i)] = {'mean':means,'covariance':covariances,'weight':weights,'precision':precisions}

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

def default_plot(*args,bins=50,alphas=None,colors=None,weighted=True,title="",ylabel="Duplication events",nums = "", plot = 'identical',**kwargs):
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
        for ax, k, f in zip(axs.flatten(), keys, funs):
            #color_random = cm.rainbow(np.linspace(0, 1, n))
            if plot == 'identical':
                for num, color in zip(range(nums),cm.rainbow(np.linspace(0, 1, nums))):
                    dist_comp = dis[dis['component']==num]
                    w = dist_comp['weightoutlierexcluded']
                    x = f(dist_comp[k])
                    y = x[np.isfinite(x)]
                    w = w[np.isfinite(x)]
                    if funs[0] == f: ax.hist(y, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, color = color, weights=w, alpha=a, rwidth=0.8, label = "component{}".format(num),**kwargs)
                    else: ax.hist(y, bins = 50, color = color, weights=w, alpha=a, rwidth=0.8, label = "component{}".format(num),**kwargs)
                    #kde_x, kde_y = get_kde(train_in,ax)
                    #ax.plot(kde_x,kde_y)
            else:
                dist_comps = [dis[dis['component']==num] for num in range(nums)]
                ws = [i['weightoutlierexcluded'] for i in dist_comps]
                xs = [f(i[k]) for i in dist_comps]
                ys = [x[np.isfinite(x)] for x in xs]
                ws = [w[np.isfinite(x)] for w,x in zip(ws,xs)]
                if funs[0] == f: ax.hist(ys, bins = np.linspace(0, 50, num=bins+1,dtype=int)/10, stacked=True, weights=ws, alpha=a, rwidth=0.8, label = ["component{}".format(int(i)) for i in range(nums)],** kwargs)
                else:
                    #maxi = int(max(list(itertools.chain.from_iterable(xs)))*100)
                    #mini = int(min(list(itertools.chain.from_iterable(xs)))*100)
                    ax.hist(ys, bins = bins, weights=ws, alpha=a, rwidth=0.8, label = ["component{}".format(int(i)) for i in range(nums)],**kwargs)
                    #ax.set_xticks(np.round_(np.linspace(mini, maxi, num=5,dtype=int)/100,decimals = 1))
                #kde_x, kde_y = get_kde(train_in,ax)
            leg = ax.legend(loc='upper right', fontsize=5,fancybox=True, framealpha=0.1,labelspacing=0.1,handlelength=2,handletextpad=0.1)
            for lh in leg.legendHandles: lh.set_alpha(0.1)
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
    kde_x = np.linspace(ks_lower, ks_upper, num=512)
    kde_y = kde(kde_x)
    kde_y = kde_y * bin_width * len(train_nonan) #Adaptable
    return kde_x, kde_y

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

#def bootstrap_dates(nd,)
def bootstrap_kde(outdir,train_in, ks_lower, ks_upper, boots, bin_width, ksdf_filtered, weight_col, weighted = False):
    train_nonan = train_in[~np.isnan(train_in)]
    modes = []
    medians = []
    kde_x = np.linspace(ks_lower, ks_upper, num=512)
    for r in range(boots):
        if weighted:
            bootstrap_sample = random.choices(ksdf_filtered['dS'], weights = weight_col, k=len(weight_col))
            #kde_x, kde_y = get_gaussian_kde(bootstrap_sample, ks_lower, ks_upper, bin_width, ksdf_filtered, weight_col, weighted)
            kde_y = NaiveKDE(bw=bin_width).fit(bootstrap_sample,weights=weight_col.tolist()).evaluate(kde_x)
        else:
            bootstrap_sample = random.choices(train_nonan, k=len(train_nonan))
            kde_y = NaiveKDE(bw=bin_width).fit(bootstrap_sample,weights=None).evaluate(kde_x)
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

def get_kde(outdir,train_in,ksdf_filtered,weighted,ks_lower,ks_upper):
    df = ksdf_filtered.dropna(subset=['weightoutlierexcluded'])
    kde_x = np.linspace(ks_lower,ks_upper, num=512)
    if weighted:
        kde_y = NaiveKDE(bw=0.1).fit(df['dS'].tolist(),weights=df['weightoutlierexcluded'].tolist()).evaluate(kde_x)
        plt.hist(df['dS'], bins = np.linspace(0, 50, num=51,dtype=int)/10, color = 'black', weights=df['weightoutlierexcluded'], alpha=0.2, rwidth=0.8)
    else:
        kde_y = NaiveKDE(bw='silverman').fit(train_in,weights=None).evaluate(kde_x)
        plt.hist(train_in, bins = np.linspace(0, 50, num=51,dtype=int)/10, color = 'black', alpha=0.2, rwidth=0.8)
    plt.plot(kde_x, kde_y, color = 'black',alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(outdir, "ksd_filtered.pdf")
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def Ten_multi(num):
    left=num%10
    return num-left

def draw_kde_CI(outdir,ksdf,boots,bw_method,date_lower = 0,date_upper=4,**kwargs):
    train_in = ksdf['PM']
    maxm = float(train_in.max())
    minm = float(train_in.min())
    kde_x = np.linspace(minm,maxm,num=512)
    modes = []
    f, ax = plt.subplots()
    if bw_method == 'silverman': logging.info("Assmuing the dates is unimodal and close to normal, applying silverman’s rule of thumb")
    else: logging.info("Assmuing the dates is far from normal or multimodal, applying the Improved Sheather Jones (ISJ) algorithm")
    for i,color in zip(range(boots),cm.rainbow(np.linspace(0, 1, boots))):
        sample = random.choices(train_in, k=len(train_in))
        #kde = stats.gaussian_kde(sample)
        kde_y = NaiveKDE(bw=bw_method).fit(sample,weights=None).evaluate(kde_x)
        mode, maxim = kde_mode(kde_x, kde_y)
        modes.append(mode)
        #kde_y = kde(kde_x)
        plt.plot(kde_x, kde_y, color = color,alpha=0.7)
    lower, upper = get_empirical_CI(0.95,modes)
    plt.axvline(x = lower, color = 'green', alpha = 0.8, ls = '--', lw = 1)
    plt.axvline(x = upper, color = 'green', alpha = 0.8, ls = '--', lw = 1)
    #https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
    logging.info("The 95% empirical confidence interval (CI) of WGD dates is {0} - {1} billion years".format(lower,upper))
    kde_y = NaiveKDE(bw=bw_method).fit(train_in.tolist(),weights=None).evaluate(kde_x)
    mode_orig, maxim_orig = kde_mode(kde_x, kde_y)
    plt.axvline(x = mode_orig, color = 'black', alpha = 0.8, ls = ':', lw = 1)
    kde_y = NaiveKDE(bw=bw_method).fit(modes).evaluate(kde_x)
    mode_of_modes, maxim_of_modes = kde_mode(kde_x, kde_y)
    plt.axvline(x = mode_of_modes, color = 'red', alpha = 0.8, ls = '-.', lw = 1)
    logging.info("The kde-mode of original WGD dates is {} billion years".format(mode_orig))
    plt.xlabel("Billion years ago", fontsize = 10)
    plt.ylabel("Frequency", fontsize = 10)
    print(Ten_multi(int(maxim_orig)))
    plt.yticks(np.linspace(0,Ten_multi(int(maxim_orig))+10,num=10,dtype=int))
    plt.hist(train_in,bins = np.linspace(minm, maxm, num=50),density=True,color = 'black', alpha=0.15, rwidth=0.8)
    props = dict(boxstyle='round', facecolor='gray', alpha=0.1)
    text = "\n".join(["Raw mode: {:4.4f}".format(mode_of_modes),"Peak: {:4.4f}".format(mode_orig),"CI: {:4.4f}-{:4.4f}".format(lower, upper),"OGs: {}".format(len(train_in))])
    plt.text(0.75,0.95,text,transform=ax.transAxes,fontsize=8,verticalalignment='top',bbox=props)
    fname = os.path.join(outdir, "WGD_peak.pdf")
    plt.tight_layout()
    plt.savefig(fname,format ='pdf', bbox_inches='tight')
    plt.close()

def draw_components_kde_bootstrap(outdir,num,ksdf_predict,weighted,boots,bin_width):
    parent = os.getcwd()
    os.chdir(outdir)
    dir_tmp = _mkdir('{}-components_model'.format(num))
    os.chdir(dir_tmp)
    for n in range(num):
        fname = "component{}.pdf".format(n)
        pretrain_in = ksdf_predict[ksdf_predict['component']==n]
        maxm = float(pretrain_in['dS'].max())
        minm = float(pretrain_in['dS'].min())
        kde_x = np.linspace(0,5,num=512)
        modes = []
        f, ax = plt.subplots()
        if weighted:
            train_in = pretrain_in.dropna(subset=['weightoutlierexcluded'])
            for i,color in zip(range(boots),cm.rainbow(np.linspace(0, 1, boots))):
                sample = random.choices(train_in['dS'],weights = train_in['weightoutlierexcluded'], k=len(train_in))
                kde_y = NaiveKDE(bw=bin_width).fit(sample,weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
                mode, maxim = kde_mode(kde_x, kde_y)
                modes.append(mode)
                plt.plot(kde_x, kde_y, color = color,alpha=0.1)
            plt.hist(train_in['dS'],bins = np.linspace(0, 50, num=51,dtype=int)/10,weights=train_in['weightoutlierexcluded'],density=True,color = 'black', alpha=0.15, rwidth=0.8)
        else:
            train_in = pretrain_in.dropna(subset=['node_averaged_dS_outlierexcluded'])
            for i,color in zip(range(boots),cm.rainbow(np.linspace(0, 1, boots))):
                sample = random.choices(train_in['node_averaged_dS_outlierexcluded'], k=len(train_in))
                kde_y = NaiveKDE(bw=bin_width).fit(sample).evaluate(kde_x)
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
            kde_y = NaiveKDE(bw=bin_width).fit(train_in['dS'].tolist(),weights=train_in['weightoutlierexcluded'].tolist()).evaluate(kde_x)
        else:
            train_in = pretrain_in.dropna(subset=['node_averaged_dS_outlierexcluded'])
            kde_y = NaiveKDE(bw=bin_width).fit(train_in['node_averaged_dS_outlierexcluded'].tolist()).evaluate(kde_x)
        mode_orig, maxim_orig = kde_mode(kde_x, kde_y)
        kde_y = NaiveKDE(bw=bin_width).fit(modes).evaluate(kde_x)
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

def Getanchor_Ksdf(anchor,ksdf,multiplicon):
    ap = pd.read_csv(anchor,header=0,index_col=0,sep = '\t')
    mp = pd.read_csv(multiplicon,header=0,index_col=0,sep = '\t')
