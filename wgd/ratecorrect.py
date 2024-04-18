from Bio import Phylo
import os
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.pyplot import cm
import random
from random import sample
from matplotlib.patches import Rectangle
import seaborn as sns
import copy
from sklearn import mixture
import math
import matplotlib.patches as patches
from scipy import interpolate,signal
from joblib import Parallel, delayed
from tqdm import trange

_labels = {"dS" : "$K_\mathrm{S}$","dN" : "$K_\mathrm{A}$", "dN/dS": "$\omega$"}

def apply_filters(df, filters):
    for key, lower, upper in filters:
        df = df[df[key] > lower]
        df = df[df[key] < upper]
    return df

def _mkdir(dirname):
    if not os.path.isdir(dirname) :
        os.mkdir(dirname)
    return dirname

def getoutinin(mrca,focusp,Ingroup_spnames):
    mrca_species_pool = [i.name for i in mrca.get_terminals()]
    outgroup_in_ingroup = set(Ingroup_spnames) - set(mrca_species_pool)
    return [i for i in outgroup_in_ingroup]

def bic_info_wgd(bic,n1):
    bestb_loc = np.argmin(bic)
    logging.info("Rules-of-thumb (Kass & Raftery, 1995) evaluates the outperformance of the BIC-best model over remaining:")
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

def addapgmm(ax,X,W,components,outdir,Hs):
    kde_x = np.linspace(0,5,num=5000)
    X_log = np.log(np.array(X)).reshape(-1, 1)
    aic_bic_fplot = os.path.join(outdir,"AIC_BIC.pdf")
    models, aic, bic, besta, bestb, N = fit_gmm(aic_bic_fplot, X_log, 2352890, components[0], components[1], em_iter=200, n_init=200)
    means,covariances,weights = besta.means_,besta.covariances_,besta.weights_
    CHF = get_totalH(Hs)
    scaling = CHF*0.1
    cs = cm.tab20b(np.linspace(0, 1, len(weights)))
    for num in range(len(weights)):
        mean,std,weight = means[num][0],np.sqrt(covariances[num][0][0]),weights[num]
        ax.plot(kde_x,scaling*weight*stats.lognorm.pdf(kde_x, scale=np.exp(mean),s=std), c=cs[num], ls='--', lw=1, alpha=0.8, label='Anchor '+'$K_\mathrm{S}$ '+'component {} (mode {:.2f})'.format(num+1,np.exp(mean - std**2)))
    return ax

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

def fit_gmm(out_file,X, seed, n1, n2, em_iter=100, n_init=1):
    """
    Compute Gaussian mixtures for different numbers of components
    """
    # The EM algorithm can't deal with weighted data, so we're actually ignoring all the associated weights.
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
    bic_info_wgd(bic,n1)
    plot_aic_bic(aic, bic, n1, n2, out_file)
    return models, aic, bic, besta, bestb, N

def aic_info(aic,n1):
    besta_loc = np.argmin(aic)
    logging.info("Rules-of-thumb (Burnham & Anderson, 2002) compares the AIC-best model and remaining:")
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
    for k in range(num_comp):
        if k == 0: prod = weights[k] * stats.expon.pdf(ks, scale=1/lambd)
        else:
            prod = weights[k] * stats.lognorm.pdf(ks, scale=np.exp(means[k-1]), s=stdevs[k-1])
        products.append(prod)
    sum_comp_perpoint = sum(products)
    log_sum_comp_perpoint = np.log(sum_comp_perpoint)
    fit_loglikelihood = sum(log_sum_comp_perpoint)
    posteriors = [products[i] / sum_comp_perpoint for i in range(num_comp)]
    return fit_loglikelihood, posteriors

def m_step(num_comp, ks, posteriors):
    new_lambda = sum(posteriors[0]) / sum(posteriors[0] * ks)
    points_per_k = [sum(posteriors[i]) for i in range(num_comp)]
    new_weights = [round(points_per_k[i]/len(ks),2) for i in range(num_comp)]
    for indice in range(len(points_per_k)):
        if not points_per_k[indice]>0:
            logging.debug("Found component weight as zero")
            points_per_k[indice] = 1e-6
    new_means = [sum(posteriors[i+1] * np.log(ks)) / points_per_k[i+1] for i in range(num_comp-1)]
    new_stdevs = [np.sqrt(sum(posteriors[i+1]*pow(np.log(ks)-new_means[i],2))/points_per_k[i+1]) for i in range(num_comp-1)]
    return new_means, new_stdevs, new_weights, new_lambda

def EM_step(num_comp,data,means,stds,lambd,weights,max_EM_iterations=200,max_num_comp = 5, reduced_gaussians_flag=None):
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
    return bic, new_means, new_stdevs, new_lambd, new_weights, convergence

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

def addelmm(ax,df,max_EM_iterations=200,num_EM_initializations=200,peak_threshold=0.1,rel_height=0.4, na = False):
    df = df.dropna(subset=['dS','weightoutlierexcluded'])
    df = df.loc[(df['dS']>0) & (df['dS']<5),:]
    ks_or = np.array(df['dS'])
    w = np.array(df['weightoutlierexcluded'])
    if na: deconvoluted_data = ks_or.copy()
    else: deconvoluted_data = get_deconvoluted_data(ks_or,w)
    hist_property = np.histogram(ks_or, weights=w, bins=50, density=True)
    init_lambd = hist_property[0][0] # as the rate parameter, higher lambd (i.e., higher bar height at Ks 0-0.1) denote faster decline of the number of gene duplicates. Besides, when x(i.e.,Ks) = 0, the y = lamda
    ks = np.log(ks_or)
    max_ks,min_ks = ks.max(),ks.min()
    ks_refed,cutoff,w_refed = reflect_logks(ks,w)
    kde = stats.gaussian_kde(ks_refed, bw_method="scott", weights=w_refed)
    bw_modifier = 0.4
    kde.set_bandwidth(kde.factor * bw_modifier)
    kde_x = np.linspace(min_ks-cutoff, max_ks+cutoff,num=500)
    kde_y = kde(kde_x)
    spl = interpolate.UnivariateSpline(kde_x, kde_y)
    spl.set_smoothing_factor(0.01)
    spl_x = np.linspace(min_ks, max_ks+0.1, num=int(round((abs(min_ks) + (max_ks+0.1)) *100)))
    spl_y = spl(spl_x)
    peaks, properties = signal.find_peaks(spl_y)
    prominences = signal.peak_prominences(spl_y, peaks)[0]
    prominences_refed_R1,width_refed_R1,prominences_refed_L1,width_refed_L1 = [],[],[],[]
    for i in range(len(peaks)):
        peak_index = peaks[i]
        spl_peak_refl_y = np.concatenate((np.flip(spl_y[peak_index+1:]), spl_y[peak_index:]))
        spl_peak_refl_x = np.concatenate((np.flip(spl_x[peak_index+1:] * -1 + 2 * spl_x[peak_index]), spl_x[peak_index:]))
        current_peak_index = int((len(spl_peak_refl_x)-1)/2)
        new_prominences = signal.peak_prominences(spl_peak_refl_y,[current_peak_index])[0][0]
        new_width,new_height,left_ips,right_ips = signal.peak_widths(spl_peak_refl_y, [current_peak_index], rel_height=rel_height)
        if new_width[0] > 150: new_width[0] = 150
        prominences_refed_R1.append(new_prominences)
        width_refed_R1.append(new_width[0])
        c = "r" if new_prominences >= peak_threshold else 'gray'
        w = new_width[0]/2/100
        spl_peak_refl_y_L = np.concatenate((spl_y[:peak_index+1], np.flip(spl_y[:peak_index])))
        spl_peak_refl_x_L = np.concatenate((spl_x[:peak_index+1], np.flip(spl_x[:peak_index]) * -1 + 2*spl_x[peak_index]))
        current_peak_index = int((len(spl_peak_refl_x_L)-1)/2)
        new_prominences = signal.peak_prominences(spl_peak_refl_y_L,[current_peak_index])[0][0]
        new_width,new_height,left_ips,right_ips = signal.peak_widths(spl_peak_refl_y_L, [current_peak_index], rel_height=rel_height)
        if new_width[0] > 150: new_width[0] = 150
        prominences_refed_L1.append(new_prominences)
        width_refed_L1.append(new_width[0])
        c = "r" if new_prominences >= peak_threshold else 'gray'
        w = new_width[0]/2/100
    good_peaks_R1,good_peaks_L1 = [i>=peak_threshold for i in prominences_refed_R1],[i>=peak_threshold for i in prominences_refed_L1]
    good_prominences,init_means,init_stdevs = [],[],[]
    for i in range(len(peaks)):
        if good_peaks_R1[i] or good_peaks_L1[i]:
            init_means.append(spl_x[peaks[i]])
            best = np.argmax((prominences_refed_R1[i], prominences_refed_L1[i]))
            good_prominences.append(max([prominences_refed_R1[i], prominences_refed_L1[i]]))
            width = [width_refed_R1[i], width_refed_L1[i]][best]
            init_stdevs.append(width/2/100)
    reduced_gaussians = False
    if len(init_stdevs) > 4:
        sor_by_prom = [(m,s) for m,s,_ in sorted(zip(init_means,init_stdevs,good_prominences), key=lambda y: y[2])]
        init_means,init_stdevs = [i for i,j in sor_by_prom[:4]],[j for i,j in sor_by_prom[:4]]
        reduced_gaussians = True
        logging.info('Too many peak signals detected among which only 4 with highest prominences are retained')
    buffer_maxks,buffer_std = 5,0.3
    logging.info('An extra buffer lognormal component with mean {:.2f} and std 0.30 is appended'.format(buffer_maxks))
    init_means.append(np.log(buffer_maxks))
    init_stdevs.append(buffer_std)
    logging.info('Found {} likely peak signals'.format(len(init_means)))
    for m,s in zip(init_means,init_stdevs): logging.info('The initiative means and stds is {:.2f} {:.2f}'.format(np.exp(m),s))
    num_comp = len(init_means)+1
    init_weights = [1/num_comp] * num_comp
    all_models_init_parameters,bic_dict,all_models_fitted_parameters = {},{},{}
    all_models_init_parameters['Model1'] = [init_means, init_stdevs, init_lambd, init_weights]
    num_comp = len(init_means) + 1
    logging.info("Performing EM algorithm from initializated data (Model1)")
    bic, new_means, new_stdevs, new_lambd, new_weights, convergence = EM_step(num_comp,deconvoluted_data,init_means, init_stdevs, init_lambd, init_weights,max_EM_iterations=max_EM_iterations,max_num_comp = 5, reduced_gaussians_flag=reduced_gaussians)
    #if convergence: logging.info('The EM algorithm has reached convergence')
    #else: logging.info("The EM algorithm hasn't reached convergence")
    all_models_fitted_parameters['Model1'] = [new_means, new_stdevs, new_lambd, new_weights]
    bic_dict['Model1'] = bic
    logging.info('BIC of Model1: {:.2f}'.format(bic))
    bic_from_same_num_comp,start_parameters,final_parameters = [],[],[]
    logging.info("Performing EM algorithm from initializated data plus a random lognormal component (Model2)")
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
        bic_from_same_num_comp.append(bic)
        final_parameters.append([new_means, new_stdevs, new_lambd, new_weights])
    updated_means, updated_stdevs, init_lambd, updated_weights = start_parameters[np.argmin(bic_from_same_num_comp)]
    all_models_init_parameters['Model2'] = [updated_means, updated_stdevs, init_lambd, updated_weights]
    final_means, final_stdevs, final_lambd, final_weights = final_parameters[np.argmin(bic_from_same_num_comp)]
    all_models_fitted_parameters['Model2'] = [final_means, final_stdevs, final_lambd, final_weights]
    bic_dict['Model2'] = min(bic_from_same_num_comp)
    logging.info('BIC of Model2 : {:.2f}'.format(min(bic_from_same_num_comp)))
    min_num_comp,max_num_comp = 2,5
    num_comp_list = np.arange(min_num_comp, max_num_comp + 1)
    model_ids = num_comp_list+1
    for num_comp, model_id in zip(num_comp_list, model_ids):
        logging.info("Performing EM algorithm from random initialization with {0} components (Model{1})".format(num_comp,model_id))
        bic_from_same_num_comp = []
        start_parameters, final_parameters = [], []
        for i in range(num_EM_initializations):
            init_means, init_stdevs, init_weights = [], [], [1/num_comp] * num_comp
            init_lambd = round(np.random.choice(np.arange(0.2, 1, 0.1)), 2)
            for j in range(num_comp-2):
                init_means.append(round(np.random.choice(np.arange(-0.5, 1, 0.01)),1))
                init_stdevs.append(round(np.random.choice(np.arange(0.3, 0.9, 0.01)),1))
            init_means.append(np.log(5))
            init_stdevs.append(0.3)
            start_parameters.append([init_means, init_stdevs, init_lambd, init_weights])
            bic, new_means, new_stdevs, new_lambd, new_weights, convergence = EM_step(num_comp,deconvoluted_data,init_means, init_stdevs, init_lambd, init_weights,max_EM_iterations=max_EM_iterations,max_num_comp = 5)
            bic_from_same_num_comp.append(bic)
            final_parameters.append([new_means, new_stdevs, new_lambd, new_weights])
        init_means, init_stdevs, init_lambd, init_weights = start_parameters[np.argmin(bic_from_same_num_comp)]
        all_models_init_parameters["Model{}".format(model_id)] = [init_means, init_stdevs, init_lambd, init_weights]
        final_means, final_stdevs, final_lambd, final_weights = final_parameters[np.argmin(bic_from_same_num_comp)]
        all_models_fitted_parameters["Model{}".format(model_id)] = [final_means, final_stdevs, final_lambd, final_weights]
        bic_dict["Model{}".format(model_id)] = min(bic_from_same_num_comp)
        logging.info('BIC of Model{} : {:.2f}'.format(model_id,min(bic_from_same_num_comp)))
    logging.info("Models are evaluated according to BIC scores")
    model_bic = [(k,v) for k,v in bic_dict.items()]
    modelist, bic_list = [k for k,v in model_bic],[v for k,v in model_bic]
    best_model_id = modelist[np.argmin(bic_list)]
    logging.info("The best fitted model via BIC is {}".format(best_model_id))
    bic_info(modelist, bic_list)
    model_bic_ordered = [(k,v) for k,v in sorted(bic_dict.items(), key=lambda y:y[0])]
    model_ordered, bic_ordered = [k for k,v in model_bic_ordered],[v for k,v in model_bic_ordered]
    final_means, final_stdevs, final_lambd, final_weights = all_models_fitted_parameters[best_model_id]
    bin_width = 0.1
    scaling = 0.1 * len(deconvoluted_data[deconvoluted_data <= 5])
    x_points = np.linspace(-5, 5, int((5 + 5) *100))
    x_points_strictly_positive = np.linspace(0, 5, int(5 * 100))
    total_pdf = final_weights[0] * stats.expon.pdf(x_points_strictly_positive, scale=1/final_lambd)
    ax.plot(x_points_strictly_positive,scaling*final_weights[0]*stats.expon.pdf(x_points_strictly_positive, scale=1/final_lambd), c='g', ls='-', lw=1.5, alpha=0.8, label='Exponential optimized')
    lognormal_peaks = {i:round(np.exp(final_means[i] - pow(final_stdevs[i], 2)), 2) for i in range(len(final_stdevs))}
    lognormals_sorted_by_peak = [k for k,v in sorted(lognormal_peaks.items(), key=lambda y:y[1])]
    letter_dict = dict(zip(lognormals_sorted_by_peak, [ "a", "b", "c", "d", "e", "f", "g"][:len(final_stdevs)]))
    colors = ["b", "r", "c", "m", "k"][:len(final_stdevs)-1] + ["y"]
    for comp, color in zip(lognormals_sorted_by_peak, colors):
        ax.plot(x_points_strictly_positive,scaling*final_weights[comp+1]*stats.lognorm.pdf(x_points_strictly_positive, scale=np.exp(final_means[comp]),s=final_stdevs[comp]), c=color, ls='-', lw=1.5, alpha=0.8, label=f'Lognormal {letter_dict[comp]} optimized (mode {lognormal_peaks[comp]})')
        total_pdf = total_pdf + final_weights[comp+1]*stats.lognorm.pdf(x_points_strictly_positive,scale=np.exp(final_means[comp]),s=final_stdevs[comp])
    ax.plot(x_points_strictly_positive, scaling*total_pdf, "k-", lw=1.5, label=f'Exp-lognormal mixture model')
    return ax

def plotmixed(focusp,df,reweight,extraPara=None,AP=None,elmm=True,mEM=20,nEM=20,na=True,pt=0.1,rh=0.4,components=(1,4),apgmm=True):
    spair = focusp+"__"+focusp
    fig,ax = plt.subplots()
    df_spair = df[df['spair']==spair].copy()
    if len(df_spair) == 0 and extraPara is None: logging.error("No paralogous Ks data was found")
    if extraPara is not None:
        df_spair = pd.read_csv(extraPara,header=0,index_col=0,sep='\t')
        df_spair = apply_filters(df_spair, [("dS", 0., 5.)])
    if reweight:
        w = reweighted(df_spair)
        df_spair['weightoutlierexcluded'] = w
    else:
        w = df_spair['weightoutlierexcluded']
    x = df_spair['dS']
    y = x[np.isfinite(x)]
    w = w[np.isfinite(x)]
    Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
    safemax = ax.get_ylim()[1] * 1.1
    CHF = get_totalH(Hs)
    scaling = CHF*0.1
    if elmm:
        logging.info("ELMM analysis on paralogous Ks distribution")
        if na:
            df_spair = df_spair.drop_duplicates(subset=['family','node'])
            df_spair = df_spair.drop(['dS'], axis=1).rename(columns={'node_averaged_dS_outlierexcluded':'dS'})
            df_spair['weightoutlierexcluded'] = 1
        ax = addelmm(ax,df_spair,max_EM_iterations=mEM,num_EM_initializations=nEM,peak_threshold=pt,rel_height=rh,na=na)
    if AP is not None:
        df_ap = pd.read_csv(AP,header=0,index_col=0,sep='\t')
        df_ap.loc[:,"pair"] = df_ap[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
        df_working = df_ap.set_index('pair').join(df_spair).dropna(subset=['dS','weightoutlierexcluded'])
        if na:
            x = df_working.groupby(['family','node'])['dS'].mean()
            y = x[np.isfinite(x)]
            w = [1 for ds in x]
        else:
            w = reweighted(df_working) if reweight else df_working['weightoutlierexcluded']
            x = df_working['dS']
            y = x[np.isfinite(x)]
            w = w[np.isfinite(x)]
        Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='g', rwidth=0.8,label='Anchor pairs')
        if apgmm: ax = addapgmm(ax,y,w,components,os.getcwd(),Hs)
    ax.set_title(focusp)
    ax.set_xlabel(_labels["dS"])
    ax.set_ylabel('Number of retained duplicates')
    if elmm or AP is not None: ax.legend(loc=0,frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,safemax)
    return fig,ax

def addcorrectline(ax_spair,corrected_ks_spair,corrected_ks_spair_std,maxim_spair,ks_spair):
    fontsize_factor = 25 / len(ax_spair) if len(ax_spair) != len(corrected_ks_spair) else 20 / len(ax_spair)
    for spair in corrected_ks_spair.keys():
        if spair not in ax_spair: continue
        ax_spair[spair] = addvvline(ax_spair[spair],corrected_ks_spair[spair],'g','--','Corrected mean')
        ax_spair[spair] = addvvline(ax_spair[spair],corrected_ks_spair[spair]+corrected_ks_spair_std[spair],'g','-.','+ Corrected std')
        ax_spair[spair] = addvvline(ax_spair[spair],corrected_ks_spair[spair]-corrected_ks_spair_std[spair],'g','-.','- Corrected std')
        if len(ax_spair)!=1: ax_spair[spair].legend(loc=0,frameon=False,fontsize=fontsize_factor)
        else: ax_spair[spair].legend(loc=0,frameon=False)
        ax_spair[spair].quiver(ks_spair[spair],maxim_spair[spair],corrected_ks_spair[spair]-ks_spair[spair],0,angles='xy', scale_units='xy', scale=1,color='k',width=0.005,headwidth=2,headlength=2,headaxislength=2)
        p = patches.Rectangle((corrected_ks_spair[spair]-corrected_ks_spair_std[spair], 0), 2*corrected_ks_spair_std[spair], ax_spair[spair].get_ylim()[1], edgecolor='none', facecolor='g', alpha=0.2)
        ax_spair[spair].add_patch(p)

def addcorrectline_mixed(ax,corrected_ks_spair,corrected_ks_spair_std,ks_spair,Outgroup_spair_ordered,focusp):
    y = lambda x:x.replace(focusp,'').replace('__','')
    cs = cm.viridis(np.linspace(0, 1, len(set(Outgroup_spair_ordered.values()))))
    for spair in sorted(corrected_ks_spair.keys(),key=lambda x:Outgroup_spair_ordered[x],reverse=True):
    #for spair in corrected_ks_spair.keys():
        ax = addvvline(ax,corrected_ks_spair[spair],cs[Outgroup_spair_ordered[spair]-1],'-.','[{0}] {1} {2:.2f} {3} {4:.2f}'.format(Outgroup_spair_ordered[spair],y(spair),float(corrected_ks_spair[spair]),'\u2190',float(ks_spair[spair])),rawid=True)
        p = patches.Rectangle((corrected_ks_spair[spair]-corrected_ks_spair_std[spair], 0), 2*corrected_ks_spair_std[spair], ax.get_ylim()[1], edgecolor='none', facecolor=cs[Outgroup_spair_ordered[spair]-1], alpha=0.2)
        ax.add_patch(p)
    #ax.legend(loc=0,frameon=False)
    ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5),frameon=False)
    sns.despine(offset=1)

def ksadjustment(Trios_dict,spairs_means_stds_samples):
    corrected_ks_spair,corrected_ks_spair_std = {},{}
    for spair,trios in Trios_dict.items():
        Ks_adjusted_all,Ks_adjusted_all_stds,Ks_adjusted_all_samples = [],[],[]
        for trio in trios:
            Ks_focus_outgroup = spairs_means_stds_samples["{}".format("__".join(sorted([trio[0],trio[2]])))][0]
            Ks_sister_outgroup = spairs_means_stds_samples["{}".format("__".join(sorted([trio[1],trio[2]])))][0]
            Ks_focus_sister = spairs_means_stds_samples["{}".format("__".join(sorted([trio[0],trio[1]])))][0]
            Ks_focus_specific = ((Ks_focus_outgroup - Ks_sister_outgroup) + Ks_focus_sister)/2
            Ks_adjusted = Ks_focus_specific * 2
            Ks_adjusted_all.append(Ks_adjusted)
            std_Ks_focus_outgroup = spairs_means_stds_samples["{}".format("__".join(sorted([trio[0],trio[2]])))][1]
            std_Ks_sister_outgroup = spairs_means_stds_samples["{}".format("__".join(sorted([trio[1],trio[2]])))][1]
            std_Ks_focus_sister = spairs_means_stds_samples["{}".format("__".join(sorted([trio[0],trio[1]])))][1]
            samples_Ks_focus_outgroup = spairs_means_stds_samples["{}".format("__".join(sorted([trio[0],trio[2]])))][2]
            samples_Ks_sister_outgroup = spairs_means_stds_samples["{}".format("__".join(sorted([trio[1],trio[2]])))][2]
            samples_Ks_focus_sister = spairs_means_stds_samples["{}".format("__".join(sorted([trio[0],trio[1]])))][2]
            samples_this_trio = [i+j-n for i,j,n in zip(samples_Ks_focus_outgroup,samples_Ks_sister_outgroup,samples_Ks_focus_sister)]
            Ks_adjusted_all_samples.append(samples_this_trio)
            #std_Ks_adjusted = np.std(samples_this_trio)
            std_Ks_adjusted = ((std_Ks_focus_outgroup)**2 + (std_Ks_sister_outgroup)**2 +(std_Ks_focus_sister)**2 +
                    2*(-np.cov(samples_Ks_focus_outgroup,samples_Ks_sister_outgroup)[0,1]
                    - np.cov(samples_Ks_sister_outgroup,samples_Ks_focus_sister)[0,1]
                    + np.cov(samples_Ks_focus_outgroup,samples_Ks_focus_sister)[0,1]))**0.5
            Ks_adjusted_all_stds.append(std_Ks_adjusted)
        added = 0
        for i in range(len(trios)):
            for j in range(i+1,len(trios)):
                added += 2*(np.cov(Ks_adjusted_all_samples[i],Ks_adjusted_all_samples[j])[0,1])
        std_final_adjusted_Ks = ((sum([std**2 for std in Ks_adjusted_all_stds])+added)/(len(trios)**2))**0.5
        final_adjusted_Ks = np.array(Ks_adjusted_all).mean()
        corrected_ks_spair[spair] = final_adjusted_Ks
        corrected_ks_spair_std[spair] = std_final_adjusted_Ks
    return corrected_ks_spair,corrected_ks_spair_std

def addvvline(ax,xvalue,color,lstyle,labell,rawid=False):
    if labell == '': ax.axvline(xvalue,color=color, ls=lstyle, lw=1)
    elif rawid: ax.axvline(xvalue,color=color, ls=lstyle, lw=1, label='{}'.format(labell))
    else: ax.axvline(xvalue,color=color, ls=lstyle, lw=1, label='{}: {:.2f}'.format(labell,xvalue))
    return ax

def aviod_singular_matrix(*args):
    new_args = []
    for arg in args:
        arg = [arg[0]+(1e-9)*random.randint(1,1000)] + list(set(arg))
        new_args.append(arg)
    return new_args

def addbt_noax(spair,df,reweight,na,num=10):
    df_spair = df[df['spair']==spair].copy()
    if na:
        df_spair = df_spair.drop_duplicates(subset=['family','node'])
        df_spair = df_spair.drop(['dS'], axis=1).rename(columns={'node_averaged_dS_outlierexcluded':'dS'})
        df_spair['weightoutlierexcluded'] = 1
        w = df_spair['weightoutlierexcluded']
    else:
        if reweight:
            w = reweighted(df_spair)
            df_spair['weightoutlierexcluded'] = w
        else:
            w = df_spair['weightoutlierexcluded']
    x = df_spair['dS']
    y = x[np.isfinite(x)]
    w = w[np.isfinite(x)]
    if len(y) == 0 or len(w) == 0:
        logging.error("Species pair {} has no Ks data, please check your data!".format(spair))
        exit()
    data = [(i,j) for i,j in zip(y,w)]
    kde_x = np.linspace(0,5,num=5000)
    modes,mus,kde_xs,kde_ys = [],[],[],[]
    for i in range(num):
        random_values = random.choices(data, k=len(y))
        new_y,new_w = [m for m,n in random_values],[n for m,n in random_values]
        if len(set(y)) == 1 and len(set(w)) == 1: new_y,new_w = aviod_singular_matrix(new_y,new_w)
        kde = stats.gaussian_kde(new_y,weights=new_w,bw_method=0.1)
        kde_y = kde(kde_x)
        kde_ys.append(kde_y)
        kde_xs.append(kde_x)
        mode, maxim = kde_mode(kde_x, kde_y)
        modes.append(mode)
    lower, upper = np.percentile(modes,5), np.percentile(modes,95)
    mean, std = np.mean(modes), np.std(modes)
    return spair,lower,upper,mean,std,modes,kde_ys,kde_xs

def addbt_precal(ax,scaling,kde_xs,kde_ys,bt):
    for kde_x, kde_y in zip(kde_xs,kde_ys):
        ax.plot(kde_x, kde_y*scaling, color='gray',alpha=2/bt, ls = '-')
    return ax

def addbt(ax,y,w,scaling,num=10):
    data = [(i,j) for i,j in zip(y,w)]
    kde_x = np.linspace(0,5,num=5000)
    modes,mus,kde_xs,kde_ys = [],[],[],[]
    for i in range(num):
        random_values = random.choices(data, k=len(y))
        new_y,new_w = [m for m,n in random_values],[n for m,n in random_values]
        kde = stats.gaussian_kde(new_y,weights=new_w,bw_method=0.1)
        kde_y = kde(kde_x)
        kde_ys.append(kde_y)
        kde_xs.append(kde_x)
        mode, maxim = kde_mode(kde_x, kde_y)
        modes.append(mode)
        ax.plot(kde_x, kde_y*scaling, color='gray',alpha=2/num, ls = '-')
    lower, upper = np.percentile(modes,5), np.percentile(modes,95)
    mean, std = np.mean(modes), np.std(modes)
    return ax,lower,upper,mean,std,modes,kde_ys,kde_xs

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

def get_totalH(Hs):
    CHF = 0
    for i in Hs: CHF = CHF + i
    return CHF

def reweighted(df_per):
    return 1 / df_per.groupby(["family", "node"])["dS"].transform('count')

def is_prime(number):
    if number <= 1:
        return False
    elif number == 2:
        return True
    elif number % 2 == 0:
        return False
    else:
        for i in range(3, int(number**0.5) + 1, 2):
            if number % i == 0:
                return False
        return True

def find_closest_divisors(number):
    if is_prime(number):
        if number>3:
            number+=1
    closest_divisors = (1, number)
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            if abs(i - number//i) < abs(closest_divisors[0] - closest_divisors[1]):
                closest_divisors = (i, number//i)
    return closest_divisors[::-1]

#def plotspair_one(i,spair,spairs,closest_divisors,fig,df,reweight,fig_fp,ax_spair_fp,fig_sigs,ax_sigs,ax_spair,maxim_spair,ks_spair,bt,spairs_means_stds_samples,order_fs_pair,fs_pairs,na,kde_x,fontsize_factor,closest_divisors_fp,fontsize_factor_fp):
    # 3 plots are made here, all species pair (together), focal-sister pair, all species pair (single plot)
    #if len(spairs) > 1:
    #    if closest_divisors[0]>1:
    #        ax = fig.add_subplot(closest_divisors[0], closest_divisors[1], i+1)
    #    else:
    #        ax = fig.add_subplot(1, closest_divisors[1], i+1)
    #else:
    #    ax = axes
    #df_spair = df[df['spair']==spair].copy()
    #if na:
    #    df_spair = df_spair.drop_duplicates(subset=['family','node'])
    #    df_spair = df_spair.drop(['dS'], axis=1).rename(columns={'node_averaged_dS_outlierexcluded':'dS'})
    #    df_spair['weightoutlierexcluded'] = 1
    #    w = df_spair['weightoutlierexcluded']
    #else:
    #    if reweight:
    #        w = reweighted(df_spair)
    #        df_spair['weightoutlierexcluded'] = w
    #    else:
    #        w = df_spair['weightoutlierexcluded']
    #x = df_spair['dS']
    #y = x[np.isfinite(x)]
    #w = w[np.isfinite(x)]
    #Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
    #CHF = get_totalH(Hs)
    #scaling = CHF*0.1
    #kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
    #kde_y = kde(kde_x)
    #mode, maxim = kde_mode(kde_x, kde_y)
    #ax,lower,upper,mean,std,modes,kde_ys,kde_xs = addbt(ax,y,w,scaling,num=bt)
    #ax.plot(kde_x, kde_y*scaling, color='k',alpha=1, ls = '-')
    #spairs_means_stds_samples[spair] = (mean,std,modes)
    #ax = addvvline(ax,lower,'k','-.','90% BTCI')
    #ax = addvvline(ax,upper,'k','-.','90% BTCI')
    #ax = addvvline(ax,mode,'k','-','Raw mode')
    #ax = addvvline(ax,mean,'k','--','Mean')
    #ax.set_title(" & ".join(sorted(spair.split('__'))),fontsize=fontsize_factor*1.8)
    #ax.legend(loc=0,frameon=False,fontsize=fontsize_factor)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #if spair in fs_pairs and len(spairs) > 1:
    #    ax_fp = fig_fp.add_subplot(closest_divisors_fp[0], closest_divisors_fp[1], order_fs_pair[spair]) #index starts from 1 here
    #    ax_fp.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
    #    for kde_x,kde_y in zip(kde_xs,kde_ys): ax_fp.plot(kde_x, kde_y*scaling, color='gray',alpha=2/bt, ls = '-')
    #    ax_fp.plot(kde_x, kde_y*scaling, color='k',alpha=1, ls = '-')
    #    ax_fp = addvvline(ax_fp,lower,'k','-.','90% BTCI')
    #    ax_fp = addvvline(ax_fp,upper,'k','-.','90% BTCI')
    #    ax_fp = addvvline(ax_fp,mode,'k','-','Raw mode')
    #    ax_fp = addvvline(ax_fp,mean,'k','--','Mean')
    #    ax_fp.set_title(" & ".join(sorted(spair.split('__'))),fontsize=fontsize_factor_fp*1.8)
    #    ax_fp.spines['top'].set_visible(False)
    #    ax_fp.spines['right'].set_visible(False)
        #ax_spair_fp[spair] = ax_fp
    #    fig_fp.text(0.5, 0.01, _labels["dS"],va='center',ha='center')
    #    fig_fp.text(0.01, 0.5, 'Number of retained duplicates',rotation='vertical',va='center',ha='center')
    #else: ax_fp = None
    #fig_sig,ax_sig = plt.subplots()
    #fig_sig,ax_sig = fig_sigs[spair],ax_sigs[spair]
    #ax_sig.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
    #for kde_x,kde_y in zip(kde_xs,kde_ys): ax_sig.plot(kde_x, kde_y*scaling, color='gray',alpha=2/10, ls = '-')
    #ax_sig.plot(kde_x, kde_y*scaling, color='k',alpha=1, ls = '-')
    #ax_sig = addvvline(ax_sig,lower,'k','-.','90% BTCI')
    #ax_sig = addvvline(ax_sig,upper,'k','-.','90% BTCI')
    #ax_sig = addvvline(ax_sig,mode,'k','-','Raw mode')
    #ax_sig = addvvline(ax_sig,mean,'k','--','Mean')
    #ax_sig.set_title(" & ".join(sorted(spair.split('__'))))
    #ax_sig.legend(loc=0,frameon=False)
    #ax_sig.spines['top'].set_visible(False)
    #ax_sig.spines['right'].set_visible(False)
    #fig_sig.text(0.5, 0.01, _labels["dS"],va='center',ha='center')
    #fig_sig.text(0.01, 0.5, 'Number of retained duplicates',rotation='vertical',va='center',ha='center')
    #maxim_scaling = maxim*scaling
    #fig_sigs[spair] = fig_sig
    #ax_sigs[spair] = ax_sig
    #ax_spair[spair] = ax
    #maxim_spair[spair] = maxim*scaling
    #ks_spair[spair] = mode
    #return ax,maxim_scaling,mean,std,mode,modes,ax_fp

def plotspair_cov(df,spairs,fs_pairs,focusp,reweight,bt=200,na=True,nthreads=4):
    order_spair = {spair:(i+1) for i,spair in enumerate(spairs) if spair in fs_pairs}
    for i in spairs:
        if i not in order_spair: order_spair[i] = 0
    if len(spairs) > 1:
        closest_divisors = find_closest_divisors(len(spairs))
        closest_divisors_fp = find_closest_divisors(len(fs_pairs))
        fig = plt.figure()
        fig_fp = plt.figure()
    else:
        fig, axes = plt.subplots()
    cs = cm.viridis(np.linspace(0, 1, len(spairs)))
    kde_x = np.linspace(0,5,num=5000)
    spairs_means_stds_samples,ax_spair,maxim_spair,ks_spair,ax_spair_fp,fig_sigs,ax_sigs = {},{},{},{},{},{},{}
    fontsize_factor = 25 / len(spairs)
    fontsize_factor_fp = 25 / len(fs_pairs) if closest_divisors_fp[1] !=1 else 15 / len(fs_pairs)
    plotted = 0
    #order_fs_pair = [spair for i,spair in enumerate(sorted(spairs,key=lambda x: order_spair[x],reverse=True)) if spair in fs_pairs]
    #order_fs_pair = {spair:(i+1) for i,spair in enumerate(order_fs_pair)}
    #for spair in spairs:
    #    fig_sig,ax_sig = plt.subplots()
    #    fig_sigs[spair] = fig_sig
    #    ax_sigs[spair] = ax_sig
    #results = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(plotspair_one)(i,spair,spairs,closest_divisors,fig,df,reweight,fig_fp,ax_spair_fp,fig_sigs,ax_sigs,ax_spair,maxim_spair,ks_spair,bt,spairs_means_stds_samples,order_fs_pair,fs_pairs,na,kde_x,fontsize_factor,closest_divisors_fp,fontsize_factor_fp) for i,spair in enumerate(sorted(spairs,key=lambda x: order_spair[x],reverse=True)))
    #for result,spair in zip(results,sorted(spairs,key=lambda x: order_spair[x],reverse=True)):
    #    ax,maxim_scaling,mean,std,mode,modes,ax_fp = result
        #fig_sigs[spair] = fig_sig
        #ax_sigs[spair] = ax_sig
    #    ax_spair[spair] = ax
    #    maxim_spair[spair] = maxim_scaling
    #    ks_spair[spair] = mode
    #    if ax_fp != None: ax_spair_fp[spair] = ax_fp
    #    spairs_means_stds_samples[spair] = (mean,std,modes)
    #for i,spair in tqdm(enumerate(sorted(spairs,key=lambda x: order_spair[x],reverse=True)),desc="Working on all considered species-pairs",unit=" species-pair finished"):
    results = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(addbt_noax)(spair,df,reweight,na,num=bt) for i,spair in zip(trange(len(spairs)),sorted(spairs,key=lambda x: order_spair[x],reverse=True)))
    results_list = [(result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]) for result in results]
    results_ordered = sorted(results_list,key=lambda x:order_spair[x[0]],reverse=True)
    logging.info("Sampling done, now making plots")
    for i,spair,result in zip(range(len(spairs)),sorted(spairs,key=lambda x: order_spair[x],reverse=True),results_ordered):
        if len(spairs) > 1:
            if closest_divisors[0]>1:
                ax = fig.add_subplot(closest_divisors[0], closest_divisors[1], i+1)
            else:
                ax = fig.add_subplot(1, closest_divisors[1], i+1)
        else:
            ax = axes
        df_spair = df[df['spair']==spair].copy()
        if na:
            df_spair = df_spair.drop_duplicates(subset=['family','node'])
            df_spair = df_spair.drop(['dS'], axis=1).rename(columns={'node_averaged_dS_outlierexcluded':'dS'})
            df_spair['weightoutlierexcluded'] = 1
            w = df_spair['weightoutlierexcluded']
        else:
            if reweight:
                w = reweighted(df_spair)
                df_spair['weightoutlierexcluded'] = w
            else:
                w = df_spair['weightoutlierexcluded']
        x = df_spair['dS']
        y = x[np.isfinite(x)]
        w = w[np.isfinite(x)]
        if len(set(y)) == 1 and len(set(w)) == 1:
            y,w = aviod_singular_matrix(y,w)
            logging.info("Species pair {} has only one unique orthologous Ks datapoint,\na random datapoint with negligible difference is thus added".format(spair))
        Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
        CHF = get_totalH(Hs)
        scaling = CHF*0.1
        kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
        kde_y = kde(kde_x)
        mode, maxim = kde_mode(kde_x, kde_y)
        # change is here
        #ax,lower,upper,mean,std,modes,kde_ys,kde_xs = addbt(ax,y,w,scaling,num=bt)
        spair,lower,upper,mean,std,modes,kde_ys,kde_xs = result
        ax = addbt_precal(ax,scaling,kde_xs,kde_ys,bt)
        # change stops here
        ax.plot(kde_x, kde_y*scaling, color='k',alpha=1, ls = '-')
        spairs_means_stds_samples[spair] = (mean,std,modes)
        ax = addvvline(ax,lower,'k','-.','90% BTCI')
        ax = addvvline(ax,upper,'k','-.','90% BTCI')
        ax = addvvline(ax,mode,'k','-','Raw mode')
        ax = addvvline(ax,mean,'k','--','Mean')
        ax.set_title(" & ".join(sorted(spair.split('__'))),fontsize=fontsize_factor*1.8)
        ax.legend(loc=0,frameon=False,fontsize=fontsize_factor)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if spair in fs_pairs and len(spairs) > 1:
            plotted += 1
            ax_fp = fig_fp.add_subplot(closest_divisors_fp[0], closest_divisors_fp[1], plotted)
            ax_fp.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
            for kde_x,kde_y in zip(kde_xs,kde_ys): ax_fp.plot(kde_x, kde_y*scaling, color='gray',alpha=2/bt, ls = '-')
            ax_fp.plot(kde_x, kde_y*scaling, color='k',alpha=1, ls = '-')
            ax_fp = addvvline(ax_fp,lower,'k','-.','90% BTCI')
            ax_fp = addvvline(ax_fp,upper,'k','-.','90% BTCI')
            ax_fp = addvvline(ax_fp,mode,'k','-','Raw mode')
            ax_fp = addvvline(ax_fp,mean,'k','--','Mean')
            ax_fp.set_title(" & ".join(sorted(spair.split('__'))),fontsize=fontsize_factor_fp*1.8)
            ax_fp.spines['top'].set_visible(False)
            ax_fp.spines['right'].set_visible(False)
            ax_spair_fp[spair] = ax_fp
            fig_fp.text(0.5, 0.01, _labels["dS"],va='center',ha='center')
            fig_fp.text(0.01, 0.5, 'Number of retained duplicates',rotation='vertical',va='center',ha='center')
        fig_sig,ax_sig = plt.subplots()
        ax_sig.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color='k', alpha=0.5, rwidth=0.8)
        for kde_x,kde_y in zip(kde_xs,kde_ys): ax_sig.plot(kde_x, kde_y*scaling, color='gray',alpha=2/10, ls = '-')
        ax_sig.plot(kde_x, kde_y*scaling, color='k',alpha=1, ls = '-')
        ax_sig = addvvline(ax_sig,lower,'k','-.','90% BTCI')
        ax_sig = addvvline(ax_sig,upper,'k','-.','90% BTCI')
        ax_sig = addvvline(ax_sig,mode,'k','-','Raw mode')
        ax_sig = addvvline(ax_sig,mean,'k','--','Mean')
        ax_sig.set_title(" & ".join(sorted(spair.split('__'))))
        ax_sig.legend(loc=0,frameon=False)
        ax_sig.spines['top'].set_visible(False)
        ax_sig.spines['right'].set_visible(False)
        fig_sig.text(0.5, 0.01, _labels["dS"],va='center',ha='center')
        fig_sig.text(0.01, 0.5, 'Number of retained duplicates',rotation='vertical',va='center',ha='center')
        fig_sigs[spair] = fig_sig
        ax_sigs[spair] = ax_sig
        ax_spair[spair] = ax
        maxim_spair[spair] = maxim*scaling
        ks_spair[spair] = mode
    fig.text(0.5, 0.01, _labels["dS"],va='center',ha='center')
    fig.text(0.01, 0.5, 'Number of retained duplicates',rotation='vertical',va='center',ha='center')
    return fig,spairs_means_stds_samples,ax_spair,maxim_spair,ks_spair,ax_spair_fp,fig_fp,fig_sigs,ax_sigs

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
    return list(set(all_spairs)),list(set(spairs)),trios,Trios_dict

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

def findoutgroup(focusp,first_children_of_root):
    for clade in first_children_of_root:
        clade.name = clade.name + "_Outgroup"
        for tip in clade.get_terminals():
            if tip.name == focusp:
                clade.name = clade.name.replace("_Outgroup","_Ingroup")
                break

def getoutorder(tree,focusp):
    good_nodes_order = {}
    spair_order = {}
    rank = 0
    occured_outspnames = []
    for i in tree.get_nonterminals():
        if i.name == 'assumed_root': continue
        if focusp in [j.name for j in i.get_terminals()]:
            order = int(i.name.replace('internal_node_',''))
            outspname = [j.name for j in i.get_terminals() if j.name!=focusp]
            good_nodes_order[order] = outspname
    for order,outspnames in sorted(good_nodes_order.items(),key=lambda x:x[0],reverse=True):
        rank +=1
        for sp in outspnames:
            if sp not in occured_outspnames:
                spair_order["{}".format("__".join(sorted([focusp,sp])))] = rank
            occured_outspnames.append(sp)
    return spair_order

def writetable(spairs_means_stds_samples,fname):
    dic = {"species pair":[],"mean":[],"std":[],"bootstrap samples":[]}
    y = lambda x:", ".join([str(i) for i in x])
    for spair,value in spairs_means_stds_samples.items():
        dic["species pair"].append(spair)
        dic["mean"].append(value[0])
        dic["std"].append(value[1])
        dic["bootstrap samples"].append(y(value[2]))
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(fname,header=True,index=False,sep='\t')

def writecortable(corrected_ks_spair,corrected_ks_spair_std,spairs_means_stds_samples,fname):
    dic = {"species pair":[],"orig_mean":[],"orig_std":[],"corr_mean":[],"corr_std":[]}
    for spair,value in corrected_ks_spair.items():
        dic["species pair"].append(spair)
        dic["corr_mean"].append(value)
        dic["corr_std"].append(corrected_ks_spair_std[spair])
        dic["orig_mean"].append(spairs_means_stds_samples[spair][0])
        dic["orig_std"].append(spairs_means_stds_samples[spair][1])
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(fname,header=True,index=False,sep='\t')

def getspairplot_cov_cor(df,focusp,speciestree,onlyrootout,reweight,extraparanomeks,anchorpoints,outdir,na=True,elmm=True,mEM=200,nEM=200,pt=0.1,rh=0.4,components=(1,4),apgmm=True,BT=200,nthreads=4):
    odir = _mkdir(outdir)
    tree = Phylo.read(speciestree, "newick")
    logging.info("Reading species tree and categorizing sister&outgroup species")
    for i,clade in enumerate(tree.get_nonterminals()): clade.name = "internal_node_{}".format(i)
    tree.root.name = 'assumed_root'
    Outgroup_spair_ordered = getoutorder(tree,focusp)
    Depths = tree.root.depths(unit_branch_lengths=True)
    first_children_of_root = []
    for clade,depth in Depths.items():
        if depth == 1: first_children_of_root.append(clade)
    findoutgroup(focusp,first_children_of_root)
    Outgroup_clade = first_children_of_root[0] if first_children_of_root[0].name.endswith('_Outgroup') else first_children_of_root[1]
    Ingroup_clade = first_children_of_root[0] if first_children_of_root[0].name.endswith('_Ingroup') else first_children_of_root[1]
    Outgroup_spnames = [i.name.replace('_Outgroup','').replace('_Ingroup','') for i in Outgroup_clade.get_terminals()]
    Ingroup_spnames = [i.name.replace('_Outgroup','').replace('_Ingroup','') for i in Ingroup_clade.get_terminals()]
    logging.info("Composing trios (outgroup,(focal,sister))")
    if onlyrootout: all_spairs,spairs,Trios,Trios_dict = gettrios(focusp,Ingroup_spnames,Outgroup_spnames)
    else: all_spairs,spairs,Trios,Trios_dict = gettrios_overall(focusp,Ingroup_spnames,Outgroup_spnames,Ingroup_clade)
    logging.info("Sampling, calculating and plotting {} bootstrap replicates for {} orthologous Ks distributions using {} threads (which might take a while..)".format(BT,len(all_spairs),nthreads))
    logging.info("Note that the number of bootstrap replicates can be adjusted via the option --bootstrap")
    fig,spairs_means_stds_samples,ax_spair,maxim_spair,ks_spair,ax_spair_fp,fig_fp,fig_sigs,ax_sigs = plotspair_cov(df,all_spairs,spairs,focusp,reweight,na=na,bt=BT,nthreads=nthreads)
    corrected_ks_spair, corrected_ks_spair_std = ksadjustment(Trios_dict,spairs_means_stds_samples)
    logging.info("Synonymous substitution rate correction done (info written in output)\nNow adding correction shade onto plots")
    addcorrectline(ax_spair,corrected_ks_spair,corrected_ks_spair_std,maxim_spair,ks_spair)
    fig.tight_layout()
    os.chdir(odir)
    writetable(spairs_means_stds_samples,"spair.original.ks.info.tsv")
    writecortable(corrected_ks_spair,corrected_ks_spair_std,spairs_means_stds_samples,"spair.corrected.ks.info.tsv")
    if na: fig.savefig("All_pairs.ks.node.averaged.pdf",bbox_inches='tight')
    else: fig.savefig("All_pairs.ks.node.weighted.pdf",bbox_inches='tight')
    plt.close()
    os.chdir("../")
    addcorrectline(ax_spair_fp,corrected_ks_spair,corrected_ks_spair_std,maxim_spair,ks_spair)
    fig_fp.tight_layout()
    os.chdir(odir)
    if na: fig_fp.savefig("Focus_sister_pairs.ks.node.averaged.pdf",bbox_inches='tight')
    else: fig_fp.savefig("Focus_sister_pairs.ks.node.weighted.pdf",bbox_inches='tight')
    plt.close()
    os.chdir(_mkdir("Simple_Ks_Distributions"))
    for key,value in fig_sigs.items():
        addcorrectline({key:ax_sigs[key]},corrected_ks_spair,corrected_ks_spair_std,maxim_spair,ks_spair)
        value.tight_layout()
        if na: value.savefig("{}.ks.node.averaged.pdf".format(key),bbox_inches='tight')
        else: value.savefig("{}.ks.node.weighted.pdf".format(key),bbox_inches='tight')
        plt.close()
    os.chdir("../../")
    logging.info("Plotting the final mixed Ks distribution")
    fig,ax = plotmixed(focusp,df,reweight,extraPara=extraparanomeks,AP=anchorpoints,na=na,elmm=elmm,mEM=mEM,nEM=nEM,pt=pt,rh=rh,components=components,apgmm=apgmm)
    addcorrectline_mixed(ax,corrected_ks_spair,corrected_ks_spair_std,ks_spair,Outgroup_spair_ordered,focusp)
    #fig.tight_layout()
    os.chdir(odir)
    if na: fig.savefig("Mixed.ks.{}.node.averaged.pdf".format(focusp),bbox_inches='tight')
    else: fig.savefig("Mixed.ks.{}.node.weighted.pdf".format(focusp),bbox_inches='tight')
    plt.close()
    os.chdir("../")

def ratediffplot(df,outdir,focusp,speciestree,onlyrootout,reweight,extraparanomeks,anchorpoints,na=True,elmm=False,mEM=200,nEM=200,pt=0.1,rh=0.4,components=(1,4),apgmm=False,BT=200,nthreads=4):
    df['sp1'] = df['g1'].apply(lambda x:"_".join(x.split('_')[:-1]))
    df['sp2'] = df['g2'].apply(lambda x:"_".join(x.split('_')[:-1]))
    df['spair'] = ["__".join(sorted([sp1,sp2])) for sp1,sp2 in zip(df['sp1'],df['sp2'])]
    getspairplot_cov_cor(df,focusp,speciestree,onlyrootout,reweight,extraparanomeks,anchorpoints,outdir,na=na,elmm=elmm,mEM=mEM,nEM=nEM,pt=pt,rh=rh,components=components,apgmm=apgmm,BT=BT,nthreads=nthreads)
