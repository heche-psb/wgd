import pandas as pd
import click
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from rich.logging import RichHandler
import logging

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

def get_totalH(Hs):
    CHF = 0
    for i in Hs: CHF = CHF + i
    return CHF

def calculateHPD(train_in,per):
    sorted_in = np.sort(train_in)
    upper_bound = np.percentile(train_in, per)
    lower_bound = np.percentile(train_in, 100-per)
    upper_bound_indice,lower_bound_indice = 0,0
    cutoff,candidates = int(np.ceil(per*len(sorted_in)/100)),[]
    for i,v in enumerate(sorted_in):
        if v >= upper_bound:
            upper_bound_indice = i
            break
    for i,v in enumerate(sorted_in):
        if v >= lower_bound:
            lower_bound_indice = i
            break
    for (x,y) in itertools.product(np.arange(0,lower_bound_indice,1,dtype=int), np.arange(upper_bound_indice,len(sorted_in),1,dtype=int)):
        if (y-x+1) >= cutoff: candidates.append((sorted_in[y] - sorted_in[x],(x,y)))
    lower,upper = sorted(candidates, key=lambda y: y[0])[0][1][0],sorted(candidates, key=lambda y: y[0])[0][1][1]
    return sorted_in[upper],sorted_in[lower]

@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbosity', '-v', type=click.Choice(['info', 'debug']), default='info', help="Verbosity level, default = info.")
def cli(verbosity):
    """
    This script is for post WGD dating visualization
    """
    logging.basicConfig(
        format='%(message)s',
        handlers=[RichHandler()],
        datefmt='%H:%M:%S',
        level=verbosity.upper())
    logging.info("Proper Initiation")
    pass

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('posterior', type=click.Path(exists=True))
@click.option('--percentile','-p',show_default=True, default=90,help = "percentile of posteriors")
@click.option('--indexcol', is_flag=True,help="first column as index col")
@click.option('--hpd', is_flag=True,help="calculate HPD CI instead of Equal-tail CI")
@click.option('--title','-t',show_default=True, default="WGD date",help = "title of plot")
@click.option('--output','-o',show_default=True, default="xx_dates_inM.pdf",help = "filename of output")
def postdis(posterior,percentile,indexcol,hpd,title,output):
    """
    Plot the posterior distribution
    """
    if indexcol: df = pd.read_csv(posterior,header=0,index_col=0,sep='\t').dropna()
    else: df = pd.read_csv(posterior,header=0,index_col=None,sep='\t').dropna()
    fig, ax = plt.subplots()
    train_in = np.array(df.loc[:,df.columns[0]])*100
    upper = np.percentile(train_in, percentile+(100-percentile)/2)
    lower = np.percentile(train_in, (100-percentile)/2)
    maxm,minm,mean,median = float(train_in.max()),float(train_in.min()),float(train_in.mean()),np.median(train_in)
    ax.set_xlim(minm-10,maxm+10)
    kde_x = np.linspace(minm,maxm,num=500)
    kde_y=stats.gaussian_kde(train_in,bw_method='silverman').pdf(kde_x)
    mode, maxim = kde_mode(kde_x, kde_y)
    Hs, Bins, patches = ax.hist(train_in,bins = np.linspace(0, 200, num=201),color='g', alpha=0.8, rwidth=0.8,label='Raw dates')
    CHF = get_totalH(Hs)
    scaling = CHF*1
    ax.plot(kde_x, kde_y*scaling, color='k',alpha=0.8, ls = '-',lw = 1)
    if hpd:
        upper_HPD,lower_HPD = calculateHPD(train_in,percentile)
        plt.axvline(x = upper_HPD, color = 'k', alpha = 0.8, ls = '-.', lw = 1,label='{}% HPD CI Upper {:.2f}'.format(percentile,upper_HPD))
        plt.axvline(x = lower_HPD, color = 'k', alpha = 0.8, ls = '-.', lw = 1,label='{}% HPD CI Lower {:.2f}'.format(percentile,lower_HPD))
    else:
        plt.axvline(x = upper, color = 'k', alpha = 0.8, ls = '-.', lw = 1,label='{}% Equal-tail CI Upper {:.2f}'.format(percentile,upper))
        plt.axvline(x = lower, color = 'k', alpha = 0.8, ls = '-.', lw = 1,label='{}% Equal-tail CI Lower {:.2f}'.format(percentile,lower))
    plt.axvline(x = mean, color = 'k', alpha = 0.8, ls = ':', lw = 1,label='Posterior Mean {:.2f}'.format(mean))
    plt.axvline(x = median, color = 'k', alpha = 0.8, ls = '--', lw = 1,label='Median {:.2f}'.format(median))
    plt.axvline(x = mode, color = 'k', alpha = 0.8, ls = '-', lw = 1,label='Mode {:.2f}'.format(mode))
    ax.legend(loc=0,fontsize='small',frameon=False)
    ax.set_xlabel("Million years ago")
    ax.set_ylabel("Counts")
    sns.despine(offset=1)
    plt.title(title)
    fig.tight_layout()
    if output != "xx_dates_inM.pdf": fig.savefig(output)
    else: fig.savefig(posterior+'_dates_inM.pdf')
    plt.close()

if __name__ == '__main__':
	cli()
