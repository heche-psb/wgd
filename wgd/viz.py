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
from scipy import stats
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

def getspair_ks(spair,df,spgenemap):
    df_perspair = {}
    allspair = []
    for i in spair:
        pair = '_'.join(sorted([j.strip() for j in i.split(';')]))
        if pair not in allspair: allspair.append(pair)
    #df['sp1'],df['sp2'] = [spgenemap[g] for g in df['gene1']],[spgenemap[g] for g in df['gene2']]
    df['spair'] = ['_'.join(sorted([spgenemap[g1],spgenemap[g2]])) for g1,g2 in zip(df['gene1'],df['gene2'])]
    for p in allspair: df_perspair[p] = df[df['spair']==p]
    return df_perspair,allspair

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

def multi_sp_plot(df,spair,spgenemap,outdir,title='',ylabel='',viz=False):
    fnames = os.path.join(outdir,'{}_per_spair.ksd.svg'.format(title))
    fnamep = os.path.join(outdir,'{}_per_spair.ksd.pdf'.format(title))
    if not viz: writespgenemap(spgenemap,outdir)
    df_perspair,allspair = getspair_ks(spair,df,spgenemap)
    cs = cm.rainbow(np.linspace(0, 1, len(allspair)))
    keys = ["dS", "dS", "dN", "dN/dS"]
    np.seterr(divide='ignore')
    funs = [lambda x: x, np.log10, np.log10, np.log10]
    #fig, axs = plt.subplots(2, 2)
    fig, ax = plt.subplots()
    df_pers = [df_perspair[i] for i in allspair]
    bins = 50
    kdesity = 100
    kde_x = np.linspace(0,5,num=bins*kdesity)
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    for i,item in enumerate(df_perspair.items()):
        pair,df_per = item[0],item[1]
        #for ax, k, f in zip(axs.flatten(), keys, funs):
        w = df_per['weightoutlierexcluded']
        x = df_per['dS']
        y = x[np.isfinite(x)]
        w = w[np.isfinite(x)]
        Hs, Bins, patches = ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color=cs[i], alpha=0.3, rwidth=0.8,label=pair)
        kde = stats.gaussian_kde(y,weights=w,bw_method=0.1)
        kde_y = kde(kde_x)
        mode, maxim = kde_mode(kde_x, kde_y)
        CHF = get_totalH(Hs)
        scale = CHF*0.1
        ax.plot(kde_x, kde_y*scale, color=cs[i],alpha=0.4, ls = '--')
        ax.axvline(x = mode, color = cs[i], alpha = 0.8, ls = ':', lw = 1)
            #if funs[0] == f: ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color=cs[i], alpha=0.3, rwidth=0.8,label=pair)
            #else: ax.hist(y, weights=w, color=cs[i], alpha=0.3, rwidth=0.8,bins=50,label=pair)
        #w = [df_per['weightoutlierexcluded'] for df_per in df_pers]
        #x = [f(df_per['dS']) for df_per in df_pers]
        #y = [i[np.isfinite(i)] for i in x]
        #w = [j[np.isfinite(n)] for j,n in zip(w,x)]
        #if funs[0] == f: ax.hist(y, bins = np.linspace(0, 50, num=51,dtype=int)/10, weights=w, color=cs, alpha=0.3, rwidth=0.8,label=allspair)
        #else: ax.hist(y, weights=w, color=cs, alpha=0.3, rwidth=0.8,bins=50,label=allspair)
            #xlabel = _labels[k]
            #if f == np.log10: xlabel = "$\log_{10}" + xlabel[1:-1] + "$"
            #ax.set_xlabel(xlabel)
            #ax.legend(loc=1,bbox_to_anchor=(1.0, 0.9),fontsize='small')
    ax.set_xlabel(_labels["dS"])
    ax.legend(loc=1,fontsize=5,bbox_to_anchor=(0.9, 0.95))
    #axs[0,0].set_ylabel(ylabel)
    #axs[1,0].set_ylabel(ylabel)
    #axs[0,0].set_xticks([0,1,2,3,4,5])
    ax.set_ylabel(ylabel)
    ax.set_xticks([0,1,2,3,4,5])
    sns.despine(offset=1)
    #fig.suptitle(title, x=0.125, y=0.9, ha="left", va="top")
    #plt.title(title,loc='center')
    ax.set_title(title)
    fig.tight_layout()
    #plt.subplots_adjust(top=0.85)
    fig.savefig(fnames)
    fig.savefig(fnamep)
    plt.close()

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
            ax.set_title("${}$:${}$".format(cols[i], cols[j]), fontsize=9)
            k += 1
    for ax in axs:
        ymn, ymx = ax.get_ylim()
        ax.set_ylim(-0.5, ymx)
        ax.set_xlabel("# segments")
    axs[0].set_ylabel("A:B ratio")
    sns.despine(trim=False, offset=3)
    fig.tight_layout()
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
    colors = cm.rainbow(np.linspace(0, 1, len(scafflength)))
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

def sankey_plot(spx, dfx, spy, dfy, minlen, outdir, seg, multi):
    lens = dfx.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1x = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1x.len.max() * 0.1
    df1x = df1x.loc[df1x.len > minlen]
    seg.loc[:,"segment"] = seg.index
    seg_unfilterded = seg.loc[seg['genome']==spx].copy()
    segs_info = seg.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    profile = segs_info.unstack(level=-1).fillna(0)
    if spy not in profile.columns:
        logging.info('No collinear segments were found involving genome of {}'.format(spy))
    elif spx not in profile.columns:
        logging.info('No collinear segments were found involving genome of {}'.format(spx))
    else:
        if spx != spy: multi_goodinuse = profile.loc[profile[spy]>0,[spy,spx]].copy()
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
            plothlines(highest_level,segs_levels,spx,gene_start,df1x.len,df1x.index,outdir,scaflabels,patchescoordif,patchescoordil,patchessegid,spy = spy,spx_level = segs_levels_spx)

def plothlines(highest_level,segs_levels,sp,gene_start,scafflength,scafflabel,outdir,patchedscaflabels,patchescoordif,patchescoordil,patchessegid,spy = None, spx_level = None):
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
        lower = (3*i)+0.5+0.1
        upper = (3*i)+3-0.75-0.1
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
                if spx_level is None: level = segs_levels[segid]
                else: level = segs_levels[segid] + spx_level[segid] - 1
                hscaled = 0.75*height_increment
                iternum = level-1 if sp == spy else level
                color = 'green' if sp == spy else 'blue'
                hatchs = '//////' if color == 'green' else '\\\\\\\\\\\\'
                if ple > 0:
                    ax.add_patch(Rectangle((left, (3*i)+1),ple,0.5,fc ='green',ec ='none',lw = 1, zorder=2,alpha=0.5))
                    if spx_level is None:
                        for lev in range(iternum):
                            ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.5, hatch=hatchs))
                    else:
                        spx_times = 0
                        level_x = spx_level[segid] - 1
                        if level_x == 0:
                            for lev in range(iternum):
                                ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.5, hatch=hatchs))
                        else:
                            for lev in range(iternum):
                                if spx_times < level_x:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc ='green',ec ='none',lw = 1, zorder=1,alpha=0.5,hatch='//////'))
                                else:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.5,hatch=hatchs))
                                spx_times = spx_times + 1
                else:
                    ax.add_patch(Rectangle((right, (3*i)+1),-ple,0.5,fc = 'green',ec ='none',lw = 1, zorder=2,alpha=0.5))
                    if spx_level is None:
                        for lev in range(iternum):
                            ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.5,hatch=hatchs))
                    else:
                        spx_times = 0
                        level_x = spx_level[segid] - 1
                        if level_x == 0:
                            for lev in range(iternum):
                                ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.5,hatch=hatchs))
                        else:
                            for lev in range(iternum):
                                if spx_times < level_x:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc ='green',ec ='none',lw = 1, zorder=1,alpha=0.5,hatch='//////'))
                                else:
                                    ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),-ple,hscaled,fc =color,ec ='none',lw = 1, zorder=1,alpha=0.5,hatch=hatchs))
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

def get_marco_whole(dfs,seg, multi, maxsize=None, minlen=None, outdir=None):
    #species = [i.loc[:,'species'][0] for i in dfs]
    sp_scaffla_scaffle = {}
    gene_start = {}
    for df in dfs:
        gene_start.update({gene:start for gene,start in zip(df.index,list(df['start']))})
        species = df.loc[:,'species'][0]
        lens = df.groupby("scaffold")["start"].agg(max)
        lens.name = "len"
        df_tmp = pd.DataFrame(lens).sort_values("len", ascending=False)
        if minlen < 0: minlen = df_tmp.len.max() * 0.1
        df_tmp = df_tmp.loc[df_tmp.len > minlen]
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
            plot_marco_whole(sp_scaffla_scaffle,seg_filtered,gene_start,outdir)

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
                        if spi_indice-spj_indice==1:
                            f2,l2 = (f2[0],f2[1]+0.75), (l2[0],l2[1]+0.75)
                        else:
                            f1,l1 = (f1[0],f1[1]+0.75), (l1[0],l1[1]+0.75)
                        vertices.append([f1,l1,l2,f2])
    return vertices,color


def plot_marco_whole(scaf_info,seg_f,gene_start,outdir):
    fig, ax = plt.subplots(1, 1, figsize=(100,10))
    fname = os.path.join(outdir, "All_species_marcosynteny.png")
    fnamep = os.path.join(outdir, "All_species_marcosynteny.pdf")
    fnames = os.path.join(outdir, "All_species_marcosynteny.svg")
    num_sp = len(scaf_info)
    colors = cm.rainbow(np.linspace(0, 1, num_sp))
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
            ax.add_patch(Rectangle((leng_done, 1+indice*10),le_scaled,0.75,fc = color,ec ='none',lw = 1, zorder=0, alpha=0.5))
            ax.text(leng_done+(le_scaled/2),1+indice*10+0.75/2,lb,size=5,zorder=1,color="w",ha="center",va="center")
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
    for coord in coordi:
        sp_occur = set(coord['genome'])
        coord_sp = {sp:[] for sp in sp_occur}
        for sp in sp_occur:
            df = coord.loc[coord['genome']==sp,['first','last','list']]
            for i in df.index:
                f,l,li = df.loc[i,'first'],df.loc[i,'last'],df.loc[i,'list']
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
    ax.tick_params(axis='both', which='major', labelsize=30, labelbottom = False, bottom = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)
    fig.savefig(fnamep)
    fig.savefig(fnames)

def get_marco(dfx, dfy, seg, multi, maxsize=None, minlen=None, outdir=None):
    spx=dfx.loc[:,'species'][0]
    spy=dfy.loc[:,'species'][0]
    lens = dfx.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1x = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1x.len.max() * 0.1
    df1x = df1x.loc[df1x.len > minlen]
    lens = dfy.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1y = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1y.len.max() * 0.1
    df1y = df1y.loc[df1y.len > minlen]
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
            plot_marco(spx,spy,df1x.index,df1x.len,df1y.index,df1y.len,outdir,genex_start,geney_start,seg_filterded)

def plot_marco(sp1,sp2,sp1_scafflabel,sp1_scafflength,sp2_scafflabel,sp2_scafflength,outdir,genex_start,geney_start,seg_f):
    ## Only consider inter-specific links
    fname = os.path.join(outdir, "{0}_{1}_marcosynteny.png".format(sp1,sp2))
    fnamep = os.path.join(outdir, "{0}_{1}_marcosynteny.pdf".format(sp1,sp2))
    fnames = os.path.join(outdir, "{0}_{1}_marcosynteny.svg".format(sp1,sp2))
    fig, ax = plt.subplots(1, 1, figsize=(100,10))
    sp1_wholelength = sum(sp1_scafflength)
    sp2_wholelength = sum(sp2_scafflength)
    sp1_length_scaled = [i/sp1_wholelength for i in sp1_scafflength]
    sp2_length_scaled = [i/sp2_wholelength for i in sp2_scafflength]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    colors = cm.rainbow(np.linspace(0, 1, 2))
    sp1_leng_done = 0
    segs1_starts = {}
    yticks = [1+0.75/2,1+10+0.75/2]
    yticklabels = [sp1,sp2]
    sp1_scafflabel_length = {lb:le for lb,le in zip(sp1_scafflabel,sp1_scafflength)}
    sp2_scafflabel_length = {lb:le for lb,le in zip(sp2_scafflabel,sp2_scafflength)}
    for i,le,lb in zip(range(len(sp1_length_scaled)),sp1_length_scaled,sp1_scafflabel):
        le_scaled = le * 0.75
        ax.add_patch(Rectangle((sp1_leng_done, 1),le_scaled,0.75,fc =colors[0],ec ='none',lw = 1, zorder=0, alpha=0.3))
        ax.text(sp1_leng_done+(le_scaled/2),1+0.75/2,lb,size=20,zorder=1,color="w",ha="center",va="center")
        segs1_starts[lb]= sp1_leng_done
        sp1_leng_done = sp1_leng_done + le
    sp2_leng_done = 0
    segs2_starts = {}
    for i,le,lb in zip(range(len(sp2_length_scaled)),sp2_length_scaled,sp2_scafflabel):
        le_scaled = le * 0.75
        ax.add_patch(Rectangle((sp2_leng_done, 1 + 10),le_scaled,0.75,fc =colors[1],ec ='none',lw = 1, zorder=0, alpha=0.3))
        ax.text(sp2_leng_done+le_scaled/2,1+10+0.75/2,lb,size=20,zorder=1,color="w",ha="center",va="center")
        segs2_starts[lb] = sp2_leng_done
        sp2_leng_done = sp2_leng_done + le
    multis = list(seg_f.reset_index(drop=True).groupby('multiplicon'))
    multi_indices = list(map(lambda x: x[0],multis))
    coordi_x = list(map(lambda x: x[1].loc[x[1]['genome']==sp1,['first','last','list']],multis))
    coordi_y = list(map(lambda x: x[1].loc[x[1]['genome']==sp2,['first','last','list']],multis))
    for cix,ciy in zip(coordi_x,coordi_y):
        for i in cix.index:
            f,l,li1 = cix.loc[i,'first'],cix.loc[i,'last'],cix.loc[i,'list']
            f1,l1 = 0.75*genex_start[f]/sum(sp1_scafflength)+segs1_starts[li1],0.75*genex_start[l]/sum(sp1_scafflength)+segs1_starts[li1]
            for j in ciy.index:
                f2,l2,li2 = ciy.loc[j,'first'],ciy.loc[j,'last'],ciy.loc[j,'list']
                if f==f2 and l==l2:
                    continue
                ratio = (geney_start[l2]-geney_start[f2])/sp2_scafflabel_length[li2]
                if ratio > 0.05:
                    if len(cix) == 1 and len(ciy) == 1: color = 'gray'
                    elif len(cix) == 2 or len(ciy) == 2: color = 'green'
                    elif len(cix) == 3 or len(ciy) == 3: color = 'blue'
                    elif len(cix) == 4 or len(ciy) == 4: color = 'red'
                    else: color = 'yellow'
                else: color = 'gray'
                f2,l2 = 0.75*geney_start[f2]/sum(sp2_scafflength)+segs2_starts[li2],0.75*geney_start[l2]/sum(sp2_scafflength)+segs2_starts[li2]
                vertices = [(f1,1+0.75),(l1,1+0.75),(l2,11),(f2,11)]
                codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
                pp = patches.PathPatch(Path(vertices,codes),fc=color,alpha=0.1,zorder=1,lw=0.1)
                ax.add_patch(pp)
    # Second the links
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize=30, labelbottom = False, bottom = False)
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
def all_dotplots(df, segs, multi, anchors=None, ancestor=None, **kwargs):
    """
    Generate dot plots for all pairs of species in `df`, coloring anchor pairs.
    """
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
            get_dots(dfx, dfy, segs, multi, dupStack = True, **kwargs)
    logging.info("Making dotplots and marco-synteny plots")
    if n > 1: get_marco_whole(list(map(lambda x:x[1],gdf)),segs, multi,**kwargs)
    for i in range(n):
        for j in range(i, n):
            fig, ax = plt.subplots(1, 1, figsize=(10,10))
            ax2 = ax.twinx()
            ax3 = ax.twiny()
            spx, dfx = gdf[i]
            spy, dfy = gdf[j]
            logging.info("{} vs. {}".format(spx, spy))
            get_marco(dfx, dfy, segs, multi, **kwargs)
            df, xs, ys, scaffxlabels, scaffylabels, scaffxtick, scaffytick = get_dots(dfx, dfy, segs, multi, dupStack = False, **kwargs)
            if df is None:  # HACK, in case we're dealing with RBH orthologs...
                continue
            ax.scatter(df.x, df.y, s=1, color="k", alpha=0.01)
            if not (anchors is None):
                andf = df.join(anchors, how="inner")
                ax.scatter(andf.x, andf.y, s=1, color="red", alpha=0.9)
            xlim = max(scaffxtick)
            ylim = max(scaffytick)
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, ylim)
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            #ax.vlines(xs, ymin=0, ymax=ys[-1], alpha=0.8, color="k")
            ax.vlines(xs, ymin=0, ymax=ylim, alpha=0.8, color="k")
            #ax.hlines(ys, xmin=0, xmax=xs[-1], alpha=0.8, color="k")
            ax.hlines(ys, xmin=0, xmax=xlim, alpha=0.8, color="k")
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
            figs[spx + "-vs-" + spy] = fig
    return figs

def Ks_dotplots(segs,dff, df, ks, an, anchors=None, color_map='Spectral',min_ks=0.05, max_ks=5, minlen=250, maxsize=25, **kwargs):
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
            df, xs, ys, scafflabels, scaffylabels, scaffxtick, scaffytick = get_dots(dfx, dfy, segs, multi, minlen=minlen, maxsize=maxsize, outdir = outdir)
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

def get_dots(dfx, dfy, seg, multi, minlen=-1, maxsize=200, outdir = '', dupStack = False):
    spx=dfx.loc[:,'species'][0]
    spy=dfy.loc[:,'species'][0]
    if dupStack: sankey_plot(spx, dfx, spy, dfy, minlen, outdir, seg, multi)
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
            return None, None, None, None
        df = pd.DataFrame.from_dict(xs).set_index("pair")
        scaffxlabels = list(dfx['scaffold'].drop_duplicates())
        scaffylabels = list(dfy['scaffold'].drop_duplicates())
        #xl = list(np.unique(dfx["scaffstart"])) + [max(df.x)]
        xl = list(dfx["scaffstart"].drop_duplicates()) + [max(df.x)]
        #yl = list(np.unique(dfy["scaffstart"])) + [max(df.y)]
        yl = list(dfy["scaffstart"].drop_duplicates()) + [max(df.y)]
        return df, xl, yl, scaffxlabels, scaffylabels, scaffxtick, scaffytick 

def filter_data_dotplot(df, minlen):
    lens = df.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    lens = pd.DataFrame(lens).sort_values("len", ascending=False)
    scaffstart = [0] + list(np.cumsum(lens.len))[0:-1]
    scafftick = list(np.cumsum(lens.len))
    lens["scaffstart"] = scaffstart
    df = df.join(lens, on="scaffold").sort_values("len", ascending=False)
    # df now contains scaffold lengths
    if minlen < 0:  # find a reasonable threshold, 5% of longest scaffold?
        minlen = df.len.max() * 0.1
        logging.info("`minlen` not set, taking 10% of longest scaffold ({})".format(minlen))
    noriginal = len(df.index)
    df = df.loc[df.len > minlen]
    logging.info("Dropped {} genes because they are on scaffolds shorter "
            "than {}".format(noriginal - len(df.index), minlen))
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
    
