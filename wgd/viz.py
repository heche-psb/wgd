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

def sankey_plot(sp, df, minlen,outdir, seg, multi):
    lens = df.groupby("scaffold")["start"].agg(max)
    lens.name = "len"
    df1 = pd.DataFrame(lens).sort_values("len", ascending=False)
    if minlen < 0: minlen = df1.len.max() * 0.1
    noriginal = len(df1.index)
    df1 = df1.loc[df1.len > minlen]
    seg = seg.loc[seg['genome']==sp].copy()
    segs = list(seg.groupby('list'))
    scaflabels = list(map(lambda x: x[0],segs))
    #for i in scaflabels:
    #    print(i)
    #    print(type(i))
    patchescoordif = list(map(lambda x: list(x[1].loc[:,'first']),segs))
    patchescoordil = list(map(lambda x: list(x[1].loc[:,'last']),segs))
    patchessegid = list(map(lambda x: list(x[1].index),segs))
    #for i,j in zip(patchescoordif,patchescoordil):
    #    print(i)
    #    print(j)
    gene_start = {gene:start for gene,start in zip(df.index,list(df['start']))}
    #for pco in patchescoordi:
    #    print(pco.index)
        #for index in pco.index: print(str(pco.loc[index,'first']))
        #for index in pco.index: print(gene_start[pco.loc[index,'first']])
    multi = multi.loc[:,['id','level']].copy()
    seg_with_level = seg.merge(multi,left_on='multiplicon', right_on='id').drop(columns='id')
    segs_levels = {scafflabel:level for scafflabel,level in zip(list(seg_with_level.index),list(seg_with_level['level']))}
    highest_level = max(segs_levels.values())
    plothlines(highest_level,segs_levels,sp,gene_start,df1.len,df1.index,outdir,scaflabels,patchescoordif,patchescoordil,patchessegid)

def plothlines(highest_level,segs_levels,sp,gene_start,scafflength,scafflabel,outdir,patchedscaflabels,patchescoordif,patchescoordil,patchessegid):
    scafflength_normalized = [i/max(scafflength) for i in scafflength]
    fname = os.path.join(outdir, "{}_multiplicons_level.png".format(sp))
    fig, ax = plt.subplots(1, 1, figsize=(10,20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, (3*len(scafflength))+1)
    common = list(set(scafflabel) & set(patchedscaflabels))
    #levelcolortable = {2:black,3:green,4:}
    for i,le,la in zip(range(len(scafflength)),scafflength_normalized,scafflabel):
        #ax.hlines(i, xmin=0, xmax=le,linestyles='solid',label=la)
        #ax.add_patch(Rectangle((0, i+1),le,0.5,fc ='black',ec ='none',lw = 1, zorder=0, alpha=0.3))
        lower = (3*i)+0.5+0.1
        upper = (3*i)+3-0.75-0.1
        height_increment = (upper-lower)/highest_level
        #print(highest_level)
        #print(height_increment)
        ax.add_patch(Rectangle((0, (3*i)+1),le,0.5,fc ='black',ec ='none',lw = 1, zorder=0, alpha=0.3))
        ax.text(0, (3*i)+0.25,la)
        #if la in patchedscaflabels:
        if la in common:
            idc = patchedscaflabels.index(la)
            #for pla,pcof,pcol in zip(patchedscaflabels,patchescoordif,patchescoordil):
            #if la == pla:
            #    for f,l in zip(pcof,pcol):
            #        left = gene_start[f]/max(scafflength)
            #        right = gene_start[l]/max(scafflength)
            #        ple = right - left
            #intervals,height = np.linspace((3*i)+1.6, (3*i)+3.15, num=segs_levels[la],retstep=True,endpoint=False)
            #for inte in intervals: print(inte)
            #hscaled = 0.75*height
            for f,l,segid in zip(patchescoordif[idc],patchescoordil[idc],patchessegid[idc]):
                left = gene_start[f]/max(scafflength)
                right = gene_start[l]/max(scafflength)
                ple = right - left
                level = segs_levels[segid]
                #intervals,height = np.linspace((3*i)+1.6, (3*i)+3.15, num=level-1,retstep=True,endpoint=False)
                #lower = (3*i)+0.5+0.05
                #upper = (3*i)+3-0.75-0.05
                #height = (lower-upper)/(level-1)
                #hscaled = 0.75*height
                #height_increment = (upper-lower)/highest_level
                #height_increment = 0.2
                hscaled = 0.75*height_increment
                if ple > 0:
                    ax.add_patch(Rectangle((left, (3*i)+1),ple,0.5,fc ='green',ec ='none',lw = 1, zorder=2,alpha=0.9))
                    for lev in range(level-1): ax.add_patch(Rectangle((left, (3*i)+1.6+height_increment*(lev)),ple,hscaled,fc ='blue',ec ='none',lw = 1, zorder=1,alpha=0.9))
                    #for i in range(level):
                    #    inter = intervals[i]
                    #    ax.add_patch(Rectangle((left, inter),ple,hscaled,fc ='blue',ec ='none',lw = 1, zorder=1,alpha=0.9))
                else:
                    ax.add_patch(Rectangle((right, (3*i)+1),-ple,0.5,fc ='green',ec ='none',lw = 1, zorder=2,alpha=0.9))
                    for i in range(level):
                        inter = intervals[i]
                        ax.add_patch(Rectangle((left, inter),-ple,hscaled,fc ='blue',ec ='none',lw = 1, zorder=1,alpha=0.9))
                    print('find inversion')
    fig.savefig(fname)
    plt.close()

# dot plot stuff
def all_dotplots(df, segs, multi, anchors=None, **kwargs):
    """
    Generate dot plots for all pairs of species in `df`, coloring anchor pairs.
    """
    gdf = list(df.groupby("species"))
    n = len(gdf)
    figs = {}
    for i in range(n):
        for j in range(i, n):
            fig, ax = plt.subplots(1, 1, figsize=(10,10))
            ax2 = ax.twinx()
            ax3 = ax.twiny()
            spx, dfx = gdf[i]
            spy, dfy = gdf[j]
            logging.info("{} vs. {}".format(spx, spy))
            df, xs, ys, scaffxlabels, scaffylabels, scaffxtick, scaffytick = get_dots(dfx, dfy, segs, multi, **kwargs)
            if df is None:  # HACK, in case we're dealing with RBH orthologs...
                continue
            ax.scatter(df.x, df.y, s=1, color="k", alpha=0.01)
            if not (anchors is None):
                andf = df.join(anchors, how="inner")
                ax.scatter(andf.x, andf.y, s=1, color="red", alpha=0.9)
            ax.vlines(xs, ymin=0, ymax=ys[-1], alpha=0.8, color="k")
            ax.hlines(ys, xmin=0, xmax=xs[-1], alpha=0.8, color="k")
            ax.set_xlim(0, xs[-1])
            ax.set_ylim(0, ys[-1])
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

def get_dots(dfx, dfy, seg, multi, minlen=-1, maxsize=50, outdir = ''):
    sp=dfx.loc[:,'species'][0]
    sankey_plot(sp, dfx, minlen, outdir, seg, multi)
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
    
