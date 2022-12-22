#!/usr/bin/python3
import click
import logging
import sys
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sp
import pkg_resources  # part of setuptools
from rich.logging import RichHandler
__version__ = pkg_resources.require("wgd")[0].version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# CLI entry point
@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbosity', '-v', type=click.Choice(['info', 'debug']),
    default='info', help="Verbosity level, default = info.")
def cli(verbosity):
    """
    wgd - Copyright (C) 2018-2020 Arthur Zwaenepoel\n
    Contact: arzwa@psb.vib-ugent.be
    """
    logging.basicConfig(
        format='%(message)s',
        handlers=[RichHandler()],
        datefmt='%H:%M:%S',
        level=verbosity.upper())
    logging.info("This is wgd v{}".format(__version__))
    pass


# Diamond and gene families
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('sequences', nargs=-1, type=click.Path(exists=True))
@click.option('--outdir', '-o', default='wgd_dmd', show_default=True,
    help='output directory')
@click.option('--tmpdir', '-t', default=None, show_default=True,
    help='tmp directory')
@click.option('--cscore', '-c', default=None, show_default=True,
    help='c-score to delineate homologs')
@click.option('--inflation', '-I', default=2.0,
    help="inflation factor for MCL")
@click.option('--eval', '-e', default=1e-10,
    help="e-value cut-off for similarity")
@click.option('--to_stop', is_flag=True, 
    help="don't translate through STOP codons")
@click.option('--cds', is_flag=True,
    help="enforce proper CDS sequences")
@click.option('--focus','-f', default=None,
    help="Species whose WGD is to be dated")
@click.option('--anchorpoints', '-ap', default=None, show_default=True,
    help='anchorpoints.txt file from i-adhore')
@click.option('--keepfasta','-kf', is_flag=True,
    help="keep the fasta file of homologs family")
@click.option('--keepduplicates','-kd', is_flag=True,
    help="Keep ID duplicates of focus species")
@click.option('--globalmrbh','-gm', is_flag=True,
    help="global MRBH regardless of focus species")
@click.option('--nthreads', '-n', default=4, show_default=True,help="number of threads to use")
def dmd(**kwargs):
    """
    All-vs-all diamond blastp + MCL clustering.

    Requires diamond and mcl. Note the two key  parameters, being the e-value
    cut-off and inflation factor. It is advised to explore the effects of these
    on your analysis.

    Example 1 - whole paranome delineation:

        wgd dmd ath.fasta

    Example 2 - one vs. one ortholog delineation:

        wgd dmd ath.fasta vvi.fasta

    Example 3 - one vs. one ortholog delineation for multiple pairs:

        wgd dmd ath.fasta vvi.fasta egr.fasta

    Example 4 - one vs. one ortholog delineation for multiple pairs with focus species:

        wgd dmd ath.fasta vvi.fasta egr.fasta --focus ath.fasta (--anchorpoints anchorpoints.txt --cscore 0.7)

    """
    _dmd(**kwargs)

def _dmd(sequences, outdir, tmpdir, cscore, inflation, eval, to_stop, cds, focus, anchorpoints, keepfasta, keepduplicates, globalmrbh, nthreads):
    from wgd.core import SequenceData, read_MultiRBH_gene_families,mrbh
    s = [SequenceData(s, out_path=outdir, tmp_path=tmpdir,
        to_stop=to_stop, cds=cds, cscore=cscore) for s in sequences]
    if len(s) == 0:
        logging.error("No sequences provided!")
        return
    logging.info("tmp_dir = {}".format(s[0].tmp_path))
    if len(s) == 1:
        logging.info("One CDS file: will compute paranome")
        s[0].get_paranome(inflation=inflation, eval=eval)
        s[0].write_paranome()
    if focus is None and len(s) != 1 and not globalmrbh:
        logging.info("Multiple CDS files: will compute RBH orthologs")
        for i in range(len(s)-1):
            for j in range(i+1, len(s)):
                logging.info("{} vs. {}".format(s[i].prefix, s[j].prefix))
                s[i].get_rbh_orthologs(s[j], cscore=cscore, eval=eval)
                s[i].write_rbh_orthologs(s[j],singletons=False)
    mrbh(globalmrbh,outdir,s,cscore,eval,keepduplicates,anchorpoints,focus,keepfasta,nthreads)
    if tmpdir is None:
        [x.remove_tmp(prompt=False) for x in s]
    logging.info("Done")

#MSA and ML tree inference for given sets of orthologous gene familes for species tree inference and WGD timing

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('families', type=click.Path(exists=True))
@click.argument('sequences', nargs=-1, type=click.Path(exists=True))
@click.option('--outdir', '-o', default="wgd_focus_post", show_default=True,help='output directory')
@click.option('--tmpdir', '-t', default=None, show_default=True,help='tmp directory')
@click.option('--nthreads', '-n', default=4, show_default=True,help="number of threads to use")
@click.option('--to_stop', is_flag=True,help="don't translate through STOP codons")
@click.option('--cds', is_flag=True,help="enforce proper CDS sequences")
@click.option('--strip_gaps', is_flag=True,help="remove all gap-containing columns in the alignment")
@click.option('--aligner', '-a', default="mafft", show_default=True,type=click.Choice(['muscle', 'prank', 'mafft']), help='aligner program to use')
@click.option('--tree_method', '-tree',type=click.Choice(['fasttree', 'iqtree', 'mrbayes']),default='fasttree',show_default=True,help="tree inference method")
@click.option('--treeset', '-ts', multiple=True, default=None, show_default=True,help='parameters setting for gene tree inference')
@click.option('--concatenation', is_flag=True,help="species tree inference using concatenation method")
@click.option('--coalescence', is_flag=True,help="species tree inference using multispecies coalescence method")
@click.option('--speciestree', '-sp', default=None, show_default=True,help='species tree for dating')
@click.option('--dating', '-d', type=click.Choice(['beast', 'mcmctree', 'r8s', 'none']),default='none',show_default=True,help="dating orthologous families")
@click.option('--datingset', '-ds', multiple=True, default=None, show_default=True,help='parameters setting for dating')
@click.option('--nsites', '-ns', default=None, show_default=True,help='nsites information for r8s dating')
@click.option('--outgroup', '-ot', default=None, show_default=True,help='outgroup species for r8s dating')
@click.option('--partition','-pt', is_flag=True,help="1st 2nd and 3rd codon partition analysis")
@click.option('--aamodel', '-am', type=click.Choice(['poisson','wag', 'lg', 'dayhoff']),default='poisson',show_default=True,help="protein model to be used in mcmctree")
@click.option('-ks', is_flag=True,help="Ks analysis for orthologous families")
@click.option('--annotation',type=click.Choice(['none','eggnog', 'hmmpfam', 'interproscan']),default='none',show_default=True,help="Functional annotation for orthologous families")
@click.option('--pairwise', is_flag=True,help="Pairwise gene-pair feeded into codeml")
@click.option('--eggnogdata', '-ed', default=None, show_default=True,help='Eggnog data dirctory for annotation')
@click.option('--pfam', type=click.Choice(['none', 'denovo', 'realign']),default='none',show_default=True,help='PFAM domains for annotation')
@click.option('--dmnb', default=None, show_default=True,help='Diamond database for annotation')
@click.option('--hmm', default=None, show_default=True,help='profile for hmmscan')
@click.option('--evalue', default=1e-3, show_default=True,help='E-value threshold for annotation')
@click.option('--exepath', default=None, show_default=True,help='Path to interproscan installation folder')
@click.option('--fossil', '-f', nargs=5, default= ('clade1;clade2', 'taxa1,taxa2;taxa3,taxa4', '4;5', '0.5;0.6', '400;500'), show_default=True, help='fossil calibration info (id,taxa,mean,std,offset)')
@click.option('--rootheight', '-rh', nargs=3,default= (4,0.5,400), show_default=True, help='root height calibration info (mean,std,offset)')
@click.option('--chainset', '-cs', nargs=2,default= (10000,100), show_default=True, help='MCMC chain info (length,frequency) for beast')
@click.option('--beastlgjar', default=None, show_default=True,help='path of beastLG.jar')
@click.option('--beagle', is_flag=True,help='using beagle')
def focus(**kwargs):
    """
    Multiply species RBH or c-score defined orthologous family's gene tree inference, species tree inference and absolute dating pipeline.

    Example 1 - Dating orthologous families containing anchor pairs with a required user-defined species tree:

        wgd focus ap_families cds1.fasta cds2.fasta cds3.fasta --dating mcmctree --speciestree sp.newick -ds 'burnin = 2000' -ds 'sigma2_gamma = 1 10 1'

    Example 2 - Dating orthologous families containing anchor pairs with or without a user-defined species tree in r8s:

        wgd focus families cds1.fasta cds2.fasta cds3.fasta -d r8s -sp sp.newick -ns 1000 -ds 'MRCA **;' -ds 'constrain **;' 
    
        wgd focus families cds1.fasta cds2.fasta cds3.fasta -d r8s -ds 'MRCA **;' -ds 'constrain **;' -ot outgroup

    Example 3 - Species tree inference under both concatenation and coalescence method:

        wgd focus families cds1.fasta cds2.fasta cds3.fasta --concatenation --coalescence

    Example 4 - How to specify user's parameters for fasttree and iqtree

        wgd focus families cds1.fasta cds2.fasta cds3.fasta -ts '-boot 100' -ts -fastest

    If you want to keep intermediate (temporary) files, please provide a directory
    name for the `--tmpdir` option.
    """
    _focus(**kwargs)

def _focus(families, sequences, outdir, tmpdir, nthreads, to_stop, cds, strip_gaps, aligner, tree_method, treeset, concatenation, coalescence, speciestree, dating, datingset, nsites, outgroup, partition, aamodel, ks, annotation, pairwise, eggnogdata, pfam, dmnb, hmm, evalue, exepath, fossil, rootheight, chainset, beastlgjar, beagle):
    from wgd.core import SequenceData, read_gene_families, get_gene_families, KsDistributionBuilder
    from wgd.core import mergeMultiRBH_seqs, read_MultiRBH_gene_families, get_MultipRBH_gene_families, Concat, _Codon2partition_, Coale, Run_MCMCTREE, Run_r8s, Reroot, eggnog, hmmer_pfam, interproscan, Run_BEAST
    if dating=='r8s' and not speciestree is None and nsites is None:
        logging.error("Please provide nsites parameter for r8s dating")
        exit(0)
    if dating=='r8s' and speciestree is None and outgroup is None:
        logging.error("Please provide outgroup species for r8s dating")
        exit(0)
    if len(sequences) < 2:
        logging.error("Please provide at least three sequence files for constructing trees")
        exit(0)
    seqs = [SequenceData(s, tmp_path=tmpdir, out_path=outdir,to_stop=to_stop, cds=cds) for s in sequences]
    for s in range(len(seqs)):
        logging.info("tmpdir = {} for {}".format(seqs[s].tmp_path,seqs[s].prefix))
    fams = read_MultiRBH_gene_families(families)
    cds_alns, pro_alns, tree_famsf, calnfs, palnfs, calnfs_length, cds_fastaf = get_MultipRBH_gene_families(seqs,fams,tree_method,treeset,outdir,nthreads,option="--auto")
    if concatenation or dating != 'none':
        cds_alns_rn, pro_alns_rn, Concat_ctree, Concat_ptree, Concat_calnf, Concat_palnf, ctree_pth, ctree_length, gsmap, Concat_caln, Concat_paln, slist = Concat(cds_alns, pro_alns, families, tree_method, treeset, outdir)
    if coalescence:
        coalescence_ctree, coalescence_treef = Coale(tree_famsf, families, outdir)
    if dating == 'beast':
        if speciestree is None or beastlgjar is None:
            logging.error("Please provide species tree and path of beastLG.jar for beast dating")
            exit(0)
        Run_BEAST(Concat_caln, Concat_paln, Concat_calnf, cds_alns_rn, pro_alns_rn, calnfs, tmpdir, outdir, speciestree, datingset, slist, nthreads, beastlgjar, beagle, fossil, chainset, rootheight)
    if dating=='mcmctree':
        if speciestree is None:
            logging.error("Please provide species tree for mcmctree dating")
            exit(0)
        Run_MCMCTREE(Concat_caln, Concat_paln, Concat_calnf, Concat_palnf, cds_alns_rn, pro_alns_rn, calnfs, palnfs, tmpdir, outdir, speciestree, datingset, aamodel, partition, slist, nthreads)
    if dating=='r8s':
        if datingset is None:
            logging.error("Please provide necessary fixage or constrain information of internal node for r8s dating")
            exit(0)
        if speciestree is None:
            logging.info("Using concatenation-inferred species tree as input for r8s")
            spt = Reroot(ctree_pth,outgroup)
            Run_r8s(spt, ctree_length, outdir, datingset)
        else:
            Run_r8s(speciestree, nsites, outdir, datingset)
    if not annotation == 'none':
        logging.info("Doing functional annotation on orthologous families")
        if annotation == 'eggnog':
            if eggnogdata is None:
                logging.error("Please provide the path to eggNOG-mapper databases")
                exit(0)
            eggnog(cds_fastaf,eggnogdata,outdir,pfam,dmnb,evalue,nthreads)
        if annotation == 'hmmpfam': hmmer_pfam(cds_fastaf,hmm,outdir,evalue,nthreads)
        if annotation == 'interproscan': interproscan(cds_fastaf,exepath,outdir,nthreads)
    if ks:
        s = mergeMultiRBH_seqs(seqs)
        fams = read_gene_families(families)
        fams = get_gene_families(s, fams, pairwise=pairwise, strip_gaps=False, tree_method=tree_method)
        ksdb = KsDistributionBuilder(fams, s, n_threads=nthreads)
        ksdb.get_distribution()
        prefix = os.path.basename(families)
        outfile = os.path.join(outdir, "{}.ks.tsv".format(prefix))
        logging.info("Ks result saved to {}".format(outfile))
        ksdb.df.fillna("NaN").to_csv(outfile,sep="\t")
    if tmpdir is None: [x.remove_tmp(prompt=False) for x in seqs]
    logging.info("Done")

# Get peak and confidence interval of Ks distribution
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('ks_distribution', type=click.Path(exists=True))
@click.option('--anchor', '-a', default=None, show_default=True,
    help='anchor pair infomation if available')
@click.option('--outdir', '-o', default='wgd_peak', show_default=True,
    help='output directory')
@click.option('--alignfilter', '-f', nargs=3, type=float, default= (0.,0,0.), show_default=True,
    help='filter alignment identity, length and coverage')
@click.option('--ksrange', '-r', nargs=2, type=float, default=(0, 5), show_default=True,
    help='range of Ks to be analyzed')
@click.option('--bin_width', '-bw',type=float, default=0.1, show_default=True,
    help='bandwidth of distribution')
@click.option('--weights_outliers_included','-ic', is_flag=True,
    help="include Ks outliers")
@click.option('--method', '-m', type=click.Choice(['gmm', 'bgmm']), default='gmm', show_default=True, help="mixture modeling method")
@click.option('--seed',type=int, default=2352890, show_default=True, help="random seed given to initialize parameters")
@click.option('--em_iter', '-ei',type=int, default=100, show_default=True, help="number of EM iterations to perform")
@click.option('--n_init', '-ni',type=int, default=1, show_default=True, help="number of initializations to perform")
@click.option('--components', '-c', nargs=2, default=(1, 4), show_default=True, help="range of number of components to fit")
@click.option('--boots', type=int, default=200, show_default=True, help="number of bootstrap replicates of kde")
@click.option('--weighted', is_flag=True,help="node-weighted instead of node-averaged method")
@click.option('--plot', '-p', type=click.Choice(['stacked', 'identical']), default='stacked', show_default=True, help="plotting method")
@click.option('--bw_method', '-bm', type=click.Choice(['silverman', 'ISJ']), default='silverman', show_default=True, help="bandwidth method")
@click.option('--multiplicon', '-mp', default=None, show_default=True,help='multiplicon infomation if available')
def peak(**kwargs):
    """
    Infer peak and CI of Ks distribution.
    """
    _peak(**kwargs)

def _peak(ks_distribution, anchor, outdir, alignfilter, ksrange, bin_width, weights_outliers_included, method, seed, em_iter, n_init, components, boots, weighted, plot, bw_method,multiplicon):
    from wgd.peak import alnfilter, group_dS, log_trans, fit_gmm, fit_bgmm, add_prediction, bootstrap_kde, default_plot, get_kde, draw_kde_CI, draw_components_kde_bootstrap
    from wgd.core import _mkdir
    outpath = _mkdir(outdir)
    ksdf = pd.read_csv(ks_distribution,header=0,index_col=0,sep='\t')
    if len(ksdf.columns) <4:
        logging.info("Begin to analyze peak of WGD dates")
        draw_kde_CI(outdir,ksdf,boots,bw_method,date_lower = 0,date_upper=4)
        exit()
    ksdf_filtered = alnfilter(ksdf,weights_outliers_included,alignfilter[0],alignfilter[1],alignfilter[2],ksrange[0],ksrange[1])
    fn_ksdf, weight_col = group_dS(ksdf_filtered)
    train_in = log_trans(fn_ksdf)
    get_kde(outdir,train_in,ksdf_filtered,weighted,ksrange[0],ksrange[1])
    if method == 'gmm':
        out_file = os.path.join(outdir, "AIC_BIC.pdf")
        models, aic, bic, besta, bestb, N = fit_gmm(out_file, train_in, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    if method == 'bgmm':
        models, N = fit_bgmm(train_in, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    for n, m in zip(N,models):
        fname = os.path.join(outpath, "Ks_{0}_{1}components_prediction.tsv".format(method,n))
        ksdf_predict = add_prediction(ksdf,fn_ksdf,train_in,m)
        ksdf_predict.to_csv(fname,header=True,index=True,sep='\t')
        logging.info("Plotting components-annotated Ks distribution for {} components model".format(n))
        fig = default_plot(ksdf_predict, title=fname, bins=50, ylabel="Duplication events", nums = int(n),plot = plot)
        fig.savefig(fname + "_Ks.svg")
        fig.savefig(fname + "_Ks.pdf")
        plt.close()
        ksdf_predict_filter = alnfilter(ksdf_predict,weights_outliers_included,alignfilter[0],alignfilter[1],alignfilter[2],ksrange[0],ksrange[1])
        draw_components_kde_bootstrap(outdir,int(n),ksdf_predict_filter,weighted,boots,bin_width)
    mean_modes, std_modes, mean_medians, std_medians = bootstrap_kde(outdir, train_in, ksrange[0], ksrange[1], boots, bin_width, ksdf_filtered, weight_col, weighted = weighted)
    logging.info("Done")

# Ks distribution construction
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('families', type=click.Path(exists=True))
@click.argument('sequences', nargs=-1, type=click.Path(exists=True))
@click.option('--outdir', '-o', default="wgd_ksd", show_default=True,
    help='output directory')
@click.option('--tmpdir', '-t', default=None, show_default=True,
    help='tmp directory')
@click.option('--nthreads', '-n', default=4, show_default=True,
    help="number of threads to use")
@click.option('--to_stop', is_flag=True, 
    help="don't translate through STOP codons")
@click.option('--cds', is_flag=True,
    help="enforce proper CDS sequences")
@click.option('--pairwise', is_flag=True,
    help="run codeml on all gene pairs separately")
@click.option('--strip_gaps', is_flag=True,
    help="remove all gap-containing columns in the alignment")
@click.option('--tree_method', '-tree', 
    type=click.Choice(['cluster', 'fasttree', 'iqtree']), 
    default='cluster', show_default=True,
    help="Tree inference method for node weighting")
def ksd(**kwargs):
    """
    Paranome and one-to-one ortholog Ks distribution inference pipeline.

    Example 1 - whole-paranome:

        wgd ksd families.mcl cds.fasta

    Example 2 - one-to-one orthologs (RBH):

        wgd ksd orthologs.rbh cds1.fasta cds2.fasta

    If you want to keep intermediate (temporary) files, please provide a directory
    name for the `--tmpdir` option.
    """
    _ksd(**kwargs)

def _ksd(families, sequences, outdir, tmpdir, nthreads, to_stop, cds, pairwise,
        strip_gaps, tree_method):
    from wgd.core import get_gene_families, SequenceData, KsDistributionBuilder
    from wgd.core import read_gene_families, merge_seqs
    from wgd.viz import default_plot, apply_filters
    if len(sequences) == 0: 
        logging.error("Please provide at least one sequence file")
        exit(0)
    if len(sequences) == 2:
        tree_method = "cluster"  # for RBH others don't make sense (and crash)
    seqs = [SequenceData(s, tmp_path=tmpdir, out_path=outdir,
            to_stop=to_stop, cds=cds) for s in sequences]
    s = merge_seqs(seqs)
    logging.info("tmpdir = {}".format(s.tmp_path))
    fams = read_gene_families(families)
    fams = get_gene_families(s, fams, 
            pairwise=pairwise, 
            strip_gaps=strip_gaps,
            tree_method=tree_method)
    ksdb = KsDistributionBuilder(fams, s, n_threads=nthreads)
    ksdb.get_distribution()
    prefix = os.path.basename(families)
    outfile = os.path.join(outdir, "{}.ks.tsv".format(prefix))
    logging.info("Saving to {}".format(outfile))
    ksdb.df.fillna("NaN").to_csv(outfile,sep="\t")
    logging.info("Making plots")
    df = apply_filters(ksdb.df, [("dS", 0., 5.)])
    ylabel = "Duplications"
    if len(sequences) == 2:
        ylabel = "RBH orthologs"
    fig = default_plot(df, title=prefix, bins=50, ylabel=ylabel)
    fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)))
    fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)))
    if tmpdir is None:
        [x.remove_tmp(prompt=False) for x in seqs]
    logging.info("Done")
    

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('families', type=click.Path(exists=True))
@click.argument('gff_files', nargs=-1, type=click.Path(exists=True))
@click.option('--ks_distribution', '-ks', default=None,
    help="ks distribution tsv file (optional, see `wgd ksd`)")
@click.option('--outdir', '-o', default='./wgd_syn', show_default=True, 
    help='output directory')
@click.option('--feature', '-f', default='gene', show_default=True,
    help="keyword for parsing the genes from the GFF file (column 3)")
@click.option('--attribute', '-a', default='ID', show_default=True,
    help="keyword for parsing the gene IDs from the GFF file (column 9)")
@click.option('--minlen', '-ml', default=250, show_default=True,
    help="minimum length of a genomic element to be included in dotplot.")
@click.option('--maxsize', '-ms', default=25, show_default=True,
    help="maximum family size to include in analysis.")
@click.option('--ks_range', '-r', nargs=2, default=(0.05, 5), show_default=True,
    type=float, help='Ks range to use for colored dotplot')
@click.option('--iadhore_options', default="",
    help="other options for I-ADHoRe, as a comma separated string, "
         "e.g. gap_size=30,q_value=0.75,prob_cutoff=0.05")
def syn(**kwargs):
    """
    Co-linearity and anchor inference using I-ADHoRe.
    """
    _syn(**kwargs)

def _syn(families, gff_files, ks_distribution, outdir, feature, attribute,
        minlen, maxsize, ks_range, iadhore_options):
    """
    Co-linearity and anchor inference using I-ADHoRe.
    """
    from wgd.syn import make_gene_table, configure_adhore, run_adhore
    from wgd.syn import get_anchors, get_anchor_ksd, get_segments_profile
    from wgd.viz import default_plot, apply_filters, syntenic_depth_plot, all_dotplots, Ks_dotplots, syntenic_dotplot_ks_colored
    # non-default options for I-ADHoRe
    iadhore_opts = {x.split("=")[0].strip(): x.split("=")[1].strip()
               for x in iadhore_options.split(",") if x != ""}
    if len(iadhore_opts) > 0:
        logging.info("I-ADHoRe 3.0 options: {}".format(iadhore_opts))
    # read families and make table
    prefix = os.path.basename(families)
    fams = pd.read_csv(families, index_col=0, sep="\t")
    table = make_gene_table(gff_files, fams, feature, attribute)
    if len(table.dropna().index) == 0:
        logging.error("No genes from families file `{}` found in the GFF file "
                "for `feature={}` and `attribute={}`, please double check command " 
                "settings.".format(families, feature, attribute))
        exit(1)
    if len(table.dropna()) < 1000:
        logging.warning("Few genes from families `{}` found in the GFF file, better "
                "Double check your command.".format(families))

    # I-ADHoRe
    logging.info("Configuring I-ADHoRe co-linearity search")
    conf, out_path = configure_adhore(table, outdir, **iadhore_opts)
    table.to_csv(os.path.join(outdir, "gene-table.csv"))
    logging.info("Running I-ADHoRe")
    run_adhore(conf)

    # general post-processing
    logging.info("Processing I-ADHoRe output")
    anchors = get_anchors(out_path)
    if anchors is None:
        logging.warning("No anchors found, terminating! Please inspect your input files "
                "and the I-ADHoRe results in `{}`".format(out_path))
        exit(1)

    anchors.to_csv(os.path.join(outdir, "anchors.csv"))
    segprofile = get_segments_profile(out_path)
    segprofile.to_csv(os.path.join(outdir, "segprofile.csv"))
    fig = syntenic_depth_plot(segprofile)
    fig.savefig(os.path.join(outdir, "{}.syndepth.svg".format(prefix)))
    fig.savefig(os.path.join(outdir, "{}.syndepth.pdf".format(prefix)))

    # dotplot
    logging.info("Generating dot plots")
    figs = all_dotplots(table, anchors, maxsize=maxsize, minlen=minlen) 
    for k, v in figs.items():
        v.savefig(os.path.join(outdir, "{}.dot.svg".format(k)))
        v.savefig(os.path.join(outdir, "{}.dot.pdf".format(k)))
        v.savefig(os.path.join(outdir, "{}.dot.png".format(k)))

    # anchor Ks distributions
    if ks_distribution:
        ylabel = "Duplications"
        if len(gff_files) == 2:
            ylabel = "RBH orthologs"
        ksd = pd.read_csv(ks_distribution, sep="\t", index_col=0)
        anchor_ks = get_anchor_ksd(ksd, anchors)
        anchor_ks.to_csv(os.path.join(outdir, "{}.anchors.ks.csv".format(prefix)))
        a = apply_filters(ksd,       [("dS", 1e-4, 5.), ("S", 10, 1e6)])
        b = apply_filters(anchor_ks, [("dS", 1e-4, 5.), ("S", 10, 1e6)])
        logging.info("Generating anchor Ks distribution")
        fig = default_plot(a, b, title=prefix, rwidth=0.8, bins=50, ylabel=ylabel)
        fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)))
        fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)))
        # Ks colored dotplot
        logging.info("Generating Ks colored (median Ks) dotplot")
        multiplicons = pd.read_csv(os.path.join(
            outdir, 'iadhore-out', 'multiplicons.txt'), sep='\t')
        anchor_points = pd.read_csv(os.path.join(
            outdir, 'iadhore-out', 'anchorpoints.txt'), sep='\t')
        dotplot_out = os.path.join(outdir, '{}.dotplot.ks.svg'.format(
                os.path.basename(families)))
        syntenic_dotplot_ks_colored(
                multiplicons, anchor_points, anchor_ks, min_ks=ks_range[0],
                max_ks=ks_range[1], output_file=dotplot_out,
                min_length=minlen
        )
        figs2=Ks_dotplots(multiplicons, table, anchor_ks, anchor_points, anchors,min_ks=ks_range[0],max_ks=ks_range[1], maxsize=maxsize, minlen=minlen)
        for k, v in figs2.items():
            v.savefig(os.path.join(outdir, "{}.ks.dot.svg".format(k)))
            v.savefig(os.path.join(outdir, "{}.ks.dot.pdf".format(k)))
            v.savefig(os.path.join(outdir, "{}.ks.dot.png".format(k)))
    logging.info("Done")    

# MIXTURE MODELING
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument(
        'ks_distribution', type=click.Path(exists=True), default=None
)
@click.option(
        '--filters', '-f', type=int, default=300,
        help="Alignment length",
        show_default=True
)
@click.option(
        '--ks_range', '-r', nargs=2, default=(0.005, 3), show_default=True,
        type=float,
        help='Ks range to use for modeling'
)
@click.option(
        '--bins', '-b', default=50, show_default=True, type=int,
        help="Number of histogram bins."
)
@click.option(
        '--output_dir', '-o', default="wgd_mix", show_default=True,
        help='output directory'
)
@click.option(
        '--method', type=click.Choice(['gmm', 'bgmm']), default='gmm',
        show_default=True, help="mixture modeling method"
)
@click.option(
        '--components', '-n', nargs=2, default=(1, 4), show_default=True,
        help='range of number of components to fit'
)
@click.option(
        '--gamma', '-g', default=1e-3, show_default=True,
        help='gamma parameter for bgmm models'
)
@click.option(
        '--n_init', '-ni', default=1, show_default=True,
        help='number of k-means initializations'
)
@click.option(
        '--max_iter', '-mi', default=1000, show_default=True,
        help='maximum number of iterations'
)
def mix(
        ks_distribution, filters, ks_range, bins, output_dir, method,
        components, gamma, n_init, max_iter
):
    """
    Mixture modeling of Ks distributions.
    Basic function
    """
    mix_(
            ks_distribution, filters, ks_range, method, components, bins,
            output_dir, gamma, n_init, max_iter
    )
def mix_(
        ks_distribution, filters, ks_range, method, components, bins,
        output_dir, gamma, n_init, max_iter
):
    """
    Mixture modeling tools.

    Note that histogram weighting is done after applying specified filters. Also
    note that mixture models are fitted to node-averaged (not weighted)
    histograms. Please interpret mixture model results with caution, for more
    info, refer to :ref:`note_on_gmms`.
    :param ks_distribution: Ks distribution data frame
    :param filters: alignment stats filters (here only alignment length)
    :param ks_range: Ks range used for models
    :param method: mixture modeling method, Bayesian/ordinary Gaussian mixtures
    :param components: number of components to use (tuple: (min, max))
    :param bins: number histogram bins for visualization
    :param output_dir: output directory
    :param gamma: gamma parameter for BGMM
    :param n_init: number of k-means initializations (best is kept)
    :param max_iter: number of iterations
    :return: nada
    """
    from wgd.mix import filter_group_data, get_array_for_mixture, fit_gmm
    from wgd.mix import inspect_aic, inspect_bic, plot_aic_bic
    from wgd.mix import plot_all_models_gmm, get_component_probabilities
    from wgd.mix import fit_bgmm,plot_all_models_bgmm 

    # make output dir if needed
    if not os.path.exists(output_dir):
        logging.info("Making directory {}".format(output_dir))
        os.mkdir(output_dir)
    # prepare data frame
    logging.info("Preparing data frame")
    df = pd.read_csv(ks_distribution, index_col=0, sep='\t')
    df = filter_group_data(df, filters,
                           ks_range[0], ks_range[1])
    X = get_array_for_mixture(df)

    logging.info(" .. max_iter = {}".format(max_iter))
    logging.info(" .. n_init   = {}".format(n_init))

    # GMM method
    if method == "gmm":
        logging.info("Method is GMM, interpret best model with caution!")
        models, bic, aic, best = fit_gmm(
                X, components[0], components[1], max_iter=max_iter,
                n_init=n_init
        )
        inspect_aic(aic)
        inspect_bic(bic)
        logging.info("Plotting AIC & BIC")
        plot_aic_bic(aic, bic, components[0], components[1],
                     os.path.join(output_dir, "aic_bic.svg"))
        logging.info("Plotting mixtures")
        plot_all_models_gmm(models, X, ks_range[0], ks_range[1], bins=bins,
                            out_file=os.path.join(output_dir, "gmms.svg"))

    # BGMM method
    else:
        logging.info("Method is BGMM, weights are informative for best model")
        logging.info(" .. gamma    = {}".format(gamma))
        models = fit_bgmm(
                X, components[0], components[1], gamma=gamma,
                max_iter=max_iter, n_init=n_init
        )
        logging.info("Plotting mixtures")
        plot_all_models_bgmm(models, X, ks_range[0], ks_range[1], bins=bins,
                             out_file=os.path.join(output_dir, "bgmms.svg"))
        logging.warning("Method is BGMM, unable to choose best model!")
        logging.info("Taking model with most components for the component-wise"
                     "probability output file.")
        logging.info("To get the output file for a particular number of "
                     "components, run wgd mix again ")
        logging.info("with the desired component number as maximum.")
        best = models[-1]

    # save component probabilities
    logging.info("Writing component-wise probabilities to file")
    new_df = get_component_probabilities(df, best)
    new_df.round(5).to_csv(os.path.join(
            output_dir, "ks_{}.tsv".format(method)), sep="\t")


if __name__ == '__main__':
    cli()
