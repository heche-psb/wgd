#!/usr/bin/python3
import click
import logging
import sys
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sp
from timeit import default_timer as timer
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
    wgd - Copyright (C) 2023-2024 Hengchi Chen\n
    Contact: heche@psb.vib-ugent.be
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
    help='anchor points datafile')
@click.option('--segments', '-sm', default=None, show_default=True,
    help='segments datafile')
@click.option('--listelements', '-le', default=None, show_default=True,
    help='list elements datafile')
@click.option('--keepfasta','-kf', is_flag=True,
    help="keep the fasta file of homologs family")
@click.option('--keepduplicates','-kd', is_flag=True,
    help="Keep ID duplicates in MRBHs")
@click.option('--globalmrbh','-gm', is_flag=True,
    help="global MRBH regardless of focus species")
@click.option('--nthreads', '-n', default=4, show_default=True,help="number of threads to use")
@click.option('--orthoinfer','-oi', is_flag=True,help="orthogroups inference")
@click.option('--onlyortho','-oo', is_flag=True,help="only run orthogroups inference")
@click.option('--getnsog','-gn', is_flag=True,help="get nested single-copy gene families")
@click.option('--tree_method', '-tree',type=click.Choice(['fasttree', 'iqtree', 'mrbayes']),default='fasttree',show_default=True,help="tree inference method")
@click.option('--treeset', '-ts', multiple=True, default=None, show_default=True,help='parameters setting for gene tree inference')
@click.option('--msogcut', '-mc', type=float, default=0.8, show_default=True,help='ratio cutoff for mostly single-copy family and species representation in collinear coalescence inference')
@click.option('--geneassign','-ga', is_flag=True,help="assign genes to given gene families")
@click.option('--assign_method', '-am',type=click.Choice(['hmmer', 'diamond']),default='hmmer',show_default=True,help="gene assignment method")
@click.option('--seq2assign', '-sa', multiple=True, default= None, show_default=True, help='sequences to be assigned')
@click.option('--fam2assign', '-fa',default= None, show_default=True, help='families to be assigned upon')
@click.option('--concat','-cc', is_flag=True,help="concatenation pipeline for orthoinfer")
@click.option('--collinearcoalescence','-coc', is_flag=True,help="collinear coalescence inference of phylogeny and WGD")
@click.option('--testsog','-te', is_flag=True,help="Unbiased test of single-copy gene families")
@click.option('--bins', '-bs', type=int, default=100, show_default=True, help='bins for gene length normalization')
@click.option('--normalizedpercent', '-np', type=int, default=5, show_default=True, help='percentage of upper hits used for normalization')
@click.option('--nonormalization','-nn', is_flag=True,help="call off the normalization process")
@click.option('--buscosog','-bsog', is_flag=True,help="get busco-guided single-copy gene family")
@click.option('--buscohmm', '-bhmm',default= None, show_default=True, help='HMM profile of given busco dataset')
@click.option('--buscocutoff', '-bctf', default= None, show_default=True, help='HMM score cutoffs of BUSCO')
@click.option('--genetable', '-gt', default= None, show_default=True, help='gene table file')
def dmd(**kwargs):
    """
    All-vs-all diamond blastp + MCL clustering.

    Requires diamond and mcl. Note the two key parameters, being the e-value
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

def _dmd(sequences, outdir, tmpdir, cscore, inflation, eval, to_stop, cds, focus, anchorpoints, keepfasta, keepduplicates, globalmrbh, nthreads, orthoinfer, onlyortho, getnsog, tree_method, treeset, msogcut, geneassign, assign_method, seq2assign, fam2assign, concat, segments, listelements, collinearcoalescence, testsog, bins, buscosog, buscohmm, buscocutoff, genetable, normalizedpercent, nonormalization):
    from wgd.core import SequenceData, read_MultiRBH_gene_families,mrbh,ortho_infer,genes2fams,endt,segmentsaps,bsog
    start = timer()
    if tmpdir != None and not os.path.isdir(tmpdir): os.mkdir(tmpdir)
    s = [SequenceData(s, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore, threads=nthreads, bins=bins, normalizedpercent=normalizedpercent, nonormalization=nonormalization) for s in sequences]
    for i in s: logging.info("tmpdir = {} for {}".format(i.tmp_path,i.prefix))
    if buscosog:
        logging.info("Constructing busco-guided families")
        bsog(s,buscohmm,outdir,eval,nthreads,buscocutoff)
        endt(tmpdir,start,s)
    if collinearcoalescence:
        logging.info("Analyzing collinear coalescence")
        segmentsaps(genetable,listelements,anchorpoints,segments,outdir,s,nthreads,tree_method,treeset,msogcut)
        endt(tmpdir,start,s)
    if geneassign:
        genes2fams(assign_method,seq2assign,fam2assign,outdir,s,nthreads,tmpdir,to_stop,cds,cscore,eval,start,normalizedpercent)
    if orthoinfer:
        logging.info("Infering orthologous gene families")
        ortho_infer(sequences,s,outdir,tmpdir,to_stop,cds,cscore,inflation,eval,nthreads,getnsog,tree_method,treeset,msogcut,concat,testsog,normalizedpercent,bins=bins,nonormalization=nonormalization)
        if onlyortho: endt(tmpdir,start,s)
    if len(s) == 0:
        logging.error("No sequences provided!")
        return
    if len(s) == 1:
        logging.info("One CDS file: will compute paranome")
        s[0].get_paranome(inflation=inflation, eval=eval)
        s[0].write_paranome(False)
    elif focus is None and not globalmrbh:
        logging.info("Multiple CDS files: will compute RBH orthologs")
        for i in range(len(s)-1):
            for j in range(i+1, len(s)):
                logging.info("{} vs. {}".format(s[i].prefix, s[j].prefix))
                s[i].get_rbh_orthologs(s[j], cscore, False, eval=eval)
                s[i].write_rbh_orthologs(s[j],singletons=False)
    mrbh(globalmrbh,outdir,s,cscore,eval,keepduplicates,anchorpoints,focus,keepfasta,nthreads)
    endt(tmpdir,start,s)

#MSA and ML tree inference for given sets of orthologous gene familes for species tree inference and WGD timing

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('families', type=click.Path(exists=True))
@click.argument('sequences', nargs=-1, type=click.Path(exists=True))
@click.option('--outdir', '-o', default="wgd_focus", show_default=True,help='output directory')
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
@click.option('--evalue', default=1e-10, show_default=True,help='E-value threshold for annotation')
@click.option('--exepath', default=None, show_default=True,help='Path to interproscan installation folder')
@click.option('--fossil', '-f', nargs=5, default= ('clade1;clade2', 'taxa1,taxa2;taxa3,taxa4', '4;5', '0.5;0.6', '400;500'), show_default=True, help='fossil calibration info (id,taxa,mean,std,offset)')
@click.option('--rootheight', '-rh', nargs=3,default= (4,0.5,400), show_default=True, help='root height calibration info (mean,std,offset)')
@click.option('--chainset', '-cs', nargs=2,default= (10000,100), show_default=True, help='MCMC chain info (length,frequency) for beast')
@click.option('--beastlgjar', default=None, show_default=True,help='path of beastLG.jar')
@click.option('--beagle', is_flag=True,help='using beagle')
@click.option('--protdating', is_flag=True,help='only run protein concatenation dating')
def focus(**kwargs):
    """
    Multiply species RBH or c-score defined orthologous family's gene tree inference, species tree inference and absolute dating pipeline.

    Example 1 - Dating orthologous families containing anchor pairs with a required user-defined species tree:

        wgd focus ap_families cds1.fasta cds2.fasta cds3.fasta --dating mcmctree --speciestree sp.newick -ds 'burnin = 2000' -ds 'sigma2_gamma = 1 10 1'
        wgd focus ap_families cds1.fasta cds2.fasta cds3.fasta --dating beast --speciestree sp.newick --fossil clade1 taxa1,taxa2 4 0.5 400 --rootheight 4 0.5 400 --chainset 10000 100 --beastlgjar path (--beagle)

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

def _focus(families, sequences, outdir, tmpdir, nthreads, to_stop, cds, strip_gaps, aligner, tree_method, treeset, concatenation, coalescence, speciestree, dating, datingset, nsites, outgroup, partition, aamodel, ks, annotation, pairwise, eggnogdata, pfam, dmnb, hmm, evalue, exepath, fossil, rootheight, chainset, beastlgjar, beagle, protdating):
    from wgd.core import SequenceData, read_gene_families, get_gene_families, KsDistributionBuilder
    from wgd.core import mergeMultiRBH_seqs, read_MultiRBH_gene_families, get_MultipRBH_gene_families, Concat, _Codon2partition_, Coale, Run_MCMCTREE, Run_r8s, Reroot, eggnog, hmmer_pfam, interproscan, Run_BEAST, get_only_protaln, Run_MCMCTREE_concprot, Concat_prot
    start = timer()
    if tmpdir != None and not os.path.isdir(tmpdir): os.mkdir(tmpdir)
    if dating=='r8s' and not speciestree is None and nsites is None:
        logging.error("Please provide nsites parameter for r8s dating")
        exit(0)
    if dating=='r8s' and speciestree is None and outgroup is None:
        logging.error("Please provide outgroup species for r8s dating")
        exit(0)
    if len(sequences) < 2:
        logging.error("Please provide at least three sequence files for constructing trees")
        exit(0)
    seqs = [SequenceData(s, tmp_path=tmpdir, out_path=outdir,to_stop=to_stop, cds=cds, threads=nthreads) for s in sequences]
    for s in seqs: logging.info("tmpdir = {} for {}".format(s.tmp_path,s.prefix))
    fams = read_MultiRBH_gene_families(families)
    if protdating:
        logging.info("Only implement protein concatenation dating via mcmctree")
        pro_alns = get_only_protaln(seqs,fams,outdir,nthreads,option="--auto")
        Concat_palnf,Concat_paln,slist = Concat_prot(pro_alns,families,outdir)
        Run_MCMCTREE_concprot(Concat_paln,Concat_palnf,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads)
        end = timer()
        logging.info("Total run time: {}s".format(int(end-start)))
        logging.info("Done")
        if tmpdir is None: [x.remove_tmp(prompt=False) for x in seqs]
        exit()
    if coalescence: cds_alns, pro_alns, tree_famsf, calnfs, palnfs, calnfs_length, cds_fastaf, tree_fams = get_MultipRBH_gene_families(seqs,fams,tree_method,treeset,outdir,nthreads,option="--auto",runtree=True)
    else: cds_alns, pro_alns, tree_famsf, calnfs, palnfs, calnfs_length, cds_fastaf, tree_fams = get_MultipRBH_gene_families(seqs,fams,tree_method,treeset,outdir,nthreads,option="--auto",runtree=False)
    if concatenation or dating != 'none':
        if concatenation:
            cds_alns_rn, pro_alns_rn, Concat_ctree, Concat_ptree, Concat_calnf, Concat_palnf, ctree_pth, ctree_length, gsmap, Concat_caln, Concat_paln, slist = Concat(cds_alns, pro_alns, families, tree_method, treeset, outdir, infer_tree=True)
        else:
            cds_alns_rn, pro_alns_rn, Concat_calnf, Concat_palnf, ctree_length, gsmap, Concat_caln, Concat_paln, slist = Concat(cds_alns, pro_alns, families, tree_method, treeset, outdir, infer_tree=False)
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
    end = timer()
    logging.info("Total run time: {}s".format(int(end-start)))
    logging.info("Done")

# Get peak and confidence interval of Ks distribution
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('ks_distribution', type=click.Path(exists=True))
@click.option('--anchorpoints', '-ap', default=None, show_default=True, help='anchor pair infomation')
@click.option('--segments', '-sm', default=None, show_default=True, help='segments information')
@click.option('--listelements', '-le', default=None, show_default=True, help='listelements information')
@click.option('--multipliconpairs', '-mp', default=None, show_default=True, help='multipliconpairs information')
@click.option('--outdir', '-o', default='wgd_peak', show_default=True,
    help='output directory')
@click.option('--alignfilter', '-af', nargs=3, type=float, default= (0.,0,0.), show_default=True,
    help='filter alignment identity, length and coverage')
@click.option('--ksrange', '-r', nargs=2, type=float, default=(0, 5), show_default=True,
    help='range of Ks to be analyzed')
@click.option('--bin_width', '-bw',type=float, default=0.1, show_default=True,
    help='bandwidth of distribution')
@click.option('--weights_outliers_included','-ic', is_flag=True,
    help="include Ks outliers")
@click.option('--method', '-m', type=click.Choice(['gmm', 'bgmm']), default='gmm', show_default=True, help="mixture modeling method")
@click.option('--seed',type=int, default=2352890, show_default=True, help="random seed given to initialize parameters")
@click.option('--em_iter', '-ei',type=int, default=200, show_default=True, help="number of EM iterations to perform")
@click.option('--n_init', '-ni',type=int, default=200, show_default=True, help="number of initializations to perform")
@click.option('--components', '-n', nargs=2, default=(1, 4), show_default=True, help="range of number of components to fit")
@click.option('--boots', type=int, default=200, show_default=True, help="number of bootstrap replicates of kde")
@click.option('--weighted', is_flag=True,help="node-weighted instead of node-averaged method")
@click.option('--plot', '-p', type=click.Choice(['stacked', 'identical']), default='identical', show_default=True, help="plotting method")
@click.option('--bw_method', '-bm', type=click.Choice(['silverman', 'ISJ']), default='silverman', show_default=True, help="bandwidth method")
@click.option('--n_medoids', type=int, default=2, show_default=True, help="number of medoids to generate")
@click.option('--kdemethod', '-km', type=click.Choice(['scipy', 'naivekde', 'treekde', 'fftkde']), default='scipy', show_default=True, help="kde method")
@click.option('--n_clusters',type=int, default=5, show_default=True, help="number of clusters to plot Elbow loss function")
@click.option('--kmedoids', is_flag=True,help="K-Medoids clustering method")
@click.option('--guide', '-gd', type=click.Choice(['multiplicon', 'basecluster', 'segment']), default='segment', show_default=True, help="regime residing anchors")
@click.option('--prominence_cutoff', '-prct', type=float, default=0.1, show_default=True, help='prominence cutoff of acceptable peaks')
@click.option('--kstodate', '-kd', nargs=2, type=float, default=(0.5, 1.5), show_default=True, help='range of Ks to be dated')
@click.option('--family', '-f', default=None, show_default=True, help='family to filter Ks upon')
@click.option('--manualset', is_flag=True,help="Manually set Ks range of anchor pairs or multiplicons as CI")
@click.option('--rel_height', '-rh', type=float, default=0.4, show_default=True, help='relative height at which the peak width is measured')
@click.option('--ci', default=95, show_default=True,type=int, help='confidence level of log-normal distribution to date')
@click.option('--hdr', default=95, show_default=True,type=int, help='highest densidy region (HDR) in a given distribution to date')
@click.option('--heuristic', is_flag=True,help="heuristic CI for dating")
@click.option('--kscutoff', '-kc', default=5, show_default=True, type=float, help='Ks Saturation cutoff for genes in Dating')
def peak(**kwargs):
    """
    Infer peak and CI of Ks distribution.
    """
    _peak(**kwargs)

def _peak(ks_distribution, anchorpoints, outdir, alignfilter, ksrange, bin_width, weights_outliers_included, method, seed, em_iter, n_init, components, boots, weighted, plot, bw_method, n_medoids, kdemethod, n_clusters, kmedoids, guide, prominence_cutoff, kstodate, family, rel_height, ci,manualset,segments,hdr,heuristic,listelements,multipliconpairs,kscutoff):
    from wgd.peak import alnfilter, group_dS, log_trans, fit_gmm, fit_bgmm, add_prediction, bootstrap_kde, default_plot, get_kde, draw_kde_CI, draw_components_kde_bootstrap, fit_kmedoids, default_plot_kde, fit_apgmm_guide, fit_apgmm_ap, find_apeak, find_mpeak, retreive95CI, formatv2
    from wgd.core import _mkdir
    outpath = _mkdir(outdir)
    ksdf = pd.read_csv(ks_distribution,header=0,index_col=0,sep='\t')
    ksdf = formatv2(ksdf)
    if len(ksdf.columns) <4:
        logging.info("Begin to analyze peak of WGD dates")
        draw_kde_CI(kdemethod, outdir,ksdf,boots,bw_method,date_lower = 0,date_upper=4)
        exit()
    ksdf_filtered = alnfilter(ksdf,weights_outliers_included,alignfilter[0],alignfilter[1],alignfilter[2],ksrange[0],ksrange[1])
    if family != None:
        retreive95CI(family,ksdf_filtered,outdir,kstodate[0],kstodate[1])
        logging.info('Done')
        exit()
    fn_ksdf, weight_col = group_dS(ksdf_filtered)
    train_in = log_trans(fn_ksdf)
    if anchorpoints!= None:
        if kmedoids:
            df_ap = fit_kmedoids(guide, anchorpoints, boots, kdemethod, bin_width, weighted, ksdf, ksdf_filtered, outdir, seed, n_medoids, em_iter=em_iter, plot=plot, n_kmedoids = n_clusters, segment = segments, multipliconpairs=multipliconpairs,listelement=listelements)
        else:
            df_ap_mp = fit_apgmm_guide(hdr,guide,anchorpoints,ksdf,ksdf_filtered,seed,components,em_iter,n_init,outdir,method,weighted,plot,segment=segments,multipliconpairs=multipliconpairs,listelement=listelements,cutoff = kscutoff)
            df_ap = fit_apgmm_ap(hdr,anchorpoints,ksdf_filtered,seed,components,em_iter,n_init,outdir,method,weighted,plot)
        if heuristic:
            find_apeak(df_ap,anchorpoints,os.path.basename(ks_distribution),outdir,peak_threshold=prominence_cutoff,na=False,rel_height=rel_height,ci=ci,user_low=kstodate[0],user_upp=kstodate[1],user=manualset)
            find_apeak(df_ap,anchorpoints,os.path.basename(ks_distribution),outdir,peak_threshold=prominence_cutoff,na=True,rel_height=rel_height,ci=ci,user_low=kstodate[0],user_upp=kstodate[1],user=manualset)
            find_mpeak(df_ap_mp,anchorpoints,os.path.basename(ks_distribution),outdir,guide,peak_threshold=prominence_cutoff,rel_height=rel_height,ci=ci,user_low=kstodate[0],user_upp=kstodate[1],user=manualset)
        logging.info('Done')
        exit()
    get_kde(kdemethod,outdir,fn_ksdf,ksdf_filtered,weighted,ksrange[0],ksrange[1])
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
        #fig = default_plot(ksdf_predict, title=os.path.basename(fname), bins=50, ylabel="Duplication events", nums = int(n),plot = plot)
        #fig.savefig(fname + "_Ks.svg")
        #fig.savefig(fname + "_Ks.pdf")
        #plt.close()
        fig,ylim,yticks = default_plot_kde(ksdf_predict, title=os.path.basename(fname), bins=50, ylabel="Duplication events", nums = int(n),plot = 'identical')
        fig.savefig(fname + "_Ks_kde.svg")
        fig.savefig(fname + "_Ks_kde.pdf")
        plt.close()
        fig = default_plot(ksdf_predict, title=os.path.basename(fname), bins=50, ylabel="Duplication events", nums = int(n),plot = plot, ylim=ylim,yticks=yticks)
        fig.savefig(fname + "_Ks.svg")
        fig.savefig(fname + "_Ks.pdf")
        plt.close()
        #ksdf_predict_filter = alnfilter(ksdf_predict,weights_outliers_included,alignfilter[0],alignfilter[1],alignfilter[2],ksrange[0],ksrange[1])
        #draw_components_kde_bootstrap(kdemethod,outdir,int(n),ksdf_predict_filter,weighted,boots,bin_width)
    #mean_modes, std_modes, mean_medians, std_medians = bootstrap_kde(kdemethod,outdir, train_in, ksrange[0], ksrange[1], boots, bin_width, ksdf_filtered, weight_col, weighted = weighted)
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
    default='fasttree', show_default=True,
    help="Tree inference method for node weighting")
@click.option('--spair', '-sr', multiple=True, default=None, show_default=True,help='species pair to be plotted')
@click.option('--speciestree', '-sp', default=None, show_default=True,help='species tree to perform rate correction')
@click.option('--reweight', '-rw', is_flag=True, help='recalculate the weight per species pair')
@click.option('--onlyrootout', '-or', is_flag=True, help='only consider the outgroup at root')
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
        strip_gaps, tree_method,spair, speciestree, reweight, onlyrootout):
    from wgd.core import get_gene_families, SequenceData, KsDistributionBuilder
    from wgd.core import read_gene_families, merge_seqs
    from wgd.viz import default_plot, apply_filters,multi_sp_plot
    start = timer()
    if tmpdir != None and not os.path.isdir(tmpdir): os.mkdir(tmpdir)
    if len(sequences) == 0: 
        logging.error("Please provide at least one sequence file")
        exit(0)
    if len(sequences) == 2:
        tree_method = "cluster"  # for RBH others don't make sense (and crash)
    seqs = [SequenceData(s, tmp_path=tmpdir, out_path=outdir,
            to_stop=to_stop, cds=cds, threads=nthreads) for s in sequences]
    spgenemap = {}
    for i in seqs: spgenemap.update(i.spgenemap())
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
    elif len(sequences) > 2:
        ylabel = "Homologous pairs"
    if len(spair)!= 0:  multi_sp_plot(df,spair,spgenemap,outdir,onlyrootout,title=prefix,ylabel=ylabel,ksd=True,reweight=reweight,sptree=speciestree)
    fig = default_plot(df, title=prefix, bins=50, ylabel=ylabel)
    fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)))
    fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)))
    plt.close()
    if tmpdir is None:
        [x.remove_tmp(prompt=False) for x in seqs]
    end = timer()
    logging.info("Total run time: {}s".format(int(end-start)))
    logging.info("Done")
    
# Ks distribution construction
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--datafile', '-d', default=None, show_default=True, help='Ks data file')
@click.option('--outdir', '-o', default="wgd_viz", show_default=True, help='output directory')
@click.option('--spair', '-sr', multiple=True, default=None, show_default=True,help='species pair to be plotted')
@click.option('--gsmap', '-gs', default=None, show_default=True, help='gene name-species name map')
@click.option('--speciestree', '-sp', default=None, show_default=True,help='species tree to perform rate correction')
@click.option('--plotkde', '-pk', is_flag=True, help='plot kde curve over histogram')
@click.option('--reweight', '-rw', is_flag=True, help='recalculate the weight per species pair')
@click.option('--onlyrootout', '-or', is_flag=True, help='only consider the outgroup at root')
@click.option('--em_iterations', '-iter', type=int, default=200, show_default=True, help='maximum EM iterations')
@click.option('--em_initializations', '-init', type=int, default=200, show_default=True, help='maximum EM initializations')
@click.option('--prominence_cutoff', '-prct', type=float, default=0.1, show_default=True, help='prominence cutoff of acceptable peaks')
@click.option('--segments', '-sm', default=None,show_default=True,help='segments.txt file')
@click.option('--minlen', '-ml', default=-1, show_default=True, help="minimum length of a genomic element to be included in dotplot")
@click.option('--maxsize', '-ms', default=200, show_default=True, help="maximum family size to include in analysis")
@click.option('--anchorpoints', '-ap', default=None, show_default=True, help='anchorpoints.txt file')
@click.option('--multiplicon', '-mt', default=None, show_default=True, help='multiplicons.txt file')
@click.option('--genetable', '-gt', default=None, show_default=True, help='gene-table.csv file')
@click.option('--rel_height', '-rh', type=float, default=0.4, show_default=True, help='relative height at which the peak width is measured')
@click.option('--minseglen', '-mg', default=10000, show_default=True, help="minimum length (ratio if <=1) of segments to show in marco-synteny")
@click.option('--keepredun', '-kr', is_flag=True, help='keep redundant multiplicons')
def viz(**kwargs):
    """
    Visualization of Ks distribution or synteny
    """
    _viz(**kwargs)

def _viz(datafile,spair,outdir,gsmap,plotkde,reweight,em_iterations,em_initializations,prominence_cutoff,segments,minlen,maxsize,anchorpoints,multiplicon,genetable,rel_height,speciestree,onlyrootout,minseglen,keepredun):
    from wgd.viz import elmm_plot, apply_filters, multi_sp_plot, default_plot,all_dotplots,filter_by_minlength
    from wgd.core import _mkdir
    from wgd.syn import get_anchors,get_multi,get_segments_profile
    if datafile!=None: prefix = os.path.basename(datafile)
    _mkdir(outdir)
    if datafile ==None:
        table = pd.read_csv(genetable,header=0,index_col=0,sep=',')
        df_anchor,df_multi = get_anchors('',userdf=anchorpoints),get_multi('',userdf2=multiplicon)
        segs = get_segments_profile(df_multi,keepredun,'',userdf3=segments)
        segs,table = filter_by_minlength(table,segs,minlen,df_multi,keepredun,outdir,minseglen)
        figs = all_dotplots(table, segs, df_multi, minseglen, anchors=df_anchor, maxsize=maxsize, minlen=minlen, outdir=outdir)
        for k, v in figs.items():
            v.savefig(os.path.join(outdir, "{}.dot.svg".format(k)))
            v.savefig(os.path.join(outdir, "{}.dot.pdf".format(k)))
            v.savefig(os.path.join(outdir, "{}.dot.png".format(k)))
        logging.info('Done')
        exit()
    ksdb_df = pd.read_csv(datafile,header=0,index_col=0,sep='\t')
    df = apply_filters(ksdb_df, [("dS", 0., 5.)])
    ylabel = "Duplications" if spair == () else "Homologous pairs"
    if len(spair)!= 0: multi_sp_plot(df,spair,gsmap,outdir,onlyrootout,title=prefix,ylabel=ylabel,viz=True,plotkde=plotkde,reweight=reweight,sptree=speciestree,ap = anchorpoints)
    fig = default_plot(df, title=prefix, bins=50, ylabel=ylabel)
    fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)))
    fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)))
    plt.close()
    if spair == ():
        logging.info('Exponential-Lognormal mixture modeling on node-weighted Ks distribution')
        elmm_plot(df,prefix,outdir,max_EM_iterations=em_iterations,num_EM_initializations=em_initializations,peak_threshold=prominence_cutoff,rel_height=rel_height)
        logging.info('Exponential-Lognormal mixture modeling on node-averaged Ks distribution')
        elmm_plot(df,prefix,outdir,max_EM_iterations=em_iterations,num_EM_initializations=em_initializations,peak_threshold=prominence_cutoff,na=True,rel_height=rel_height)
    logging.info('Done')

@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('families', type=click.Path(exists=True))
@click.argument('gff_files', nargs=-1, type=click.Path(exists=True))
@click.option('--ks_distribution', '-ks', default=None,
    help="ks distribution tsv file (optional, see `wgd ksd`)")
@click.option('--outdir', '-o', default='wgd_syn', show_default=True, 
    help='output directory')
@click.option('--feature', '-f', default='gene', show_default=True,
    help="keyword for parsing the genes from the GFF file (column 3)")
@click.option('--attribute', '-a', default='ID', show_default=True,
    help="keyword for parsing the gene IDs from the GFF file (column 9)")
@click.option('--minlen', '-ml', default=-1, show_default=True,
    help="minimum length of a genomic element to be included in dotplot.")
@click.option('--maxsize', '-ms', default=200, show_default=True,
    help="maximum family size to include in analysis.")
@click.option('--ks_range', '-r', nargs=2, default=(0, 5), show_default=True,
    type=float, help='Ks range to use for colored dotplot')
@click.option('--iadhore_options', default="",
    help="other options for I-ADHoRe, as a comma separated string, "
         "e.g. gap_size=30,q_value=0.75,prob_cutoff=0.05")
@click.option('--ancestor', '-ac', default=None,show_default=True,help='assumed ancestor species')
@click.option('--minseglen', '-mg', default=100000, show_default=True, help="min length of segments in ratio if <= 1")
@click.option('--keepredun', '-kr', is_flag=True, help='keep redundant multiplicons')
def syn(**kwargs):
    """
    Co-linearity and anchor inference using I-ADHoRe.
    """
    _syn(**kwargs)

def _syn(families, gff_files, ks_distribution, outdir, feature, attribute,
        minlen, maxsize, ks_range, iadhore_options, ancestor, minseglen, keepredun):
    """
    Co-linearity and anchor inference using I-ADHoRe.
    """
    from wgd.syn import make_gene_table, configure_adhore, run_adhore
    from wgd.syn import get_anchors, get_anchor_ksd, get_segments_profile, get_multi
    from wgd.viz import default_plot, apply_filters, all_dotplots, syntenic_dotplot_ks_colored,filter_by_minlength
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
    multi = get_multi(out_path)
    if anchors is None:
        logging.warning("No anchors found, terminating! Please inspect your input files "
                "and the I-ADHoRe results in `{}`".format(out_path))
        exit(1)

    anchors.to_csv(os.path.join(outdir, "anchors.csv"))
    segs = get_segments_profile(multi,keepredun,out_path)
    segs,table = filter_by_minlength(table,segs,minlen,multi,keepredun,outdir,minseglen)
    # dotplot
    #logging.info("Generating dot plots")
    figs = all_dotplots(table, segs, multi, anchors, minseglen, maxsize=maxsize, minlen=minlen, outdir=outdir, ancestor=ancestor) 
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
        anchor_ks.to_csv(os.path.join(outdir, "{}.anchors.ks.tsv".format(prefix)),sep='\t')
        a = apply_filters(ksd,       [("dS", 1e-4, 5.), ("S", 10, 1e6)])
        b = apply_filters(anchor_ks, [("dS", 1e-4, 5.), ("S", 10, 1e6)])
        logging.info("Generating anchor Ks distribution")
        fig = default_plot(a, b, title=prefix, bins=50, ylabel=ylabel)
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
        #figs2=Ks_dotplots(segs,multiplicons, table, anchor_ks, anchor_points, anchors,min_ks=ks_range[0],max_ks=ks_range[1], maxsize=maxsize, minlen=minlen,outdir = outdir)
        #for k, v in figs2.items():
        #    v.savefig(os.path.join(outdir, "{}.ks.dot.svg".format(k)))
        #    v.savefig(os.path.join(outdir, "{}.ks.dot.pdf".format(k)))
        #    v.savefig(os.path.join(outdir, "{}.ks.dot.png".format(k)))
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
        '--n_init', '-ni', default=200, show_default=True,
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
    start = time.time()
    cli()
    end = time.time()
    logging.info("Total run time: {}s".format(int(end-start)))
