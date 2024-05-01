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
from wgd.core import memory_reporter_initial
import tracemalloc
__version__ = pkg_resources.require("wgd")[0].version
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# CLI entry point
@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbosity', '-v', type=click.Choice(['info', 'debug']),
    default='info', help="Verbosity level, default = info.")
def cli(verbosity):
    """
    wgd v2 - Copyright (C) 2024-2025 Hengchi Chen\n
    Contact: heche@psb.vib-ugent.be
    """
    logging.basicConfig(
        format='%(message)s',
        handlers=[RichHandler()],
        datefmt='%Y-%m-%d %H:%M:%S',
        level=verbosity.upper())
    logging.info("This is wgd v{}".format(__version__))
    memory_reporter_initial()
    pass


# Diamond and gene families
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('sequences', nargs=-1, type=click.Path(exists=True))
@click.option('--outdir', '-o', default='wgd_dmd', show_default=True,
    help='output directory')
@click.option('--tmpdir', '-t', default=None, show_default=True,
    help='tmp directory')
@click.option('--prot', '-p', is_flag=True, help="protein sequence instead of cds")
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
@click.option('--genetable', '-gt', default= None, show_default=True, help='gene table file')
@click.option('--collinearcoalescence','-coc', is_flag=True,help="collinear coalescence inference of phylogeny and WGD")
@click.option('--keepfasta','-kf', is_flag=True,
    help="keep the fasta file of homologs family")
@click.option('--keepduplicates','-kd', is_flag=True,
    help="Keep ID duplicates in MRBHs")
@click.option('--globalmrbh','-gm', is_flag=True,
    help="global MRBH regardless of focal species")
@click.option('--nthreads', '-n', default=4, show_default=True,help="number of threads to use")
@click.option('--orthoinfer','-oi', is_flag=True,help="orthogroups inference")
@click.option('--onlyortho','-oo', is_flag=True,help="only run orthogroups inference")
@click.option('--getnsog','-gn', is_flag=True,help="get nested single-copy gene families")
@click.option('--tree_method', '-tree',type=click.Choice(['fasttree', 'iqtree', 'mrbayes']),default='fasttree',show_default=True,help="tree inference method")
@click.option('--treeset', '-ts', multiple=True, default=None, show_default=True,help='parameters setting for gene tree inference')
@click.option('--msogcut', '-mc', type=float, default=0.8, show_default=True,help='ratio cutoff for mostly single-copy family and species representation in collinear coalescence inference')
@click.option('--geneassign','-ga', is_flag=True,help="assign genes to given gene families")
@click.option('--seq2assign', '-sa', multiple=True, default= None, show_default=True, help='sequences to be assigned')
@click.option('--fam2assign', '-fa',default= None, show_default=True, help='families to be assigned upon')
@click.option('--concat','-cc', is_flag=True,help="concatenation pipeline for orthoinfer")
@click.option('--testsog','-te', is_flag=True,help="Unbiased test of single-copy gene families")
@click.option('--bins', '-bs', type=int, default=100, show_default=True, help='bins for gene length normalization')
@click.option('--normalizedpercent', '-np', type=int, default=5, show_default=True, help='percentage of upper hits used for normalization')
@click.option('--nonormalization','-nn', is_flag=True,help="call off the normalization process")
@click.option('--buscosog','-bsog', is_flag=True,help="get busco-guided single-copy gene family")
@click.option('--buscohmm', '-bhmm',default= None, show_default=True, help='HMM profile of given busco dataset')
@click.option('--buscocutoff', '-bctf', default= None, show_default=True, help='HMM score cutoffs of BUSCO')
@click.option('--ogformat','-of', is_flag=True,help="get RBH gene families with index")
def dmd(**kwargs):
    """
    All-vs-all diamond blastp + MCL clustering.

    Requires diamond and mcl. Note the two key parameters, being the e-value
    cut-off and inflation factor. It is advised to explore the effects of them
    on your analysis.

    Example 1 - whole paranome delineation:

        wgd dmd ath.fasta

    Example 2 - one vs. one ortholog delineation:

        wgd dmd ath.fasta vvi.fasta

    Example 3 - one vs. one ortholog delineation for multiple pairs:

        wgd dmd ath.fasta vvi.fasta egr.fasta

    Example 4 - one vs. one ortholog delineation for multiple pairs with focal species:

        wgd dmd ath.fasta vvi.fasta egr.fasta --focus ath.fasta (--anchorpoints anchorpoints.txt --cscore 0.7)

    """
    _dmd(**kwargs)

def _dmd(sequences, outdir, tmpdir, prot, cscore, inflation, eval, to_stop, cds, focus, anchorpoints, keepfasta, keepduplicates, globalmrbh, nthreads, orthoinfer, onlyortho, getnsog, tree_method, treeset, msogcut, geneassign, seq2assign, fam2assign, concat, segments, listelements, collinearcoalescence, testsog, bins, buscosog, buscohmm, buscocutoff, genetable, normalizedpercent, nonormalization, ogformat):
    from wgd.core import SequenceData, read_MultiRBH_gene_families,mrbh,ortho_infer,genes2fams,endt,segmentsaps,bsog,parallelrbh
    from joblib import Parallel, delayed
    start = timer()
    if tmpdir != None and not os.path.isdir(tmpdir): os.mkdir(tmpdir)
    s = [SequenceData(s, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore, threads=nthreads, bins=bins, normalizedpercent=normalizedpercent, nonormalization=nonormalization, prot=prot) for s in sequences]
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
        genes2fams(seq2assign,fam2assign,outdir,s,nthreads,tmpdir,to_stop,cds,cscore,eval,start,normalizedpercent,tree_method,treeset,assign_method='hmmer', prot=prot)
    if orthoinfer:
        logging.info("Infering orthologous gene families")
        if not getnsog: logging.info("Note that the option --getnsog can be set to further retrieve nested single-copy gene families")
        if not testsog: logging.info("Note that the option --testsog can be set to verify the inferred single-copy gene families")
        ortho_infer(sequences,s,outdir,tmpdir,to_stop,cds,cscore,inflation,eval,nthreads,getnsog,tree_method,treeset,msogcut,concat,testsog,normalizedpercent,bins=bins,nonormalization=nonormalization)
        if onlyortho: endt(tmpdir,start,s)
    if len(s) == 0:
        logging.error("No sequences provided!")
        return
    if len(s) == 1:
        if prot: logging.info("One protein file: will compute paranome")
        else: logging.info("One cds file: will compute paranome")
        s[0].get_paranome(inflation=inflation, eval=eval)
        s[0].write_paranome(False)
    elif focus is None and not globalmrbh:
        if prot: logging.info("Multiple protein files: will compute RBH orthologs")
        else: logging.info("Multiple cds files: will compute RBH orthologs")
        if nthreads!=(len(s)-1)*len(s)/2: logging.info("Note that setting the number of threads as {} is the most efficient".format(int((len(s)-1)*len(s)/2)))
        pairs = sum(map(lambda i:[(i,j) for j in range(i+1,len(s))],range(len(s)-1)),[])
        Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(parallelrbh)(s,i,j,ogformat,cscore,eval) for i,j in pairs)
    mrbh(globalmrbh,outdir,s,cscore,eval,keepduplicates,anchorpoints,focus,keepfasta,nthreads)
    endt(tmpdir,start,s)


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
@click.option('--pairwise', is_flag=True,help="Pairwise gene-pair feeded into codeml")
@click.option('--annotation',type=click.Choice(['none','eggnog', 'hmmpfam', 'interproscan']),default='none',show_default=True,help="Functional annotation for orthologous families")
@click.option('--eggnogdata', '-ed', default=None, show_default=True,help='Eggnog data dirctory for annotation')
@click.option('--pfam', type=click.Choice(['none', 'denovo', 'realign']),default='none',show_default=True,help='PFAM domains for annotation')
@click.option('--dmnb', default=None, show_default=True,help='Diamond database for annotation')
@click.option('--hmm', default=None, show_default=True,help='profile for hmmscan')
@click.option('--evalue', '-e', default=1e-10, show_default=True,help='E-value threshold for annotation')
@click.option('--exepath', default=None, show_default=True,help='Path to interproscan installation folder')
@click.option('--fossil', '-f', nargs=5, default= ('clade1;clade2', 'taxa1,taxa2;taxa3,taxa4', '4;5', '0.5;0.6', '400;500'), show_default=True, help='fossil calibration info (id,taxa,mean,std,offset)')
@click.option('--rootheight', '-rh', nargs=3,default= (4,0.5,400), show_default=True, help='root height calibration info (mean,std,offset)')
@click.option('--chainset', '-cs', nargs=2,default= (10000,100), show_default=True, help='MCMC chain info (length,frequency) for beast')
@click.option('--beastlgjar', default=None, show_default=True,help='path of beastLG.jar')
@click.option('--beagle', is_flag=True,help='using beagle')
@click.option('--protcocdating', is_flag=True,help='only run protein concatenation dating')
@click.option('--protdating', is_flag=True,help='only run protein dating')
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

def _focus(families, sequences, outdir, tmpdir, nthreads, to_stop, cds, strip_gaps, aligner, tree_method, treeset, concatenation, coalescence, speciestree, dating, datingset, nsites, outgroup, partition, aamodel, ks, annotation, pairwise, eggnogdata, pfam, dmnb, hmm, evalue, exepath, fossil, rootheight, chainset, beastlgjar, beagle, protdating, protcocdating):
    from wgd.core import SequenceData, read_gene_families, get_gene_families, KsDistributionBuilder
    from wgd.core import mergeMultiRBH_seqs, read_MultiRBH_gene_families, get_MultipRBH_gene_families, Concat, _Codon2partition_, Coale, Run_MCMCTREE, Run_r8s, Reroot, eggnog, hmmer_pfam, interproscan, Run_BEAST, get_only_protaln, Run_MCMCTREE_concprot, Concat_prot, Parallel_MCMCTREE_Prot, endt
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
    if nthreads < len(fams): logging.info("{} threads are used for {} gene families\nNote that adding threads can significantly accelerate the analysis".format(int(nthreads),int(len(fams))))
    if protcocdating:
        logging.info("Only implement protein concatenation dating via mcmctree")
        pro_alns,pro_alnfs = get_only_protaln(seqs,fams,outdir,nthreads,option="--auto")
        Concat_palnf,Concat_paln,slist = Concat_prot(pro_alns,families,outdir)
        Run_MCMCTREE_concprot(Concat_paln,Concat_palnf,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads)
        endt(tmpdir,start,seqs)
    if protdating:
        logging.info("Only implement protein dating via mcmctree")
        pro_alns,pro_alnfs = get_only_protaln(seqs,fams,outdir,nthreads,option="--auto")
        Concat_palnf,Concat_paln,slist = Concat_prot(pro_alns,families,outdir)
        Parallel_MCMCTREE_Prot(Concat_palnf,Concat_paln,pro_alns,pro_alnfs,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads)
        endt(tmpdir,start,seqs)
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
    endt(tmpdir,start,seqs)

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
@click.option('--gamma', '-g', default=1e-3, show_default=True, help='gamma parameter for bgmm models')
@click.option('--boots', type=int, default=200, show_default=True, help="number of bootstrap replicates of kde")
@click.option('--weighted', is_flag=True,help="node-weighted instead of node-averaged method")
@click.option('--plot', '-p', type=click.Choice(['stacked', 'identical']), default='identical', show_default=True, help="plotting method")
@click.option('--bw_method', '-bm', type=click.Choice(['silverman', 'ISJ']), default='silverman', show_default=True, help="bandwidth method")
@click.option('--n_medoids', type=int, default=2, show_default=True, help="number of medoids to generate")
@click.option('--kdemethod', '-km', type=click.Choice(['scipy', 'naivekde', 'treekde', 'fftkde']), default='scipy', show_default=True, help="kde method")
@click.option('--n_clusters',type=int, default=5, show_default=True, help="number of clusters to plot Elbow loss function")
@click.option('--guide', '-gd', type=click.Choice(['Multiplicon','Segment']), default='Segment', show_default=True, help="regime residing anchors")
@click.option('--prominence_cutoff', '-prct', type=float, default=0.1, show_default=True, help='prominence cutoff of acceptable peaks')
@click.option('--rel_height', '-rh', type=float, default=0.4, show_default=True, help='relative height at which the peak width is measured')
@click.option('--kstodate', '-kd', nargs=2, type=float, default=(0.5, 1.5), show_default=True, help='range of Ks to be dated')
@click.option('--xlim', '-xl', nargs=2, type=float, default=(None, None), show_default=True, help='xlim of GMM Ks distribution')
@click.option('--ylim', '-yl', nargs=2, type=float, default=(None, None), show_default=True, help='ylim of GMM Ks distribution')
@click.option('--manualset', is_flag=True,help="Manually set Ks range of anchor pairs or multiplicons as CI")
@click.option('--ci', default=95, show_default=True,type=int, help='confidence level of log-normal distribution to date')
@click.option('--hdr', default=95, show_default=True,type=int, help='highest density region (HDR)')
@click.option('--heuristic', is_flag=True,help="heuristic CI for dating")
@click.option('--kscutoff', '-kc', default=5, show_default=True, type=float, help='Ks Saturation cutoff for genes in Dating')
@click.option('--keeptmpfig', is_flag=True,help="keep temporary figures in peak finding process")
def peak(**kwargs):
    """
    Infer peak and CI of Ks distribution.
    """
    _peak(**kwargs)

def _peak(ks_distribution, anchorpoints, outdir, alignfilter, ksrange, bin_width, weights_outliers_included, method, seed, em_iter, n_init, components, boots, weighted, plot, bw_method, n_medoids, kdemethod, n_clusters, guide, prominence_cutoff, kstodate, rel_height, ci,manualset,segments,hdr,heuristic,listelements,multipliconpairs,kscutoff,gamma,xlim,ylim,keeptmpfig,family = None):
    from wgd.peak import alnfilter, group_dS, log_trans, fit_gmm, fit_bgmm, add_prediction, bootstrap_kde, default_plot, get_kde, draw_kde_CI, draw_components_kde_bootstrap, default_plot_kde, fit_apgmm_guide, fit_apgmm_ap, find_apeak, find_mpeak, retreive95CI
    from wgd.core import _mkdir,endtime
    from wgd.utils import formatv2
    start = timer()
    outpath = _mkdir(outdir)
    ksdf = pd.read_csv(ks_distribution,header=0,index_col=0,sep='\t')
    ksdf = formatv2(ksdf)
    if len(ksdf.columns) <4:
        logging.info("Begin to analyze peak of WGD dates")
        draw_kde_CI(kdemethod, outdir,ksdf,boots,bw_method,date_lower = 0,date_upper=4)
        endtime(start)
    ksdf_filtered = alnfilter(ksdf,weights_outliers_included,alignfilter[0],alignfilter[1],alignfilter[2],ksrange[0],ksrange[1])
    if family != None:
        retreive95CI(family,ksdf_filtered,outdir,kstodate[0],kstodate[1])
        endtime(start)
    fn_ksdf, weight_col = group_dS(ksdf_filtered)
    train_in = log_trans(fn_ksdf)
    if anchorpoints!= None:
        df_ap_mp = fit_apgmm_guide(hdr,guide,anchorpoints,ksdf,ksdf_filtered,seed,components,em_iter,n_init,outdir,method,gamma,weighted,plot,segment=segments,multipliconpairs=multipliconpairs,listelement=listelements,cutoff = kscutoff,user_xlim=xlim,user_ylim=ylim,peak_threshold=prominence_cutoff,rel_height=rel_height,keeptmp=keeptmpfig)
        df_ap = fit_apgmm_ap(hdr,anchorpoints,ksdf_filtered,seed,components,em_iter,n_init,outdir,method,gamma,weighted,plot,cutoff = kscutoff,peak_threshold=prominence_cutoff,rel_height=rel_height,user_xlim=xlim,user_ylim=ylim)
        if heuristic:
            find_apeak(df_ap,anchorpoints,os.path.basename(ks_distribution),outdir,peak_threshold=prominence_cutoff,na=False,rel_height=rel_height,ci=ci,user_low=kstodate[0],user_upp=kstodate[1],user=manualset, kscutoff=kscutoff,keeptmp=keeptmpfig)
            find_apeak(df_ap,anchorpoints,os.path.basename(ks_distribution),outdir,peak_threshold=prominence_cutoff,na=True,rel_height=rel_height,ci=ci,user_low=kstodate[0],user_upp=kstodate[1],user=manualset, kscutoff=kscutoff,keeptmp=keeptmpfig)
            find_mpeak(df_ap_mp,anchorpoints,os.path.basename(ks_distribution),outdir,guide,peak_threshold=prominence_cutoff,rel_height=rel_height,ci=ci,user_low=kstodate[0],user_upp=kstodate[1],user=manualset,kscutoff=kscutoff,keeptmp=keeptmpfig)
        endtime(start)
    get_kde(kdemethod,outdir,fn_ksdf,ksdf_filtered,weighted,ksrange[0],ksrange[1])
    if method == 'gmm':
        out_file = os.path.join(outdir, "AIC_BIC.pdf")
        models, aic, bic, besta, bestb, N = fit_gmm(out_file, train_in, seed, components[0], components[1], em_iter=em_iter, n_init=n_init)
    if method == 'bgmm':
        models, N = fit_bgmm(train_in, seed, gamma, components[0], components[1], em_iter=em_iter, n_init=n_init)
    for n, m in zip(N,models):
        fname = os.path.join(outpath, "Ks_{0}_{1}components_prediction.tsv".format(method,n))
        ksdf_predict = add_prediction(ksdf,fn_ksdf,train_in,m)
        ksdf_predict.to_csv(fname,header=True,index=True,sep='\t')
        logging.info("Plotting components-annotated Ks distribution for {} components model".format(n))
        #fig = default_plot(ksdf_predict, title=os.path.basename(fname), bins=50, ylabel="Duplication events", nums = int(n),plot = plot)
        #fig.savefig(fname + "_Ks.svg")
        #fig.savefig(fname + "_Ks.pdf")
        #plt.close()
        fig,safe_ylim,yticks = default_plot_kde(ksdf_predict, title=os.path.basename(fname), bins=50, ylabel="Duplication events", nums = int(n),plot = 'identical',user_xlim=xlim,user_ylim=ylim)
        fig.savefig(fname + "_Ks_kde.svg")
        fig.savefig(fname + "_Ks_kde.pdf")
        plt.close()
        fig = default_plot(ksdf_predict, title=os.path.basename(fname), bins=50, ylabel="Duplication events", nums = int(n),plot = plot, ylim=safe_ylim,yticks=yticks,user_xlim=xlim,user_ylim=ylim)
        fig.savefig(fname + "_Ks.svg")
        fig.savefig(fname + "_Ks.pdf")
        plt.close()
        #ksdf_predict_filter = alnfilter(ksdf_predict,weights_outliers_included,alignfilter[0],alignfilter[1],alignfilter[2],ksrange[0],ksrange[1])
        #draw_components_kde_bootstrap(kdemethod,outdir,int(n),ksdf_predict_filter,weighted,boots,bin_width)
    #mean_modes, std_modes, mean_medians, std_medians = bootstrap_kde(kdemethod,outdir, train_in, ksrange[0], ksrange[1], boots, bin_width, ksdf_filtered, weight_col, weighted = weighted)
    endtime(start)

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
@click.option('--aligner', '-a', type=click.Choice(['mafft', 'muscle', 'prank']), default='mafft', show_default=True, help="aligner method for MSA")
@click.option('--aln_options', default='--auto', show_default=True, help="options in aligning, as a comma separated string")
@click.option('--tree_method', '-tree', 
    type=click.Choice(['cluster', 'fasttree', 'iqtree']), 
    default='fasttree', show_default=True,
    help="Tree inference method for node weighting")
@click.option('--tree_options', default=None, show_default=True, help="options in tree inference, as a comma separated string")
@click.option('--node_average', is_flag=True, help="node-average way of de-redundancy instead of node-weighted")
@click.option('--spair', '-sr', multiple=True, default=None, show_default=True,help='species pair to be plotted')
@click.option('--speciestree', '-sp', default=None, show_default=True,help='species tree to perform rate correction')
@click.option('--reweight', '-rw', is_flag=True, help='recalculate the weight per species pair')
@click.option('--onlyrootout', '-or', is_flag=True, help='only consider the outgroup at root')
@click.option('--extraparanomeks', '-epk', default=None, help='extra paranome ks data')
@click.option('--anchorpoints', '-ap', default=None, show_default=True, help='anchorpoints.txt file to plot anchor Ks')
@click.option('--plotkde', '-pk', is_flag=True, help='plot kde curve over histogram')
@click.option('--plotapgmm', '-pag', is_flag=True, help='plot mixture modeling of anchor Ks')
@click.option('--plotelmm', '-pem', is_flag=True, help='plot elmm mixture modeling')
@click.option('--components', '-c', nargs=2, default=(1, 4), show_default=True, help="range of the number of components to fit in anchor Ks mixture modeling")
@click.option('--xlim', '-xl', nargs=2, type=float, default=(None, None), show_default=True, help='xlim of Ks distribution')
@click.option('--ylim', '-yl', nargs=2, type=float, default=(None, None), show_default=True, help='ylim of Ks distribution')
@click.option('--adjustortho', '-ado', is_flag=True, help='adjust the histogram height of orthologous Ks')
@click.option('--adjustfactor', '-adf', type=float, default=1, show_default=True, help='adjustment factor of orthologous Ks')
@click.option('--okalpha', '-oa', type=float, default=0.5, show_default=True, help='opacity of ortholog Ks distribution in mixed plot')
@click.option('--focus2all', '-fa', default=None, show_default=True, help='set focal species and let species pair to be between focal and all the remaining species')
@click.option('--kstree', '-ks', is_flag=True, help='infer Ks tree')
@click.option('--onlyconcatkstree', '-ock', is_flag=True, help='only infer Ks tree under concatenated alignment')
@click.option('--classic', '-cs', is_flag=True, help='mixed plot in a classic manner where the full orthologous Ks distribution is drawed')
@click.option('--toparrow', '-ta', is_flag=True, help='rate correction arrow set at top instead of at the maximum of kde')
@click.option('--bootstrap', '-bs', type=int, default=200, show_default=True, help='Number of bootstrap replicates of ortholog Ks distribution in mixed plot')
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
        strip_gaps, aligner, aln_options, tree_method, tree_options, node_average, spair, speciestree, reweight, onlyrootout, extraparanomeks, anchorpoints, plotkde, plotapgmm, plotelmm, components,xlim,ylim,adjustortho,adjustfactor,okalpha,focus2all,kstree,onlyconcatkstree,classic,toparrow, bootstrap):
    from wgd.core import get_gene_families, SequenceData, KsDistributionBuilder
    from wgd.core import read_gene_families, merge_seqs, get_MultipRBH_gene_families, getconcataln, endt
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
    if not (speciestree is None) and kstree:
        getconcataln(seqs, families, nthreads, outdir, speciestree, spgenemap, onlyconcatkstree, tree_options, option="--auto",tree_method=tree_method)
        if tmpdir is None: [x.remove_tmp(prompt=False) for x in seqs]
        exit()
    s = merge_seqs(seqs)
    logging.info("tmpdir = {}".format(s.tmp_path))
    fams = read_gene_families(families)
    fams = get_gene_families(s, fams, 
            pairwise=pairwise, 
            strip_gaps=strip_gaps, aligner=aligner, aln_options = aln_options,
            tree_method=tree_method, tree_options=tree_options)
    if len(fams) == 0:
        logging.error("All families are singleton families, No Ks can be calculated")
        exit(0)
    #if not (speciestree is None) and kstree:
        #getconcataln(s, families, nthreads, outdir, option="--auto")
        #cds_alns, pro_alns, tree_famsf, calnfs, palnfs, calnfs_length, cds_fastaf, tree_fams = get_MultipRBH_gene_families(seqs,fams,tree_method,treeset,outdir,nthreads,option="--auto",runtree=False)
        #calculatekstree(fams,s,spgenemap)
        #exit()
    ksdb = KsDistributionBuilder(fams, s, n_threads=nthreads)
    ksdb.get_distribution()
    prefix = os.path.basename(families)
    outfile = os.path.join(outdir, "{}.ks.tsv".format(prefix))
    logging.info("Saving to {}".format(outfile))
    ksdb.df.fillna("NaN").to_csv(outfile,sep="\t")
    logging.info("Making plots")
    if "dS" not in ksdb.df.columns or len(ksdb.df.dropna(subset=["dS"]))==0:
        logging.info("No valid Ks values for plotting")
        exit(0)
    df = apply_filters(ksdb.df, [("dS", 0., 5.)])
    ylabel = "Duplications"
    if len(sequences) == 2:
        ylabel = "RBH orthologs"
    elif len(sequences) > 2:
        ylabel = "Homologous pairs"
    if len(spair)!= 0 or (not focus2all is None):
        multi_sp_plot(df,spair,spgenemap,outdir,onlyrootout,title=prefix,ylabel=ylabel,ksd=True,reweight=reweight,sptree=speciestree,extraparanomeks=extraparanomeks, ap = anchorpoints,plotkde=plotkde,plotapgmm=plotapgmm,plotelmm=plotelmm,components=components,na=node_average,user_xlim=xlim,user_ylim=ylim,adjustortho=adjustortho,adfactor=adjustfactor,okalpha=okalpha,focus2all=focus2all,clean=classic,toparrow=toparrow,BT=bootstrap,nthreads=nthreads)
        #multi_sp_plot(df,spair,spgenemap,outdir,onlyrootout,title=prefix,ylabel=ylabel,ksd=True,reweight=reweight,sptree=speciestree,extraparanomeks=extraparanomeks, ap = anchorpoints,plotkde=plotkde,plotapgmm=plotapgmm,plotelmm=plotelmm,components=components,user_xlim=xlim,user_ylim=ylim,adjustortho=adjustortho,adfactor=adjustfactor,okalpha=okalpha,focus2all=focus2all)
    fig = default_plot(df, title=prefix, bins=50, ylabel=ylabel, nodeaverage=node_average)
    fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)))
    fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)))
    plt.close()
    endt(tmpdir,start,seqs)
    
# Ks distribution and synteny visualization
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--datafile', '-d', default=None, show_default=True, help='Ks data file')
@click.option('--outdir', '-o', default="wgd_viz", show_default=True, help='output directory')
@click.option('--spair', '-sr', multiple=True, default=None, show_default=True,help='species pair to be plotted')
@click.option('--focus2all', '-fa', default=None, show_default=True, help='set focal species and let species pair to be between focal and all the remaining species')
@click.option('--gsmap', '-gs', default=None, show_default=True, help='gene name-species name map')
@click.option('--speciestree', '-sp', default=None, show_default=True,help='species tree to perform rate correction')
@click.option('--plotkde', '-pk', is_flag=True, help='plot kde curve over histogram')
@click.option('--reweight', '-rw', is_flag=True, help='recalculate the weight per species pair')
@click.option('--onlyrootout', '-or', is_flag=True, help='only consider the outgroup at root')
@click.option('--em_iterations', '-iter', type=int, default=200, show_default=True, help='maximum EM iterations')
@click.option('--em_initializations', '-init', type=int, default=200, show_default=True, help='maximum EM initializations')
@click.option('--prominence_cutoff', '-prct', type=float, default=0.1, show_default=True, help='prominence cutoff of acceptable peaks')
@click.option('--rel_height', '-rh', type=float, default=0.4, show_default=True, help='relative height at which the peak width is measured')
@click.option('--segments', '-sm', default=None,show_default=True,help='segments.txt file')
@click.option('--minlen', '-ml', default=-1, show_default=True, help="minimum length of a genomic element to be included in dotplot")
@click.option('--maxsize', '-ms', default=200, show_default=True, help="maximum family size to include in analysis")
@click.option('--anchorpoints', '-ap', default=None, show_default=True, help='anchorpoints.txt file')
@click.option('--multiplicon', '-mt', default=None, show_default=True, help='multiplicons.txt file')
@click.option('--genetable', '-gt', default=None, show_default=True, help='gene-table.csv file')
@click.option('--minseglen', '-mg', default=10000, show_default=True, help="minimum length (ratio if <=1) of segments to show in marco-synteny")
@click.option('--mingenenum', '-mgn', default=30, type=int, show_default=True, help="minimum number of genes on segments to be considered")
@click.option('--keepredun', '-kr', is_flag=True, help='keep redundant multiplicons')
@click.option('--extraparanomeks', '-epk', default=None, help='extra paranome ks data')
@click.option('--plotapgmm', '-pag', is_flag=True, help='plot mixture modeling of anchor pairs')
@click.option('--plotelmm', '-pem', is_flag=True, help='plot elmm mixture modeling')
@click.option('--components', '-n', nargs=2, default=(1, 4), show_default=True, help="range of number of components to fit")
@click.option('--plotsyn', '-psy', is_flag=True, help='plot synteny')
@click.option('--dotsize', '-ds', type=float, default=0.3, show_default=True, help='size of dots')
@click.option('--apalpha', '-aa', type=float, default=1, show_default=True, help='opacity of anchor dots')
@click.option('--hoalpha', '-ha', type=float, default=0, show_default=True, help='opacity of homolog dots')
@click.option('--showrealtick', '-srt', is_flag=True, help='show the real tick in genes or bases')
@click.option('--ticklabelsize', '-tls', type=float, default=5, show_default=True, help='label size of tick')
@click.option('--xlim', '-xl', nargs=2, type=float, default=(None, None), show_default=True, help='xlim of Ks distribution')
@click.option('--ylim', '-yl', nargs=2, type=float, default=(None, None), show_default=True, help='ylim of Ks distribution')
@click.option('--adjustortho', '-ado', is_flag=True, help='adjust the histogram height of ortholog Ks')
@click.option('--adjustfactor', '-adf', type=float, default=0.5, show_default=True, help='adjust factor of ortholog Ks')
@click.option('--okalpha', '-oa', type=float, default=0.5, show_default=True, help='opacity of ortholog Ks distribution in mixed plot')
@click.option('--classic', '-cs', is_flag=True, help='mixed plot in a classic manner')
@click.option('--toparrow', '-ta', is_flag=True, help='rate correction arrow set at top instead of at the maximum of kde')
@click.option('--nodeaveraged', '-na', is_flag=True, help='node averaged way of de-redundancy')
@click.option('--bootstrap', '-bs', type=int, default=200, show_default=True, help='Number of bootstrap replicates of ortholog Ks distribution in mixed plot')
@click.option('--gistrb', '-gr', is_flag=True, help='whether to use gist_rainbow as color map of dotplot')
@click.option('--nthreads', '-n', default=1, show_default=True,help="number of threads to use")
def viz(**kwargs):
    """
    Visualization of Ks distribution or synteny
    """
    _viz(**kwargs)

def _viz(datafile,spair,outdir,gsmap,plotkde,reweight,em_iterations,em_initializations,prominence_cutoff,segments,minlen,maxsize,anchorpoints,multiplicon,genetable,rel_height,speciestree,onlyrootout,minseglen,keepredun,extraparanomeks,plotapgmm,plotelmm,components,mingenenum,plotsyn,dotsize,apalpha,hoalpha,showrealtick,ticklabelsize,xlim,ylim,adjustortho,adjustfactor,okalpha,focus2all,classic,nodeaveraged,toparrow,bootstrap,gistrb,nthreads):
    from wgd.viz import elmm_plot, apply_filters, multi_sp_plot, default_plot,all_dotplots,filter_by_minlength,dotplotunitgene,dotplotingene,filter_mingenumber,dotplotingeneoverall
    from wgd.core import _mkdir,endtime
    from wgd.syn import get_anchors,get_multi,get_segments_profile,get_chrom_gene,get_mp_geneorder,transformunit
    from wgd.utils import formatv2
    start = timer()
    if datafile!=None: prefix = os.path.basename(datafile)
    _mkdir(outdir)
    if plotsyn:
        df = None
        if datafile!=None:
            ksdb_df = pd.read_csv(datafile,header=0,index_col=0,sep='\t')
            ksdb_df = formatv2(ksdb_df)
            df = apply_filters(ksdb_df, [("dS", 0., 5.)])
        table = pd.read_csv(genetable,header=0,index_col=0,sep=',')
        table_orig = table.copy()
        df_anchor,orig_anchors = get_anchors('',userdf=anchorpoints)
        df_multi = get_multi('',userdf2=multiplicon)
        ordered_genes_perchrom_allsp, gene_orders = get_chrom_gene(table,outdir)
        #ordered_mp = get_mp_geneorder(gene_orders,'',outdir,table,userdf4=multipliconpairs)
        segs = get_segments_profile(df_multi,keepredun,'',userdf3=segments)
        segs,table,df_multi,removed_scfa = filter_by_minlength(table,segs,minlen,df_multi,keepredun,outdir,minseglen)
        segs_gene_unit, gene_order_dict_allsp = transformunit(segs,ordered_genes_perchrom_allsp,outdir)
        segs = filter_mingenumber(segs_gene_unit,mingenenum,outdir,len(gene_order_dict_allsp),start)
        dotplotingene(ordered_genes_perchrom_allsp,removed_scfa,outdir,table,gene_orders,anchor=df_anchor,ksdf=df,maxsize=maxsize,dotsize=dotsize, apalpha=apalpha, hoalpha=hoalpha,showrealtick=showrealtick, las=ticklabelsize, gistrb=gistrb)
        if len(ordered_genes_perchrom_allsp)>1: dotplotingeneoverall(ordered_genes_perchrom_allsp,removed_scfa,outdir,table,gene_orders,anchor=df_anchor,ksdf=df,maxsize=maxsize,dotsize=dotsize, apalpha=apalpha, hoalpha=hoalpha,showrealtick=showrealtick, las=ticklabelsize, gistrb=gistrb)
        #dotplotunitgene(ordered_genes_perchrom_allsp,segs_gene_unit,removed_scfa,outdir,mingenenum,table_orig,ordered_mp,ksdf=df)
        figs = all_dotplots(table, segs, df_multi, minseglen, anchors=df_anchor, maxsize=maxsize, minlen=minlen, outdir=outdir, Ks = df, dotsize=dotsize, apalpha=apalpha, hoalpha=hoalpha, showrealtick=showrealtick, las=ticklabelsize, gistrb=gistrb)
        for k, v in figs.items():
            v.savefig(os.path.join(outdir, "{}.dot.svg".format(k)))
            v.savefig(os.path.join(outdir, "{}.dot.pdf".format(k)))
            v.savefig(os.path.join(outdir, "{}.dot.png".format(k)),dpi=500)
        endtime(start)
    ksdb_df = pd.read_csv(datafile,header=0,index_col=0,sep='\t')
    ksdb_df = formatv2(ksdb_df)
    df = apply_filters(ksdb_df, [("dS", 0., 5.)])
    ylabel = "Duplications" if spair == () else "Homologous pairs"
    if adjustortho: ylabel = "Homologous pairs (adjusted)"
    if len(spair)!= 0 or not (focus2all is None):
        multi_sp_plot(df,spair,gsmap,outdir,onlyrootout,title=prefix,ylabel=ylabel,viz=True,plotkde=plotkde,reweight=False,sptree=speciestree,ap = anchorpoints, extraparanomeks=extraparanomeks,plotapgmm=plotapgmm,plotelmm=plotelmm,components=components,max_EM_iterations=em_iterations,num_EM_initializations=em_initializations,peak_threshold=prominence_cutoff,rel_height=rel_height, na=nodeaveraged,user_xlim=xlim,user_ylim=ylim,adjustortho=adjustortho,adfactor=adjustfactor,okalpha=okalpha,focus2all=focus2all,clean=classic,toparrow=toparrow,BT=bootstrap,nthreads=nthreads)
        #multi_sp_plot(df,spair,gsmap,outdir,onlyrootout,title=prefix,ylabel=ylabel,viz=True,plotkde=plotkde,reweight=reweight,sptree=speciestree,ap = anchorpoints, extraparanomeks=extraparanomeks,plotapgmm=plotapgmm,plotelmm=plotelmm,components=components,max_EM_iterations=em_iterations,num_EM_initializations=em_initializations,peak_threshold=prominence_cutoff,rel_height=rel_height,user_xlim=xlim,user_ylim=ylim,adjustortho=adjustortho,adfactor=adjustfactor,okalpha=okalpha,focus2all=focus2all,clean=plot2)
    fig = default_plot(df, title=prefix, bins=50, ylabel=ylabel,user_xlim=xlim,user_ylim=ylim)
    fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)))
    fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)))
    plt.close()
    if spair == () and focus2all is None:
        logging.info('Exponential-Lognormal mixture modeling on node-weighted Ks distribution')
        elmm_plot(df,prefix,outdir,max_EM_iterations=em_iterations,num_EM_initializations=em_initializations,peak_threshold=prominence_cutoff,rel_height=rel_height,user_xlim=xlim,user_ylim=ylim)
        logging.info('Exponential-Lognormal mixture modeling on node-averaged Ks distribution')
        elmm_plot(df,prefix,outdir,max_EM_iterations=em_iterations,num_EM_initializations=em_initializations,peak_threshold=prominence_cutoff,na=True,rel_height=rel_height,user_xlim=xlim,user_ylim=ylim)
    endtime(start)

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
@click.option('--additionalgffinfo', '-atg', default=None,multiple=True, show_default=True, help='the feature and attribute info of additional gff3 if different in the format of (feature;attribute)')
@click.option('--minlen', '-ml', default=-1, show_default=True,
    help="minimum length of a genomic element (scaffold) to be included in dotplot")
@click.option('--maxsize', '-ms', default=200, show_default=True,
    help="maximum family size to include in analysis.")
@click.option('--ks_range', '-r', nargs=2, default=(0, 5), show_default=True,
    type=float, help='Ks range in the colored dotplot')
@click.option('--iadhore_options', default="",
    help="other options for I-ADHoRe, as a comma separated string, "
         "e.g. gap_size=30,q_value=0.75,prob_cutoff=0.05")
@click.option('--minseglen', '-mg', default=10000, show_default=True, help="minimum length of segments in ratio if <= 1")
@click.option('--keepredun', '-kr', is_flag=True, help='keep redundant multiplicons')
@click.option('--mingenenum', '-mgn', default=30, type=int, show_default=True, help="minimum number of genes on segments to be considered")
@click.option('--dotsize', '-ds', type=float, default=0.3, show_default=True, help='size of dots')
@click.option('--apalpha', '-aa', type=float, default=1, show_default=True, help='opacity of anchor dots')
@click.option('--hoalpha', '-ha', type=float, default=0, show_default=True, help='opacity of homolog dots')
@click.option('--showrealtick', '-srt', is_flag=True, help='show the real tick in genes or bases')
@click.option('--ticklabelsize', '-tls', type=float, default=10, show_default=True, help='label size of tick')
@click.option('--gistrb', '-gr', is_flag=True, help='whether to use gist_rainbow as color map of dotplot')
@click.option('--nthreads', '-n', default=4, show_default=True,help="number of threads to use")
def syn(**kwargs):
    """
    Co-linearity and anchor inference using I-ADHoRe.
    """
    _syn(**kwargs)

def _syn(families, gff_files, ks_distribution, outdir, feature, attribute,
        minlen, maxsize, ks_range, iadhore_options, minseglen, keepredun, mingenenum, dotsize, apalpha, hoalpha, additionalgffinfo, showrealtick, ticklabelsize, gistrb, nthreads, ancestor=None):
    """
    Co-linearity and anchor inference using I-ADHoRe.
    """
    from wgd.syn import make_gene_table, configure_adhore, run_adhore
    from wgd.syn import get_anchors, get_anchor_ksd, get_segments_profile, get_multi, get_chrom_gene, transformunit, get_mp_geneorder
    from wgd.viz import default_plot, apply_filters, all_dotplots, filter_by_minlength,dotplotunitgene,dotplotingene,filter_mingenumber,dotplotingeneoverall
    from wgd.utils import formatv2
    from wgd.core import endtime
    start = timer()
    # non-default options for I-ADHoRe
    iadhore_opts = {x.split("=")[0].strip(): x.split("=")[1].strip()
               for x in iadhore_options.split(",") if x != ""}
    if len(iadhore_opts) > 0:
        logging.info("I-ADHoRe 3.0 options: {}".format(iadhore_opts))
    iadhore_opts.update({"number_of_threads":nthreads})
    # read families and make table
    prefix = os.path.basename(families)
    fams = pd.read_csv(families, index_col=0, sep="\t")
    table = make_gene_table(gff_files, fams, feature, attribute, additionalgffinfo)
    table_orig = table.copy()
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
    ordered_genes_perchrom_allsp, gene_orders = get_chrom_gene(table,outdir)
    table.to_csv(os.path.join(outdir, "gene-table.csv"))
    logging.info("Running I-ADHoRe")
    run_adhore(conf)

    # general post-processing
    logging.info("Processing I-ADHoRe output")
    #ordered_mp = get_mp_geneorder(gene_orders,out_path,outdir,table)
    anchors,orig_anchors = get_anchors(out_path)
    multi = get_multi(out_path)
    if anchors is None:
        logging.warning("No anchors found, terminating! Please inspect your input files "
                "and the I-ADHoRe results in `{}`".format(out_path))
        endtime(start)
        exit(1)
    anchors.to_csv(os.path.join(outdir, "anchors.csv"))
    #ap_order_permlt = getmltorder(orig_anchors,multi,gene_orders)
    segs = get_segments_profile(multi,keepredun,out_path)
    #segmentpair_order = get_segmentpair_order(orig_anchors,segs,table,gene_orders)
    segs,table,multi,removed_scfa = filter_by_minlength(table,segs,minlen,multi,keepredun,outdir,minseglen)
    segs_gene_unit, gene_order_dict_allsp = transformunit(segs,ordered_genes_perchrom_allsp,outdir)
    segs = filter_mingenumber(segs_gene_unit,mingenenum,outdir,len(gene_order_dict_allsp),start)
    #if ks_distribution: segs_gene_unit_ks = getsegks(segs_gene_unit,ks_distribution,ordered_genes_perchrom_allsp)
    df_ks = None
    if ks_distribution!=None:
        ksdb_df = pd.read_csv(ks_distribution,header=0,index_col=0,sep='\t')
        ksdb_df = formatv2(ksdb_df)
        df_ks = apply_filters(ksdb_df, [("dS", 0., 5.)])
    dotplotingene(ordered_genes_perchrom_allsp,removed_scfa,outdir,table,gene_orders,anchor=anchors,ksdf=df_ks,maxsize=maxsize,dotsize=dotsize,apalpha=apalpha, hoalpha=hoalpha, showrealtick=showrealtick, las = ticklabelsize, gistrb=gistrb)
    if len(gff_files)>1: dotplotingeneoverall(ordered_genes_perchrom_allsp,removed_scfa,outdir,table,gene_orders,anchor=anchors,ksdf=df_ks,maxsize=maxsize,dotsize=dotsize, apalpha=apalpha, hoalpha=hoalpha, showrealtick=showrealtick, las = ticklabelsize, gistrb=gistrb)
    #dotplotunitgene(ordered_genes_perchrom_allsp,segs_gene_unit,removed_scfa,outdir,mingenenum,table_orig,ordered_mp,ksdf=df_ks)
    # dotplot
    #logging.info("Generating dot plots")
    figs = all_dotplots(table, segs, multi, minseglen, anchors=anchors, maxsize=maxsize, minlen=minlen, outdir=outdir, ancestor=ancestor, Ks = df_ks, dotsize=dotsize, apalpha=apalpha, hoalpha=hoalpha, las=ticklabelsize, gistrb=gistrb) 
    for k, v in figs.items():
        v.savefig(os.path.join(outdir, "{}.dot.svg".format(k)))
        v.savefig(os.path.join(outdir, "{}.dot.pdf".format(k)))
        v.savefig(os.path.join(outdir, "{}.dot.png".format(k)),dpi=500)
    plt.close()

    # anchor Ks distributions
    if ks_distribution:
        ylabel = "Duplications"
        if len(gff_files) == 2:
            ylabel = "RBH orthologs"
        ksd = pd.read_csv(ks_distribution, sep="\t", index_col=0)
        ksd = formatv2(ksd)
        anchor_ks = get_anchor_ksd(ksd, anchors)
        anchor_ks.to_csv(os.path.join(outdir, "{}.anchors.ks.tsv".format(prefix)),sep='\t')
        a = apply_filters(ksd,       [("dS", 0, 5.)])
        b = apply_filters(anchor_ks, [("dS", 0, 5.)])
        logging.info("Generating anchor Ks distribution")
        fig = default_plot(a, b, title=prefix, bins=50, ylabel=ylabel)
        fig.savefig(os.path.join(outdir, "{}.ksd.svg".format(prefix)),dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(outdir, "{}.ksd.pdf".format(prefix)),dpi=300, bbox_inches='tight')
    endtime(start)

# MIXTURE MODELING
@cli.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('ks_distribution', type=click.Path(exists=True), default=None)
@click.option('--filters', '-f', type=int, default=300, help="Alignment length", show_default=True)
@click.option('--ks_range', '-r', nargs=2, default=(0, 5), show_default=True, type=float, help='Ks range to use for modeling')
@click.option('--bins', '-b', default=50, show_default=True, type=int, help="Number of histogram bins")
@click.option('--outdir', '-o', default="wgd_mix", show_default=True, help='output directory')
@click.option('--method', type=click.Choice(['gmm', 'bgmm']), default='gmm', show_default=True, help="mixture modeling method")
@click.option('--components', '-n', nargs=2, default=(1, 4), show_default=True, help='range of number of components to fit')
@click.option('--gamma', '-g', default=1e-3, show_default=True, help='gamma parameter for bgmm models')
@click.option('--n_init', '-ni', default=200, show_default=True, help='number of k-means initializations')
@click.option('--max_iter', '-mi', default=200, show_default=True, help='maximum number of iterations')
def mix(**kwargs):
    """
    Mixture modeling of Ks distributions.
    Basic function
    """
    _mix(**kwargs)
def _mix(ks_distribution, filters, ks_range, method, components, bins, outdir, gamma, n_init, max_iter):
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
    :param outdir: output directory
    :param gamma: gamma parameter for BGMM
    :param n_init: number of k-means initializations (best is kept)
    :param max_iter: number of iterations
    :return: nada
    """
    from wgd.mix import filter_group_data, get_array_for_mixture, fit_gmm
    from wgd.mix import inspect_aic, inspect_bic, plot_aic_bic
    from wgd.mix import plot_all_models_gmm, get_component_probabilities
    from wgd.mix import fit_bgmm,plot_all_models_bgmm 
    from wgd.core import endtime
    start = timer()
    # make output dir if needed
    if not os.path.exists(outdir):
        logging.info("Making directory {}".format(outdir))
        os.mkdir(outdir)
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
                     os.path.join(outdir, "aic_bic.svg"))
        logging.info("Plotting mixtures")
        plot_all_models_gmm(models, X, ks_range[0], ks_range[1], bins=bins,
                            out_file=os.path.join(outdir, "gmms.svg"))

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
                             out_file=os.path.join(outdir, "bgmms.svg"))
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
            outdir, "ks_{}.tsv".format(method)), sep="\t")
    endtime(start)
#@cli.command(context_settings={'help_option_names': ['-h', '--help']})
#@click.argument('config', type=click.Path(exists=True))
#def tree(**kwargs):
#    """
#    Gene tree reconstruction and reconciliation still in test
#    """
#    _tree(**kwargs)
#
#def _tree(config):
#    from wgd.tree import Config_Hauler
#    start = timer()
#    Config_Hauler(config)
#    end = timer()
#    logging.info("Total run time: {} min".format(round((end-start)/60,2)))


if __name__ == '__main__':
    cli()

