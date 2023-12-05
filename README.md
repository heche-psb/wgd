<div align="center">

# `wgd v2` : a suite tool of WGD inference and timing

[![Build Status](https://app.travis-ci.com/heche-psb/wgd.svg?branch=phylodating)](https://app.travis-ci.com/heche-psb/wgd)
[![Documentation Status](https://readthedocs.org/projects/wgdv2/badge/?version=latest)](https://wgdv2.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/pypi/l/wgd.svg)](https://pypi.python.org/pypi/wgd)
[![Latest PyPI version](https://img.shields.io/pypi/v/wgd.svg)](https://pypi.python.org/pypi/wgd)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/wgd/README.html)
[![Downloads](https://pepy.tech/badge/wgd)](https://pepy.tech/project/wgd)

**Hengchi Chen, Arthur Zwaenepoel, Yves Van de Peer**

[**Bioinformatics & Evolutionary Genomics Group**](https://www.vandepeerlab.org/)**, VIB-UGent Center for Plant Systems Biology**

[**Introduction**](#introduction) | 
[**Installation**](#installation) | 
[**Parameters**](#parameters) | 
[**Usage**](#usage) | 
[**Illustration**](#illustration) |
[**Documentation**](https://wgdv2.readthedocs.io/en/latest/?badge=latest) |
[**Citation**](#citation)

</div>

`wgd v2` is a python package upgraded from the original `wgd` package aiming for the inference and timing of ancient whole-genome duplication (WGD) events. For the propose of illustrating the principle and usage of `wgd v2`, we compiled this documentation. Below we first give an introduction over the scope and mechanism of `wgd v2` and then the practical information of installation and usage. An examplar workflow is provided in the tutorial section on how to seek evidence for a putative WGD event and perform proper timing with a freshly obtained genome assembly in hand. For those who are interested, we recommend turning to our paper and book chapter for more detailed description and insightful discussions. If you use `wgd v2` in your research, please cite us. 

## Introduction

Polyploidizations, the evolutionary process that the entire genome of an organism is duplicated, also named as whole-genome duplications (WGDs), occur recurrently across the tree of life. There are two modes of polyploidizations, autopolyploidizations and allopolyploidizations. Autopolyploidizations are the duplication of the same genome, resulting in two identical subgenomes at the time it emerged. While the allopolyploidizations are normally achieved in two steps, first the hybridization between two different species, resulting in the arising of transient homoploidy,second the duplication of the homoploidy, resulting in the emergence of allopolyploidy. Due to the unstability and unbalanced tetrasomic inheritance, for instance the nuclear-cytoplasmic incompatibility, the polyploidy genome will then experience a process called diploidization, also named as fractionation, during which a large portion of gene duplicates will get lost and only a fraction can be retained. The traces of polyploidizations can be thus unearthed from these retained gene duplicates. Three approaches based on gene duplicates, namely, *K*<sub>S</sub> method, gene tree - species tree reconciliation method and synteny method, are commonly used in detecting evidence for WGDs. The gene tree - species tree reconciliation method is not within the scope of `wgd v2`, but we kindly refer readers who are interested to the phylogenomic program developed by Arthur Zwaenepoel named [WHALE](https://github.com/arzwa/Whale.jl) and the associated [paper](https://doi.org/10.1093/molbev/msz088) for more technical and theoretical details.

The *K*<sub>S</sub> method is established on a model of gene family evolution that each gene family is allowed to evolve via gene duplication and loss. Note that the gene family here is assumed to be the cluster of all genes descended from an ancestral gene in a single genome. Recovering the gene tree of such gene family informs the timing, scilicet the age, of gene duplication events. The age refered here, is not in real geological time, but in the unit of evolutionary distance, i.e., the number of substitutions per site. When the evolutionary rate remains approximately constant, the evolutionary distance is then supposed to be proportional to the real evolutionary time. The synonymous distance *K*<sub>S</sub>, the number of synonymous substitutions per synonymous site, is such candidate that synonymous substitutions would not incur the change of amino acid and are thus regarded as neutral, which according to the neutral theory should occur in constant rate. Given a model of gene family that allows the gene to duplicate and get lost in a fixed rate, one can derive that the probability density function of the *K*<sub>S</sub> age distribution of retained gene duplicates is a quasi-exponential function that most retained gene duplicates are recently borned with ~0 age while as the age going older the associated number of retained gene duplicates decay quasi-exponentially. Therefore, the occurance of large-scale gene duplication events, for instane WGDs, with varied retention rate, will leave an age peak from the burst of gene duplicates in a short time-frame upon the initial age distribution, and can be unveiled from mixture modeling analysis. However, WGDs identified from the paralogous *K*<sub>S</sub> age distributions can only inform the WGD timing in the time-scale of that specific species, which is not comparable in the phylogenetic context. Only with the orthologous *K*<sub>S</sub> age distributions, which convert the estimated body from paralogues to orthologues and inform the relative timing of speciation events, can we decipher the phylogenetic placement of WGDs after proper rate correction. `wgd v2` is such program that helps users construct paralogous and orthologous *K*<sub>S</sub> age distributions and realize both the identification and placement of WGDs.

## Installation

The easiest way to install `wgd v2` is using PYPI. Note that if you want to get the latest update, we suggest installing from the source, since the update on PYPI will be delayed compared to here in the source.

```
pip install wgd
```

To install `wgd v2` in a virtual environment, the following command lines could be used.

```
git clone <wgd repo>
cd wgd
virtualenv -p=python3 ENV (or python3 -m venv ENV)
source ENV/bin/activate
pip install -r requirements.txt
pip install .
```

When met with permission problem in installation, please try the following command line.

```
pip install -e .
```

If multiply versions of `wgd` were installed in the system, please add the right path of interested version into the environment variables, for example

```
export PATH="$PATH:~/.local/bin/wgd"
```

Note that the version of `numpy` is important (for many other packages are the same of course), especially for `fastcluster` package. In our test, the `numpy` 1.19.0 works fine on `python3.6/8`. If you met some errors or warnings about numpy, maybe considering pre-install `numpy` as 1.19.0 or other close-by versions before you install `wgd`.

## Parameters

There are 7 main programs in `wgd v2`: `dmd`,`focus`,`ksd`,`mix`,`peak`,`syn`,`viz`. Hereafter we will provide a detailed elucidation on each of the program and its associated parameters. Please refer to the [Usage](#usage) for the scenarios to which each parameter applies.

The program `wgd dmd` can realize the delineation of whole paranome, RBHs (Reciprocal Best Hits), MRBHs (Multiple Reciprocal Best Hits), orthogroups and some other orthogroup-related functions, including circumscription of nested single-copy orthogroups (NSOGs), unbiased uest of single-copy orthogroups (SOGs) over missing inparalogs, construction of BUSCO-guided single-copy orthogroups (SOGs),and the collinear coalescence inference of phylogeny.
```
wgd dmd sequences (option)
--------------------------------------------------------------------------------
-o, --outdir, the output directory, default wgd_dmd
-t, --tmpdir, the temporary working directory, default None, if None was given, the tmpdir will be assigned random names in current directory and automately removed at the completion of program, else the tmpdir will be kept
-c, --cscore, the c-score to restrict the homologs of MRBHs, default None, if None was given, the c-score funcion won't be activated
-I, --inflation, the inflation factor for MCL program, default 2.0
-e, --eval, the e-value cut-off for similarity in diamond and/or hmmer, default 1e-10
--to_stop, flag option, whether to translate through STOP codons, if the flag was set, translation will be terminated at the first in-frame stop codon, else a full translation continuing on passing any stop codons will be initiated
--cds, flag option, whether to only translate the complete CDS that starts with a valid start codon and only contains a single in-frame stop codon at the end and must be dividable by three, if the flag was set, only the complete CDS will be translated
-f, --focus, the species to be merged on local MRBHs, default None, if None was given, the local MRBHs won't be inferred
-ap, --anchorpoints, the anchor points data file, default None
-coc, --collinearcoalescence, flag option, whether to initiate the collinear coalescence analysis, if the flag was set, the analysis will be initiated
-sm, --segments, the segments data file used in collinear coalescence analysis if initiated, default None
-le, --listelements, the listsegments data file used in collinear coalescence analysis if initiated, default None
-gt, --genetable, the gene table datafile used in collinear coalescence analysis if initiated, default None
-kf, --keepfasta, flag option, whether to output the sequence information of MRBHs, if the flag was set, the sequences of MRBHs will be in output
-kd, --keepduplicates, flag option, whether to allow the same gene to occur in different MRBHs, if the flag was set, the same gene can be assigned to different MRBHs
-gm, --globalmrbh, flag option, whether to initiate global MRBHs construction, if the flag was set, the --focus option will be ignored and only global MRBHs will be built
-n, --nthreads, the number of threads to use, default 4
-oi, --orthoinfer, flag option, whether to initiate orthogroup infernece, if the flag was set, the orthogroup infernece program will be initiated
-oo, --onlyortho, flag option, whether to only conduct orthogroup infernece, if the flag was set, only the orthogroup infernece program will be conducted while the other analysis won't be initiated
-gn, --getnsog, flag option, whether to initiate the searching for nested single-copy gene families (NSOGs), if the flag was set, additional NSOGs analysis will be performed besides the basic orthogroup infernece
-tree, --tree_method, which gene tree inference program to invoke, default fasttree
-ts, --treeset, the parameters setting for gene tree inference, default None, this option can be provided multiple times
-mc, --msogcut, the ratio cutoff for mostly single-copy family and species representation in collinear coalescence inference, default 0.8.
-ga, --geneassign, flag option, whether to initiate the gene-to-family assignment analysis, if the flag was set, the analysis will be initiated
-am, --assign_method, which method to conduct the gene-to-family assignment analysis, default hmmer
-sa, --seq2assign, the queried sequences data file in gene-to-family assignment analysis, default None, this option can be provided multiple times
-fa, --fam2assign, the queried familiy data file in gene-to-family assignment analysis, default None
-cc, --concat, flag option, whether to initiate the concatenation pipeline for orthogroup infernece, if the flag was set, the analysis will be initiated
-te, --testsog, flag option, whether to initiate the unbiased test of single-copy gene families, if the flag was set, the analysis will be initiated
-bs, --bins, the number of bins divided in gene length normalization, default 100
-np, --normalizedpercent, the percentage of upper hits used for gene length normalization, default 5
-nn, --nonormalization, flag option, whether to call off the normalization, if the flag was set, no normalization will be conducted
-bsog, --buscosog, flag option, whether to initiate the busco-guided single-copy gene family analysis, if the flag was set, the analysis will be initiated
-bhmm, --buscohmm, the HMM profile datafile in the busco-guided single-copy gene family analysis, default None
-bctf, --buscocutoff, the HMM score cutoff datafile in the busco-guided single-copy gene family analysis, default None
```

The program `wgd focus` can realize the concatenation-based and coalescence-based phylogenetic inference, functional annotation of gene families and phylogenetic dating of WGDs.
```
wgd focus families sequences (option)
--------------------------------------------------------------------------------
-o, --outdir, the output directory, default wgd_focus
-t, --tmpdir, the temporary working directory, default None, if None was given, the tmpdir will be assigned random names in current directory and automately removed at the completion of program, else the tmpdir will be kept
-n, --nthreads, the number of threads to use, default 4
--to_stop, flag option, whether to translate through STOP codons, if the flag was set, translation will be terminated at the first in-frame stop codon, else a full translation continuing on past any stop codons will be initiated
--cds, flag option, whether to only translate the complete CDS that starts with a valid start codon and only contains a single in-frame stop codon at the end and must be dividable by three, if the flag was set, only the complete CDS will be translated
--strip_gaps, flag option, whether to drop all gaps in multiple sequence alignment, if the flag was set, all gaps will be dropped
-a, --aligner, which alignment program to use, default mafft
-tree, --tree_method, which gene tree inference program to invoke, default fasttree
-ts, --treeset, the parameters setting for gene tree inference, default None, this option can be provided multiple times
--concatenation, flag option, whether to initiate the concatenation-based species tree inference, if the flag was set, concatenation-based species tree will be infered
--coalescence, flag option, whether to initiate the coalescence-based species tree inference, if the flag was set, coalescence-based species tree will be infered
-sp, --speciestree, species tree datafile for dating, default None
-d, --dating, which molecular dating program to use, default none
-ds, --datingset, the parameters setting for dating program, default None, this option can be provided multiple times
-ns, --nsites, the nsites information for r8s dating, default None
-ot, --outgroup, the outgroup information for r8s dating, default None
-pt, --partition, flag option, whether to initiate partition dating analysis for codon, if the flag was set, an additional partition dating analysis will be initiated
-am, --aamodel, which protein model to be used in mcmctree, default poisson
-ks, flag option, whether to initiate Ks calculation
--annotation, which annotation program to use, default None
--pairwise, flag option, whether to initiate pairwise Ks estimation, if the flag was set, pairwise Ks values will be estimated
-ed, --eggnogdata, the eggnog annotation datafile, default None
--pfam, which option to use for pfam annotation, default None
--dmnb, the diamond database for annotation, default None
--hmm, the HMM profile for annotation, default None
--evalue, the e-value cut-off for annotation, default 1e-10
--exepath, the path to the interproscan executable, default None
-f, --fossil, the fossil calibration information in Beast, default ('clade1;clade2', 'taxa1,taxa2;taxa3,taxa4', '4;5', '0.5;0.6', '400;500')
 -rh, --rootheight, the root height calibration info in Beast, default (4,0.5,400)
-cs, --chainset, the parameters of MCMC chain in Beast, default (10000,100)
--beastlgjar, the path to beastLG.jar, default None
--beagle, flag option, whether to use beagle in Beast, if the flag was set, beagle will be used
--protdating, flag option, whether to only initiate the protein-concatenation based dating analysis, if the flag was set, the analysis will be initiated
```

The program `wgd ksd` can realize the construction of *K*<sub>S</sub> age distribution and rate correction.
```
wgd ksd families sequences (option)
--------------------------------------------------------------------------------
-o, --outdir, the output directory, default wgd_ksd
-t, --tmpdir, the temporary working directory, default None, if None was given, the tmpdir will be assigned random names in current directory and automately removed at the completion of program, else the tmpdir will be kept
-n, --nthreads, the number of threads to use, default 4
--to_stop, flag option, whether to translate through STOP codons, if the flag was set, translation will be terminated at the first in-frame stop codon, else a full translation continuing on past any stop codons will be initiated
--cds, flag option, whether to only translate the complete CDS that starts with a valid start codon and only contains a single in-frame stop codon at the end and must be dividable by three, if the flag was set, only the complete CDS will be translated
--pairwise, flag option, whether to initiate pairwise Ks estimation, if the flag was set, pairwise Ks values will be estimated
--strip_gaps, flag option, whether to drop all gaps in multiple sequence alignment, if the flag was set, all gaps will be dropped
-tree, --tree_method, which gene tree inference program to invoke, default fasttree
-sr, --spair, the species pair to be plotted, default None, this option can be provided multiple times
-sp, --speciestree, the species tree to perform rate correction, default None, if None was given, the rate correction analysis will be called off
-rw, --reweight, flag option, whether to recalculate the weight per species pair, if the flag was set, the weight will be recalculated
-or, --onlyrootout, flag option, whether to only conduct rate correction using the outgroup at root as outgroup, if the flag was set, only the outgroup at root will be used as outgroup
-epk, --extraparanomeks, extra paranome Ks data to plot in the mixed Ks distribution, default None
-ap, --anchorpoints, anchorpoints.txt file to plot anchor Ks in the mixed Ks distribution, default None
-pk, --plotkde, flag option, whether to plot kde curve of orthologous Ks distribution over histogram in the mixed Ks distribution, if the flag was set, the kde curve will be plotted
-pag, --plotapgmm, flag option, whether to plot mixture modeling of anchor Ks in the mixed Ks distribution, if the flag was set, the mixture modeling of anchor Ks will be plotted
-pem, --plotelmm, flag option, whether to plot elmm mixture modeling of paranome Ks in the mixed Ks distribution, if the flag was set, the elmm mixture modeling of paranome Ks will be plotted
-c, --components, the range of the number of components to fit in anchor Ks mixture modeling, default (1,4)
```

The program `wgd mix` can realize the mixture model clustering analysis of *K*<sub>S</sub> age distribution.
```
wgd mix ks_datafile (option)
--------------------------------------------------------------------------------
-f, --filters, the cutoff alignment length, default 300
-r, --ks_range, the Ks range to be considered, default (0, 5)
-b, --bins, the number of bins in Ks distribution, default 50
-o, --outdir, the output directory, default wgd_mix
--method, which mixture model to use, default gmm
-n, --components, the range of the number of components to fit, default (1, 4)
-g, --gamma, the gamma parameter for bgmm models, default 0.001
-ni, --n_init, the number of k-means initializations, default 200
-mi, --max_iter, the maximum number of iterations, default 1000
```

The program `wgd peak` can realize the search of crediable *K*<sub>S</sub> range used in WGD dating.
```
wgd peak ks_datafile (option)
--------------------------------------------------------------------------------
-ap, --anchorpoints, the anchor points datafile, default None
-sm, --segments, the segments datafile, default None
-le, --listelements, the listsegments datafile, default None 
-mp, --multipliconpairs, the multipliconpairs datafile, default None
-o, --outdir, the output directory, default wgd_peak
-af, --alignfilter, cutoff for alignment identity, length and coverage, default 0.0, 0, 0.0
-r, --ksrange, range of Ks to be analyzed, default (0, 5)
-bw, --bin_width, bandwidth of Ks distribution, default 0.1
-ic, --weights_outliers_included, flag option, whether to include Ks outliers, if the flag was set, Ks outliers will be included in the analysis
-m, --method, which mixture model to use, default gmm
--seed, random seed given to initialization, default 2352890
-ei, --em_iter, the number of EM iterations to perform, default 200
-ni, --n_init, the number of k-means initializations, default 200
-n, --components, the range of the number of components to fit, default (1, 4)
--boots, the number of bootstrap replicates of kde, default 200
--weighted, flag option, whether to use node-weighted method of de-redundancy, if the flag was set, the node-weighted method will be used
-p, --plot, the plotting method to be used, default identical
-bm, --bw_method, the bandwidth method to be used, default silverman
--n_medoids, the number of medoids to fit, default 2
-km, --kdemethod, the kde method to be used, default scipy
--n_clusters, the number of clusters to plot Elbow loss function, default 5
--kmedoids, flag option, whether to initiate K-Medoids clustering analysis, if the flag was set, the analysis will be initiated
-gd, --guide, the regime residing anchors, default: segment
-prct, --prominence_cutoff, the prominence cutoff of acceptable peaks, default 0.1
-kd, --kstodate, the range of Ks to be dated in heuristic search, default (0.5, 1.5)
-f, --family, the family to filter Ks upon, default None
--manualset, flag option, whether to output anchor pairs with manually set Ks range, if the flag was set, manually set Ks range will be used
-rh, --rel_height, the relative height at which the peak width is measured, default 0.4
--ci, the confidence level of log-normal distribution to date, default 95
--hdr, the highest density region (HDR) in a given distribution to date, default 95
--heuristic, flag option, whether to initiate heuristic method of defining CI for dating, if the flag was set, the heuristic method will be initiated
-kc, --kscutoff, the Ks saturation cutoff in dating, default 5
-g, --gamma, the gamma parameter for bgmm models, default 1e-3
```

The program `wgd syn` can realize the intra- and inter-specific synteny inference.
```
wgd syn families gffs (option)
--------------------------------------------------------------------------------
-ks, --ks_distribution, ks distribution datafile, default None
-o, --outdir, the output directory, default wgd_syn
-f, --feature, the feature for parsing gene IDs from GFF files, default gene
-a, --attribute, the attribute for parsing the gene IDs from the GFF files, default ID
-ml, --minlen, the minimum length of a scaffold to be included in dotplot, default -1, if -1 was set, the 10% of the longest scaffold will be set
-ms, --maxsize, the maximum family size to include, default 200
-r, --ks_range, the Ks range in colored dotplot, default (0, 5)
--iadhore_options, the parameter setting in iadhore, default 
-ac, --ancestor, the assumed ancestor species, default None, a option that is still in development
-mg, --minseglen, the minimum length of segments to include in ratio if <= 1, default 100000
-kr, --keepredun, flag option, whether to keep redundant multiplicons, if the flag was set, the redundant multiplicons will be kept
-mgn, --mingenenum, the minimum number of genes for a segment to be considered, default 30
-ds, --dotsize, the dot size in dot plot, default 1
-aa, --apalpha, the opacity of anchor dots, default 1
-ha, --hoalpha, the opacity of homolog dots, default 0.1
```

The program `wgd viz` can realize the visualization of *K*<sub>S</sub> age distribution and synteny.
```
wgd viz (option)
--------------------------------------------------------------------------------
-d, --datafile, the Ks datafile, default None
-o, --outdir, the output directory, default wgd_viz
-sr, --spair, the species pair to be plotted, default None, this option can be provided multiple times
-gs, --gsmap, the gene name-species name map, default None
-sp, --speciestree, the species tree to perform rate correction, default None, if None was given, the rate correction analysis will be called off
-pk, --plotkde, flag option, whether to plot kde curve upon histogram, if the flag was set, kde curve will be added
-rw, --reweight, flag option, whether to recalculate the weight per species pair, if the flag was set, the weight will be recalculated
-or, --onlyrootout, flag option, whether to only conduct rate correction using the outgroup at root as outgroup, if the flag was set, only the outgroup at root will be used as outgroup
-iter, --em_iterations, the maximum EM iterations, default 200
-init, --em_initializations, the maximum EM initializations, default 200
-prct, --prominence_cutoff, the prominence cutoff of acceptable peaks, default 0.1
-sm, --segments, the segments data file, default None
-ml, --minlen, the minimum length of a scaffold to be included in dotplot, default -1, if -1 was set, the 10% of the longest scaffolds will be set
-ms, --maxsize, the maximum family size to include, default 200
-ap, --anchorpoints, the anchor points datafile, default None
-mt, --multiplicon, the multiplicons datafile, default None
-gt, --genetable, the gene table datafile, default None
-rh, --rel_height, the relative height at which the peak width is measured, default 0.4
-mg, --minseglen, the minimum length of segments to include in ratio if <= 1, default 100000
-kr, --keepredun, flag option, whether to keep redundant multiplicons, if the flag was set, the redundant multiplicons will be kept
-epk, --extraparanomeks, extra paranome Ks data to plot in the mixed Ks distribution, default None
-pag, --plotapgmm, flag option, whether to plot mixture modeling of anchor Ks in the mixed Ks distribution, if the flag was set, the mixture modeling of anchor Ks will be plotted
-pem, --plotelmm, flag option, whether to plot elmm mixture modeling of paranome Ks in the mixed Ks distribution, if the flag was set, the elmm mixture modeling of paranome Ks will be plotted
-c, --components, the range of the number of components to fit in anchor Ks mixture modeling, default (1,4)
-mgn, --mingenenum, the minimum number of genes for a segment to be considered, default 30
-psy, --plotsyn, flag option, whether to initiate the synteny plot, if the flag was set, the synteny plot will be produced
-ds, --dotsize, the dot size in dot plot, default 1
-aa, --apalpha, the opacity of anchor dots, default 1
-ha, --hoalpha, the opacity of homolog dots, default 0.1
```

## Usage

Here we provided the basic usage for each program and the relevant parameters.

### wgd dmd

**The delineation of whole paranome**
```
wgd dmd Aquilegia_coerulea -I 2 -e 1e-10 -bs 100 -np 5 (-nn) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
``` 

Note that we don't provide the data of this cds file `Aquilegia_coerulea` but it can be downloaded at [Phytozome](https://phytozome-next.jgi.doe.gov/info/Acoerulea_v3_1) (same for other Usage doc). In principle, only the primary transcript per gene of genome data should be used. Transcriptome data should be carefully treated with de-redundancy so as to reduce the false positive duplication bump caused by massive alternative splicing. Here the inflation factor parameter, given by `-I` or `--inflation`, affects the granularity or resolution of the clustering outcome and implicitly controlls the number of clusters, with low values such as 1.3 or 1.4 leading to fewer and larger clusters and high values such as 5 or 6 leading to more and smaller clusters. We set the default value as 2 as suggested by [MCL](https://micans.org/mcl/). The e-value cut-off for sequence similarity, given by `-e` or `--eval`, which denotes the expected value of the hit quantifies the number of alignments of similar or better quality that you expect to find searching this query against a database of random sequences the same size as the actual target database, is the key parameter measuring the significance of a hit, is set here as default 1e-10. Note that [DIAMOND](https://github.com/bbuchfink/diamond/wiki/1.-Tutorial) will by default only report all alignments with e-value < 0.001. The percentage of upper hits used for gene length normalization, given by `-np` or `--normalizedpercent`, which determines the upper percentile of hits per bin (categorized by gene length) used in the fit of linear regression, considering that not all hits per bin show apparent linear relationship, is set as default 5, indicating the usage of top 5% hits per bin. The number of bins divided in gene length normalization, given by `-bs` or `--bins`, determines the number of bins to categorize the gene length, is set as default 100. The parameter `-nn` or `--nonormalization` can be set to call off the normalization process, although it's suggested to conduct the normalization to acquire more accurate gene family clustering result. The parameters `--to_stop` and `--cds` control the behaviour of translating coding sequence into amino acid sequence. If the `--to_stop` was set, the translation would be terminated at the first in-frame stop codon, otherwise the translation would simply skip any stop codons. If the `--cds` was set, sequences that doesn't start with a valid start codon, or contains more than 1 in-frame stop codon, or is not dividable by 3, would be simply dropped, such that only strict complete coding sequences would be included in the subsequent analysis. The directory of output or intermediate files is determined by the parameter `-o` or `--outdir`, and `-t` or `--tmpdir`, which will be created by the program itself and be overwritten if the folder has already been created. Note that the software `diamond` should be installed and set in the environment path in all the analysis performed by `wgd dmd` except for the final collinear coalescence analysis.

**The delineation of RBHs**
```
wgd dmd sequence1 sequence2 -e 1e-10 -bs 100 -np 5 (-nn) (-n 4) (-c 0.9) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

To delineate RBHs between two cds sequence files, the relevant parameter is mostly the same as whole paranome inference, except for the parameter `-c` or `--cscore`, which is ranging from 0 to 1 and used to relax the similarity cutoff from only reciprocal best hits to a certain ratio as to the best hits. For instance, if the gene b1 from genome B has the best hit gene a1 from genome A with the bit score as 100, which is a scoring matrix independent measure of the (local) similarity of the two aligned sequences, with higher numbers meaning more similar, given the `-c 0.9`, genes from genome A which has the bit score with gene b1 higher than 0.9x100 will all be written in the result file, which in a sense is not RBH anymore of course, but the highly similar homologue pairs. If more than 2 sequence files were provided, every pair-wise RBHs would be calculated except for querying the same sequence itself. The number of parallel threads to booster the running speed can be set by `-n` or `--nthreads`.

**The delineation of local MRBHs**
```
wgd dmd sequence1 sequence2 sequence3 -f sequence1 -e 1e-10 -bs 100 -np 5 (-nn) (-n 4) (-kf) (-kd) (-c 0.9) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

Two types of MRBHs can be delineated by `wgd dmd`, the local MRBHs and the global MRBHs. The local MRBHs is constructed by merging all the relevant RBHs only with the focus species, which is set by `-f` or `--focus`. For instance, given 3 genome (A,B,C) and focus species C, two RBHs would be calculated, i.e., (A.vs.C) and (B.vs.C), and then the two RBHs tables would be merged on the species C to acquire the local MRBHs of C. The parameter `-kf` or `--keepfasta` can be set to write the sequence information of each MRBHs. The parameter `-kd` or `--keepduplicates` determines whether the same genes can appear in different local MRBHs. Normally there will be no duplicates in the local MRBHs but if users set the `-c` as 0.9 (for instance), it's likely that the same gene can appear multiply times in different local MRBHs. That is to say, the parameter `-kd` is meaningful only when it's set together with the parameter `-c`.

**The delineation of global MRBHs**
```
wgd dmd sequence1 sequence2 sequence3 -gm -e 1e-10 -bs 100 -np 5 (-nn) (-n 4) (-kf) (-kd) (-c 0.9) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

The global MRBHs is constructed by exhaustively merging all the possible pair-wise RBHs except for querying the same sequence itself, which can be initiated by add the flag `-gm` or `--globalmrbh`. For instance, given 3 genome (A,B,C), three RBHs would be calculated, i.e., (A.vs.C), (A.vs.B) and (B.vs.C), and then they would be merged progressively and exhaustively. The rest of relevant parameters stays the same as the local MRBHs.

**The delineation of orthogroups**
```
wgd dmd sequence1 sequence2 sequence3 -oi -oo -e 1e-10 -bs 100 -np 5 (-nn) (-cc) (-te) (-mc 0.8) (-gn) (-tree 'fasttree') (-ts '-fastest') (-n 4) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

In `wgd v2`, we also implemented an algorithm of delineating orthogroups, which can be initiated with the parameter `-oi` or `--orthoinfer`. Two ways of delineation can be chosen, the concatenation way (set by the parameter `-cc` or `--concat`) or the non-concatenation (default) way. In brief, the concatenation way of delineating orthogroups starts with concatenating all the sequences into a single sequence file and then inferring the whole paranome of this single sequence file with the clustering results mapped back to the belonging species. While the non-concatenation way starts with respective pair-wise diamond search (including querying the same sequence itself) and then all the sequence similarity tables will be added up and clustered into orthogroups. Some other possibly useful post-clustering functions can be initiated, including the parameter `-te` or `--testsog`, which can be set to start the unbiased test of single-copy gene families (note that this function needs `hmmer` (v3.1b2) to be installed in the environment path), the parameter `-mc` or `--msogcut`, ranging from 0 to 1, which can be set to search the so-called mostly single-copy family which has higher than certain cut-off percentage of species coverage, the parameter `-gn` or `--getnsog`, which can be set to search for nested single-copy gene families (NSOGs) which is originally multiy-copy but has a (mostly) single-copy branch (which requires the chosen tree-inference program set by `-tree` or `--tree_method` to be pre-installed in the environment path with the parameters setting for gene tree inference controlled by `-ts` or `--treeset`). The program `wgd dmd` will still conduct the RBHs calculation unless the parameter `-oo` or `--onlyortho` was set. If one only wants to infer the orthogroups, it's suggested to add the flag `-oo` to just implement the orthogroups delineation analysis.

**The collinear coalescence inference of phylogeny**
```
wgd dmd sequence1 sequence2 sequence3 -ap apdata -sm smdata -le ledata -gt gtdata -coc (-tree 'fasttree') (-ts '-fastest') (-n 4) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

A novel phylogenetic inference method named "collinear coalescence inference" is also implemented in `wgd v2`. For this analysis, users need to provide the anchor points file by `-ap` or `--anchorpoints`, the collinear segment file by `-sm` or `--segments`, the listsegments file by `-le` or `--listelements`, and the gene table file by `-gt` or `--genetable`, all of which can be produced in the program `wgd syn`. The parameter `-coc` or `--collinearcoalescence` needs to be set for starting this analysis. The tree-inference program and the associated parameters can be set just as above by `-tree` or `--tree_method` and `-ts` or `--treeset`. Please also make sure the chosen tree-inference program is installed in the environment path. The program `astral-pro` is required to be installed in the environment path too. Note that there should be no duplicated gene IDs in the sequence file.

### wgd focus

**The concatenation-based/coalescence-based phylogenetic inference**
```
wgd focus families sequence1 sequence2 sequence3 (--concatenation) (--coalescence) (-tree 'fasttree') (-ts '-fastest') (-n 4) (--to_stop) (--cds) (-o wgd_focus) (-t working_tmp)
```

The program `wgd focus` implemented two basic phylogenetic inference methods, i.e., concatenation-based and coalescence-based methods. To 

**The functional annotation of gene families**
```
wgd focus families sequence1 sequence2 sequence3 --annotation eggnog -ed eddata --dmnb dbdata
```

**The phylogenetic dating of WGDs**
```
wgd focus families sequence1 sequence2 sequence3 -d mcmctree -sp spdata
```

### wgd ksd

**The construction of whole paranome *K*<sub>S</sub> age distribution**
```
wgd ksd families sequence
```

**The construction of orthologous *K*<sub>S</sub> age distribution**
```
wgd ksd families sequence1 sequence2
```

**The construction of *K*<sub>S</sub> age distribution with rate correction**
```
wgd ksd families sequence1 sequence2 sequence3 -sr srdata -sp spdata
```

### wgd mix

**The mixture model clustering analysis of *K*<sub>S</sub> age distribution**
```
wgd mix ksdata
```

### wgd peak

**The search of crediable *K*<sub>S</sub> range used in WGD dating**
```
wgd peak ksdata -ap apdata -sm smdata -le ledata -mp mpdata
```
Note that users can add the flag --heuristic to implement the heuristic search analysis

### wgd syn

**The intra-specific synteny inference**
```
wgd syn families gff
```

**The inter-specific synteny inference**
```
wgd syn families gff1 gff2
```

### wgd viz

**The visualization of *K*<sub>S</sub> age distribution**
```
wgd viz -d ksdata
```

**The visualization of *K*<sub>S</sub> age distribution with rate correction**
```
wgd viz -d ksdata -sr srdata -sp spdata -gs gsdata
```

**The visualization of synteny**
```
wgd viz -ap apdata -sm smdata -mt mtdata -gt gtdata
```

## Illustration

We illustrate our program on an exemplary WGD inference and dating upon species *Aquilegia coerulea*.

The *Aquilegia coerulea* was reported to experience an paleo-polyploidization event after the divergence of core eudicots, which is likely shared by all Ranunculales.

First above all, let's delineate the whole paranome *K*<sub>S</sub> age distribution and have a basic observation for potentially conceivable WGDs, using the command line below.

```
wgd dmd Aquilegia_coerulea
wgd ksd wgd_dmd/Aquilegia_coerulea.tsv Aquilegia_coerulea
```

The constructed whole paranome *K*<sub>S</sub> age distribution of *Aquilegia coerulea* is as below, we can see that there seems to be a hump at *K*<sub>S</sub> 1 but not clear.

![](data/Aquilegia_coerulea.tsv.ksd_wp.svg)

We then construct the anchor *K*<sub>S</sub> age distribution using the command line below.

```
wgd syn -f mRNA -a Name wgd_dmd/Aquilegia_coerulea.tsv Aquilegia_coerulea.gff3 -ks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv
```

As shown below, there are some retained anchor pairs with *K*<sub>S</sub> between 1 and 2, which seems to suggest a WGD event.

![](data/Aquilegia_coerulea.tsv.ksd_wp_ap.svg)

The associated `dupStack` plot shows that there are numerous duplicated segments across most of the chromosomes.

![](data/Aquilegia_coerulea_Aquilegia_coerulea_multiplicons_level.svg)

We implemented two types of dot plots in oxford grid: one in the unit of bases and the other in the unit of genes, which can be colored by *K*<sub>S</sub> values given *K*<sub>S</sub> data.

![](data/Aquilegia_coerulea-vs-Aquilegia_coerulea_Ks.dot_unit_gene.png)

As shown above, the dot plot in the unit of genes presents numerous densely aggregated (line-like) anchor points at most of the chromosomes with consistent *K*<sub>S</sub> age between 1 and 2. The dot plot in the unit of bases shows the same pattern, as manifested below.

![](data/Aquilegia_coerulea-vs-Aquilegia_coerulea_Ks.dot.png)

The dot plots without *K*<sub>S</sub> annotation will also be automately produced, as shown below.

![](data/Aquilegia_coerulea-vs-Aquilegia_coerulea.dot_unit_gene.png)

![](data/Aquilegia_coerulea-vs-Aquilegia_coerulea.dot.png)

Note that the opacity of anchor dots and all homolog dots can be set by the option `apalpha` and `hoalpha` separately. If one just wants to see the anchor dots, setting the `hoalpha` as 0 (or other minuscule values) will do. If one wants to see the distribution of whole dots better, setting the `hoalpha` higher (and `apalpha` lower) will do. The `dotsize` option can be called to adjust the size of dots.

A further associated Syndepth plot shows that there are more than 80 duplicated segments, which dominates the whole collinear ratio category.

![](data/Syndepth.svg)

We can fit an ELMM mixture model upon the whole paranome *K*<sub>S</sub> age distribution to see more accurately the significance and location of potential WGDs, using the command line below.

```
wgd viz -d wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv
```

The result of ELMM mixture model clustering shows that there is a likely WGD component at *K*<sub>S</sub> 1.19.

![](data/elmm_Aquilegia_coerulea.tsv.ks.tsv_best_models_weighted.svg)

Let's do a mixture model clustering for anchor *K*<sub>S</sub> too, using the command line below.

```
wgd peak wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv --anchorpoints wgd_syn/iadhore-out/anchorpoints.txt --segments wgd_syn/iadhore-out/segments.txt --listelements wgd_syn/iadhore-out/list_elements.txt --multipliconpairs wgd_syn/iadhore-out/multiplicon_pairs.txt --weighted
```

The anchor *K*<sub>S</sub> age distribution also has a likely WGD component with mode 1.28.

![](data/Original_AnchorKs_GMM_Component3_node_weighted_Lognormal.svg)

Now that we have seen the evidence of numerous duplicated segments and the aggregation of duplicates age in *K*<sub>S</sub> around 1.2 for anchor pairs and non-anchor pairs throughout the whole genome. We can claim with some confidence that *Aquilegia coerulea* might have experienced a paleo-polyploidization event. Next, Let's have a further look about its phylogenetic location. We know that there are uncertainties about whether this putative paleo-polyploidization event is shared with all eudicots or not. We can choose some other eudicot genomes to see the ordering of speciation and polyploidization events. Here we choose *Vitis vinifera*, *Protea cynaroides* and *Acorus americanus* in the following *K*<sub>S</sub> analysis. First, we built a global MRBH family using the command below.

```
wgd dmd --globalmrbh Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera -o wgd_globalmrbh
```

In the global MRBH family, every pair of orthologous genes is the reciprocal best hit, suggesting true orthologous relationships. We would use the *K*<sub>S</sub> values associated with these orthologous pairs to delimit the divergence *K*<sub>S</sub> peak. Together with the whole paranome *K*<sub>S</sub> distribution, we conduct the rate correction using the command below.

!!Since `wgd` version 2.0.24, we rewrote a cleaner and quicker way of doing substitution rate correction. It's not required to type in any speices pair and a series of *K*<sub>S</sub> plots will be produced. The required files are orthologous *K*<sub>S</sub> table, paralogous *K*<sub>S</sub> table, a species tree and a focused species (the one inputted with paralogous *K*<sub>S</sub> data). Users can choose to add one more layer of elmm modeling on paralogous *K*<sub>S</sub> values and/or gmm modeling on anchor *K*<sub>S</sub> values. The orthologous *K*<sub>S</sub> values can be calculated using the command below.

```
wgd ksd wgd_globalmrbh/global_MRBH.tsv seqs* -o wgd_globalmrbh_ks
```

With the calculated orthologous *K*<sub>S</sub> table, we can use the command below to conduct the rate correction and/or mixture modeling analysis.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv -fa Aquilegia_coerulea -epk Aquilegia_coerulea.ks.tsv -ap anchorpoints.txt -sp speciestree.nw -o wgd_viz_mixed_Ks --plotelmm --plotapgmm
```

```
wgd ksd wgd_globalmrbh/global_MRBH.tsv seqs* --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -o wgd_globalmrbh_ks --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Aquilegia_coerulea;Aquilegia_coerulea" --plotkde (-ap wgd_syn/iadhore-out/anchorpoints.txt)
```

The file `speciestree.nw` is the text file of species tree in newick that rate correction would be conducted on. Its content is as below. Users need to provide the species pairs to be plotted. We suggest adding the option `--reweight` to recalculate the weight per species pair such that the weight of orthologous gene pairs will become 1 as the paralogous gene pairs. The flag `--plotkde` can be added when the kde curve of orthologous *K*<sub>S</sub> is desired. Extra collinear data can be added by the option `-ap`.

```
(((Vitis_vinifera,Protea_cynaroides),Aquilegia_coerulea),Acorus_americanus);
```

![](data/Aquilegia_coerulea_GlobalmrbhKs_Corrected.ksd.svg)

As shown above, because of the higher substitution rate of *Aquilegia coerulea*, the original orthologous *K*<sub>S</sub> values were actually underestimated in the time-frame of *Aquilegia coerulea*. When we recovered the divergence substitution distance in terms of two times of the branch-specific contribution of *Aquilegia coerulea* since its divergence with the sister species plus the shared substitution distance before divergence (in relative to the outgroup), the corrected *K*<sub>S</sub> mode became larger.

If one had the orthologous *K*<sub>S</sub> data already, one could also apply the program `wgd viz` to conduct the rate correction analysis using the command below. Note that the order of given `spair` options decides the color of the *K*<sub>S</sub> distribution of each species pair. The additional option `focus2all` in `wgd ksd` and `wgd viz` can be used to tell the program that the species pairs are between the focus species and all the other species such that users don't need to type in each pair individually.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -ap wgd_syn/iadhore-out/anchorpoints.txt -o wgd_viz_mixed_Ks --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Aquilegia_coerulea;Aquilegia_coerulea" --plotkde (--gsmap gene_species.map)
```

With the option `focus2all`, the command can be also like this, same with `wgd ksd`.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -ap wgd_syn/iadhore-out/anchorpoints.txt -o wgd_viz_mixed_Ks --focus2all Aquilegia_coerulea --plotkde
```

Note that we can easily show that *Aquilegia coerulea* has higher substitution rate than *Protea cynaroides* and *Vitis vinifera* by comparing their substitution distance in regard to the same divergence event with outgroup species *Acorus_americanus*, using command below.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv -sp speciestree.nw --reweight -o wgd_viz_Compare_rate --spair "Acorus_americanus;Protea_cynaroides" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Vitis_vinifera;Acorus_americanus" --gsmap gene_species.map --plotkde
```

![](data/Raw_Orthologues_Compare_rate.ksd.svg)

As displayed above, the orthologous *K*<sub>S</sub> values bewteen *Aquilegia coerulea* and *Acorus americanus* has the highest mode, indicatingthe faster substitution rate of *Aquilegia coerulea* compared to *Protea cynaroides* and *Vitis vinifera*.

Before v2.0.21, the gene-species map file is neccessarily needed for its implementation in `wgd viz`, which should be automately produced by the last `wgd ksd` step given the `spair` and `speciestree` parameters. The `gene_species.map` has contents as below in which each line is the joined string of gene name and species name by space. After v2.0.21 (included), the gene-species map file is not neccessarily needed anymore.

```
Aqcoe6G057800.1 Aquilegia_coerulea
Vvi_VIT_201s0011g01530.1 Vitis_vinifera
Pcy_Procy01g08510 Protea_cynaroides
Aam_Acora.04G142900.1 Acorus_americanus
```

A more complex plot can be made by add the flag `--plotelmm` such that the ELMM mixture modeling of provided paranome *K*<sub>S</sub> can be superimposed, using the command below.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -ap wgd_syn/iadhore-out/anchorpoints.txt -o wgd_viz_mixed_Ks_elmm --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Aquilegia_coerulea;Aquilegia_coerulea" --gsmap gene_species.map --plotkde --plotelmm
```

![](data/Aquilegia_coerulea_GlobalmrbhKs_Elmm_Corrected.ksd.svg)

From the mixed *K*<sub>S</sub> plot above, we can see that the optimized lognormal component b with mode 1.2 is younger than the corrected orthologous *K*<sub>S</sub> mode with *Protea cynaroides* and *Vitis vinifera* (1.39 and 1.47, respectively).

Besides, we can also add the GMM mixture modeling of anchor *K*<sub>S</sub> values with the flag `--plotapgmm`, using the command below.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -ap wgd_syn/iadhore-out/anchorpoints.txt -o wgd_viz_mixed_Ks_elmm --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Aquilegia_coerulea;Aquilegia_coerulea" --gsmap gene_species.map --plotkde --plotapgmmm
```

![](data/Aquilegia_coerulea_GlobalmrbhKs_Apgmm_Corrected.ksd.svg)

As manifested above, the anchor *K*<sub>S</sub> component 2 with mode 1.28 is also younger than the corrected orthologous *K*<sub>S</sub> mode with *Protea cynaroides* and *Vitis vinifera*. But we need to be of course cautious that such distinction comes with the uncertainties introduced from the applied mixture modeling methodology in terms of for instance different initialization points and the issue of overfitting and the sister speciess adopted in that there might be species with more disparate substitution rates than the one we chose.

Adding both ELMM result for paranome and GMM result for anchor *K*<sub>S</sub> can be achieved just by add the two flags mentioned above, using the command below.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -ap wgd_syn/iadhore-out/anchorpoints.txt -o wgd_viz_mixed_Ks_elmm --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Aquilegia_coerulea;Aquilegia_coerulea" --gsmap gene_species.map --plotkde --plotelmm --plotapgmmm
```

![](data/Aquilegia_coerulea_GlobalmrbhKs_Elmm_Apgmm_Corrected.ksd.svg)

An alternative way to calculate the orthologous *K*<sub>S</sub> is to directly use the orthogroups instead of global MRBH family. That way we don't use the pre-inferred paranome *K*<sub>S</sub> but the paralogous gene pairs inside each orthogroup instead. To achieve that, we first need to infer orthogroups using the command below.

```
wgd dmd Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera --orthoinfer -o wgd_ortho (--onlyortho) 
```

Users can decide to only conduct the orthogroup analysis while skipping other analysis by adding the flag `--onlyortho`. Next step is the same with global MRBH family except that we don't use the extra pre-inferred paranome *K*<sub>S</sub> anymore. We infer the *K*<sub>S</sub> using the command below. Note that the program `wgd viz` can plot the same just as shown above.

```
wgd ksd wgd_ortho/Orthogroups.sp.tsv -sp speciestree.nw --reweight -o wgd_ortho_ks --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Aquilegia_coerulea;Aquilegia_coerulea" --plotkde (-ap wgd_syn/iadhore-out/anchorpoints.txt)
```

![](data/Aquilegia_coerulea_OrthoKs_Corrected.ksd.svg)

As shown above, the number of both paralogous gene pairs and orthologous gene pairs is different than the one from global MRBH family in that here we plotted all orthologous gene pairs instead of only global MRBH and potentially new paralogous gene pairs might be produced in the orthogroup inference step, together with different recalculated weights.

If one only wanted to plot the orthologous *K*<sub>S</sub> distribution, as we have already shown above, it can be easily achieved by removing the paralogous species pair `Aquilegia_coerulea;Aquilegia_coerulea`, using the command below.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv -sp speciestree.nw --reweight -o wgd_globalmrbh_onlyortho_ks --spair "Aquilegia_coerulea;Protea_cynaroides" --spair "Aquilegia_coerulea;Vitis_vinifera" --spair "Aquilegia_coerulea;Acorus_americanus" --gsmap gene_species.map
```

We can clearly see that *Vitis vinifera* has higher substitution rate than *Protea cynaroides* in that their orthologous *K*<sub>S</sub> peaks with *Aquilegia coerulea*, although representing the same divergence event, differed in substitution distance.
![](data/Raw_Orthologues.ksd.svg)

After the phylogenetic timing of the Ranunculales WGD, we can further infer its absolute age. First we infer the credible range of anchor pairs by *K*<sub>S</sub> heuristically using the program `wgd peak`.

```
wgd peak --heuristic wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -ap wgd_syn/iadhore-out/anchorpoints.txt -sm wgd_syn/iadhore-out/segments.txt -le wgd_syn/iadhore-out/list_elements.txt -mp wgd_syn/iadhore-out/multiplicon_pairs.txt -o wgd_peak
```

![](data/AnchorKs_PeakCI_Aquilegia_coerulea.tsv.ks.tsv_node_weighted.svg)

As shown above, we assumed a lognormal distribution at the peak location detected by the `signal` module of `scipy` library. The 95% confidence level of the lognormal distribution was applied, i.e., 0.68-2.74, in further molecular dating. The file `Aquilegia_coerulea.tsv.ks.tsv_95%CI_AP_for_dating_weighted_format.tsv` is what we need for next step. To build the orthogroups used in phylogenetic dating, we need to select some species and form a starting tree with proper fossil calibrations. We provide one in mcmctree format as below.

```
17 1
((((Potamogeton_acutifolius,(Spirodela_intermedia,Amorphophallus_konjac)),(Acanthochlamys_bracteata,(Dioscorea_alata,Dioscorea_rotundata))'>0.5600<1.2863')'>0.8360<1.2863',(Acorus_americanus,Acorus_tatarinowii))'>0.8360<1.2863',((((Tetracentron_sinense,Trochodendron_aralioides),(Buxus_austroyunnanensis,Buxus_sinica))'>1.1080<1.2863',(Nelumbo_nucifera,(Telopea_speciosissima,Protea_cynaroides)))'>1.1080<1.2863',(Aquilegia_coerulea_ap1,Aquilegia_coerulea_ap2))'>1.1080<1.2863')'>1.2720<2.4720';
```

As presented above, the focus species that is about to be dated needs to be replaced with `(Aquilegia_coerulea_ap1,Aquilegia_coerulea_ap2)`. With this starting tree and predownloaded cds files of all the species, we can build the orthogroup used in the final molecular dating using the command as below.

```
wgd dmd -f Aquilegia_coerulea -ap wgd_peak/Aquilegia_coerulea.tsv.ks.tsv_95%CI_AP_for_dating_weighted_format.tsv -o wgd_dmd_ortho Potamogeton_acutifolius Spirodela_intermedia Amorphophallus_konjac Acanthochlamys_bracteata Dioscorea_alata Dioscorea_rotundata Acorus_americanus Acorus_tatarinowii Tetracentron_sinense Trochodendron_aralioides Buxus_austroyunnanensis Buxus_sinica Nelumbo_nucifera Telopea_speciosissima Protea_cynaroides Aquilegia_coerulea
```

The result file `merge_focus_ap.tsv` is what we need for the final step of molecular dating in program `wgd focus`.

```
wgd focus --protdating --aamodel lg wgd_dmd_ortho/merge_focus_ap.tsv -sp dating_tree.nw -o wgd_dating -d mcmctree -ds 'burnin = 2000' -ds 'sampfreq = 1000' -ds 'nsample = 20000' Potamogeton_acutifolius Spirodela_intermedia Amorphophallus_konjac Acanthochlamys_bracteata Dioscorea_alata Dioscorea_rotundata Acorus_americanus Acorus_tatarinowii Tetracentron_sinense Trochodendron_aralioides Buxus_austroyunnanensis Buxus_sinica Nelumbo_nucifera Telopea_speciosissima Protea_cynaroides Aquilegia_coerulea
```

Here we only implemented the concatenation analysis using protein sequence by adding the flag `--protdating` and we set the parameter for `mcmctree` via the option `-ds`. Note that other dating program such as `r8s` and `beast` are also available given some mandatory parameters. The final log of the successful run is as below.

```
16:04:25 INFO     Running mcmctree using Hessian matrix of LG+Gamma  core.py:967
                  for protein model
23:49:37 INFO     Posterior mean for the ages of wgd is 1.128945 mcmctree.py:296
                  billion years from Concatenated peptide
                  alignment and 95% credibility intervals (CI)
                  is 1.01224-1.23121 billion years
         INFO     Total run time: 29175s                              cli.py:241
         INFO     Done                                                cli.py:242
```

To visualize the date, we also provided a python script to plot the WGD dates in the `wgd` folder. Users need to extract the raw dates from the `mcmc.txt` for the WGD node first and save it as file `dates.txt` (or whatever preferred name). An example command is as below.

```
python $PATH/postplot.py postdis dates.txt --percentile 90 --title "WGD date" --hpd -o "Ranunculales_WGD_date.svg"
```

![](data/Ranunculales_WGD_date.svg)

The posterior mean, median and mode of the Ranunculales WGD age is 112.92, 113.44 and 112.54 mya, with 90% HPD 105.07 - 122.32 mya as manifested above.

### Kstree

In addition to pairwise *K*<sub>S</sub> estimation, a *K*<sub>S</sub> tree with branch length in *K*<sub>S</sub> unit can also be derived from the program `wgd ksd` given the option `--kstree` and `--speciestree`. Note that the additional option `--onlyconcatkstree` will only call the *K*<sub>S</sub> estimation for the concatenated alignment rather than all the alignments. Users need to provide a preset species tree for the *K*<sub>S</sub> tree inference of the concatenated alignment while the remaining alignments will be against an automately inferred tree from `fasttree` or `iqtree`. In the end, users will get a *K*<sub>S</sub> tree, a *K*<sub>A</sub> tree and a ω tree per fam and for the concatenated alignment.

```
wgd ksd data/kstree_data/fam.tsv data/kstree_data/Acorus_tatarinowii data/kstree_data/Amborella_trichopoda data/kstree_data/Aquilegia_coerulea data/kstree_data/Aristolochia_fimbriata data/kstree_data/Cycas_panzhihuaensi --kstree --speciestree data/kstree_data/species_tree1.nw --onlyconcatkstree -o wgd_kstree_topology1
```

![](data/kstree_results/kstree.svg)

Above we used three alternative topologies to infer the *K*<sub>S</sub> tree which led to different branch length estimation. Note that the families we used were only two global MRBH families for the purpose of illustration. To acquire an accurate profile of the substitution rate variation, orthologues at the whole genome scale should be used.

## Citation
 
Please cite us at https://doi.org/10.1007/978-1-0716-2561-3_1.

```
Hengchi Chen, Arthur Zwaenepoel (2023). Inference of Ancient Polyploidy from Genomic Data. In: Van de Peer, Y. (eds) Polyploidy. Methods in Molecular Biology, vol 2545. Humana, New York, NY. https://doi.org/10.1007/978-1-0716-2561-3_1
```

For citation of the tools used in wgd, please consult the documentation at
https://wgdv2.readthedocs.io/en/latest/citation.html.

