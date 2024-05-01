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

`wgd v2` is a python package upgraded from the original `wgd` package aiming for the inference and timing of ancient whole-genome duplication (WGD) events. For the propose of illustrating the principle and usage of `wgd v2`, we compiled this documentation. Below we first gave an introduction over the scope and mechanism of `wgd v2` and then the practical information pertaining to the installation and usage. An examplar workflow is provided in the tutorial section on how to seek evidence for a putative WGD event and perform proper timing with a freshly obtained genome assembly in hand. For those who are interested in more theoretical details, we recommend turning to our paper and book chapter for more detailed description and insightful discussions. The key improved features of `wgd v2` are demonstrated in our latest manuscript on [Bioinformatics](https://doi.org/10.1093/bioinformatics/btae272). If you use `wgd v2` in your research, please cite us as suggested in [Citation](#citation) section.

## Introduction

Polyploidizations, the evolutionary process that the entire genome of an organism is duplicated, also named as whole-genome duplications (WGDs), occur recurrently across the tree of life. There are two modes of polyploidizations, autopolyploidizations and allopolyploidizations. Autopolyploidizations are the duplication of the same genome, resulting in two identical subgenomes at the time it emerged. While the allopolyploidizations are normally achieved in two steps, first the hybridization between two different species, resulting in the arising of transient homoploidy,second the duplication of the homoploidy, resulting in the emergence of allopolyploidy. Due to the unstability and unbalanced tetrasomic inheritance, for instance the nuclear-cytoplasmic incompatibility, the polyploidy genome will then experience a process called diploidization, also named as fractionation, during which a large portion of gene duplicates will get lost and only a fraction can be retained. The traces of polyploidizations can be thus unearthed from these retained gene duplicates. Three approaches based on gene duplicates, namely, *K*<sub>S</sub> method, gene tree - species tree reconciliation method and synteny method, are commonly used in detecting evidence for WGDs. The gene tree - species tree reconciliation method is not within the scope of `wgd v2`, but we kindly refer readers who are interested to the phylogenomic program developed by Arthur Zwaenepoel named [WHALE](https://github.com/arzwa/Whale.jl) and the associated [paper](https://doi.org/10.1093/molbev/msz088) for more technical and theoretical details.

The *K*<sub>S</sub> method is established on a model of gene family evolution that each gene family is allowed to evolve via gene duplication and loss. Note that the gene family here is assumed to be the cluster of all genes descended from an ancestral gene in a single genome. Recovering the gene tree of such gene family informs the timing, scilicet the age, of gene duplication events. The age refered here, is not in real geological time, but in the unit of evolutionary distance, i.e., the number of substitutions per site. When the evolutionary rate remains approximately constant, the evolutionary distance is then supposed to be proportional to the real evolutionary time. The synonymous distance *K*<sub>S</sub>, the number of synonymous substitutions per synonymous site, is such candidate that synonymous substitutions would not incur the change of amino acid and are thus regarded as neutral, which according to the neutral theory should occur in constant rate. Given a model of gene family that allows the gene to duplicate and get lost in a fixed rate, one can derive that the probability density function of the *K*<sub>S</sub> age distribution of retained gene duplicates is a quasi-exponential function that most retained gene duplicates are recently borned with ~0 age while as the age going older the associated number of retained gene duplicates decay quasi-exponentially. Therefore, the occurance of large-scale gene duplication events, for instane WGDs, with varied retention rate, will leave an age peak from the burst of gene duplicates in a short time-frame upon the initial age distribution, and can be unveiled from mixture modeling analysis. However, WGDs identified from the paralogous *K*<sub>S</sub> age distributions can only inform the WGD timing in the time-scale of that specific species, which is not comparable in the phylogenetic context. Only with the orthologous *K*<sub>S</sub> age distributions, which convert the estimated body from paralogues to orthologues and inform the relative timing of speciation events, can we decipher the phylogenetic placement of WGDs after proper rate correction. `wgd v2` is such program that helps users construct paralogous and orthologous *K*<sub>S</sub> age distributions and realize both the identification and placement of WGDs.

In the premise of phylogenetically located WGDs, the absolute age (in geological time) of WGDs can also be inferred from those WGD-retained gene duplicates, although there has been no easy or straightforward pipeline for this job so far. In `wgd v2`, we developed a feasible integrated pipleline for absolute dating of WGDs. The pipeline can be roughly divided into three main steps. 1) The construction of anchor *K*<sub>S</sub> distribution and the delineation of crediable *K*<sub>S</sub> range adopted for phylogenetic dating, using `wgd dmd`, `wgd ksd`, `wgd syn` and `wgd peak`. Note that here we only consider genome assembly because for transcriptome assembly it's impossible to distinguish WGD-derived duplicates from small-scale duplication-derived duplicates, which happened in a continuous time-frame instead of only a separate short time-frame and thus reflects the duration of that branch rather than the time at which WGD occurred. 2) The formulation of a starting tree used in the phylogenetic dating, composed of a few species and annotated with fossil calibration information. This step is essential for the result of absolute WGD dating that we suggest users of taking great caution to assure the tree topology and proper bounds for fossil calibrations. 3) The construction of orthogroups consisting of collinear duplicates of the focal species and their reciprocal best hits (RBHs) against other species in the starting tree and the phylogenetic dating using a molecular dating program for instance `mcmctree`, via program `wgd dmd` and `wgd focus`. We recommend the usage of Bayesian molecular dating program `mcmctree`, which provides a variety of different substitution and rate models. Nonetheless, we urge users to set prior distribution of different parameters with caution and assure adequate sampling of different parameters.

## Installation

The easiest way to install `wgd v2` is using `PYPI`. Note that if you want to get the latest update, we suggest installing from the source, since the update on `PYPI` will be delayed compared to here of source.

```
pip install wgd
```

To install `wgd v2` in a virtual environment (which we strongly recommend, no matter you install from source, `PYPI` or `bioconda`), the following command lines could be used.

```
git clone https://github.com/heche-psb/wgd
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

Note that the version of `numpy` is important (for many other packages are the same of course), especially for `fastcluster` package. In our test, the `numpy` 1.19.0 works fine on `python3.6/8`. If you met some errors or warnings about `numpy`, maybe considering pre-install `numpy` as 1.19.0 or other close-by versions before you install `wgd`. `wgd` relies on external softwares including `diamond` and `mcl` for `wgd dmd`, `paml v4.9j`, `mafft` (`muscle` and `prank` if set), `fasttree` (or `iqtree` if set) for `wgd ksd` and optionally `mrbayes` for the phylogenetic inference function in `wgd dmd` and `wgd focus` (`mafft`, `muscle` and `prank` as well when the analysis requires sequence alignment). Some other optional softwares including `paml v4.9j`, `r8s`, `beast`, `eggnog`, `diamond`, `interproscan`, `hmmer v3.1b2` and `astral-pro` (`hmmer v3.1b2` is also for the [orthogroup assignment](https://wgdv2.readthedocs.io/en/latest/usage.html#orthogroup-assignment) function in `wgd dmd` and `astral-pro` is also for the function `--collinearcoalescence` in `wgd dmd`) are for the molecular dating, gene family function annotation or phylogenetic inference in `wgd focus`.

## Parameters

There are 7 main programs in `wgd v2`: `dmd`,`focus`,`ksd`,`mix`,`peak`,`syn`,`viz`. Hereafter we will provide a detailed elucidation on each of the program and its associated parameters. Please refer to the [Usage](#usage) for the scenarios to which each parameter applies.

The program `wgd dmd` can realize the delineation of whole paranome, RBHs (Reciprocal Best Hits), MRBHs (Multiple Reciprocal Best Hits), orthogroups and some other orthogroup-related functions, including circumscription of nested single-copy orthogroups (NSOGs), unbiased test of single-copy orthogroups (SOGs) over missing inparalogs, construction of BUSCO-guided single-copy orthogroups (SOGs),and the collinear coalescence inference of phylogeny.
```
wgd dmd sequences (option)
--------------------------------------------------------------------------------
-o, --outdir, the output directory, default wgd_dmd
-t, --tmpdir, the temporary working directory, default None, if None was given, the tmpdir would be assigned random names in current directory and automately removed at the completion of program, else the tmpdir would be kept
-p, --prot, flag option, whether using protein or nucleotide sequences
-c, --cscore, the c-score to restrict the homolog similarity of MRBHs, default None, if None was given, the c-score funcion wouldn't be activated, else expecting a decimal within the range of 0 and 1
-I, --inflation, the inflation factor for MCL program, default 2.0, with higher value leading to more but smaller clusters
-e, --eval, the e-value cut-off for similarity in diamond and/or hmmer, default 1e-10
--to_stop, flag option, whether to translate through STOP codons, if the flag was set, translation will be terminated at the first in-frame stop codon, else a full translation continuing on passing any stop codons would be initiated
--cds, flag option, whether to only translate the complete CDS that starts with a valid start codon and only contains a single in-frame stop codon at the end and must be dividable by three, if the flag was set, only the complete CDS would be translated
-f, --focus, the species to be merged on local MRBHs, default None, if None was given, the local MRBHs wouldn't be inferred
-ap, --anchorpoints, the anchor points data file from i-adhore for constructing the orthogroups with anchor pairs, default None
-sm, --segments, the segments datafile used in collinear coalescence analysis if initiated, default None
-le, --listelements, the listsegments data file used in collinear coalescence analysis if initiated, default None
-gt, --genetable, the gene table datafile used in collinear coalescence analysis if initiated, default None
-coc, --collinearcoalescence, flag option, whether to initiate the collinear coalescence analysis, if the flag was set, the analysis would be initiated
-kf, --keepfasta, flag option, whether to output the sequence information of MRBHs, if the flag was set, the sequences of MRBHs would be in output
-kd, --keepduplicates, flag option, whether to allow the same gene to occur in different MRBHs (only meaningful when the cscore was used), if the flag was set, the same gene could be assigned to different MRBHs
-gm, --globalmrbh, flag option, whether to initiate global MRBHs construction, if the flag was set, the --focus option would be ignored and only global MRBHs would be built
-n, --nthreads, the number of threads to use, default 4
-oi, --orthoinfer, flag option, whether to initiate orthogroup infernece, if the flag was set, the orthogroup infernece program would be initiated
-oo, --onlyortho, flag option, whether to only conduct orthogroup infernece, if the flag was set, only the orthogroup infernece pipeline would be performed while the other analysis wouldn't be initiated
-gn, --getnsog, flag option, whether to initiate the searching for nested single-copy gene families (NSOGs) (only meaningful when the orthogroup infernece pipeline was activated), if the flag was set, additional NSOGs analysis would be performed besides the basic orthogroup infernece
-tree, --tree_method, which gene tree inference program to invoke (only meaningful when the collinear coalescence, gene-to-family assignment or NSOGs analysis were activated), default fasttree
-ts, --treeset, the parameters setting for gene tree inference, default None, this option can be provided multiple times
-mc, --msogcut, the ratio cutoff for mostly single-copy family (meaningful when activating the orthogroup infernece pipeline) and species representation in collinear coalescence analysis, default 0.8.
-ga, --geneassign, flag option, whether to initiate the gene-to-family assignment analysis, if the flag was set, the analysis would be initiated
-sa, --seq2assign, the queried sequences data file in gene-to-family assignment analysis, default None, this option can be provided multiple times
-fa, --fam2assign, the queried familiy data file in gene-to-family assignment analysis, default None
-cc, --concat, flag option, whether to initiate the concatenation pipeline for orthogroup infernece, if the flag was set, the analysis would be initiated
-te, --testsog, flag option, whether to initiate the unbiased test of single-copy gene families, if the flag was set, the analysis would be initiated
-bs, --bins, the number of bins divided in the gene length normalization, default 100
-np, --normalizedpercent, the percentage of upper hits used for gene length normalization, default 5
-nn, --nonormalization, flag option, whether to call off the normalization, if the flag was set, no normalization would be conducted
-bsog, --buscosog, flag option, whether to initiate the busco-guided single-copy gene family analysis, if the flag was set, the analysis would be initiated
-bhmm, --buscohmm, the HMM profile datafile in the busco-guided single-copy gene family analysis, default None
-bctf, --buscocutoff, the HMM score cutoff datafile in the busco-guided single-copy gene family analysis, default None
-of ,--ogformat, flag option, whether to add index to the RBH families
```

The program `wgd focus` can realize the concatenation-based and coalescence-based phylogenetic inference and phylogenetic dating of WGDs etc.
```
wgd focus families sequences (option)
--------------------------------------------------------------------------------
-o, --outdir, the output directory, default wgd_focus
-t, --tmpdir, the temporary working directory, default None, if None was given, the tmpdir will be assigned random names in current directory and automately removed at the completion of program, else the tmpdir would be kept
-n, --nthreads, the number of threads to use, default 4
--to_stop, flag option, whether to translate through STOP codons, if the flag was set, translation will be terminated at the first in-frame stop codon, else a full translation continuing on past any stop codons would be initiated
--cds, flag option, whether to only translate the complete CDS that starts with a valid start codon and only contains a single in-frame stop codon at the end and must be dividable by three, if the flag was set, only the complete CDS would be translated
--strip_gaps, flag option, whether to drop all gaps in multiple sequence alignment, if the flag was set, all gaps would be dropped
-a, --aligner, which alignment program to use, default mafft
-tree, --tree_method, which gene tree inference program to invoke, default fasttree
-ts, --treeset, the parameters setting for gene tree inference, default None, this option can be provided multiple times
--concatenation, flag option, whether to initiate the concatenation-based species tree inference, if the flag was set, concatenation-based species tree would be infered
--coalescence, flag option, whether to initiate the coalescence-based species tree inference, if the flag was set, coalescence-based species tree would be infered
-sp, --speciestree, species tree datafile for dating, default None
-d, --dating, which molecular dating program to use, default none
-ds, --datingset, the parameters setting for dating program, default None, this option can be provided multiple times
-ns, --nsites, the nsites information for r8s dating, default None
-ot, --outgroup, the outgroup information for r8s dating, default None
-pt, --partition, flag option, whether to initiate partition dating analysis for codon, if the flag was set, an additional partition dating analysis would be initiated
-am, --aamodel, which protein model to be used in mcmctree, default poisson
-ks, flag option, whether to initiate Ks calculation for homologues in the provided orthologous gene family
--annotation, which annotation program to use, default None
--pairwise, flag option, whether to initiate pairwise Ks estimation, if the flag was set, pairwise Ks values would be estimated
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
--beagle, flag option, whether to use beagle in Beast, if the flag was set, beagle would be used
--protcocdating, flag option, whether to only initiate the protein-concatenation-based dating analysis, if the flag was set, the analysis would be initiated
--protdating, flag option, whether to only initiate the protein-based dating analysis, if the flag was set, the analysis would be initiated
```

The program `wgd ksd` can realize the construction of *K*<sub>S</sub> age distribution and rate correction.
```
wgd ksd families sequences (option)
--------------------------------------------------------------------------------
-o, --outdir, the output directory, default wgd_ksd
-t, --tmpdir, the temporary working directory, default None, if None was given, the tmpdir will be assigned random names in current directory and automately removed at the completion of program, else the tmpdir would be kept
-n, --nthreads, the number of threads to use, default 4
--to_stop, flag option, whether to translate through STOP codons, if the flag was set, translation will be terminated at the first in-frame stop codon, else a full translation continuing on past any stop codons would be initiated
--cds, flag option, whether to only translate the complete CDS that starts with a valid start codon and only contains a single in-frame stop codon at the end and must be dividable by three, if the flag was set, only the complete CDS would be translated
--pairwise, flag option, whether to initiate pairwise Ks estimation, if the flag was set, pairwise Ks values would be estimated
--strip_gaps, flag option, whether to drop all gaps in multiple sequence alignment, if the flag was set, all gaps would be dropped
-a, --aligner, which alignment program to use, default mafft 
-tree, --tree_method, which gene tree inference program to invoke, default fasttree
--tree_options, options in tree inference as a comma separated string, default None
--node_average, flag option, whether to initiate node-average way of de-redundancy instead of node-weighted, if the flag was set, the node-averaging de-redundancy would be initiated
-sr, --spair, the species pair to be plotted, default None, this option can be provided multiple times
-sp, --speciestree, the species tree to perform rate correction, default None, if None was given, the rate correction analysis would be called off
-rw, --reweight, flag option, whether to recalculate the weight per species pair, if the flag was set, the weight would be recalculated
-or, --onlyrootout, flag option, whether to only conduct rate correction using the outgroup at root as outgroup, if the flag was set, only the outgroup at root would be used as outgroup
-epk, --extraparanomeks, extra paranome Ks data to plot in the mixed Ks distribution, default None
-ap, --anchorpoints, anchorpoints.txt file to plot anchor Ks in the mixed Ks distribution, default None
-pk, --plotkde, flag option, whether to plot kde curve of orthologous Ks distribution over histogram in the mixed Ks distribution, if the flag was set, the kde curve would be plotted
-pag, --plotapgmm, flag option, whether to perform and plot mixture modeling of anchor Ks in the mixed Ks distribution, if the flag was set, the mixture modeling of anchor Ks would be plotted
-pem, --plotelmm, flag option, whether to perform and plot elmm mixture modeling of paranome Ks in the mixed Ks distribution, if the flag was set, the elmm mixture modeling of paranome Ks would be plotted
-c, --components, the range of the number of components to fit in anchor Ks mixture modeling, default (1,4)
-xl, --xlim, the x axis limit of Ks distribution
-yl, --ylim, the y axis limit of Ks distribution
-ado, --adjustortho, flag option, whether to adjust the histogram height of orthologous Ks as to match the height of paralogous Ks, if the flag was set, the adjustment would be conducted
-adf, --adjustfactor, the adjustment factor of orthologous Ks, default 0.5
-oa, --okalpha, the opacity of orthologous Ks distribution in mixed plot, default 0.5
-fa, --focus2all, set focal species and let species pair to be between focal and all the remaining species, default None
-ks, --kstree, flag option, whether to infer Ks tree, if the flag was set, the Ks tree inference analysis would be initiated
-ock, --onlyconcatkstree, flag option, whether to only infer Ks tree under concatenated alignment, if the flag was set, only the Ks tree under concatenated alignment would be calculated
-cs, --classic, flag option, whether to draw mixed Ks plot in a classic manner where the full orthologous Ks distribution is drawed, if the flag was set, the classic mixed Ks plot would be drawn
-ta, --toparrow, flag option, whether to adjust the arrow to be at the top of the plot, instead of being coordinated as the KDE of the orthologous Ks distribution, if the flag was set, the arrow would be set at the top
-bs, --bootstrap, the number of bootstrap replicates of ortholog Ks distribution in mixed plot
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
-mi, --max_iter, the maximum number of iterations, default 200
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
-ic, --weights_outliers_included, flag option, whether to include Ks outliers, if the flag was set, Ks outliers would be included in the analysis
-m, --method, which mixture model to use, default gmm
--seed, random seed given to initialization, default 2352890
-ei, --em_iter, the number of EM iterations to perform, default 200
-ni, --n_init, the number of k-means initializations, default 200
-n, --components, the range of the number of components to fit, default (1, 4)
-g, --gamma, the gamma parameter for bgmm models, default 1e-3
--boots, the number of bootstrap replicates of kde, default 200
--weighted, flag option, whether to use node-weighted method of de-redundancy, if the flag was set, the node-weighted method would be used
-p, --plot, the plotting method to be used, default identical
-bm, --bw_method, the bandwidth method to be used in analyzing the peak of WGD dates, default silverman
--n_medoids, the number of medoids to fit, default 2
-km, --kdemethod, the kde method to be used in analyzing the peak of WGD dates, kmedoids analysis or the basic Ks plotting, default scipy
--n_clusters, the number of clusters to plot Elbow loss function, default 5
-gd, --guide, the regime residing anchors, default Segment
-prct, --prominence_cutoff, the prominence cutoff of acceptable peaks in peak finding steps, default 0.1
-rh, --rel_height, the relative height at which the peak width is measured, default 0.4
-kd, --kstodate, the range of Ks to be dated in heuristic search, default (0.5, 1.5)
-xl, --xlim, the x axis limit of GMM Ks distribution
-yl, --ylim, the y axis limit of GMM Ks distribution
--manualset, flag option, whether to output anchor pairs with manually set Ks range, if the flag was set, manually set Ks range would be outputted
--ci, the confidence level of log-normal distribution to date, default 95
--hdr, the highest density region (HDR) applied in the segment-guided anchor pair Ks distribution, default 95
--heuristic, flag option, whether to initiate heuristic method of defining CI for dating, if the flag was set, the heuristic method would be initiated
-kc, --kscutoff, the Ks saturation cutoff in dating, default 5
--keeptmpfig, flag option, whether to keep temporary figures in peak finding process, if the flag was set, those figures would be kept
```

The program `wgd syn` can realize the intra- and inter-specific synteny inference.
```
wgd syn families gffs (option)
--------------------------------------------------------------------------------
-ks, --ks_distribution, ks distribution datafile, default None
-o, --outdir, the output directory, default wgd_syn
-f, --feature, the feature for parsing gene IDs from GFF files, default gene
-a, --attribute, the attribute for parsing the gene IDs from the GFF files, default ID
-atg, --additionalgffinfo, the feature and attribute information of additional gff3 files if different in the format of (feature;attribute)', default None
-ml, --minlen, the minimum length of a scaffold to be included in dotplot, default -1, if -1 was set, the 10% of the longest scaffold would be set
-ms, --maxsize, the maximum family size to be included, default 200
-r, --ks_range, the Ks range in colored dotplot, default (0, 5)
--iadhore_options, the parameter setting in iadhore, default as a string of length zero
-mg, --minseglen, the minimum length of segments to include in ratio if <= 1, default 10000
-kr, --keepredun, flag option, whether to keep redundant multiplicons, if the flag was set, the redundant multiplicons would be kept
-mgn, --mingenenum, the minimum number of genes for a segment to be considered, default 30
-ds, --dotsize, the dot size in dot plot, default 0.3
-aa, --apalpha, the opacity of anchor dots, default 1
-ha, --hoalpha, the opacity of homolog dots, default 0
-srt, --showrealtick, flag option, whether to show the real tick in genes or bases, if the flag was set, the real tick would be showed
-tls, --ticklabelsize, the label size of tick, default 5
-gr, --gistrb, flag option, whether to use gist_rainbow as color map of dotplot
-n, --nthreads, the number of threads to use in synteny inference, default 4
```

The program `wgd viz` can realize the visualization of *K*<sub>S</sub> age distribution and synteny.
```
wgd viz (option)
--------------------------------------------------------------------------------
-d, --datafile, the Ks datafile, default None
-o, --outdir, the output directory, default wgd_viz
-sr, --spair, the species pair to be plotted, default None, this option can be provided multiple times
-fa, --focus2all, set focal species and let species pair to be between focal and all the remaining species, default None
-gs, --gsmap, the gene name-species name map, default None
-sp, --speciestree, the species tree to perform rate correction, default None, if None was given, the rate correction analysis would be called off
-pk, --plotkde, flag option, whether to plot kde curve upon histogram, if the flag was set, kde curve would be added
-rw, --reweight, flag option, whether to recalculate the weight per species pair, if the flag was set, the weight would be recalculated
-or, --onlyrootout, flag option, whether to only conduct rate correction using the outgroup at root as outgroup, if the flag was set, only the outgroup at root would be used as outgroup
-iter, --em_iterations, the maximum EM iterations, default 200
-init, --em_initializations, the maximum EM initializations, default 200
-prct, --prominence_cutoff, the prominence cutoff of acceptable peaks, default 0.1
-rh, --rel_height, the relative height at which the peak width is measured, default 0.4
-sm, --segments, the segments datafile, default None
-ml, --minlen, the minimum length of a scaffold to be included in dotplot, default -1, if -1 was set, the 10% of the longest scaffolds will be set
-ms, --maxsize, the maximum family size to be included, default 200
-ap, --anchorpoints, the anchor points datafile, default None
-mt, --multiplicon, the multiplicons datafile, default None
-gt, --genetable, the gene table datafile, default None
-mg, --minseglen, the minimum length of segments to include, in ratio if <= 1, default 10000
-mgn, --mingenenum, the minimum number of genes for a segment to be considered, default 30
-kr, --keepredun, flag option, whether to keep redundant multiplicons, if the flag was set, the redundant multiplicons would be kept
-epk, --extraparanomeks, extra paranome Ks data to plot in the mixed Ks distribution, default None
-pag, --plotapgmm, flag option, whether to conduct and plot mixture modeling of anchor Ks in the mixed Ks distribution, if the flag was set, the mixture modeling of anchor Ks would be conducted and plotted
-pem, --plotelmm, flag option, whether to conduct and plot elmm mixture modeling of paranome Ks in the mixed Ks distribution, if the flag was set, the elmm mixture modeling of paranome Ks would be conducted and plotted
-c, --components, the range of the number of components to fit in anchor Ks mixture modeling, default (1,4)
-psy, --plotsyn, flag option, whether to initiate the synteny plot, only when the flag was set, the synteny plot would be produced
-ds, --dotsize, the dot size in dot plot, default 0.3
-aa, --apalpha, the opacity of anchor dots, default 1
-ha, --hoalpha, the opacity of homolog dots, default 0
-srt, --showrealtick, flag option, whether to show the real tick in genes or bases, if the flag was set, the real tick would be showed
-tls, --ticklabelsize, the label size of tick, default 5
-xl, --xlim, the x axis limit of Ks distribution
-yl, --ylim, the y axis limit of Ks distribution
-ado, --adjustortho, flag option, whether to adjust the histogram height of orthologous Ks as to match the height of paralogous Ks, if the flag was set, the adjustment would be conducted
-adf, --adjustfactor, the adjustment factor of orthologous Ks, default 0.5
-oa, --okalpha, the opacity of orthologous Ks distribution in mixed plot, default 0.5
-cs, --classic, flag option, whether to draw mixed Ks plot in a classic manner where the full orthologous Ks distribution is drawed, if the flag was set, the classic mixed Ks plot would be drawn
-ta, --toparrow, flag option, whether to adjust the arrow to be at the top of the plot, instead of being coordinated as the KDE of the orthologous Ks distribution, if the flag was set, the arrow would be set at the top
-na, --nodeaveraged, flag option, whether to use node-averaged method for de-redundancy, if the flag was set, the node-averaged method would be initiated
-bs, --bootstrap, the number of bootstrap replicates of ortholog Ks distribution in mixed plot
-gr, --gistrb, flag option, whether to use gist_rainbow as color map of dotplot
-n, --nthreads, the number of threads to use in bootstrap sampling, default 1
```

## Usage

Here we provided the basic usage for each program and the relevant parameters and suggestions on parameterization. A reminder that the given cores and threads can significantly impact the run time and thus we added some report information pertaining to the system of users to facilitate the efficient setting of threads and memory. The logical CPUs reported represents the number of physical cores multiplied by the number of threads that can run on each core, also known as Hyper Threading. The number of logical CPUs may not necessarily be equivalent to the actual number of CPUs the current process can use. The available memory refers to the memory that can be given instantly to processes without the system going into swap and reflects the actual memory available. The free memory refers to the memory not being used at all (zeroed) that is readily available. The description above refers to the documentation of `psutil`.

### wgd dmd

**The delineation of whole paranome**
```
wgd dmd Aquilegia_coerulea -I 2 -e 1e-10 -bs 100 -np 5 (-nn) (--to_stop) (--cds) (-n 4) (-o wgd_dmd) (-t working_tmp)
``` 

Note that we don't provide the data of this coding sequence (cds) file `Aquilegia_coerulea` but it can be easily downloaded at [Phytozome](https://phytozome-next.jgi.doe.gov/info/Acoerulea_v3_1) (same for other Usage doc). A reminder that in the issues some users didn't download the `Acoerulea_322_v3.1.cds_primaryTranscriptOnly.fa.gz` but instead the `Acoerulea_322_v3.1.cds.fa.gz` file. For the construction of whole paranome *K*<sub>S</sub> distribution, **only one transcript (the primary one) per gene should be included** such that the *K*<sub>S</sub> is really indicating the age of gene duplication event, instead of alternative splicing. Transcriptome data should be carefully treated with de-redundancy so as to reduce the false positive duplication bump caused by pervasive alternative splicing. Here the inflation factor parameter, given by `-I` or `--inflation`, affects the granularity or resolution of the clustering outcome and implicitly controlls the number of clusters, with low values such as 1.3 or 1.4 leading to fewer but larger clusters and high values such as 5 or 6 leading to more but smaller clusters. We set the default value as 2 as suggested by [MCL](https://micans.org/mcl/). The e-value cut-off for sequence similarity, given by `-e` or `--eval`, which denotes the expected value of the hit quantifies the number of alignments of similar or better quality that you expect to find searching this query against a database of random sequences the same size as the actual target database, is the key parameter measuring the significance of a hit, which is set here as default 1e-10. Note that [DIAMOND](https://github.com/bbuchfink/diamond/wiki/1.-Tutorial) itself by default only reports all alignments with e-value < 0.001. The percentage of upper hits used for gene length normalization, given by `-np` or `--normalizedpercent`, which determines the upper percentile of hits per bin (categorized by gene length) used in the fit of linear regression, considering that not all hits per bin show apparent linear relationship, is set as default 5, indicating the usage of top 5% hits per bin. The number of bins divided in gene length normalization, given by `-bs` or `--bins`, determines the number of bins to categorize the gene length, is set as default 100. The parameter `-nn` or `--nonormalization` can be set to call off the normalization process, although it's suggested to conduct the normalization to acquire more accurate gene family clustering result. The parameters `--to_stop` and `--cds` control the behaviour of translating coding sequence into amino acid sequence. If the `--to_stop` was set, the translation would be terminated at the first in-frame stop codon, otherwise the translation would simply skip any stop codons. If the `--cds` was set, sequences that doesn't start with a valid start codon, or contains more than one in-frame stop codon, or is not dividable by 3, would be simply dropped, such that only strict complete coding sequences would be included in the subsequent analysis. The exact behaviour of `--to_stop` and `--cds` is defined and described in the [biopython](https://biopython.org/docs/1.75/api/Bio.Seq.html#Bio.Seq.Seq.translate) library. The number of parallel threads by the option `-n` or `--nthreads` can be set to accelerate the calculation within `diamond`. The directory of output or intermediate files is determined by the parameter `-o` or `--outdir`, and `-t` or `--tmpdir`, which will be created by the program itself and be overwritten if the folder has already been created. Note that the software `diamond` should be pre-installed and set in the environment path in all the analysis performed by `wgd dmd` except for the collinear coalescence analysis.

**We suggest that the default setting in which the inflation factor is set as 2, e-value cut-off as 1e-10 and other parameters in default is a good starting point, unless you specifically want to explore the effects of different parameters.** Such that the command for the delineation of whole paranome is simply as below.

```
wgd dmd Aquilegia_coerulea
```

**The delineation of RBHs**
```
wgd dmd sequence1 sequence2 -e 1e-10 -bs 100 -np 5 (-nn) (-n 4) (-c 0.9) (--ogformat) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

To delineate RBHs between two cds sequence files, the relevant parameter is mostly the same as whole paranome inference, except for the parameter `-c` or `--cscore`, which ranges between 0 and 1 and is used to relax the similarity cutoff from the exclusive reciprocal best hits to a certain ratio as to the best hits. For instance, if the gene b1 from genome B has the best hit gene a1 from genome A with the bit score as 100, which is a scoring matrix independent measure of the (local) similarity of the two aligned sequences, with larger values indicating higher similarities, given the `-c 0.9`, genes from genome A which has the bit score with gene b1 higher than 0.9x100 will also be written in the result file, which in a sense are not RBHs anymore of course, but the highly similar homologue pairs. If more than 2 sequence files were provided, every pair-wise RBHs would be calculated except for querying the same sequence itself. The number of parallel threads to booster the running speed can be set by the option `-n` or `--nthreads` which is suggested to be set as (N-1)N/2 where N is the number of cds files to achieve the highest efficiency. The option `--ogformat` can be set to add index (for instance GF00000001) to the output RBH gene families which can be further used in the *K*<sub>S</sub> calculation by `wgd ksd`.

**The suggested command to start with is also under the default setting with the command shown below**

```
wgd dmd sequence1 sequence2
```

**The delineation of local MRBHs**
```
wgd dmd sequence1 sequence2 sequence3 -f sequence1 -e 1e-10 -bs 100 -np 5 (-nn) (-n 4) (-kf) (-kd) (-c 0.9) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

**The distinction between local and global (hereunder) MRBHs is that local MRBHs are the results of merged RBHs on a joint focal species, for instance in a three species system (A,(B,C)), the local MRBHs of A only require the calculation of RBHs between A and C (denoted as AC) and AB and then the merging of AB and AC at the axis of A, while the gloabl MRBHs are independent of focal species in that it just calls the calculations of all possible species pair (not self to self species pair), such that AB, AC, and BC are all to be calculated and merged.**

Two types of MRBHs as intepretated above can be delineated by `wgd dmd`, the local MRBHs and the global MRBHs. The local MRBHs are constructed by merging all the relevant RBHs only with the focal species, which is set by `-f` or `--focus`. The parameter `-kf` or `--keepfasta` can be set to retain the sequence information of each MRBH. The parameter `-kd` or `--keepduplicates` determines whether the same genes can appear in different local MRBHs. Normally there will be no duplicates in the local MRBHs but if users set the `-c` as 0.9 (or any value smaller than 1), it's likely that the same gene will have chance to appear multiple times in different local MRBHs. That is to say, the parameter `-kd` is meaningful only when it's set together with the parameter `-c`. The number of parallel threads is suggested to be set as the number of cds files minus 1.

**A suggested starting run is under the default parameter with the command shown as below.**

```
wgd dmd sequence1 sequence2 sequence3 -f sequence1
```

**The delineation of global MRBHs**
```
wgd dmd sequence1 sequence2 sequence3 -gm -e 1e-10 -bs 100 -np 5 (-nn) (-n 4) (-kf) (-kd) (-c 0.9) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

The global MRBHs is constructed by exhaustively merging all the possible pair-wise RBHs except for querying the sequence itself, which can be initiated by add the flag `-gm` or `--globalmrbh`. The rest of relevant parameters stays the same as the local MRBHs. The number of parallel threads is suggested to be set as (N-1)N/2 too where N is the number of cds files to achieve the highest efficiency.

**A suggested starting run is under the default parameter with the command shown as below.**

```
wgd dmd sequence1 sequence2 sequence3 -gm
```

**The delineation of orthogroups**
```
wgd dmd sequence1 sequence2 sequence3 -oi -oo -e 1e-10 -bs 100 -np 5 (-nn) (-cc) (-te) (-mc 0.8) (-gn) (-tree 'fasttree') (-ts '-fastest') (-n 4) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

In `wgd v2`, we also implemented an algorithm of delineating orthogroups, which can be initiated with the parameter `-oi` or `--orthoinfer`. Two ways of delineation can be chosen, the concatenation way (set by the parameter `-cc` or `--concat`) or the non-concatenation (default) way. In brief, the concatenation way of delineating orthogroups starts with concatenating all the sequences into a single sequence file and then inferring the whole paranome of this single sequence file with the clustering results mapped back to the belonging species. While the non-concatenation way starts with respective pair-wise diamond search (including querying the same sequence itself) and then all the sequence similarity tables will be concatenated and clustered into orthogroups. Some other possibly useful post-clustering functions can be initiated, including the parameter `-te` or `--testsog`, which can be set to start the unbiased test of single-copy gene families (note that this function needs `hmmer` (v3.1b2) to be installed in the environment path), the parameter `-mc` or `--msogcut`, ranging between 0 to 1, which can be set to search the so-called mostly single-copy family which has higher than certain cut-off percentage of species coverage, the parameter `-gn` or `--getnsog`, which can be set to search for nested single-copy gene families (NSOGs) which is originally multiy-copy but has a (mostly) single-copy branch (which requires the chosen tree-inference program set by `-tree` or `--tree_method` to be pre-installed in the environment path with the parameters setting for gene tree inference controlled by `-ts` or `--treeset`). The program `wgd dmd` would still conduct the RBHs calculation unless the parameter `-oo` or `--onlyortho` was set. If one only wants to infer the orthogroups, it's suggested to add the flag `-oo` to just implement the orthogroups delineation analysis. The number of parallel threads is suggested to be set as (N+1)N/2 where N is the number of cds files to achieve the highest efficiency since the self-comparison is also included.

**The default setting of parameters is a reasonable starting point with the command as below.**

```
wgd dmd sequence1 sequence2 sequence3 -oi -oo
```

**The collinear coalescence inference of phylogeny**
```
wgd dmd sequence1 sequence2 sequence3 -ap apdata -sm smdata -le ledata -gt gtdata -coc (-tree 'fasttree') (-ts '-fastest') (-n 4) (--to_stop) (--cds) (-o wgd_dmd) (-t working_tmp)
```

A novel phylogenetic inference method named "collinear coalescence inference" is also implemented in `wgd v2`. For this analysis, users need to provide the anchor points file by `-ap` or `--anchorpoints`, the collinear segments file by `-sm` or `--segments`, the listsegments file by `-le` or `--listelements`, and the gene table file by `-gt` or `--genetable`, all of which can be produced in the program `wgd syn`. The parameter `-coc` or `--collinearcoalescence` needs to be set to start this analysis. The tree-inference program and the associated parameters can be set just as above by `-tree` or `--tree_method` and `-ts` or `--treeset`. Please also make sure the chosen tree-inference program is installed in the environment path. The program `astral-pro` is required to be installed in the environment path too. Note that there should be no duplicated gene IDs in the sequence file. The parallel threads here are to accelerate the sequence alignment and gene tree inference for each gene family and thus suggested to be set as much as the number of gene families.

**A suggested starting run of this analysis is with the simple command below.**

```
wgd dmd sequence1 sequence2 sequence3 -ap apdata -sm smdata -le ledata -gt gtdata -coc
```

### wgd focus

**The concatenation-based/coalescence-based phylogenetic inference**
```
wgd focus families sequence1 sequence2 sequence3 (--concatenation) (--coalescence) (-tree 'fasttree') (-ts '-fastest') (-n 4) (--to_stop) (--cds) (-o wgd_focus) (-t working_tmp)
```

The program `wgd focus` implemented two basic phylogenetic inference methods, i.e., concatenation-based and coalescence-based methods. To initiate these analysis, users need to set the flag option `--concatenation` or `--coalescence`. The concatenation-based method includes a few major steps, i.e., the multiple sequence alignment (MSA) of each gene family, the concatenation of all gene families and then the gene tree (also species tree in this case) inference. The coalescence-based method will instead perform the MSA of each gene family and the gene tree inference based on each MSA, and then infer the species tree based on these individual gene trees. The tree-inference program and the associated parameters can be set just as above by `-tree` or `--tree_method` and `-ts` or `--treeset`. Please also make sure the chosen tree-inference program is installed in the environment path. The program `astral-pro` is required to be installed in the environment path if the coalescence-based method is chosen. The parallel threads here are to accelerate the sequence alignment and gene tree inference for each gene family too and thus suggested to be set as much as the number of gene families.

**A suggested starting run of this analysis is with the simple command below.**

```
wgd focus families sequence1 sequence2 sequence3 --concatenation
wgd focus families sequence1 sequence2 sequence3 --coalescence
```

**The functional annotation of gene families**
```
wgd focus families sequence1 sequence2 sequence3 --annotation eggnog -ed eddata --dmnb dbdata
```

The program `wgd focus` also added some wrapping functions for functional annotation of gene families on the hood of databases and softwares for instance `EggNOG` and `EggNOG-mapper`. For the annotation using `EggNOG-mapper`, users need to provide the path to the eggNOG annotation database via the option `-ed`, the path to the diamond-compatible database via the option `--dmnd_db` and the option `--annotation` set as "eggnog". The manner how PFAM annotation will be performed can be controlled via the option `pfam`, either "none", "realign" or "denovo", the detailed explanation can be found at the wiki of [EggNOG-mapper](https://github.com/eggnogdb/eggnog-mapper/wiki). Please pre-install the `EggNOG-mapper` python package if using this function. For the annotation using `hmmscan`, what is implemented in `wgd v2` is a simple bundle function to perform `hmmscan` analysis for a given hmm profile and set of gene families such that users need to set the option `--annotation` as "hmmpfam" and provide the hmmprofile via the option `--hmm`. For the annotation using `interproscan`, users need to provide the path to the interproscan installation folder where there is a `interproscan.sh` file via the option `--exepath` and set the the option `--annotation` as "interproscan". The parallel threads here are to parallelize the analysis for each gene family and thus suggested to be set as much as the number of gene families.

**A suggested starting run of this analysis can be with the command below.**

```
wgd focus families sequence1 sequence2 sequence3 --annotation eggnog -ed eddata --dmnb dbdata
wgd focus families sequence1 sequence2 sequence3 --annotation hmmpfam --hmm hmmdata
wgd focus families sequence1 sequence2 sequence3 --annotation interproscan --exepath $PATH
```

**The phylogenetic dating of WGDs**
```
wgd focus families sequence1 sequence2 sequence3 -d mcmctree -sp spdata (--protcocdating) (--partition) (--aamodel lg) (-ds 'burnin = 2000') (-ds 'sampfreq = 1000') (-ds 'nsample = 20000')  (-n 4) (--to_stop) (--cds) (-o wgd_focus) (-t working_tmp)
```

The absolute dating of WGDs is a specific pipeline implemented in `wgd v2` using the method of phylogenetic dating. The families used in this step can be produced from `wgd dmd` and `wgd peak`. Note that here we only discuss how to date WGDs with genome assembly, instead of transcriptome assembly (which will be discussed in a separate section hereunder). The assumption we made here is that not all anchor pairs (collinear duplicates) are suitable for phylogenetic dating, for instance those fastly or slowly evolving gene duplicates, because they're prone to give biased dating estimation. So as to retrieve the "reliable" anchor pairs, we implemented some methods of identifying crediable anchor pairs based on their Ks values and/or residing collinear segments. For a genome with clear signals of putative WGDs, such as the *Aquilegia coerulea* in the example, a heuristic method that applies the principle of how [ksrates](https://github.com/VIB-PSB/ksrates) find the initial peaks and their parameters was implemented to find the 95% confidence level of the assumed lognormal distribution of the anchor pair *K*<sub>S</sub> age distribution to filter anchor pairs with too high or low *K*<sub>S</sub> ages, the examplar command of which is showed in the `wgd peak` section of [Illustration](https://github.com/heche-psb/wgd?tab=readme-ov-file#illustration). If this heuristic method failed to give reasonable results, which is usually due to the effect of multiple adjacent WGDs that blurs the peak-finding process, users can turn to the collinear segments-guided anchor pair clustering implemented in `wgd v2`, in which a collinear segment-wise GMM clustering will be first conducted based on the "so-called" segment *K*<sub>S</sub> age represented by the median *K*<sub>S</sub> age of all the residing syntelogs (note that the gene duplicate pairs adopted in this step is from the file `multiplicon_pairs.txt` which contains the full set of syntelogs), instead of the smaller gene set of anchor pairs. The distinction between anchor pairs and syntelogs is that the latter refers to multiple sets of genes derived from the same ancestral genomic region while the former implies the latter but requires additionally the conserved gene order, both of which are, within a genome, assumed to be originated from the duplication of a common ancestral genomic region and as such deemed evidence for WGD. Then the syntelogs will be mapped back according to the clustering results of their affiliated segments. Since the Gaussian shape of the segment cluster doesn't necessarily imitate the shape of the residing syntelogs, so as to retrieve the "reliable" gene sets for phylogenetic dating, we adpoted the (95%) highest density region (HDR) of the syntelog *K*<sub>S</sub> age distribution for the phylogenetic dating, the calculation of the (95%) HDR in the function [calculateHPD](https://github.com/heche-psb/wgd/blob/phylodating/wgd/peak.py#L1757) seeks the shortest *K*<sub>S</sub> range (a,b) which satisfies the requirement of spanning more than (95%) of all the *K*<sub>S</sub> values. On the premise of identified anchor pairs (note that the syntelogs are also referred as anchor pairs hereafter), we implemented in the program `wgd dmd` the so-called anchor-aware local MRBHs or orthogroups, in which the original local MRBHs are further merged with anchor pairs such that each orthogroup contains the anchor pair and the orthologues. With this orthogroup, users then need one starting tree file (as shown in the [Illustration](https://github.com/heche-psb/wgd/tree/phylodating?tab=readme-ov-file#illustration) part) indicating the tree topology and fossil calibration information for the final phylogenetic dating using the program `wgd focus`. Users can set the flag option `--protcocdating` to only conduct the dating of the concatenation protein MSA or the flag option `--protdating` to only conduct the dating of the protein MSAs, noted that these two options only work for the `mcmctree` option so far. The flag option `--partition` can be set to perform `mcmctree` analysis using the partitioned data (i.e., 1st, 2nd and 3rd position of codon) instead of using the codon as a whole. The option `--aamodel` can be set to determine the amino acid model applied in `mcmctree` analysis. The option `-ds` can be used to set parameters for the molecular dating program. The parallel threads here are to parallelize the analysis for each gene family and thus suggested to be set as much as the number of gene families.

**A suggested starting run can be with the command below.**

```
wgd focus families sequence1 sequence2 sequence3 -d mcmctree -sp spdata
wgd focus families sequence1 sequence2 sequence3 -d r8s -sp spdata --nsites nsiteinfo
wgd focus families sequence1 sequence2 sequence3 -d beast --fossil fossilinfo --rootheight rootheightinfo --chainset chainsetinfo --beastlgjar $PATH
```

### wgd ksd

**The construction of whole paranome *K*<sub>S</sub> age distribution**
```
wgd ksd families sequence (-o wgd_ksd -t wgd_ksd_tmp --nthreads 4 --to_stop --cds --pairwise --strip_gaps --aligner mafft --tree_method fasttree --node_average)
```

The program `wgd ksd`, as impied by its name, is for the construction of *K*<sub>S</sub> age distribution. Except for the aforementioned parameters such as `--to_stop` and `--cds`, there are some important parameters that have crucial impact on the *K*<sub>S</sub> estimation. The option `--pairwise` is a very important parameter for the *K*<sub>S</sub> estimation, with which the `CODEML` will calculate the *K*<sub>S</sub> for each gene pair separately based on the alignment of only these two genes instead of the whole alignment of the family, such that less gaps are expected and thus the alignment in the consideration of `CODEML` will be longer (because `CODEML` will automately skip every column with gap, regardless of whether it's an overall or partial gap), without which the `CODEML` will calculate the *K*<sub>S</sub> based on the whole alignment of the family, which might have no *K*<sub>S</sub> result at all if the stripped alignment length (removing all gap-containing columns) was zero, a cause of different number of *K*<sub>S</sub> estimates between "pairwise mode" and "non-pairwise mode". It's difficult to say which mode is more ideal, although the "non-pairwise mode" (default setting) which runs on the whole alignment instead of a local alignment, might be more biologically conserved in that it assures the evolution of each column to be started from the root of the family and all the gene duplicates are taken into account in the *K*<sub>S</sub> estimation process. The option `--strip_gaps` can remove all the gap-containing columns, with or without which the result of "non-pairwise mode" won't be affected, while with which the result of "pairwise mode" will be altered. The option `--aligner` and `--aln_options` which decide which alignment program to be used and which parameter to be set, will have impact on the *K*<sub>S</sub> results, noted that the default program is `mafft` and the parameter is `--auto`. The option `--tree_method` and `--tree_options` decide which gene tree inference program to be used and which parameter to be set, won't affect the *K*<sub>S</sub> estimation itself but the result of de-redundancy, noted that we implemented a built-in gene tree inference method based on the Average Linkage Clustering (ALC) (thus a distance-based tree) with the `--tree_method` set as "cluster". Two methods of de-redundancy are implemented in `wgd v2`, namely node-weighted and node-averaged methods. The node-weighted method achieves the de-redundancy via weighing the *K*<sub>S</sub> value associated with each gene pair such that the weight of a single duplication event sums up to 1 (noted that the number of *K*<sub>S</sub> estimation remains the same) while the node-averaged method realizes the de-redundancy via calculating per gene tree node one averaged *K*<sub>S</sub> value to represent the age of each duplication event. The option `--node_average` can be set to choose the node-averaged way of de-redundancy. Different methods of de-redundancy will have impact on the detection of WGD signals, which has been investigated in this [literature](https://doi.org/10.1093/gbe/evy200). The parallel threads here are to parallelize the analysis for each gene family and thus suggested to be set as much as the number of gene families.

**A suggested starting run can use command as below**

```
wgd ksd families sequence
```

**The construction of orthologous *K*<sub>S</sub> age distribution**
```
wgd ksd families sequence1 sequence2 sequence3 (--reweight)
```

From paralogous to orthologous *K*<sub>S</sub> age distribution, users only need to provide more cds files. Note that with orthologous gene families the weighting method can be set to be calculated per species pair instead of considering the whole family because when plotting orthologous *K*<sub>S</sub> age distribution between two species the weight calculated from this specific species pair should be conserved while the one calculated from the whole family will vary with the number of species. To initiate the weighting per species pair, the option `--reweight` can be set.

**A suggested starting run can use command as below**

```
wgd ksd families sequence1 sequence2 sequence3
```

**The construction of *K*<sub>S</sub> age distribution with rate correction**
```
wgd ksd families sequence1 sequence2 sequence3 --focus2all sequence1 -sp spdata --extraparanomeks paranomeKsdata (--plotelmm --plotapgmm --anchorpoints apdata --reweight --onlyrootout)
```

Inspired by the rate correction algorithm in [ksrates](https://github.com/VIB-PSB/ksrates), we implemented the rate correction analysis also in `wgd v2`, which is mostly the same as [ksrates](https://github.com/VIB-PSB/ksrates) but differs in the calculation of the standard deviation of rescaled *K*<sub>S</sub> ages. To perform the rate correction analysis, users can use both the `wgd ksd` and `wgd viz` program. For the `wgd ksd` program, users need to provide a species tree via the option `--speciestree` on which rate correction can be conducted. Note that unnecessary brackets might lead to unexpected errors, for instance a tree `(A,(B,C));` should not be represented as `(A,((B,C)));`. The set of species pairs to be shown is flexible that the most convenient option is `--focus2all` which simply shows all the possible focal-sister species pairs, or users can manually set the species pairs via the option `--spair`. Note that if the species pairs were manually set, it would be needed to co-set the option `--classic`. The option `--onlyrootout` can be set to only consider outgroup at the root, instead of all the possible outgroups per focal-sister species pair, which has impact on the final corrected *K*<sub>S</sub> ages. We suggest of using all the possible outgroups per focal-sister species pair as for a less biased result. The paranome *K*<sub>S</sub> data should be provided via the option `--extraparanomeks`. 

Some other options which have no impact on the rate correction but add more layers or change the appearance on the mixed *K*<sub>S</sub> age distribution, include `--plotapgmm` and `--plotelmm` etc. The option `--plotapgmm` can be set to call the GMM analysis upon the anchor pair *K*<sub>S</sub> age distribution and plot the clustering result upon the mixed *K*<sub>S</sub> age distribution, which has to be co-set with the option `--anchorpoints` providing the anchor pairs information. The option `--plotelmm` can be set to call the ELMM analysis upon the whole paranome *K*<sub>S</sub> age distribution. **Note that the species names present in the species tree file should match the names of the corresponding sequence files** For instance, given the cds file names 'A.cds','B.cds','C.cds', the species tree could be '(A.cds,(B.cds,C.cds));' rather than '(A,(B,C));'. There is no requirement for the name of the paranome *K*<sub>S</sub> datafile which can be named in whatever manner users prefer. The 'GMM' is the abbreviation of Gaussian Mixture Modeling while the 'ELMM' refers to Exponential-Lognormal Mixture Modeling as [ksrates](https://github.com/VIB-PSB/ksrates) interprets.

There are 21 columns in the result `.ks.tsv` file besides the index columns `pair` as the unique identifier for each gene pair. The `N`, `S`, `dN`, `dN/dS`, `dS`, `l` and `t` are from the codeml results, representing the N estimate, the S estimate, the dN estimate, the dN/dS (omega) estimate, the dS estimate, the log-likelihood and the t estimate, respectively. The `alignmentcoverage`, `alignmentidentity` and `alignmentlength` are the information pertaining to the alignment for each family, representing the ratio of the stripped alignment length compared to the full alignment length, the ratio of columns with identical nucleotides compared to the overall columns of the stripped alignment, and the length of the full alignment, respectively.
### wgd mix

**The mixture model clustering analysis of *K*<sub>S</sub> age distribution**
```
wgd mix ksdata (--n_init 200 --max_iter 200 --ks_range 0 5 --filters 300 --bins 50 --components 1 4 --gamma 0.001)
```

This part of Gaussian mixture modeling (GMM) analysis is inherited from the original `wgd` program, but writes additionally the probability of each *K*<sub>S</sub> value into the final dataframe. Basically, users need to provide with a (normally from whole-paranome or anchor-pairs) *K*<sub>S</sub> datafile and the GMM analysis will be conducted upon the datafile. Some parameters can affect the results, including `--n_init`, which sets the number of k-means initializations (default 200), `--max_iter`, which sets the maximum number of iterations (default 200), `--method`, which determines which clustering method to use (default gmm), `--gamma`, which sets the gamma parameter for the bgmm model (default 0.001), `--components`, which sets the range of the number of components to fit (default 1 4), and the data filtering parameters `--filters` which filters data based on alignment length, `--ks_range` which filters data based on *K*<sub>S</sub> values and the parameter `--bins` which sets the number of bins in *K*<sub>S</sub> distribution (default 50).

**A suggested starting run can use command simply as below**

```
wgd mix ksdata
```

### wgd peak

**The search of crediable *K*<sub>S</sub> range used in WGD dating**
```
wgd peak ksdata -ap apdata -sm smdata -le ledata -mp mpdata --heuristic (--alignfilter 0.0 0 0.0 --ksrange 0 5 --bin_width 0.1 --guide segment --prominence_cutoff 0.1 --rel_height 0.4 --ci 95 --hdr 95 --kscutoff 5)
```

As mentioned previously, a heuristic method and a collinear segments-guided anchor pair clustering for the search of crediable *K*<sub>S</sub> range used in WGD dating are implemented in `wgd v2`. Users need to provide the anchor points, segments, listsegments, multipliconpairs datafile from `i-adhore` to achieve the clustering function. Some parameters that can impact the results include `--alignfilter`, which filters the data based on alignment identity, length and coverage, `--ksrange`, which sets the range of Ks to be analyzed, `--bin_width`, which sets the bandwidth of *K*<sub>S</sub> distribution, `--weights_outliers_included` which determines whether to include *K*<sub>S</sub> outliers (whose value is over 5) in analysis, `--method` which determines which clustering method to use (default gmm), `--seed` which sets the random seed given to initialization (default 2352890), `--n_init`, which sets the number of k-means initializations (default 200), `--em_iter`, which sets the maximum number of iterations (default 200), `--gamma`, which sets the gamma parameter for the bgmm model (default 0.001), `--components`, which sets the range of the number of components to fit (default 1 4), `--weighted` which determines whether to use node-weighted method for de-redundancy, `--guide` which determines which regime residing anchors to be used (default Segment), `--prominence_cutoff` which sets the prominence cutoff of acceptable peaks in peak finding process, `--rel_height` which sets the relative height at which the peak width is measured, `--kstodate` which manually sets the range of *K*<sub>S</sub> to be dated in heuristic search and needs to be co-set with option `--manualset`, `--xlim` and `--ylim` determining the x and y axis limit of GMM *K*<sub>S</sub> distribution, `--ci` setting the confidence level of log-normal distribution to date (default 95), `--hdr` setting the highest density region (HDR) applied in the segment-guided anchor pair *K*<sub>S</sub> distribution (default 95), `--heuristic` determining whether to initiate heuristic method of defining CI for dating, `--kscutoff` setting the *K*<sub>S</sub> saturation cutoff in dating (default 5).

Four result subfloders will be produced, namely `AnchorKs_FindPeak`, `AnchorKs_GMM`, `SegmentGuideKs_GMM` and `SegmentKs_FindPeak`. The `AnchorKs_FindPeak` subfloder contains results of the detected peaks by the `signal` module of `SciPy` library and the assumed highest mass part (referred to as HighMass hereafter) of each peak, which can be used for further WGD dating. The `AnchorKs_GMM` shows the GMM results upon the original anchor *K*<sub>S</sub> distribution by the `mixture` module of `scikit-learn` library and two subfloders, `LogGMM_CI` containing the results of 95% CI of each component, `HighMass_CI` containing the HighMass of each component, which can be used for further WGD dating. The `SegmentGuideKs_GMM` subfolder presents results of segment *K*<sub>S</sub> GMM which are mapped back to the residing anchor pairs and the associated 95% HDR and HighMass of each segment cluster in subfloders of `HDR_CI` and `HighMass_CI`. The `SegmentKs_FindPeak` subfolder is similar to `AnchorKs_FindPeak` but with segment *K*<sub>S</sub> instead. The *K*<sub>S</sub> in `Multiplicon` can also be calculated in place of `Segment` using the option `--guide` as such the result title, label, file and folder names will be changed accordingly.

**A suggested starting run can use command simply as below**

```
wgd peak ksdata -ap apdata -sm smdata -le ledata -mp mpdata --heuristic
```

### wgd syn

**The intra-specific synteny inference**
```
wgd syn families gff (--ks_distribution ksdata -f gene -a ID --minlen -1 --minseglen 10000 --mingenenum 30)
```

The program `wgd syn` is mainly dealing with collinearity or synteny (both referred to as synteny hereafter) analysis. Two input files are essential, the gene family file and the gff3 file. The gene family file is in the format as `OrthoFinder`. The software `i-adhore` is a prerequisite. With default parameters, the program basically conducts 1) filtering gene families based on maximum family size 2) retrieving gene position and scaffold information from gff3 file 3) producing the configuration file and associated datafiles for `i-adhore` 4) calling `i-adhore` given the parameters set to infer synteny 5) visualizing the synteny in "dotplot" in the unit of genes and bases, in "Syndepth" plot showing the distribution of different categories of collinearity ratios within and between species, in "dupStack" plot showing multiplicons with different multiplication levels. 6) if with *K*<sub>S</sub> data, a "*K*<sub>S</sub> dotplot" with dots annotated in *K*<sub>S</sub> values and a *K*<sub>S</sub> distribution with anchor pairs denoted will be produced. The gene information in the gene family file and gff3 file should be matched which requires users to set proper `--feature` and `--attribute`. The maximum family size to be included can be set via the option `--maxsize`, noted that this filtering is mainly to drop those huge tandem duplicates family and transposable elements (TEs) family, and not mandatory. Users can filter those fragmentary scaffolds via the option `--minlen`. The minimum length and number of genes for a segment to be considered can be set via the option `--minseglen` and `--mingenenum`. Redundant multiplicons can be kept by set the flag option `--keepredun`.

**A suggested starting run can use command simply as below**

```
wgd syn families gff
```

**The inter-specific synteny inference**
```
wgd syn families gff1 gff2 (--additionalgffinfo "mRNA;Name" --additionalgffinfo "gene;ID")
```

For multi-species synteny inference, if users have gff3 files which have different features or attributes for gene position information retrieval, the option `--additionalgffinfo` can be set to provide the additional information. The remaining parameter setting is the same as the intra-specific synteny inference.

**A suggested starting run can use command simply as below**

```
wgd syn families gff1 gff2
```

### wgd viz

**The visualization of *K*<sub>S</sub> age distribution and ELMM analysis**
```
wgd viz -d ksdata
```

The program `wgd viz` is mainly for the purpose of *K*<sub>S</sub> distribution and synteny visualization, with some optional mixture modeling analysis. The basic function is just to plot the *K*<sub>S</sub> distribution and conduct an ELMM analysis in search of potential WGD components. Some key parameters affecting the ELMM result include `--prominence_cutoff` and `--rel_height`, which have been explained ahead, `--em_iterations` and `--em_initializations` determining the maximum iterations and initializations in the EM algorithm.

**A suggested starting run can use command simply as below**

```
wgd viz -d ksdata
```

**The visualization of *K*<sub>S</sub> age distribution with rate correction**
```
wgd viz -d ksdata -sp spdata --focus2all focal_species --extraparanomeks ksdata (--anchorpoints apdata --plotapgmm --plotelmm)
```

Besides the basic *K*<sub>S</sub> plot, substitution rate correction can also be achieved given at least a species tree (via the option `--speciestree`) and a focal species (either via the option `--focus2all` or via the option `--spair` in the form of "$focal_species;$focal_species"). It's suggested that the orthologous *K*<sub>S</sub> data is provided by the `--datafile` option and the paralogous *K*<sub>S</sub> data is provided by the `--extraparanomeks` option, although it's allowed to only provide *K*<sub>S</sub> data via the `--datafile` option and deposit both orthologous and paralogous *K*<sub>S</sub> data thereon. There are two types of mixed plots (the "mixed" here refers to mixed orthologous and paralogous *K*<sub>S</sub> distributions), one of which is similar to what `ksrates` plots, while the other of which is like the conventional *K*<sub>S</sub> plot that both the original orthologous and paralogous *K*<sub>S</sub> distributions are truthfully plotted instead of just being represented by some vertical lines. We suggest users to adopt the `ksrates`-like plots, which is the default. Otherwise the option `--classic` will be needed to set. Extra mixture modeling analysis can be initiated via the option `--plotelmm` and `--plotapgmm` with the anchor points datafile provided by the `--anchorpoints` option.

**A suggested starting run can use command simply as below**

```
wgd viz -d ksdata -sp spdata --focus2all focal_species --extraparanomeks ksdata
```

**The visualization of synteny**
```
wgd viz -ap apdata -sm smdata -mt mtdata -gt gtdata --plotsyn (--minlen -1 --minseglen 10000 --mingenenum 30)
```

Compared to the original `wgd`, the `wgd viz` program added synteny visualization pipeline. Users need to provide the flag option `--plotsyn` to initiate this part of pipeline. This step assumes that users have obtained already the syntenic results from `i-adhore` and uses those result files to realize the synteny visualization. Simliar to `wgd syn`, the extra *K*<sub>S</sub> data file can be transmitted via the option `--datafile` (instead of `--extraparanomeks` option). Basically, the syntenic result files required are the anchor points datafile, multiplicons datafile, gene table datafile (automately produced by `wgd syn`), and segments datafile.

**A suggested starting run can use command simply as below**

```
wgd viz -ap apdata -sm smdata -mt mtdata -gt gtdata --plotsyn
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

![](data/syn_results/Aquilegia_coerulea_Aquilegia_coerulea_multiplicons_level.png)

We implemented two types of dot plots in oxford grid: one in the unit of bases and the other in the unit of genes, which can be colored by *K*<sub>S</sub> values given *K*<sub>S</sub> data.

![](data/syn_results/Aquilegia_coerulea-vs-Aquilegia_coerulea_Ks.dot_unit_gene.png)

As shown above, the dot plot in the unit of genes presents numerous densely aggregated (line-like) anchor points at most of the chromosomes with consistent *K*<sub>S</sub> age between 1 and 2. The dot plot in the unit of bases shows the same pattern, as manifested below.

![](data/syn_results/Aquilegia_coerulea-vs-Aquilegia_coerulea_Ks.dot.png)

The dot plots without *K*<sub>S</sub> annotation will also be automately produced, as shown below.

![](data/syn_results/Aquilegia_coerulea-vs-Aquilegia_coerulea.dot_unit_gene.png)

![](data/syn_results/Aquilegia_coerulea-vs-Aquilegia_coerulea.dot.png)

Note that the opacity of anchor dots and all homolog dots can be set by the option `--apalpha` and `--hoalpha` separately. If one just wants to see the anchor dots, setting the `hoalpha` as 0 (or other minuscule values) will do. If one wants to see the distribution of whole dots better, setting the `hoalpha` higher (and `apalpha` lower) will do. The `dotsize` option can be called to adjust the size of dots.

A further associated Syndepth plot shows that there are more than 50 duplicated segments longer than 10000 bp and 30 genes (so as to drop fragmentary segments), which dominates the whole collinear ratio category.

![](data/syn_results/Syndepth.svg)

More exquisite plots including both intra-specific and inter-specific comparisons using the orthogroups (composed of *Aquilegia coerulea*, *Protea cynaroides*, *Acorus americanus* and *Vitis vinifera*, see further for the context) inferred hereinafter can be also produced using `wgd syn`. Note that different genome assemblies might have different features and attributes which can be accommodated via the option `--additionalgffinfo` for each genome assembly whose order needs to follow the order of gff3 files, for instance 'mNRA;Name' for `Aquilegia_coerulea.gff3`, 'mNRA;ID' for `Protea_cynaroides.gff3`, 'mNRA;Name' for `Acorus_americanus.gff3` and 'mNRA;Name' for `Vitis_vinifera.gff3`.

```
wgd syn wgd_ortho/Orthogroups.sp.tsv -ks wgd_ortho_ks/Orthogroups.sp.tsv.ks.tsv Aquilegia_coerulea.gff3 --additionalgffinfo 'mNRA;Name' Protea_cynaroides.gff3 --additionalgffinfo 'mNRA;ID' Acorus_americanus.gff3 --additionalgffinfo 'mNRA;Name' Vitis_vinifera.gff3 --additionalgffinfo 'mNRA;Name' -o wgd_ortho_syn
```

Upon the acquisition of the collinear results using `wgd syn`, the same collinear plots can be also produced by `wgd viz` using the command below.
```
wgd viz --plotsyn -sm wgd_ortho_syn/iadhore-out/segments.txt -ap wgd_ortho_syn/iadhore-out/anchorpoints.txt -mt wgd_ortho_syn/iadhore-out/multiplicons.txt -gt wgd_ortho_syn/gene-table.csv -d wgd_ortho_ks/Orthogroups.sp.tsv.ks.tsv -o wgd_ortho_viz
```

![](data/ortho_syn_results/Aquilegia_coerulea_Vitis_vinifera_multiplicons_level.png)
The above `dupStack` plot shows the distribution of duplicated segments of *Aquilegia coerulea* compared to itself (in green) and compared to *Vitis vinifera* (in blue) over the chromosomes of *A. coerulea*.
![](data/ortho_syn_results/Overallspecies_Ks.dot_unit_gene.png)
The above *K*<sub>S</sub> dotplot in unit of gene shows the overall distribution of collinearity acorss the four species involved.
![](data/ortho_syn_results/Overallspecies.dot_unit_gene.png)
The above dotplot is without the annotation of *K*<sub>S</sub> ages compared to the last one.
![](data/ortho_syn_results/Syndepth.svg)
The above Syndepth plot shows the collinear ratio acorss all species pairs (intra-specific comparison in green while inter-specific comparison in blue).

We can fit an ELMM mixture model upon the whole paranome *K*<sub>S</sub> age distribution to see more accurately the significance and location of potential WGDs, using the command line below.

```
wgd viz -d wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv
```

The result of ELMM mixture model clustering shows that there is a likely WGD component at *K*<sub>S</sub> 1.19.

![](data/elmm_Aquilegia_coerulea.tsv.ks.tsv_best_models_weighted.svg)

Let's do a mixture model clustering for anchor *K*<sub>S</sub> too, using the command line below. Note that this step will automately call the segment *K*<sub>S</sub> clustering analysis too.

```
wgd peak wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv --anchorpoints wgd_syn/iadhore-out/anchorpoints.txt --segments wgd_syn/iadhore-out/segments.txt --listelements wgd_syn/iadhore-out/list_elements.txt --multipliconpairs wgd_syn/iadhore-out/multiplicon_pairs.txt (--weighted)
```

The anchor *K*<sub>S</sub> age distribution also has a likely WGD component with mode 1.28.

![](data/Original_AnchorKs_GMM_Component3_node_averaged_Lognormal.svg)

Now that we have seen the evidence of numerous duplicated segments and the aggregation of duplicates age at *K*<sub>S</sub> 1.28 or 1.19 for anchor pairs and non-anchor pairs throughout the whole genome. We can claim with some confidence that *Aquilegia coerulea* might have experienced a paleo-polyploidization event. Next, Let's have a further look about its phylogenetic location. We know that there are uncertainties about whether this putative paleo-polyploidization event is shared with all eudicots or not. We can choose some other eudicot genomes to see the ordering of speciation and polyploidization events. Here we choose *Vitis vinifera*, *Protea cynaroides* and *Acorus americanus* in the following *K*<sub>S</sub> analysis. First, we built a global MRBH family using the command below.

```
wgd dmd --globalmrbh Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera -o wgd_globalmrbh
```

In the global MRBH family, every pair of orthologous genes is the reciprocal best hit, suggesting true orthologous relationships. We would use the *K*<sub>S</sub> values associated with these orthologous pairs to delimit the divergence *K*<sub>S</sub> peak. Together with the whole paranome *K*<sub>S</sub> distribution, we conduct the rate correction using the command below.

!!Since `wgd` version 2.0.24, we rewrote a cleaner and quicker way of doing substitution rate correction. It's not required to type in any speices pair and a series of *K*<sub>S</sub> plots will be produced. The required files are orthologous *K*<sub>S</sub> datafile, paralogous *K*<sub>S</sub> datafile, a species tree and a focal species (the one inputted with paralogous *K*<sub>S</sub> data). Users can choose to add one more layer of ELMM analysis on paralogous *K*<sub>S</sub> values and/or GMM analysis on anchor *K*<sub>S</sub> distribution. The orthologous *K*<sub>S</sub> distribution can be calculated using the command below.

```
wgd ksd wgd_globalmrbh/global_MRBH.tsv Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera -o wgd_globalmrbh_ks
```

With the calculated orthologous *K*<sub>S</sub> distribution, we can use the command below to conduct the rate correction and/or mixture modeling analysis.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv -fa Aquilegia_coerulea -epk wgd_ksd/Aquilegia_coerulea.ks.tsv -ap wgd_syn/iadhore-out/anchorpoints.txt -sp speciestree.nw -o wgd_viz_mixed_Ks --plotelmm --plotapgmm --reweight
```

or using the command below, which combines the two steps above in one. Note that we suggest of taking two separate steps in which `wgd ksd` undertakes the calculation of orthologous *K*<sub>S</sub> distribution while `wgd viz` carries out the rate correction and GMM analysis such that it's easier to debug.
```
wgd ksd wgd_globalmrbh/global_MRBH.tsv Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera --extraparanomeks wgd_ksd/Aquilegia_coerulea.tsv.ks.tsv -sp speciestree.nw --reweight -o wgd_globalmrbh_ks_rate_correction -fa Aquilegia_coerulea -ap wgd_syn/iadhore-out/anchorpoints.txt --plotelmm --plotapgmm
```

The file `speciestree.nw` is the text file of species tree in newick that rate correction would be conducted on. Its content is as below. Users can optionally provide the species pairs to be plotted but we suggest of just using `-fa Aquilegia_coerulea` to plot all possible focal-sister species pairs. We suggest adding the option `--reweight` to recalculate the weight per species pair such that the weight of orthologous gene pairs will become 1. Extra collinear data can be added by the option `-ap` and additional clustering analysis can be initiated by setting the option `--plotapgmm` and `--plotelmm`.

```
(((Vitis_vinifera,Protea_cynaroides),Aquilegia_coerulea),Acorus_americanus);
```

![](data/Mixed.ks.Aquilegia_coerulea.node.weighted.png)

The mixed *K*<sub>S</sub> distribution shown above is a publication-ready figure that assembles the results of ELMM, GMM and rate correction. The one-vs-one orthologous *K*<sub>S</sub> distributions is also automatically produced with rate correction results superimposed where available as shown below.

![](data/Focus_sister_pairs.ks.node.weighted.png)
![](data/All_pairs.ks.node.weighted.png)

Besides the above ksrates-like *K*<sub>S</sub> plot, a more classic *K*<sub>S</sub> plot can be made by adding the option `--classic` to tap more detailedly into the variation of synonymous substitution rate.

Using the command below, the direction of rate correction and degree of rate variation can be observed more directly.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv -fa Aquilegia_coerulea -epk wgd_ksd/Aquilegia_coerulea.ks.tsv -ap wgd_syn/iadhore-out/anchorpoints.txt -sp speciestree.nw -o wgd_viz_mixed_Ks --plotelmm --plotapgmm --reweight --plotkde --classic
```

![](data/Aquilegia_coerulea_GlobalmrbhKs_Elmm_Apgmm_Corrected.ksd.svg)

As shown above, because of the higher substitution rate of *Aquilegia coerulea*, the original orthologous *K*<sub>S</sub> values were actually underestimated in the time-frame of *Aquilegia coerulea*. When we recovered the divergence substitution distance in terms of two times of the branch-specific contribution of *A. coerulea* since its divergence with the sister species plus the shared substitution distance before divergence (in relative to the outgroup), the corrected *K*<sub>S</sub> mode became larger.

Note that we can easily show that *Aquilegia coerulea* has higher substitution rate than *Protea cynaroides* and *Vitis vinifera* by comparing their substitution distance in regard to the same divergence event with outgroup species *Acorus_americanus*, using command below.

```
wgd viz -d wgd_globalmrbh_ks/global_MRBH.tsv.ks.tsv -sp speciestree.nw --reweight -o wgd_viz_Compare_rate --spair "Acorus_americanus;Protea_cynaroides" --spair "Aquilegia_coerulea;Acorus_americanus" --spair "Vitis_vinifera;Acorus_americanus" --plotkde --classic
```

![](data/Raw_Orthologues_Compare_rate.ksd.svg)

As displayed above, the orthologous *K*<sub>S</sub> values bewteen *Aquilegia coerulea* and *Acorus americanus* has the highest mode, indicating the faster substitution rate of *A. coerulea* compared to *Protea cynaroides* and *Vitis vinifera*.

Before v2.0.21, the gene-species map file is neccessarily needed for its implementation in `wgd viz`, which should be automately produced by the last `wgd ksd` step given the `spair` and `speciestree` parameters. The `gene_species.map` has contents as below in which each line is the joined string of gene name and species name by space. After v2.0.21 (included), the gene-species map file is not neccessarily needed anymore.

```
Aqcoe6G057800.1 Aquilegia_coerulea
Vvi_VIT_201s0011g01530.1 Vitis_vinifera
Pcy_Procy01g08510 Protea_cynaroides
Aam_Acora.04G142900.1 Acorus_americanus
```

An alternative way to calculate the orthologous *K*<sub>S</sub> is to directly use the orthogroups instead of global MRBH family. That way we don't use the pre-inferred paranome *K*<sub>S</sub> but the paralogous gene pairs inside each orthogroup instead. To achieve that, we first need to infer orthogroups using the command below.

```
wgd dmd Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera --orthoinfer -o wgd_ortho (--onlyortho) 
```

Users can decide to only conduct the orthogroup analysis while skipping other analysis by adding the flag `--onlyortho`. Next step is the same with global MRBH family.

```
wgd ksd wgd_ortho/Orthogroups.sp.tsv Aquilegia_coerulea Protea_cynaroides Acorus_americanus Vitis_vinifera -o wgd_ortho_ks
wgd viz -d wgd_ortho_ks/Orthogroups.sp.tsv.ks.tsv -epk wgd_ksd/Aquilegia_coerulea.ks.tsv -sp speciestree.nw --reweight -ap wgd_syn/iadhore-out/anchorpoints.txt -plotelmm --plotapgmm -o wgd_ortho_ks_rate_correction
```

![](data/Mixed.ks.Aquilegia_coerulea.node.weighted_Ortho_Ks.png)
![](data/Focus_sister_pairs.ks.node.weighted_Ortho_Ks.png)
![](data/All_pairs.ks.node.weighted_Ortho_Ks.png)

As shown above, the number of orthologous gene pairs is different than the one from global MRBH families in that here we plotted all orthologous gene pairs instead of only global MRBH families, together with different recalculated weights.

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
23:49:37 INFO     Posterior mean for the ages of wgd is 112.8945 mcmctree.py:296
                  million years from Concatenated peptide
                  alignment and 95% credibility intervals (CI)
                  is 101.224-123.121 million years
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
 
Please cite us at https://doi.org/10.1007/978-1-0716-2561-3_1 and https://doi.org/10.1093/bioinformatics/btae272.

```
Hengchi Chen, Arthur Zwaenepoel (2023). Inference of Ancient Polyploidy from Genomic Data. In: Van de Peer, Y. (eds) Polyploidy. Methods in Molecular Biology, vol 2545. Humana, New York, NY. https://doi.org/10.1007/978-1-0716-2561-3_1
Hengchi Chen, Arthur Zwaenepoel, Yves Van de Peer (2024). wgd v2: a suite of tools to uncover and date ancient polyploidy and whole-genome duplication. Bioinformatics, https://doi.org/10.1093/bioinformatics/btae272
```

For citation of the tools used in wgd, please consult the documentation at
https://wgdv2.readthedocs.io/en/latest/citation.html.

