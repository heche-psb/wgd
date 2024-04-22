import pandas as pd
import numpy as np
import subprocess as sp
import logging
import os
import re
from Bio.Align import MultipleSeqAlignment
from Bio import Phylo

def _mkdir(dirname):
    if not os.path.isdir(dirname) :
        os.mkdir(dirname)
    return dirname

def _cp2tmp(alnf, tree, tmpdir):
    """
    cp the aln and tree file to mcmctree_tmp file
    """
    if not os.path.isdir(tmpdir):
        logging.error("tmpdir not existing!")
    cmd = ["cp", alnf, tmpdir]
    cmdt = ["cp", tree, tmpdir]
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    sp.run(cmdt, stdout=sp.PIPE, stderr=sp.PIPE)

def _mv_(fname, dirname):
    cmd = ["mv", fname, dirname]
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

def writedayhoff(dirname):
    """
    write Dayhoff-DCMut empirical rate matrix with gamma rates among sites for protein model
    """
    dayhoff = os.path.join(dirname, 'dayhoff.dat')
    with open(dayhoff,'w') as f:
        f.write('0.267828\n')
        f.write('0.984474 0.327059\n')
        f.write('1.199805 0.000000 8.931515\n')
        f.write('0.360016 0.232374 0.000000 0.000000\n')
        f.write('0.887753 2.439939 1.028509 1.348551 0.000000\n')
        f.write('1.961167 0.000000 1.493409 11.388659 0.000000 7.086022\n')
        f.write('2.386111 0.087791 1.385352 1.240981 0.107278 0.281581 0.811907\n')
        f.write('0.228116 2.383148 5.290024 0.868241 0.282729 6.011613 0.439469 0.106802\n')
        f.write('0.653416 0.632629 0.768024 0.239248 0.438074 0.180393 0.609526 0.000000 0.076981\n')
        f.write('0.406431 0.154924 0.341113 0.000000 0.000000 0.730772 0.112880 0.071514 0.443504 2.556685\n')
        f.write('0.258635 4.610124 3.148371 0.716913 0.000000 1.519078 0.830078 0.267683 0.270475 0.460857 0.180629\n')
        f.write('0.717840 0.896321 0.000000 0.000000 0.000000 1.127499 0.304803 0.170372 0.000000 3.332732 5.230115 2.411739\n')
        f.write('0.183641 0.136906 0.138503 0.000000 0.000000 0.000000 0.000000 0.153478 0.475927 1.951951 1.565160 0.000000 0.921860\n')
        f.write('2.485920 1.028313 0.419244 0.133940 0.187550 1.526188 0.507003 0.347153 0.933709 0.119152 0.316258 0.335419 0.170205 0.110506\n')
        f.write('4.051870 1.531590 4.885892 0.956097 1.598356 0.561828 0.793999 2.322243 0.353643 0.247955 0.171432 0.954557 0.619951 0.459901 2.427202\n')
        f.write('3.680365 0.265745 2.271697 0.660930 0.162366 0.525651 0.340156 0.306662 0.226333 1.900739 0.331090 1.350599 1.031534 0.136655 0.782857 5.436674\n')
        f.write('0.000000 2.001375 0.224968 0.000000 0.000000 0.000000 0.000000 0.000000 0.270564 0.000000 0.461776 0.000000 0.000000 0.762354 0.000000 0.740819 0.000000\n')
        f.write('0.244139 0.078012 0.946940 0.000000 0.953164 0.000000 0.214717 0.000000 1.265400 0.374834 0.286572 0.132142 0.000000 6.952629 0.000000 0.336289 0.417839 0.608070\n')
        f.write('2.059564 0.240368 0.158067 0.178316 0.484678 0.346983 0.367250 0.538165 0.438715 8.810038 1.745156 0.103850 2.565955 0.123606 0.485026 0.303836 1.561997 0.000000 0.279379\n')
        f.write('\n')
        f.write('\n')
        f.write('0.087127 0.040904 0.040432 0.046872 0.033474 0.038255 0.049530 0.088612 0.033619 0.036886 0.085357 0.080481 0.014753 0.039772 0.050680 0.069577 0.058542 0.010494 0.029916 0.064718\n')

def writelg(dirname):
    """
    write LG empirical rate matrix with gamma rates among sites for protein model
    """
    lg = os.path.join(dirname, 'lg.dat')
    with open(lg,'w') as f:
        f.write('0.425093\n')
        f.write('0.276818 0.751878\n')
        f.write('0.395144 0.123954 5.076149\n')
        f.write('2.489084 0.534551 0.528768 0.062556\n')
        f.write('0.969894 2.807908 1.695752 0.523386 0.084808\n')
        f.write('1.038545 0.363970 0.541712 5.243870 0.003499 4.128591\n')
        f.write('2.066040 0.390192 1.437645 0.844926 0.569265 0.267959 0.348847\n')
        f.write('0.358858 2.426601 4.509238 0.927114 0.640543 4.813505 0.423881 0.311484\n')
        f.write('0.149830 0.126991 0.191503 0.010690 0.320627 0.072854 0.044265 0.008705 0.108882\n')
        f.write('0.395337 0.301848 0.068427 0.015076 0.594007 0.582457 0.069673 0.044261 0.366317 4.145067\n')
        f.write('0.536518 6.326067 2.145078 0.282959 0.013266 3.234294 1.807177 0.296636 0.697264 0.159069 0.137500\n')
        f.write('1.124035 0.484133 0.371004 0.025548 0.893680 1.672569 0.173735 0.139538 0.442472 4.273607 6.312358 0.656604\n')
        f.write('0.253701 0.052722 0.089525 0.017416 1.105251 0.035855 0.018811 0.089586 0.682139 1.112727 2.592692 0.023918 1.798853\n')
        f.write('1.177651 0.332533 0.161787 0.394456 0.075382 0.624294 0.419409 0.196961 0.508851 0.078281 0.249060 0.390322 0.099849 0.094464\n')
        f.write('4.727182 0.858151 4.008358 1.240275 2.784478 1.223828 0.611973 1.739990 0.990012 0.064105 0.182287 0.748683 0.346960 0.361819 1.338132\n')
        f.write('2.139501 0.578987 2.000679 0.425860 1.143480 1.080136 0.604545 0.129836 0.584262 1.033739 0.302936 1.136863 2.020366 0.165001 0.571468 6.472279\n')
        f.write('0.180717 0.593607 0.045376 0.029890 0.670128 0.236199 0.077852 0.268491 0.597054 0.111660 0.619632 0.049906 0.696175 2.457121 0.095131 0.248862 0.140825\n')
        f.write('0.218959 0.314440 0.612025 0.135107 1.165532 0.257336 0.120037 0.054679 5.306834 0.232523 0.299648 0.131932 0.481306 7.803902 0.089613 0.400547 0.245841 3.151815\n')
        f.write('2.547870 0.170887 0.083688 0.037967 1.959291 0.210332 0.245034 0.076701 0.119013 10.649107 1.702745 0.185202 1.898718 0.654683 0.296501 0.098369 2.188158 0.189510 0.249313\n')
        f.write('\n')
        f.write('0.079066 0.055941 0.041977 0.053052 0.012937 0.040767 0.071586 0.057337 0.022355 0.062157 0.099081 0.064600 0.022951 0.042302 0.044040 0.061197 0.053287 0.012066 0.034155 0.069147\n')

def writewag(dirname):
    """
    write WAG empirical rate matrix with gamma rates among sites for protein model
    """
    wagf = os.path.join(dirname, 'wag.dat')
    with open(wagf,'w') as f:
        f.write('0.551571\n')
        f.write('0.509848  0.635346\n')
        f.write('0.738998  0.147304  5.429420\n')
        f.write('1.027040  0.528191  0.265256  0.0302949\n')
        f.write('0.908598  3.035500  1.543640  0.616783  0.0988179\n')
        f.write('1.582850  0.439157  0.947198  6.174160  0.021352  5.469470\n')
        f.write('1.416720  0.584665  1.125560  0.865584  0.306674  0.330052  0.567717\n')
        f.write('0.316954  2.137150  3.956290  0.930676  0.248972  4.294110  0.570025  0.249410\n')
        f.write('0.193335  0.186979  0.554236  0.039437  0.170135  0.113917  0.127395  0.0304501 0.138190\n')
        f.write('0.397915  0.497671  0.131528  0.0848047 0.384287  0.869489  0.154263  0.0613037 0.499462  3.170970\n')
        f.write('0.906265  5.351420  3.012010  0.479855  0.0740339 3.894900  2.584430  0.373558  0.890432  0.323832  0.257555\n')
        f.write('0.893496  0.683162  0.198221  0.103754  0.390482  1.545260  0.315124  0.174100  0.404141  4.257460  4.854020  0.934276\n')
        f.write('0.210494  0.102711  0.0961621 0.0467304 0.398020  0.0999208 0.0811339 0.049931  0.679371  1.059470  2.115170  0.088836  1.190630\n')
        f.write('1.438550  0.679489  0.195081  0.423984  0.109404  0.933372  0.682355  0.243570  0.696198  0.0999288 0.415844  0.556896  0.171329  0.161444\n')
        f.write('3.370790  1.224190  3.974230  1.071760  1.407660  1.028870  0.704939  1.341820  0.740169  0.319440  0.344739  0.967130  0.493905  0.545931  1.613280\n')
        f.write('2.121110  0.554413  2.030060  0.374866  0.512984  0.857928  0.822765  0.225833  0.473307  1.458160  0.326622  1.386980  1.516120  0.171903  0.795384  4.378020\n')
        f.write('0.113133  1.163920  0.0719167 0.129767  0.717070  0.215737  0.156557  0.336983  0.262569  0.212483  0.665309  0.137505  0.515706  1.529640  0.139405  0.523742  0.110864\n')
        f.write('0.240735  0.381533  1.086000  0.325711  0.543833  0.227710  0.196303  0.103604  3.873440  0.420170  0.398618  0.133264  0.428437  6.454280  0.216046  0.786993  0.291148  2.485390\n')
        f.write('2.006010  0.251849  0.196246  0.152335  1.002140  0.301281  0.588731  0.187247  0.118358  7.821300  1.800340  0.305434  2.058450  0.649892  0.314887  0.232739  1.388230  0.365369  0.314730\n')
        f.write('\n')
        f.write('0.0866279 0.043972  0.0390894 0.0570451 0.0193078 0.0367281 0.0580589 0.0832518 0.0244313 0.048466  0.086209  0.0620286 0.0195027 0.0384319 0.0457631 0.0695179 0.0610127 0.0143859 0.0352742 0.0708956\n')
def _run_mcmctree(control_file):
    """
    Run mcmctree assuming all necessary files are written and we are in the
    directory with those files.
    """
    sp.run(['mcmctree', control_file], stdout=sp.PIPE)
    #if not os.path.isfile(out_file):
    #    raise FileNotFoundError('Mcmctree output file not found')
    #os.remove(control_file)
    #if not preserve:
    #    os.remove(out_file)
    #return max_results




class mcmctree:
    """
    Implementation of mcmctree provided a MRBH family for phylogenetic dating
    """
    def __init__(self, calnf_rn, palnf_rn, tmpdir, outdir, speciestree, datingset, aamodel, partition):
        self.tree = os.path.abspath(speciestree)
        self.partition = partition
        self.aamodel = aamodel
        self.calnf_rn = calnf_rn
        if self.calnf_rn is not None: self.prefix = os.path.basename(calnf_rn).replace('.caln','').replace('.rename','').replace('.paml','').replace('.','_')
        else: self.prefix = os.path.basename(palnf_rn).replace('.paln','').replace('.rename','').replace('.paml','').replace('.','_')
        if tmpdir == None:
            tmp_path = os.path.join(outdir, "mcmctree", self.prefix)
        else:
            tmp_path = os.path.join(tmpdir, "mcmctree", self.prefix)
        _mkdir(os.path.join(outdir, "mcmctree"))
        self.tmp_path = _mkdir(tmp_path)
        self.calnf_rn = calnf_rn
        if self.calnf_rn is not None:
            tmpc_path = os.path.join(tmp_path, "cds")
            self.tmpc_path = _mkdir(tmpc_path)
            _cp2tmp(self.calnf_rn,self.tree,self.tmpc_path)
            self.controlcf = os.path.join(tmpc_path, 'mcmctree.ctrl')
            self.controlc = {
                'seqfile': os.path.basename(self.calnf_rn),
                'treefile':self.tree,
                'outfile': 'mcmctree.out',
                'ndata':1,
                'seqtype':0,
                'usedata':1,
                'clock': 2,
                'RootAge': '<5.00',
                'model': 4,
                'alpha': 0.5,
                'ncatG': 5,
                'cleandata': 0,
                'BDparas': '1 1 0.1',
                'kappa_gamma': '6 2',
                'alpha_gamma': '1 1',
                'rgene_gamma': '2 20 1',
                'sigma2_gamma': '1 10 1',
                'finetune': '1: .1 .1 .1 .1 .1 .1',
                'print': 1,
                'burnin': 1,
                'sampfreq': 1,
                'nsample': 10,}
            if self.partition:
                self.controlc['ndata'] = 3
        if not self.partition:
            tmpp_path = os.path.join(tmp_path, "pep")
            self.tmpp_path = _mkdir(tmpp_path)
            self.palnf_rn = palnf_rn
            _cp2tmp(self.palnf_rn,self.tree,self.tmpp_path)
            self.controlpf = os.path.join(tmpp_path, 'mcmctree.ctrl')
            self.controlp = {
            'seqfile': os.path.basename(self.palnf_rn),
            'treefile':self.tree,
            'outfile': 'mcmctree.out',
            'ndata':1,
            'seqtype':2,
            'usedata':'3  * 0: no data; 1:seq; 2:approximation; 3:out.BV (in.BV)',
            'clock': 2,
            'RootAge': '<5.00',
            'model': 1,
            'alpha': 0.5,
            'ncatG': 5,
            'cleandata': 0,
            'BDparas': '1 1 0.1',
            'kappa_gamma': '6 2',
            'alpha_gamma': '1 1',
            'rgene_gamma': '2 20 1',
            'sigma2_gamma': '1 10 1',
            'finetune': '1: .1 .1 .1 .1 .1 .1',
            'print': 1,
            'burnin': 1,
            'sampfreq': 1,
            'nsample': 10,}
        if not datingset is None:
            for i in datingset:
                i.strip('\t').strip('\n').strip(' ')
                if self.calnf_rn is not None:
                    for key in self.controlc.keys():
                        if key in i:
                            self.controlc[key] = i.replace(key,'').replace('=','').strip(' ')
                            if not self.partition:
                                self.controlp[key] = i.replace(key,'').replace('=','').strip(' ')
                elif not self.partition:
                    for key in self.controlp.keys():
                        if key in i:
                            self.controlp[key] = i.replace(key,'').replace('=','').strip(' ')
        #for x in kwargs.keys():
        #    if x not in self.control:
        #        raise KeyError("{} is not a valid codeml param.".format(x))
        #    else:
        #        self.control.get(x) = kwargs[x]
    def write_ctrl(self):
        if self.calnf_rn is not None:
            c = ['{0} = {1}'.format(k, v) for (k,v) in self.controlc.items()]
            c = "\n".join(c)
            with open(self.controlcf, "w") as f:
                f.write(c)
        if not self.partition:
            p = ['{0} = {1}'.format(k, v) for (k,v) in self.controlp.items()]
            p = "\n".join(p)
            with open(self.controlpf, "w") as f:
                f.write(p)
    def run_mcmctree(self,CI_table,PM_table,wgd_mrca):
        """
        Run mcmctree on the codon and peptide alignment.
        """
        self.write_ctrl()
        parentdir = os.getcwd()  # where we are currently
        if self.calnf_rn is not None:
            os.chdir(self.tmpc_path)  # go to tmpdir
            _run_mcmctree('mcmctree.ctrl')
            self.CI, self.PM = self.get_dates(CI_table,PM_table,wgd_mrca,cds=True)
            CI_table[self.prefix] = [[float(i) for i in self.CI]]
            PM_table[self.prefix] = [self.PM]
            os.chdir(parentdir)
        if not self.partition:
            os.chdir(self.tmpp_path)
            _run_mcmctree('mcmctree.ctrl')
            cmd = ['rm','out.BV','rst']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            if self.aamodel == 'wag':
                writewag(os.getcwd())
                cmd = ['sed','-i','s/{}/{}/g'.format('aaRatefile =','aaRatefile = wag.dat'),'tmp0001.ctl']
                sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            elif self.aamodel == 'lg':
                writelg(os.getcwd())
                cmd = ['sed','-i','s/{}/{}/g'.format('aaRatefile =','aaRatefile = lg.dat'),'tmp0001.ctl']
                sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            elif self.aamodel == 'dayhoff':
                writedayhoff(os.getcwd())
                cmd = ['sed','-i','s/{}/{}/g'.format('aaRatefile =','aaRatefile = dayhoff.dat'),'tmp0001.ctl']
                sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            #if self.aamodel != 'poisson':
            #    cmd = ['sed', '-i', 's/{}/{}/g'.format('model = 0','model = 2'), 'tmp0001.ctl']
            #    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            if self.aamodel != 'poisson':
                cmd = ['sed', '-i', 's/{}/{}/g'.format('model = 1','model = 2'), 'tmp0001.ctl']
                sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            cmd = ['codeml', 'tmp0001.ctl']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            cmd = ['mv','rst2','in.BV']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            cmd = ['sed','-i','s/{}/{}/g'.format('usedata = 3','usedata = 2'),'mcmctree.ctrl']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            _run_mcmctree('mcmctree.ctrl')
            self.CI, self.PM = self.get_dates(CI_table,PM_table,wgd_mrca,cds=False)
            if self.calnf_rn is not None:
                CI_table[self.prefix].append([float(i) for i in self.CI])
                PM_table[self.prefix].append(self.PM)
            else:
                CI_table[self.prefix] = [[float(i) for i in self.CI]]
                PM_table[self.prefix] = [self.PM]
            os.chdir(parentdir)
        #return results, []
    def get_dates(self,CI_table,PM_table,wgd_mrca,cds=True):
        Figtree = Phylo.read('FigTree.tre','nexus')
        wgd_node = Figtree.common_ancestor({"name": wgd_mrca[0]}, {"name": wgd_mrca[1]})
        self.CI = wgd_node.comment.strip('[&95%HPD={').strip('[&95%={').strip('}]').split(', ')
        self.CI = [float(i)*100 for i in self.CI]
        self.PM = wgd_node.clades[0].branch_length
        if cds:
            logging.info("Posterior mean for the age of wgd is {0:.4f} million years from {1} codon alignment and 95% credibility intervals (CI) is {2:.4f}-{3:.4f} million years".format(self.PM*100,self.prefix,self.CI[0],self.CI[1]))
        else:
            logging.info("Posterior mean for the age of wgd is {0:.4f} million years from {1} peptide alignment and 95% credibility intervals (CI) is {2:.4f}-{3:.4f} million years".format(self.PM*100,self.prefix,self.CI[0],self.CI[1]))
        return self.CI, self.PM
