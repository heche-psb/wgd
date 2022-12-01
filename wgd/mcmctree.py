import pandas as pd
import numpy as np
import subprocess as sp
import logging
import os
import re
from Bio.Align import MultipleSeqAlignment

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
    def __init__(self, calnf_rn, palnf_rn, tmpdir, outdir, speciestree, datingset, partition):
        self.tree = speciestree
        self.partition = partition
        if tmpdir == None:
            tmp_path = os.path.join(outdir, "mcmctree", os.path.basename(calnf_rn).replace('.caln','').replace('.rename','').replace('.paml',''))
        else:
            tmp_path = os.path.join(tmpdir, "mcmctree", os.path.basename(calnf_rn).replace('.caln','').replace('.rename','').replace('.paml',''))
        _mkdir(os.path.join(outdir, "mcmctree"))
        self.tmp_path = _mkdir(tmp_path)
        tmpc_path = os.path.join(tmp_path, "cds")
        self.tmpc_path = _mkdir(tmpc_path)
        self.calnf_rn = calnf_rn
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
                for key in self.controlc.keys():
                    if key in i:
                        self.controlc[key] = i.replace(key,'').replace('=','').strip(' ')
                        if not self.partition:
                            self.controlp[key] = i.replace(key,'').replace('=','').strip(' ')
        #for x in kwargs.keys():
        #    if x not in self.control:
        #        raise KeyError("{} is not a valid codeml param.".format(x))
        #    else:
        #        self.control.get(x) = kwargs[x]
    def write_ctrl(self):
        c = ['{0} = {1}'.format(k, v) for (k,v) in self.controlc.items()]
        c = "\n".join(c)
        with open(self.controlcf, "w") as f:
            f.write(c)
        if not self.partition:
            p = ['{0} = {1}'.format(k, v) for (k,v) in self.controlp.items()]
            p = "\n".join(p)
            with open(self.controlpf, "w") as f:
                f.write(p)
    def run_mcmctree(self):
        """
        Run mcmctree on the codon and peptide alignment.
        """
        self.write_ctrl()
        parentdir = os.getcwd()  # where we are currently
        os.chdir(self.tmpc_path)  # go to tmpdir
        _run_mcmctree('mcmctree.ctrl')
        os.chdir(parentdir)
        if not self.partition:
            os.chdir(self.tmpp_path)
            _run_mcmctree('mcmctree.ctrl')
            cmd = ['rm','out.BV','rst']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            writewag(os.getcwd())
            cmd = ['sed','-i','s/{}/{}/g'.format('aaRatefile =','aaRatefile = wag.dat'),'tmp0001.ctl']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            cmd = ['codeml', 'tmp0001.ctl']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            cmd = ['mv','rst2','in.BV']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            cmd = ['sed','-i','s/{}/{}/g'.format('usedata = 3','usedata = 2'),'mcmctree.ctrl']
            sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            logging.info('Running mcmctree using Hessian matrix of WAG+Gamma for protein model')
            _run_mcmctree('mcmctree.ctrl')
            os.chdir(parentdir)
        #return results, []
