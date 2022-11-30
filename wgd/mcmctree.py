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

def _mv2tmp(calnf_rn, palnf_rn, tree, tmpcdir, tmppdir):
    """
    mv the caln and paln to mcmctree_tmp file
    """
    if not os.path.isdir(tmpcdir) or not os.path.isdir(tmppdir) :
        logging.error("tmpdir not existing!")
    cmdc = ["mv", calnf_rn, tmpcdir]
    cmdp = ["mv", palnf_rn, tmppdir]
    cmdtc = ["cp", tree, tmpcdir]
    cmdtp = ["cp", tree, tmppdir]
    sp.run(cmdc, stdout=sp.PIPE, stderr=sp.PIPE)
    sp.run(cmdp, stdout=sp.PIPE, stderr=sp.PIPE)
    sp.run(cmdtc, stdout=sp.PIPE, stderr=sp.PIPE)
    sp.run(cmdtp, stdout=sp.PIPE, stderr=sp.PIPE)
    #for i in range(len(families)):
    #    gfname = 'GF_' + str(i+1)
    #    gfpath = os.path.join(tmpdir, gfname) 
    #    gfloc = _mkdir(gfpath)
    #    gfsloc.append(gfloc)
def _mv_(fname, dirname):
    cmd = ["mv", fname, dirname]
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

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
    def __init__(self, calnf_rn, palnf_rn, tmpdir, outdir, speciestree, datingset):
        self.tree = speciestree
        if tmpdir == None:
            tmp_path = os.path.join(outdir, "mcmctree", os.path.basename(calnf_rn).strip('.caln.rename'))
        else:
            tmp_path = os.path.join(tmpdir, "mcmctree", os.path.basename(calnf_rn).strip('.caln.rename'))
        _mkdir(os.path.join(outdir, "mcmctree"))
        self.tmp_path = _mkdir(tmp_path)
        tmpc_path = os.path.join(tmp_path, "cds")
        tmpp_path = os.path.join(tmp_path, "pep")
        self.tmpc_path = _mkdir(tmpc_path)
        self.tmpp_path = _mkdir(tmpp_path)
        self.calnf_rn = calnf_rn
        self.palnf_rn = palnf_rn
        _mv2tmp(self.calnf_rn, self.palnf_rn, self.tree, self.tmpc_path, self.tmpp_path)
        self.controlcf = os.path.join(tmpc_path, 'mcmctree.ctrl')
        self.controlpf = os.path.join(tmpp_path, 'mcmctree.ctrl')
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
            'finetune': '1: 0.1  0.1  0.1  0.01 .5',
            'print': 1,
            'burnin': 1,
            'sampfreq': 1,
            'nsample': 10,}
        self.controlp = {
            'seqfile': os.path.basename(self.palnf_rn),
            'treefile':self.tree,
            'outfile': 'mcmctree.out',
            'ndata':1,
            'seqtype':2,
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
            'finetune': '1: 0.1  0.1  0.1  0.01 .5',
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
                        self.controlp[key] = i.replace(key,'').replace('=','').strip(' ')
        #for x in kwargs.keys():
        #    if x not in self.control:
        #        raise KeyError("{} is not a valid codeml param.".format(x))
        #    else:
        #        self.control.get(x) = kwargs[x]
    def write_ctrl(self):
        c = ['{0} = {1}'.format(k, v) for (k,v) in self.controlc.items()]
        p = ['{0} = {1}'.format(k, v) for (k,v) in self.controlp.items()]
        c = "\n".join(c)
        p = "\n".join(p)
        with open(self.controlcf, "w") as f:
            f.write(c)
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
        os.chdir(self.tmpp_path)
        #Protein run for later
        #_run_mcmctree('mcmctree.ctrl')
        os.chdir(parentdir)
        #return results, []
