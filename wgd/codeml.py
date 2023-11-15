# TODO: implement pair-by-pair codeml analysis for a given input alignment.
import pandas as pd
import numpy as np
import subprocess as sp
import logging
import os
import re
from io import StringIO
from Bio import Phylo
from Bio.Align import MultipleSeqAlignment
import seaborn as sns
import matplotlib.pyplot as plt


GAPS = ["-", "?", "X"]  # these should be treated as gaps

def _strip_gaps(aln):
    new_aln = aln[:,0:0]
    for j in range(aln.get_alignment_length()):
        if any([x in GAPS for x in aln[:,j]]):
            continue
        else:
            new_aln += aln[:,j:j+1]
    return new_aln

def _all_pairs(aln):
    pairs = []
    for i in range(len(aln)-1):
        for j in range(i+1, len(aln)):
            pairs.append([aln[i].id, aln[j].id])
    return pairs

def _write_aln_codeml(aln, fname):
    with open(fname, "w") as f:
        f.write("{} {}\n".format(len(aln), aln.get_alignment_length()))
        for s in aln:
            f.write("{}\n".format(s.id))
            f.write("{}\n".format(s.seq))

def _parse_pairwise(codeml_out):
    with open(codeml_out, "r") as f:
        content = f.read()
    x = content.split("pairwise comparison")[2].split("\n")
    y = [x for x in map(lambda i: _parse_pair(x, i), range(3, len(x), 7))]
    d = pd.DataFrame.from_dict(y).set_index("pair")
    return d

def _parse_pair(lines, start):
    x = lines[start].split()
    a = x[1].strip("()")
    b = x[4].strip("()")
    l = float(lines[start+1].split("=")[1].strip())
    x = lines[start+4].split("=")
    y = [x[0]]
    for elem in x[1:]:
        y += elem.split()
    x = {y[i].strip(): float(y[i+1].strip()) for i in range(0, len(y)-1, 2)}
    p = "__".join(sorted([a, b]))
    x.update({"pair": p, "gene1": a, "gene2": b, "l": l})
    return x

def _parse_tree(codeml_out,kstree_dir, katree_dir, wtree_dir, prefix, parentdir):
    content = []
    with open(codeml_out, "r") as f:
        for i in f.readlines(): content.append(i)
    ds,dn,w = content[-9],content[-7],content[-4].replace(' #',': ')
    os.chdir(parentdir)
    _write_basic(ds,os.path.join(kstree_dir,'{}.dS.tree'.format(prefix)))
    _write_basic(dn,os.path.join(katree_dir,'{}.dN.tree'.format(prefix)))
    _write_basic(w,os.path.join(wtree_dir,'{}.w.tree'.format(prefix)))
    #handle_ds,handle_dn,handle_w = StringIO(ds),StringIO(dn),StringIO(w)
    #tree_ds,tree_dn,tree_w = Phylo.read(handle_ds, "newick"),Phylo.read(handle_dn, "newick"),Phylo.read(handle_w, "newick")
    tree_ds,tree_dn,tree_w = Phylo.read(os.path.join(kstree_dir,'{}.dS.tree'.format(prefix)), "newick"),Phylo.read(os.path.join(katree_dir,'{}.dN.tree'.format(prefix)), "newick"),Phylo.read(os.path.join(wtree_dir,'{}.w.tree'.format(prefix)), "newick")
    drawtree(tree_ds,os.path.join(kstree_dir,prefix+".dS.tree.pdf"))
    drawtree(tree_dn,os.path.join(katree_dir,prefix+".dN.tree.pdf"))
    drawtree(tree_w,os.path.join(wtree_dir,prefix+".w.tree.pdf"))
    return ds,dn,w

def drawtree(tree,fname):
    fig,ax = plt.subplots()
    Phylo.draw(tree,do_show=False,axes=ax)
    sns.despine(offset=1)
    fig.savefig(fname)
    plt.close()

def _write_basic(content,fname):
    with open(fname,'w') as f: f.write(content)

def check_noneresult(codeml_out):
    with open(codeml_out, "r") as f: content = f.read()
    if "pairwise comparison" not in content:
        return False
    else:
        return True

def _run_codeml_tree(exe, control_file, out_file, prefix, kstree_dir, katree_dir, wtree_dir, parentdir, preserve=False, times=1):
    #logging.debug("Performing codeml {} times".format(times))
    #max_results = None
    #max_likelihood = None
    #Noresults = False
    for i in range(times):
        #logging.debug("Codeml iteration {0} for {1}".format(str(i+1), control_file))
        sp.run([exe, control_file], stdout=sp.PIPE)
        #sp.run(['rm', '2ML.dN', '2ML.dS', '2ML.t', '2NG.dN', '2NG.dS','2NG.t', 'rst', 'rst1', 'rub'], stdout=sp.PIPE, stderr=sp.PIPE)
        if not os.path.isfile(out_file):
            raise FileNotFoundError('Codeml output file not found')
    ds,dn,w = _parse_tree(out_file,kstree_dir,katree_dir,wtree_dir,prefix,parentdir)
    #_write_basic(ds,os.path.join(kstree_dir,'{}.dS.tree'.format(prefix)))
    #_write_basic(dn,os.path.join(katree_dir,'{}.dN.tree'.format(prefix)))
    #_write_basic(w,os.path.join(wtree_dir,'{}.w.tree'.format(prefix)))


def _run_codeml(exe, control_file, out_file, preserve=False, times=1):
    """
    Run codeml assuming all necessary files are written and we are in the
    directory with those files. Set `preserve` to true to keep intermediary
    files. When `times > 1`, codeml will be run `times` times and the best
    result (highest likelihood) will be returned.
    """
    logging.debug("Performing codeml {} times".format(times))
    max_results = None 
    max_likelihood = None
    Noresults = False
    for i in range(times):
        logging.debug("Codeml iteration {0} for {1}".format(str(i+1), control_file))
        sp.run([exe, control_file], stdout=sp.PIPE)
        sp.run(['rm', '2ML.dN', '2ML.dS', '2ML.t', '2NG.dN', '2NG.dS',
            '2NG.t', 'rst', 'rst1', 'rub'], stdout=sp.PIPE, stderr=sp.PIPE)
        if not os.path.isfile(out_file):
            raise FileNotFoundError('Codeml output file not found')
        if not check_noneresult(out_file):
            Noresults = True
            break
        results = _parse_pairwise(out_file)
        likelihood = sum(results["l"])
        if not max_likelihood or likelihood > max_likelihood:
            max_likelihood = likelihood
            max_results = results
    if Noresults:
        if not preserve:
            os.remove(out_file)
        return None
    logging.debug('Best MLE: ln(L) = {}'.format(max_likelihood))
    #os.remove(control_file)
    if not preserve:
        os.remove(out_file)
    return max_results


class Codeml:
    """
    Class for codeml (PAML Yang 2007) python wrapper. Defines the controle file
    and enables running codeml from within python in one line of code.

    :param codeml: path to codeml executable (by default will look for codeml
        in the system PATH)
    :param tmp: path to temporary directory, will default to the current working
        directory ('./')
    :param id: filename prefix for output/tmp files
    :param kwargs: other codeml keyword arguments (see PAML user guide)::

        'seqfile': None,
        'outfile': self.out_file,
        'noisy': 0,
        'verbose': 0,
        'runmode': -2,
        'seqtype': 1,
        'CodonFreq': 2,
        'clock': 0,
        'aaDist': 0,
        'aaRatefile': 'dat/jones.dat',
        'model': 0,
        'NSsites': 0,
        'icode': 0,
        'Mgene': 0,
        'fix_kappa': 0,
        'kappa': 2,
        'fix_omega': 0,
        'omega': .4,
        'fix_alpha': 1,
        'alpha': 0,
        'Malpha': 0,
        'ncatG': 8,
        'getSE': 0,
        'RateAncestor': 1,
        'Small_Diff': .5e-6,
        'cleandata': 1,
        'method': 0
    """
    def __init__(self, aln, exe='codeml', tmp='./', prefix='codeml', treefile=None, **kwargs):
        """
        Codeml wrapper init. Initializes the default control file for Ks
        analysis as proposed by Vanneste et al. (2013). Takes as keyword
        arguments the options from the normal codeml distribution. Control
        settings are stored in a dictionary that can be accessed with the
        `.control` attribute

        :param aln: alignment file
        :param exe: path to codeml executable (by default will look for
            codeml in the system PATH)
        :param tmp: path to temporary directory, will default to the current
            working directory ('./')
        :param prefix: filename prefix for output/tmp files
        :param kwargs: any codeml control option (see PAML user guide)
        """
        if not os.path.isdir(tmp):
            raise NotADirectoryError('tmp directory {} not found!'.format(tmp))
        self.prefix = prefix
        self.aln = aln
        self.exe = exe
        self.tmp = tmp
        self.treefile = treefile
        self.control_file = self.prefix + '.ctrl'
        self.aln_file = self.prefix + '.cdsaln'
        self.out_file = self.prefix + '.codeml'
        self.control = {
            'seqfile': self.aln_file, 
            'outfile': self.out_file,
            'noisy': 0, 
            'verbose': 0, 
            'runmode': -2, 
            'seqtype': 1,
            'CodonFreq': 2, 
            'clock': 0, 
            'aaDist': 0,
            'aaRatefile': 'dat/jones.dat', 
            'model': 0, 
            'NSsites': 0, 
            'icode': 0,
            'Mgene': 0, 
            'fix_kappa': 0,
            'kappa': 2, 
            'fix_omega': 0, 
            'omega': .4, 
            'fix_alpha': 1, 
            'alpha': 0,
            'Malpha': 0, 
            'ncatG': 8,
            'getSE': 0, 
            'RateAncestor': 1, 
            'Small_Diff': .5e-6, 
            'cleandata': 1,
            'method': 0}
        # update the control with kwargs
        for x in kwargs.keys():
            if x not in self.control:
                raise KeyError("{} is not a valid codeml param.".format(x))
            else:
                self.control[x] = kwargs[x]

    def __str__(self):
        """
        String method for Codeml wrapper, prints current control settings

        :return: string representation of the control file
        """
        x = ['{0} = {1}\n'.format(k, v) for (k,v) in sorted(self.control.items())]
        return "\n".join(x)

    def write_ctrl(self):
        with open(self.control_file, "w") as f:
            f.write(str(self))

    # We should output the pairs for which we couldn't estimate Ks separately,
    # so that we can make a nan_result for those, but work with those we could
    # estimate without much trouble
    def run_codeml(self, kstree_dir=None, katree_dir=None, wtree_dir=None, **kwargs):
        """
        Run codeml on the full alignment. This will exclude all gap-containing
        columns, which may lead to a significant loss of data.
        """
        stripped_aln = _strip_gaps(self.aln)  # codeml does this anyway
        if stripped_aln.get_alignment_length() == 0:
            logging.warning("Stripped alignment length == 0 for {}".format(self.prefix))
            return None, _all_pairs(self.aln)
        parentdir = os.path.abspath(os.curdir)  # where we are currently
        os.chdir(self.tmp)  # go to tmpdir
        if not (self.treefile is None):
            self.control['treefile'] = self.treefile
            self.control['model'] = 1
            self.control['runmode'] = 0
            self.control['fix_blength'] = 1
        self.write_ctrl() 
        _write_aln_codeml(self.aln, self.aln_file)
        if not (self.treefile is None):
            _run_codeml_tree(self.exe, self.control_file, self.out_file, self.prefix, kstree_dir, katree_dir, wtree_dir,parentdir,**kwargs)
            return
        results = _run_codeml(self.exe, self.control_file, self.out_file, **kwargs)
        os.chdir(parentdir)
        if results is None:
            logging.warning("No codeml result for {} due to no resolved nucleotides".format(self.prefix))
            return None, _all_pairs(self.aln)
        else:
            return results, []

    def run_codeml_pairwise(self, **kwargs):
        """
        Run codeml on each pair of the alignment separately.
        """
        parentdir = os.path.abspath(os.curdir)  # where we are currently
        os.chdir(self.tmp)  # go to tmpdir
        results = []
        no_results = []
        for i in range(len(self.aln)-1):
            for j in range(i+1, len(self.aln)):
                pair = MultipleSeqAlignment([self.aln[i], self.aln[j]])
                stripped_pair = _strip_gaps(pair)
                if stripped_pair.get_alignment_length() == 0:
                    no_results.append([p.id for p in pair])
                else:
                    self.write_ctrl() 
                    _write_aln_codeml(stripped_pair, self.aln_file)
                    tmp_result = _run_codeml(self.exe, self.control_file,self.out_file, **kwargs)
                    if tmp_result is None:
                        no_results.append([p.id for p in pair])
                    else:
                        results.append(tmp_result)
        if len(no_results) > 0:
            logging.warning("Alignment length of 0 for {} pairs in {}".format(
                len(no_results), self.prefix))
        os.chdir(parentdir)
        return pd.concat(results), no_results

