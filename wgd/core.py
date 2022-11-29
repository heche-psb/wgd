# Arthur Zwaenepoel (2020)
import uuid
import os
import logging
import numpy as np
import pandas as pd
import subprocess as sp
import itertools
from Bio import SeqIO
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio.Align import MultipleSeqAlignment
from Bio.Alphabet import IUPAC
from Bio.Data.CodonTable import TranslationError
from Bio import Phylo
from joblib import Parallel, delayed
from wgd.codeml import Codeml
from wgd.cluster import cluster_ks
from wgd.mcmctree import mcmctree
# Reconsider the renaming, more a pain than helpful?

# helper functions
def _write_fasta(fname, seq_dict):
    with open(fname, "w") as f:
        for k, v in seq_dict.items():
            f.write(">{}\n{}\n".format(k, v.seq))
    return fname

def _mkdir(dirname):
    #if os.path.isdir(dirname) :
    #    logging.warning("dir {} exists!".format(dirname))
    #else:
    if not os.path.isdir(dirname) :
        os.mkdir(dirname)
    return dirname

def _strip_gaps(aln):
    new_aln = aln[:,0:0]
    for j in range(aln.get_alignment_length()):
        if any([x == "-" for x in aln[:,j]]):
            continue
        else:
            new_aln += aln[:,j:j+1]
    return new_aln

def _pal2nal(pro_aln, cds_seqs):
    aln = {}
    for i, s in enumerate(pro_aln):
        cds_aln = ""
        cds_seq = cds_seqs[s.id].seq
        k = 0
        for j in range(pro_aln.get_alignment_length()):
            if pro_aln[i, j] == "-":
                cds_aln += "---"
            elif pro_aln[i, j] == "X":
                cds_aln += "???"  # not sure what best choice for codeml is
                k += 3
            else:
                cds_aln += cds_seq[k:k+3]
                k += 3
        aln[s.id] = cds_aln
    return MultipleSeqAlignment([SeqRecord(v, id=k) for k, v in aln.items()])

def _log_process(o, program=""):
    logging.debug("{} stderr: {}".format(program.upper(), o.stderr.decode()))
    logging.debug("{} stdout: {}".format(program.upper(), o.stdout.decode()))

def _label_internals(tree):
    for i, c in enumerate(tree.get_nonterminals()):
        c.name = str(i)

def _label_families(df):
    df.index = ["GF{:0>5}".format(i+1) for i in range(len(df.index))]

def _process_unrooted_tree(treefile, fformat="newick"):
    tree = Phylo.read(treefile, fformat).root_at_midpoint()
    _label_internals(tree)
    return tree


class SequenceData:
    """
    Sequence data container for Ks distribution computation pipeline. A helper
    class that bundles sequence manipulation methods.
    """
    def __init__(self, cds_fasta,
            tmp_path=None, out_path="wgd_dmd",
            to_stop=True, cds=True, cscore=None):
        if tmp_path == None:
            tmp_path = "wgdtmp_" + str(uuid.uuid4())
        self.tmp_path  = _mkdir(tmp_path)
        self.out_path  = _mkdir(out_path)
        self.cds_fasta = cds_fasta
        self.prefix    = os.path.basename(self.cds_fasta)
        self.pro_fasta = os.path.join(tmp_path, self.prefix + ".tfa")
        self.pro_db    = os.path.join(tmp_path, self.prefix + ".db")
        self.cds_seqs  = {}
        self.pro_seqs  = {}
        self.dmd_hits  = {}
        self.rbh       = {}
        self.mcl       = {}
        self.cds_sequence  = {}
        self.pro_sequence  = {}
        self.idmap     = {}  # map from the new safe id to the input seq id
        self.read_cds(to_stop=to_stop, cds=cds)
        _write_fasta(self.pro_fasta, self.pro_seqs)

    def read_cds(self, to_stop=True, cds=True):
        """
        Read a CDS fasta file. We give each input record a unique safe ID, and
        keep the full records in a dict with these IDs. We use the newly assigned
        IDs in further analyses, but can reconvert at any time.
        """
        for i, record in enumerate(SeqIO.parse(self.cds_fasta, 'fasta')):
            gid = "{0}_{1:0>5}".format(self.prefix, i)
            try:
                aa_seq = record.translate(to_stop=to_stop, cds=cds, id=record.id,
                                       stop_symbol="")
                aa_sequence = aa_seq.seq
                na_sequence = record.seq
            except TranslationError as e:
                logging.warning("Translation error ({}) in seq {}".format(
                    e, record.id))
                continue
            self.cds_seqs[gid] = record
            self.pro_seqs[gid] = aa_seq
            self.cds_sequence[gid] = na_sequence
            self.pro_sequence[gid] = aa_sequence
            self.idmap[record.id] = gid
        return

    def merge(self, other):
        """
        Merge other into self, keeping the paths etc. of self.
        """
        self.cds_seqs.update(other.cds_seqs)
        self.pro_seqs.update(other.pro_seqs)
        self.idmap.update(other.idmap)

    def make_diamond_db(self):
        cmd = ["diamond", "makedb", "--in",
               self.pro_fasta, "-d", self.pro_db]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        logging.debug(out.stderr.decode())
        if out.returncode == 1:
            logging.error(out.stderr.decode())

    def run_diamond(self, seqs, eval=1e-10):
        self.make_diamond_db()
        run = "_".join([self.prefix, seqs.prefix + ".tsv"])
        outfile = os.path.join(self.tmp_path, run)
        cmd = ["diamond", "blastp", "-d", self.pro_db, "-q",
            seqs.pro_fasta, "-o", outfile]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        logging.debug(out.stderr.decode())
        df = pd.read_csv(outfile, sep="\t", header=None)
        df = df.loc[df[0] != df[1]]
        self.dmd_hits[seqs.prefix] = df = df.loc[df[10] <= eval]
        return df

    def get_rbh_orthologs(self, seqs, cscore, eval=1e-10):
        if self == seqs:
            raise ValueError("RBH orthologs only defined for distinct species")
        df = self.run_diamond(seqs, eval=eval)
        if cscore == None:
            df1 = df.sort_values(10).drop_duplicates([0])
            df2 = df.sort_values(10).drop_duplicates([1])
            self.rbh[seqs.prefix] = df1.merge(df2)
        else:
            cscore = float(cscore)
            df_species1_best=df.sort_values(10).drop_duplicates([0])
            df_species2_best=df.sort_values(10).drop_duplicates([1])
            df_Tomerge_species1=df_species1_best[[0,11]]
            df_Tomerge_species2=df_species2_best[[1,11]]
            df_Tomerge_species1_rn = df_Tomerge_species1.rename(columns={11: 'species1_best'})
            df_Tomerge_species2_rn = df_Tomerge_species2.rename(columns={11: 'species2_best'})
            df_with_best=df.merge(df_Tomerge_species1_rn,on=0).merge(df_Tomerge_species2_rn,on=1)
            df_with_best_c=df_with_best.loc[(df_with_best[11]  >= cscore*df_with_best['species2_best']) & (df_with_best[11]  >= cscore*df_with_best['species1_best'])]
            df_c_score=df_with_best_c.iloc[:,0:12]
            self.rbh[seqs.prefix] = df_c_score



        # self.rbh[seqs.prefix] = seqs.rbh[self.prefix] = df1.merge(df2)
        # write to file using original ids for next steps

    def get_paranome(self, inflation=1.5, eval=1e-10):
        df = self.run_diamond(self, eval=eval)
        gf = self.get_mcl_graph(self.prefix)
        mcl_out = gf.run_mcl(inflation=inflation)
        with open(mcl_out, "r") as f:
            for i, line in enumerate(f.readlines()):
                self.mcl[i] = line.strip().split()

    def get_mcl_graph(self, *args):
        # args are keys in `self.dmd_hits` to use for building MCL graph
        gf = os.path.join(self.tmp_path, "_".join([self.prefix] + list(args)))
        df = pd.concat([self.dmd_hits[x] for x in args])
        df.to_csv(gf, sep="\t", header=False, index=False, columns=[0,1,10])
        return SequenceSimilarityGraph(gf)

    def write_paranome(self, fname=None, singletons=True):
        if singletons: 
            self.add_singletons_paranome()
        if not fname:
            fname = os.path.join(self.out_path, "{}.tsv".format(self.prefix))
        with open(fname, "w") as f:
            f.write("\t" + self.prefix + "\n")
            for i, (k, v) in enumerate(sorted(self.mcl.items())):
                # We report original gene IDs
                f.write("GF{:0>5}\t".format(i+1))
                f.write(", ".join([self.cds_seqs[x].id for x in v]))
                f.write("\n")
        return fname

    def add_singletons_paranome(self):
        xs = set(itertools.chain.from_iterable(self.mcl.values()))  # all genes in families
        gs = set(self.cds_seqs.keys())  # all genes
        ys = gs - xs 
        i = max(self.mcl.keys()) + 1
        for j, y in enumerate(ys):
            self.mcl[i + j] = [y]

    def write_rbh_orthologs(self, seqs, singletons=True):
        fname = "{}_{}.rbh.tsv".format(self.prefix, seqs.prefix)
        fname = os.path.join(self.out_path, fname)
        df = self.rbh[seqs.prefix]
        df[seqs.prefix] = df[0].apply(lambda x: seqs.cds_seqs[x].id)
        df[self.prefix] = df[1].apply(lambda x: self.cds_seqs[x].id)
        if singletons:  # this must be here, cannot be before renaming, not after labeling fams
            df = pd.concat([df, self.add_singletons_rbh(seqs)]) 
        #_label_families(df)
        df.to_csv(fname, columns=[seqs.prefix, self.prefix], sep="\t",index=False)
        return df.loc[:,[seqs.prefix, self.prefix]]

    def get_seq(self):
        focusname = os.path.join(self.out_path, 'merge_focus.tsv')
        fastaname = os.path.join(self.out_path, 'merge_focus_fasta.tsv')
        seqid_table = []
        with open (focusname,'r') as orthotable:
            next(orthotable)
            for row in orthotable:
               seqid = [s.strip('\n') for s in row.split('\t')]
               seqid_table.append(seqid)
        return seqid_table
    def get_seq_ap(self):
        focusapname = os.path.join(self.out_path, 'merge_focus_ap.tsv')
        seqid_table = []
        with open (focusapname,'r') as orthotable:
            next(orthotable)
            for row in orthotable:
                seqid = [s.strip('\n') for s in row.split('\t')]
                seqid_table.append(seqid)
        return seqid_table

    def add_singletons_rbh(self, seqs):
        # note this is implemented to work before the rbh table is modified
        gs1 = set(self.cds_seqs.keys())  # all genes species 1
        gs2 = set(seqs.cds_seqs.keys())  # all genes species 2
        df  = self.rbh[seqs.prefix]
        ys1 = gs1 - set(df[1])
        ys2 = gs2 - set(df[0])
        d = []
        for y in ys1:
            d.append({self.prefix: self.cds_seqs[y].id, seqs.prefix: ""})
        for y in ys2:
            d.append({seqs.prefix: seqs.cds_seqs[y].id, self.prefix: ""})
        return pd.DataFrame.from_dict(d)

    def remove_tmp(self, prompt=True):
        if prompt:
            ok = input("Removing {}, sure? [y|n]".format(self.tmp_path))
            if ok != "y":
                return
        out = sp.run(["rm", "-r", self.tmp_path], stdout=sp.PIPE, stderr=sp.PIPE)
        logging.debug(out.stderr.decode())


class SequenceSimilarityGraph:
    def __init__(self, graph_file):
        self.graph_file = graph_file

    def run_mcl(self, inflation=1.5):
        g1 = self.graph_file
        g2 = g1 + ".tab"
        g3 = g1 + ".mci"
        g4 = g2 + ".I{}".format(inflation*10)
        outfile = g1 + ".mcl"
        command = ['mcxload', '-abc', g1, '--stream-mirror',
            '--stream-neg-log10', '-o', g3, '-write-tab', g2]
        logging.debug(" ".join(command))
        out = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out)
        command = ['mcl', g3, '-I', str(inflation), '-o', g4]
        logging.debug(" ".join(command))
        out = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out)
        command = ['mcxdump', '-icl', g4, '-tabr', g2, '-o', outfile]
        _log_process(out)
        out = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out)
        return outfile


# Gene family i/o
def _rename(family, ids):
    return [ids[x] for x in family]

def read_gene_families(fname):
    """
    Read gene families from OrthoFinder format.
    """
    df = pd.read_csv(fname, sep="\t", index_col=0).fillna("")
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.split(", "))
    return df

def read_MultiRBH_gene_families(fname):
    """
    Read gene families from dmd -focus, in the format that each column contains seqid of each species and header of column is cds filename of each species
    """
    seqid_table = []
    with open (fname,'r') as orthotable:
        next(orthotable)
        for row in orthotable:
            seqid = [s.strip('\n') for s in row.split('\t')]
            seqid_table.append(seqid)
    return seqid_table

def merge_seqs(seqs):
    if type(seqs) == list:
        if len(seqs) > 2:
            raise ValueError("More than two sequence data objects?")
        if len(seqs) == 2:
            seqs[0].merge(seqs[1])
        seqs = seqs[0]
    return seqs

def mergeMultiRBH_seqs(seqs):
    if type(seqs) == list:
        for i in range(len(seqs)):
            if not i==0:
                seqs[0].merge(seqs[i])
        seqs = seqs[0]
    return seqs

def get_gene_families(seqs, families, rename=True, **kwargs):
    """
    Get the `GeneFamily` objects from a list of families (list with lists of
    gene IDs) and sequence data. When `rename` is set to True, it is assumed
    the gene IDs in the families are the original IDs (not those assigned 
    when reading the CDS from file).

    Note: currently this is only defined for one or two genomes (paranome 
    and one-to-one orthologs), but it should easily generalize to arbitrary
    gene families.
    """
    gene_families = []
    for fid in families.index:
        family = []
        for col in families.columns:
            ids = families.loc[fid][col]
            if ids == ['']: continue
            if rename:
                family += _rename(ids, seqs.idmap)
            else:
                family += ids
        if len(family) > 1:
            cds = {x: seqs.cds_seqs[x] for x in family}
            pro = {x: seqs.pro_seqs[x] for x in family}
            tmp = os.path.join(seqs.tmp_path, fid)
            gene_families.append(GeneFamily(fid, cds, pro, tmp, **kwargs))
        else:
            logging.debug("Skipping singleton family {}{}".format(fid,family))
    return gene_families

def get_MultipRBH_gene_families(seqs, families, tree_method, treeset, outdir, option="--auto", **kwargs):
    seqid_table = families
    cds = {}
    pro = []
    idmap = {}
    seq_cds = {}
    seq_pro = {}
    tree_fams = {}
    tree_famsf = []
    cds_alns = {}
    pro_alns = {}
    calnfs = []
    palnfs = []
    calnfs_length = []
    for i in range(len(seqs)):
        seq_cds.update(seqs[i].cds_sequence)
        seq_pro.update(seqs[i].pro_sequence)
    for i in range(len(seqs)):
        idmap.update(seqs[i].idmap)
    for i, fam in enumerate(seqid_table):
        family = []
        famid = 'GF_' + str(i+1)
        for seqid in fam:
            safeid = idmap.get(seqid)
            family.append(safeid)
            fnamep =os.path.join(outdir, famid + ".pep")
            fnamec =os.path.join(outdir, famid + ".cds")
            with open(fnamep,'a') as f:
                f.write(">{}\n{}\n".format(seqid, seq_pro.get(safeid)))
            with open(fnamec,'a') as f:
                f.write(">{}\n{}\n".format(seqid, seq_cds.get(safeid)))
        cmd = ["mafft"] + option.split() + ["--amino", fnamep]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        fnamepaln =os.path.join(outdir, famid + ".paln")
        palnfs.append(fnamepaln)
        with open(fnamepaln, 'w') as f: f.write(out.stdout.decode('utf-8'))
        _log_process(out, program="mafft")
        pro_aln = AlignIO.read(fnamepaln, "fasta")
        pro_alns[famid] = pro_aln
        fnamecaln =os.path.join(outdir, famid + ".caln")
        calnfs.append(fnamecaln)
        #cmd = ["trimal"] + ["-in", fnamepaln, "-backtrans", fnamec, "-out", fnamecaln, "-automated1"]
        #print(cmd)
        #out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        #p = sp.Popen(cmd)
        #print(p.stdout)
        #fnamecaln =os.path.join(outdir, 'GF_' + str(i+1) + ".caln")
        #with open(fnamecaln, 'w') as f: f.write(p.stdout.decode('utf-8'))
        #_log_process(out, program="trimal")
        aln = {}
        for i, s in enumerate(pro_aln):
            cds_aln = ""
            safeid = idmap.get(s.id)
            cds_seq = seq_cds.get(safeid)
            k = 0
            for j in range(pro_aln.get_alignment_length()):
                if pro_aln[i,j] == "-":
                    cds_aln += "---"
                elif pro_aln[i,j] == "X":
                    cds_aln += "???"
                else:
                    cds_aln += cds_seq[k:k+3]
                    k = k + 3
            aln[s.id] = cds_aln
        #fnamecaln =os.path.join(outdir, 'GF_' + str(i+1) + ".caln")
        with open(fnamecaln, 'a') as f:
            for k, v in aln.items():
                f.write(">{}\n{}\n".format(k, v))
        #Note that here the backtranslated codon-alignment will be shorter than the original cds file by a stop codon
        #iq_tree = os.path.join(outdir, 'GF_' + str(i+1) + ".caln.treefile")
        cds_aln = AlignIO.read(fnamecaln, "fasta")
        calnfs_length.append(cds_aln.get_alignment_length())
        cds_alns[famid] = cds_aln
        if tree_method == "mrbayes":
            fnamepalnnexus =os.path.join(outdir, famid + ".paln.nexus")
            AlignIO.convert(fnamepaln, 'fasta', fnamepalnnexus, 'nexus', IUPAC.extended_protein)
            cwd = os.getcwd()
            tmppath = os.path.join(cwd, outdir)
            os.chdir(tmppath)
            conf = os.path.join(cwd, outdir, famid + ".config.mb")
            logf = os.path.join(cwd, outdir, famid + ".mb.log")
            bashf = os.path.join(cwd, outdir, famid + ".bash.mb")
            config = {'set':'autoclose=yes nowarn=yes','execute':'./{}'.format(os.path.basename(fnamepalnnexus)),'prset':'aamodelpr=fixed(lg)','lset':'rates=gamma','mcmcp':['diagnfreq=100','samplefreq=10'],'mcmc':'ngen=1100 savebrlens=yes nchains=1','sumt':'','sump':'','quit':''}
            with open(conf,"w") as f:
                para = []
                for (k,v) in config.items():
                    if isinstance(v, list):
                        para.append('{0} {1}'.format(k, v[0]))
                        para.append('{0} {1}'.format(k, v[1]))
                    else:
                        para.append('{0} {1}'.format(k, v))
                #para = ['{0} {1}'.format(k, v) for (k,v) in config.items()]
                para = "\n".join(para)
                f.write(para)
            with open(bashf,"w") as f:
                f.write('mb <{0}> {1}'.format(os.path.basename(conf),os.path.basename(logf)))
            mb_cmd = ["sh", "{}".format(os.path.basename(bashf))]
            sp.run(mb_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            os.chdir(cwd)
        if tree_method == "iqtree":
            if not treeset is None:
                treesetfull = []
                iq_cmd = ["iqtree", "-s", fnamecaln]
                for i in treeset:
                    i = i.strip(" ").split(" ")
                    if type(i) == list:
                        treesetfull = treesetfull + i
                    else:
                        treesetfull.append(i)
                iq_cmd = iq_cmd + treesetfull
                print(iq_cmd)
            else:
                iq_cmd = ["iqtree", "-s", fnamecaln] + ["-st","CODON"] + ["-fast"]#+ ["-bb", "1000"] + ["-bnni"]
            iq_out = sp.run(iq_cmd, stdout=sp.PIPE)
            tree_pth = fnamecaln + ".treefile"
            tree = Phylo.read(tree_pth,'newick')
            tree_fams[famid] = tree
            tree_famsf.append(tree_pth)
        if tree_method == "fasttree":
            tree_pth = fnamecaln + ".fasttree"
            if not treeset is None:
                treesetfull = []
                ft_cmd = ["FastTree", '-out', tree_pth, fnamecaln]
                for i in treeset:
                    i = i.strip(" ").split(" ")
                    if type(i) == list:
                        treesetfull = treesetfull + i
                    else:
                        treesetfull.append(i)
                ft_cmd = ft_cmd[:1] + treesetfull + ft_cmd[1:]
            else:
                ft_cmd = ["FastTree", '-out', tree_pth, fnamecaln]
            ft_out = sp.run(ft_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            tree = Phylo.read(tree_pth,'newick')
            tree_fams[famid] = tree
            tree_famsf.append(tree_pth)
    return cds_alns, pro_alns, tree_famsf, calnfs, palnfs, calnfs_length
        #iq_out2 = sp.Popen(iq_cmd,shell=True)
        #with open(iq_tree, 'a') as f:
        #    f.write(iq_out.stdout.decode('utf-8'))
        #_log_process(iq_out, program="iqtree")
        #return _process_unrooted_tree(fnamecaln + ".treefile")
        #MultipleSeqAlignment([SeqRecord(v, id=k) for k, v in aln.items()])
            #print(safeid)
        #print(family)
            #Recordpro = seq_pro.get(x)
            #cds.update(seq_cds[x])
            #pro.append(Recordpro)
            #Recordcds = seq_cds.get(x)
            #cds.append(Recordpro)
            #cds = {sid: seqs[0].cds_seqs[sid] for sid in safeid}
            #for j in cds:
                #print(j)
            #pro.update(sid: seqs[0].pro_seqs[sid] for sid in safeid)
# Concatenate all MRBH family align in fasta format and feed into iqtree for species tree inference
def GetG2SMap(families, outdir):
    df = pd.read_csv(families,header=0,index_col=False,sep='\t')
    G2SMap = os.path.join(outdir, "G2S.Map")
    Slist = []
    for i in df.columns:
        Slist.append(i)
        for j in df[i]:
            j = j.strip(" ").strip("\n").strip("\t")
            with open(G2SMap, "a") as f:
                f.write(j + " "+ i + "\n")
    return G2SMap, Slist

def FileRn(cds_alns, pro_alns, tree_famsf, families, outdir):
    gsmap, slist = GetG2SMap(families, outdir)
    famnum = len(pro_alns)
    cds_alns_rn = {}
    pro_alns_rn = {}
    tree_rns = {}
    tree_rn_fs = []
    calnfs_rn = []
    palnfs_rn = []
    for i in range(famnum):
        famid = 'GF_' + str(i+1)
        cds_aln = cds_alns[famid]
        pro_aln = pro_alns[famid]
        calnpath = os.path.join(outdir, famid + ".caln.rename")
        palnpath = os.path.join(outdir, famid + ".paln.rename")
        for j in range(len(pro_aln)):
            with open(gsmap,"r") as f:
                lines = f.readlines()
                for k in lines:
                    k = k.strip('\n').strip(' ').split(' ')
                    if k[0] == cds_aln[j].id:
                        spn = k[1]
                        cds_aln[j].id = spn
                        with open(calnpath, "a") as f:
                            f.write(">{}\n{}\n".format(spn,cds_aln[j].seq))
                    if k[0] == pro_aln[j].id:
                        spn = k[1]
                        pro_aln[j].id = spn
                        with open(palnpath, "a") as f:
                            f.write(">{}\n{}\n".format(spn,pro_aln[j].seq))
        cds_alns_rn[famid] = cds_aln
        pro_alns_rn[famid] = pro_aln
        #calnf = AlignIO.write(cds_aln, calnpath, "fasta")
        #palnf = AlignIO.write(pro_aln, palnpath, "fasta")
        calnfs_rn.append(calnpath)
        palnfs_rn.append(palnpath)
        treef = tree_famsf[i]
        treecontent = ""
        treef_rn_f = os.path.join(outdir, famid + ".tree.rename")
        with open(treef,"r") as f:
            lines = f.readlines()
            for line in lines:
                treecontent = line
        with open(gsmap,"r") as f:
            lines = f.readlines()
            for k in lines:
                k = k.strip('\n').strip(' ').split(' ')
                if k[0] in treecontent:
                    treecontent = treecontent.replace(k[0],k[1])
        with open(treef_rn_f,"w") as f:
            f.write(treecontent)
        tree_rn = Phylo.read(treef_rn_f,'newick')
        tree_rns[famid] = tree_rn
        tree_rn_fs.append(treef_rn_f)
    return cds_alns_rn, pro_alns_rn, calnfs_rn, palnfs_rn, tree_rns, tree_rn_fs

def Concat(cds_alns, pro_alns, families, tree_method, outdir):
    gsmap, slist = GetG2SMap(families, outdir)
    famnum = len(pro_alns)
    cds_alns_rn = {}
    pro_alns_rn = {}
    Concat_calnf = os.path.join(outdir, "Concatenated.caln")
    Concat_palnf = os.path.join(outdir, "Concatenated.paln")
    cdsseq = {}
    proseq = {}
    ctree_length = 0
    for i in range(famnum):
        famid = 'GF_' + str(i+1)
        cds_aln = cds_alns[famid]
        pro_aln = pro_alns[famid]
        for j in range(len(pro_aln)):
            with open(gsmap,"r") as f:
                lines = f.readlines()
                for k in lines:
                    k = k.strip('\n').strip(' ').split(' ')
                    if k[0] == cds_aln[j].id:
                        spn = k[1]
                        cds_aln[j].id = spn
                        sequence = cds_aln[j].seq
                        if cdsseq.get(spn) is None:
                            cdsseq[spn] = str(sequence)
                        else:
                            cdsseq[spn] = cdsseq[spn] + str(sequence)
                    if k[0] == pro_aln[j].id:
                        spn = k[1]
                        pro_aln[j].id = spn
                        sequence = pro_aln[j].seq
                        if proseq.get(spn) is None:
                            proseq[spn] = str(sequence)
                        else:
                            proseq[spn] = proseq[spn] + str(sequence)
        cds_alns_rn[famid] = cds_aln
        pro_alns_rn[famid] = pro_aln
    for spname in range(len(slist)):
        spn = slist[spname]
        with open(Concat_palnf,"a") as f:
            sequence = proseq[spn]
            f.write(">{}\n{}\n".format(spn, sequence))
        with open(Concat_calnf,"a") as f:
            sequence = cdsseq[spn]
            f.write(">{}\n{}\n".format(spn, sequence))
    Concat_caln = AlignIO.read(Concat_calnf, "fasta")
    ctree_length = Concat_caln.get_alignment_length()
    Concat_paln = AlignIO.read(Concat_palnf, "fasta")
    if tree_method == "iqtree":
        ctree_pth = Concat_calnf + ".treefile"
        ptree_pth = Concat_palnf + ".treefile"
        iq_cmd = ["iqtree", "-s", Concat_calnf] + ["-st","CODON"] + ["-fast"]
        iq_cout = sp.run(iq_cmd, stdout=sp.PIPE)
        iq_cmd = ["iqtree", "-s", Concat_palnf] + ["-fast"]
        iq_pout = sp.run(iq_cmd, stdout=sp.PIPE)
        Concat_ptree = Phylo.read(ptree_pth, 'newick')
        Concat_ctree = Phylo.read(ctree_pth, 'newick')
    if tree_method == "fasttree":
        ctree_pth = Concat_calnf + ".fasttree"
        ft_cmd = ["FastTree", '-out', ctree_pth, Concat_calnf]
        ft_out = sp.run(ft_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        ptree_pth = Concat_palnf + ".fasttree"
        ft_cmd = ["FastTree", '-out', ptree_pth, Concat_palnf]
        ft_out = sp.run(ft_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        Concat_ctree = Phylo.read(ctree_pth,'newick')
        Concat_ptree = Phylo.read(ptree_pth,'newick')
    return cds_alns_rn, pro_alns_rn, Concat_ctree, Concat_ptree, Concat_calnf, ctree_pth, ctree_length

def _Codon2partition_(Concat_calnf, outdir):
    Concatpos_1 = os.path.join(outdir, "Concatenated.caln.pos1")
    Concatpos_2 = os.path.join(outdir, "Concatenated.caln.pos2")
    Concatpos_3 = os.path.join(outdir, "Concatenated.caln.pos3")
    with open(Concat_calnf,"r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('>'):
                with open(Concatpos_1,"a") as f1:
                    f1.write(line)
                with open(Concatpos_2,"a") as f2:
                    f2.write(line)
                with open(Concatpos_3,"a") as f3:
                    f3.write(line)
            else:
                Seq = line
                Seq = Seq.strip('\n')
                Seq_1 = Seq[0:-1:3]
                Seq_2 = Seq[1:-1:3]
                Seq_3 = Seq[2:-1:3]
                Seq_3 = Seq_3 + Seq[-1]
                with open(Concatpos_1,"a") as f1:
                    f1.write(Seq_1+'\n')
                with open(Concatpos_2,"a") as f2:
                    f2.write(Seq_2+'\n')
                with open(Concatpos_3,"a") as f3:
                    f3.write(Seq_3+'\n')
    Concatpos_1_aln = AlignIO.read(Concatpos_1, "fasta")
    Concatpos_2_aln = AlignIO.read(Concatpos_2, "fasta")
    Concatpos_3_aln = AlignIO.read(Concatpos_3, "fasta")
    return Concatpos_1_aln, Concatpos_2_aln, Concatpos_3_aln

def Coale(tree_famsf, families, outdir):
    whole_tree = ""
    whole_treef = os.path.join(outdir, "Whole.ctree")
    coalescence_treef = os.path.join(outdir, "Coalescence.ctree")
    for tree in tree_famsf:
        with open(tree,"r") as f:
            tree_content = f.readlines()
            for i in tree_content:
                whole_tree = whole_tree + i
    with open(whole_treef,"w") as f:
        f.write(whole_tree)
    if not os.path.isfile("G2S.Map"):
        gsmap, slist = GetG2SMap(families, outdir)
    else:
        gsmap = os.path.join(outdir, "G2S.Map")
    ASTER_cmd = ["astral-pro", "-i", whole_treef, "-a", gsmap, "-o", coalescence_treef]
    ASTER_cout = sp.run(ASTER_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    coalescence_ctree = Phylo.read(coalescence_treef,'newick')
    return coalescence_ctree, coalescence_treef

# Run MCMCtree
def Run_MCMCTREE(cds_alns, pro_alns, calnfs, palnfs, tree_famsf, families, tmpdir, outdir, speciestree):
    famnum = len(calnfs)
    cds_alns_rn, pro_alns_rn, calnfs_rn, palnfs_rn, tree_rns, tree_rn_fs = FileRn(cds_alns, pro_alns, tree_famsf, families, outdir)
    for fam in range(famnum):
        calnf = calnfs_rn[fam]
        palnf = palnfs_rn[fam]
        treef = tree_rn_fs[fam]
        McMctree = mcmctree(calnf, palnf, treef, tmpdir, outdir, speciestree)
        McMctree.run_mcmctree()
#Run r8s
def Run_r8s(inputtree, nsites, outdir, setting):
    prefix = os.path.basename(inputtree)
    r8s_inf = os.path.join(outdir, prefix + "_r8s_in.txt")
    treecontent = ""
    with open(inputtree,"r") as f:
        lines = f.readlines()
        for line in lines:
            treecontent = line.strip('\n').strip('\t').strip(' ')
    config = {'#nexus':'','begin trees;':'','tree inputtree = ':'{}'.format(treecontent),'end;':'','begin r8s;':'','blformat lengths=persite nsites={} ultrametric=no;'.format(nsites):'','set smoothing=':'100;', 'divtime method=':'pl;', 'end;':''}
    configcontent = ['{0}{1}'.format(k, v) for (k,v) in config.items()]
    configcontent = "\n".join(configcontent)
    with open(r8s_inf,"w") as f:
        f.write(configcontent)
        f.write('\nend;')



# NOTE: It would be nice to implement an option to do a complete approach
# where we use the tree in codeml to estimate Ks-scale branch lengths?
class GeneFamily:
    def __init__(self, gfid, cds, pro, tmp_path,
            aligner="mafft", tree_method="cluster", ks_method="GY94",
            eq_freq="F3X4", kappa=None, prequal=False, strip_gaps=False,
            min_length=3, codeml_iter=1, aln_options="--auto", 
            tree_options="-m LG", pairwise=False):
        self.id = gfid
        self.cds_seqs = cds
        self.pro_seqs = pro
        self.tmp_path = _mkdir(tmp_path)
        self.cds_fasta = os.path.join(self.tmp_path, "cds.fasta")
        self.pro_fasta = os.path.join(self.tmp_path, "pro.fasta")
        self.cds_alnf = os.path.join(self.tmp_path, "cds.aln")
        self.pro_alnf = os.path.join(self.tmp_path, "pro.aln")
        self.cds_aln = None
        self.pro_aln = None
        self.codeml_results = None
        self.no_codeml_results = None
        self.tree = None
        self.out = os.path.join(self.tmp_path, "{}.csv".format(gfid))

        # config
        self.aligner = aligner  # mafft | prank | muscle
        self.tree_method = tree_method  # iqtree | fasttree | alc
        self.ks_method = ks_method  # GY | NG
        self.kappa = kappa
        self.eq_freq = eq_freq
        self.prequal = prequal
        self.strip_gaps = strip_gaps  # strip gaps based on overall alignment
        self.codeml_iter = codeml_iter
        self.min_length = min_length  # minimum length of codon alignment
        self.aln_options = aln_options
        self.tree_options = tree_options
        self.pairwise = pairwise

    def get_ks(self):
        logging.info("Analysing family {}".format(self.id))
        self.align()
        self.run_codeml()
        if self.codeml_results is not None:
            self.get_tree()
            self.compile_dataframe()
        self.combine_results()

    def combine_results(self):
        if self.no_codeml_results is None:
            return
        self.codeml_results = pd.concat([self.codeml_results, self.no_codeml_results])
    
    def nan_result(self, pairs):
        """
        For a bunch of pairs obtain a data frame with missing data.
        """
        if len(pairs) == 0: return None
        data = []
        for pair in pairs:
            pairid = "__".join(sorted(pair))
            data.append({
                "pair": pairid, 
                "gene1": pair[0], 
                "gene2": pair[1], 
                "family": self.id})
        return pd.DataFrame.from_records(data).set_index("pair")

    # NOT TESTED
    def run_prequal(self):
        cmd = ["prequal", self.pro_fasta]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="prequal")
        self.pro_fasta = "{}.filtered".format(self.pro_fasta)

    def align(self):
        _write_fasta(self.pro_fasta, self.pro_seqs)
        if self.prequal:
            self.run_prequal()
        if self.aligner == "mafft":
            self.run_mafft(options=self.aln_options)
        else:
            logging.error("Unsupported aligner {}".format(self.aligner))
        self.get_codon_alignment()

    def run_mafft(self, options="--auto"):
        cmd = ["mafft"] + options.split() + ["--amino", self.pro_fasta]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        with open(self.pro_alnf, 'w') as f: f.write(out.stdout.decode('utf-8'))
        _log_process(out, program="mafft")
        self.pro_aln = AlignIO.read(self.pro_alnf, "fasta")

    def get_codon_alignment(self):
        self.cds_aln = _pal2nal(self.pro_aln, self.cds_seqs)
        if self.strip_gaps:
            self.cds_aln = _strip_gaps(self.cds_aln)

    def run_codeml(self):
        codeml = Codeml(self.cds_aln, exe="codeml", tmp=self.tmp_path, prefix=self.id)
        # TODO, do something with `no_result`
        if self.pairwise:
            result, no_result = codeml.run_codeml_pairwise(
                    preserve=True, times=self.codeml_iter)
        else:
            result, no_result = codeml.run_codeml(
                    preserve=True, times=self.codeml_iter)
        self.codeml_results = result
        self.no_codeml_results = self.nan_result(no_result)

    def get_tree(self):
        # dispatch method
        # This likely will have to catch families of only two or three members.
        if self.tree_method == "cluster":
            tree = self.cluster()
        elif self.tree_method == "iqtree":
            tree = self.run_iqtree(options=self.tree_options)
        elif self.tree_method == "fasttree":
            tree = self.run_fasttree()
        self.tree = tree

    def run_iqtree(self, options="-m LG"):
        cmd = ["iqtree", "-s", self.pro_alnf] + options.split()
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="iqtree")
        return _process_unrooted_tree(self.pro_alnf + ".treefile")

    def run_fasttree(self):
        tree_pth = self.pro_alnf + ".nw"
        cmd = ["fasttree", '-out', tree_pth, self.pro_alnf]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="fasttree")
        return _process_unrooted_tree(self.pro_alnf + ".nw")

    def cluster(self):
        return cluster_ks(self.codeml_results)

    def compile_dataframe(self):
        n = len(self.cds_seqs)
        d = {}
        l = self.tree.get_terminals()
        for i in range(len(l)):
            gi = l[i].name
            for j in range(i+1, len(l)):
                gj = l[j].name
                pair = "__".join(sorted([gi, gj]))
                node = self.tree.common_ancestor(l[i], l[j])
                length = self.cds_aln.get_alignment_length()
                d[pair] = {"node": node.name, "family": self.id, "alnlen": length}
        df = pd.DataFrame.from_dict(d, orient="index")
        self.codeml_results = self.codeml_results.join(df)

def _get_ks(family):
    family.get_ks()
    family.codeml_results.to_csv(family.out)


class KsDistributionBuilder:
    def __init__(self, gene_families, seqs, n_threads=4):
        self.families = gene_families
        self.df = None
        self.seqs = seqs
        self.n_threads = n_threads

    def get_distribution(self):
        Parallel(n_jobs=self.n_threads)(
            delayed(_get_ks)(family) for family in self.families)
        df = pd.concat([pd.read_csv(x.out, index_col=0) 
            for x in self.families], sort=True)
        self.df = add_original_ids(df, self.seqs)

def reverse_map(seqs):
    return {v: k for k, v in seqs.items()}

def add_original_ids(df, seqs):
    df.index.name = "p"
    df = df.rename({"gene1": "g1", "gene2": "g2"}, axis=1)
    revmap = reverse_map(seqs.idmap)
    df["gene1"] = _rename(df["g1"], revmap)
    df["gene2"] = _rename(df["g2"], revmap)
    df["pair"] = df[["gene1","gene2"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1) 
    return df.set_index("pair")

#class focusRBHTreeDating:
    #def __init__(self, seqid_table, seq, aligner="mafft", tree_method="fasttree", min_length=3, aln_options="--auto", n_threads=4):
        #self.cds_seqs = s


