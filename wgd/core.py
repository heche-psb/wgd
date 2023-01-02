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
from Bio.Align import MultipleSeqAlignment
from Bio.Alphabet import IUPAC
from Bio.Data.CodonTable import TranslationError
from Bio import Phylo
from joblib import Parallel, delayed
from wgd.codeml import Codeml
from wgd.cluster import cluster_ks
from wgd.mcmctree import mcmctree
from wgd.beast import beast
from timeit import default_timer as timer
import copy
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
    df.index = ["GF{:0>8}".format(i+1) for i in range(len(df.index))]

def _process_unrooted_tree(treefile, fformat="newick"):
    tree = Phylo.read(treefile, fformat)
    tree.root_at_midpoint()
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

    def orig_profasta(self):
        self.orig_pro_fasta = os.path.join(self.tmp_path, self.prefix+'.pep')
        with open(self.orig_pro_fasta,'w') as f:
            for k,v in self.idmap.items(): f.write('>{}\n{}\n'.format(k,self.pro_sequence[v]))

    def spgenemap(self):
        self.sgmap = {i:self.prefix for i in self.idmap.keys()}
        return self.sgmap

    def merge(self, other):
        """
        Merge other into self, keeping the paths etc. of self.
        """
        self.cds_seqs.update(other.cds_seqs)
        self.pro_seqs.update(other.pro_seqs)
        self.idmap.update(other.idmap)

    def merge_seq(self,other):
        self.cds_seqs.update(other.cds_seqs)
        self.cds_sequence.update(other.cds_sequence)
        self.pro_sequence.update(other.pro_sequence)
        self.idmap.update(other.idmap)

    def merge_dmd_hits(self,other):
        self.dmd_hits.update(other.dmd_hits)

    def make_diamond_db(self):
        if not os.path.isfile(self.pro_db + '.dmnd'):
            cmd = ["diamond", "makedb", "--in", self.pro_fasta, "-d", self.pro_db]
            out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            logging.debug(out.stderr.decode())
            if out.returncode == 1: logging.error(out.stderr.decode())

    def run_diamond(self, seqs, orthoinfer, eval=1e-10):
        self.make_diamond_db()
        run = "_".join([self.prefix, seqs.prefix + ".tsv"])
        outfile = os.path.join(self.tmp_path, run)
        if not orthoinfer:
            cmd = ["diamond", "blastp", "-d", self.pro_db, "-q", seqs.pro_fasta, "-o", outfile]
            out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            logging.debug(out.stderr.decode())
        df = pd.read_csv(outfile, sep="\t", header=None)
        df = df.loc[df[0] != df[1]]
        self.dmd_hits[seqs.prefix] = df = df.loc[df[10] <= eval]
        return df

    def get_rbh_orthologs(self, seqs, cscore, orthoinfer, eval=1e-10):
        if self == seqs:
            raise ValueError("RBH orthologs only defined for distinct species")
        df = self.run_diamond(seqs, orthoinfer, eval=eval)
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

    def rndmd_hit(self):
        self.dmd_hits = {'_'.join([self.prefix,k]):v for k,v in self.dmd_hits.items()}
        #for key in self.dmd_hits.copy().keys(): self.dmd_hits['_'.join([self.prefix,key])] = self.dmd_hits.pop(key)

    def get_para_skip_dmd(self, inflation=1.5, eval=1e-10):
        gf = os.path.join(self.tmp_path, 'Concated')
        df = pd.concat([v for v in self.dmd_hits.values()])
        df.to_csv(gf, sep="\t", header=False, index=False, columns=[0,1,10])
        gf = SequenceSimilarityGraph(gf)
        mcl_out = gf.run_mcl(inflation=inflation)
        with open(mcl_out, "r") as f:
            for i, line in enumerate(f.readlines()): self.mcl[i] = line.strip().split()

    def get_paranome(self, inflation=1.5, eval=1e-10):
        df = self.run_diamond(self, False, eval=eval)
        gf = self.get_mcl_graph(self.prefix)
        mcl_out = gf.run_mcl(inflation=inflation)
        with open(mcl_out, "r") as f:
            for i, line in enumerate(f.readlines()): self.mcl[i] = line.strip().split()

    def get_mcl_graph(self, *args):
        # args are keys in `self.dmd_hits` to use for building MCL graph
        gf = os.path.join(self.tmp_path, "_".join([self.prefix] + list(args)))
        df = pd.concat([self.dmd_hits[x] for x in args])
        df.to_csv(gf, sep="\t", header=False, index=False, columns=[0,1,10])
        return SequenceSimilarityGraph(gf)

    def write_paranome(self, orthoinfer, fname=None, singletons=True):
        if singletons: 
            self.add_singletons_paranome()
        if not fname:
            fname = os.path.join(self.out_path, "{}.tsv".format(self.prefix))
        with open(fname, "w") as f:
            if not orthoinfer:
                f.write("\t" + self.prefix + "\n")
            for i, (k, v) in enumerate(sorted(self.mcl.items())):
                # We report original gene IDs
                f.write("GF{:0>8}\t".format(i+1))
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
    Read gene MRBH families
    derived from dmd -focus or --globalmrbh in the format that each column contains seqid of each species and header of column is cds filename of each species
    """
    seqid_table = []
    df = pd.read_csv(fname,header=0,index_col=0,sep='\t')
    yids = lambda i: ', '.join(list(df.loc[i,:].dropna())).split(', ')
    seqid_table = [yids(i) for i in df.index]
    #with open (fname,'r') as orthotable:
    #    next(orthotable)
    #    for row in orthotable:
    #        seqid = []
    #        for s in row.split('\t'):
    #            s = s.strip('\n')
    #            if s:
    #                for i in s.split(', '): seqid.append(i)
    #        seqid_table.append(seqid[1:])
    return seqid_table

def merge_seqs(seqs):
    if type(seqs) == list:
        if len(seqs) > 2: raise ValueError("More than two sequence data objects?")
        if len(seqs) == 2: seqs[0].merge(seqs[1])
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
            if rename: family += _rename(ids, seqs.idmap)
            else: family += ids
        if len(family) > 1:
            cds = {x: seqs.cds_seqs[x] for x in family}
            pro = {x: seqs.pro_seqs[x] for x in family}
            tmp = os.path.join(seqs.tmp_path, fid)
            gene_families.append(GeneFamily(fid, cds, pro, tmp, **kwargs))
        else: logging.debug("Skipping singleton family {}{}".format(fid,family))
    return gene_families

def identity_ratio(aln):
    identity = [i for i in range(aln.get_alignment_length()) if len(set(aln[:,i]))==1]
    ratio = len(identity)/aln.get_alignment_length()
    return ratio

def Aligninfo(aln):
    aln_strip = _strip_gaps(aln)
    aln_length = aln.get_alignment_length()
    aln_strip_length = aln_strip.get_alignment_length()
    Coverage = float(aln_strip_length/aln_length)
    info={'alignmentcoverage':Coverage,'alignmentidentity':identity_ratio(aln_strip),'alignmentlength':aln_length,'strippedalignmentlength':aln_strip_length}
    return info

def Global2Pair(info):
    info['PairAlignmentCoverage'] = info.pop('AlignmentCoverage')
    info['PairAlignmentIdentity'] = info.pop('AlignmentIdentity')
    info['PairStrippedAlignmentLength'] = info.pop('StrippedAlignmentLength')
    info.pop(AlignmentLength)
    return info

def Pairaligninfo(aln):
    num = len(aln)
    pairs_info = []
    for i in range(num-1):
        for j in range(i+1,num):
            pair_aln = MultipleSeqAlignment([aln[i], aln[j]])
            pair_info = Aligninfo(pair_aln)
            pair_id = "__".join(sorted[aln[i].id, aln[j].id])
            pairinfo.append({'pair':pair_id}.update(Global2Pair(pair_info)))
    df_pairs_info = pd.DataFrame.from_dict(pairs_info).set_index("pair")
    return df_pairs_info

def add2table(i,outdir,cds_fastaf,palnfs,pro_alns,calnfs,calnfs_length,cds_alns,fnamecalns,fnamepalns):
    famid = "GF{:0>8}".format(i+1)
    cds_fastaf.append(os.path.join(outdir, famid + ".pep"))
    fnamepaln =os.path.join(outdir, famid + ".paln")
    fnamepalns[famid]=fnamepaln
    palnfs.append(fnamepaln)
    pro_aln = AlignIO.read(fnamepaln, "fasta")
    pro_alns[famid] = pro_aln
    fnamecaln =os.path.join(outdir, famid + ".caln")
    fnamecalns[famid] = fnamecaln
    calnfs.append(fnamecaln)
    cds_aln = AlignIO.read(fnamecaln, "fasta")
    calnfs_length.append(cds_aln.get_alignment_length())
    cds_alns[famid] = cds_aln

def mafft_cmd(fpep,o,fpaln):
    cmd = ["mafft"] + o.split() + ["--amino", fpep]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    _log_process(out, program="mafft")
    with open(fpaln, 'w') as f: f.write(out.stdout.decode('utf-8'))

def backtrans(fpaln,fcaln,idmap,seq_cds):
    aln = {}
    pro_aln = AlignIO.read(fpaln, "fasta")
    for i, s in enumerate(pro_aln):
        cds_aln = ""
        safeid = idmap.get(s.id)
        cds_seq = seq_cds.get(safeid)
        k = 0
        for j in range(pro_aln.get_alignment_length()):
            if pro_aln[i,j] == "-": cds_aln += "---"
            elif pro_aln[i,j] == "X": cds_aln += "???"
            else:
                cds_aln += cds_seq[k:k+3]
                k = k + 3
        aln[s.id] = cds_aln
    with open(fcaln, 'w') as f:
        for k, v in aln.items(): f.write(">{}\n{}\n".format(k, v))
    return pro_aln

def getseqmetaln(i,fam,outdir,idmap,seq_pro,seq_cds,option):
    famid = "GF{:0>8}".format(i+1)
    fnamep =os.path.join(outdir, famid + ".pep")
    fnamec =os.path.join(outdir, famid + ".cds")
    for seqid in fam:
        safeid = idmap.get(seqid)
        with open(fnamep,'a') as f: f.write(">{}\n{}\n".format(seqid, seq_pro.get(safeid)))
        with open(fnamec,'a') as f: f.write(">{}\n{}\n".format(seqid, seq_cds.get(safeid)))
    fnamepaln =os.path.join(outdir, famid + ".paln")
    mafft_cmd(fnamep,option,fnamepaln)
    fnamecaln =os.path.join(outdir, famid + ".caln")
    backtrans(fnamepaln,fnamecaln,idmap,seq_cds)
    #Note that here the backtranslated codon-alignment will be shorter than the original cds file by a stop codon

def addmbtree(outdir,tree_fams,tree_famsf,i=0,concat=False):
    if not concat: famid = "GF{:0>8}".format(i+1)
    else: famid = 'Concat' 
    tree_pth = famid + ".paln.nexus" + ".con.tre.backname"
    tree_pth = os.path.join(outdir, tree_pth)
    tree = Phylo.read(tree_pth,'newick')
    tree_fams[famid]=tree
    tree_famsf.append(tree_pth)

def mrbayes_run(outdir,famid,fnamepaln,pro_aln,treeset):
    fnamepalnnexus =os.path.join(outdir, famid + ".paln.nexus")
    AlignIO.convert(fnamepaln, 'fasta', fnamepalnnexus, 'nexus', IUPAC.extended_protein)
    cwd = os.getcwd()
    os.chdir(outdir)
    conf = os.path.join(cwd, outdir, famid + ".config.mb")
    logf = os.path.join(cwd, outdir, famid + ".mb.log")
    bashf = os.path.join(cwd, outdir, famid + ".bash.mb")
    config = {'set':'autoclose=yes nowarn=yes','execute':'./{}'.format(os.path.basename(fnamepalnnexus)),'prset':'aamodelpr=fixed(lg)','lset':'rates=gamma','mcmcp':['diagnfreq=100','samplefreq=10'],'mcmc':'ngen=1100 savebrlens=yes nchains=1','sumt':'','sump':'','quit':''}
    if not treeset is None:
        diasam = [100,10]
        ngnc = [1100,1]
        for i in treeset:
            i = i.strip('\t').strip(' ')
            if 'diagnfreq' in i: diasam[0] = i[10:]
            if 'samplefreq' in i: diasam[1] = i[11:]
            if 'ngen' in i: ngnc[0] = i[5:]
            if 'nchains' in i: ngnc[1] = i[8:]
        config['mcmcp'] = ['diagnfreq={}'.format(diasam[0]),'samplefreq={}'.format(diasam[1])]
        config['mcmc'] = 'ngen={0} savebrlens=yes nchains={1}'.format(ngnc[0],ngnc[1])
    with open(conf,"w") as f:
        para = []
        for (k,v) in config.items():
            if isinstance(v, list):
                para.append('{0} {1}'.format(k, v[0]))
                para.append('{0} {1}'.format(k, v[1]))
            else: para.append('{0} {1}'.format(k, v))
        para = "\n".join(para)
        f.write(para)
    with open(bashf,"w") as f:
        f.write('mb <{0}> {1}'.format(os.path.basename(conf),os.path.basename(logf)))
    mb_cmd = ["sh", os.path.basename(bashf)]
    sp.run(mb_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    genenumber = len(pro_aln)
    linenumber = genenumber + 3
    mb_out = famid + ".paln.nexus" + ".con.tre"
    mb_out_content = []
    with open(mb_out,"r") as f:
        lines = f.readlines()
        for line in lines: mb_out_content.append(line.strip(' ').strip('\t').strip('\n').strip(','))
    mb_useful = mb_out_content[-linenumber:-1]
    mb_id = mb_useful[:-2]
    mb_tree = mb_useful[-1]
    mb_id_dict = {}
    tree_pth = famid + ".paln.nexus" + ".con.tre.backname"
    for i in mb_id:
        i = i.split("\t")
        mb_id_dict[i[0]]=i[1]
    with open(tree_pth,'w') as f:
        for (k,v) in mb_id_dict.items(): mb_tree = mb_tree.replace('{}[&prob='.format(k),'{}[&prob='.format(v))
        f.write(mb_tree[27:])
    os.chdir(cwd)

def addiqfatree(famid,tree_fams,fnamecaln,tree_famsf,postfix):
    tree_pth = fnamecaln + postfix
    tree = Phylo.read(tree_pth,'newick')
    tree_fams[famid] = tree
    tree_famsf.append(tree_pth)

def iqtree_run(treeset,fnamecaln):
    if not treeset is None:
        treesetfull = []
        iq_cmd = ["iqtree", "-s", fnamecaln]
        for i in treeset:
            i = i.strip(" ").split(" ")
            if type(i) == list: treesetfull = treesetfull + i
            else: treesetfull.append(i)
        iq_cmd = iq_cmd + treesetfull
    else: iq_cmd = ["iqtree", "-s", fnamecaln] + ["-fast"] #+ ["-st","CODON"] + ["-bb", "1000"] + ["-bnni"]
    sp.run(iq_cmd, stdout=sp.PIPE)

def fasttree_run(fnamecaln,treeset):
    tree_pth = fnamecaln + ".fasttree"
    if not treeset is None:
        treesetfull = []
        ft_cmd = ["FastTree", '-out', tree_pth, fnamecaln]
        for i in treeset:
            i = i.strip(" ").split(" ")
            if type(i) == list: treesetfull = treesetfull + i
            else: treesetfull.append(i)
        ft_cmd = ft_cmd[:1] + treesetfull + ft_cmd[1:]
    else: ft_cmd = ["FastTree", '-out', tree_pth, fnamecaln]
    sp.run(ft_cmd, stdout=sp.PIPE, stderr=sp.PIPE)

def get_mrbh(s_i,s_j,cscore,eval):
    logging.info("{} vs. {}".format(s_i.prefix, s_j.prefix))
    s_i.get_rbh_orthologs(s_j, cscore, False, eval=eval)
    s_i.write_rbh_orthologs(s_j,singletons=False)

def getrbhf(s_i,s_j,outdir):
    fname = os.path.join(outdir, "{}_{}.rbh.tsv".format(s_i.prefix, s_j.prefix))
    df = pd.read_csv(fname,header = 0, index_col = False,sep = '\t')
    return df

def getfastaf(i,fam,rbhgfdirname,seq_pro,idmap,seq_cds):
    for seqs in fam:
        fname = os.path.join(rbhgfdirname, 'GF{:0>8}'.format(i+1) + ".pep")
        with open(fname,'a') as f:
            Record = seq_pro.get(idmap.get(seqs))
            f.write(">{}\n{}\n".format(seqs, Record))
        fname2 = os.path.join(rbhgfdirname, 'GF{:0>8}'.format(i+1) + ".cds")
        with open(fname2,'a') as f:
            Record = seq_cds.get(idmap.get(seqs))
            f.write(">{}\n{}\n".format(seqs, Record))

def mrbh(globalmrbh,outdir,s,cscore,eval,keepduplicates,anchorpoints,focus,keepfasta,nthreads):
    if globalmrbh:
        logging.info("Multiple CDS files: will compute globalMRBH orthologs or cscore-defined homologs regardless of focus species")
        table = pd.DataFrame()
        gmrbhf = os.path.join(outdir, 'global_MRBH.tsv')
        for i in range(len(s)-1):
            tables = []
            Parallel(n_jobs=nthreads)(delayed(get_mrbh)(s[i],s[j],cscore,eval) for j in range(i+1,len(s)))
            for j in range(i+1,len(s)):
                df = getrbhf(s[i],s[j],outdir)
                if table.empty: table = df
                else: table = table.merge(df)
        gfid = ['GF{:0>8}'.format(str(i+1)) for i in range(table.shape[0])]
        table.insert(0,'GF', gfid)
        if not keepduplicates:
            for i in table.columns: table.drop_duplicates(subset=[i],inplace=True)
        table.to_csv(gmrbhf, sep="\t",index=False)
    elif not focus is None:
        logging.info("Multiple CDS files: will compute RBH orthologs or cscore-defined homologs between focus species and remaining species")
        x = 0
        table = pd.DataFrame()
        focusname = os.path.join(outdir, 'merge_focus.tsv')
        for i in range(len(s)):
            if s[i].prefix == focus: x = x+i
        if x == 0:
            Parallel(n_jobs=nthreads)(delayed(get_mrbh)(s[0],s[j],cscore,eval) for j in range(1,len(s)))
            for j in range(1, len(s)):
                df = getrbhf(s[0],s[j],outdir)
                if table.empty: table = df
                else: table = table.merge(df)
            if not keepduplicates: table = table.drop_duplicates([focus])
            table.insert(0, focus, table.pop(focus))
        else:
            Parallel(n_jobs=nthreads)(delayed(get_mrbh)(s[x],s[k],cscore,eval) for k in range(0,x))
            for k in range(0,x):
                df = getrbhf(s[x],s[k],outdir)
                if table.empty: table = df
                else: table = table.merge(df)
            if not len(s) == 2 and not x+1 == len(s):
                Parallel(n_jobs=nthreads)(delayed(get_mrbh)(s[x],s[l],cscore,eval) for l in range(x+1,len(s)))
                for l in range(x+1,len(s)):
                    df = getrbhf(s[x],s[l],outdir)
                    table = table.merge(df)
            if not keepduplicates: table = table.drop_duplicates([focus])
            table.insert(0, focus, table.pop(focus))
        gfid = ['GF{:0>8}'.format(str(i+1)) for i in range(table.shape[0])]
        table.insert(0,'GF', gfid)
        table.to_csv(focusname, sep="\t",index=False)
    if not anchorpoints is None:
        ap = pd.read_csv(anchorpoints,header=0,index_col=False,sep='\t')
        ap = ap.loc[:,'gene_x':'gene_y']
        ap_reverse = ap.rename(columns = {'gene_x' : 'gene_y', 'gene_y' : 'gene_x'})
        ap_combined = pd.concat([ap,ap_reverse])
        focusapname = os.path.join(outdir, 'merge_focus_ap.tsv')
        table.insert(1, focus, table.pop(focus))
        table_ap = table.merge(ap_combined,left_on = focus,right_on = 'gene_x')
        table_ap.drop('gene_x', inplace=True, axis=1)
        table_ap.insert(2, 'gene_y', table_ap.pop('gene_y'))
        #table_ap.columns = table_ap.columns.str.replace(focus, focus + '_ap1')
        #table_ap.columns = table_ap.columns.str.replace('gene_y', focus + '_ap2')
        table_ap.rename(columns = {focus : focus + '_ap1', 'gene_y' : focus + '_ap2'}, inplace = True)
        table_ap.to_csv(focusapname, sep="\t",index=False)
    if globalmrbh or not focus is None:
        if keepfasta:
            idmap = {}
            for i in range(len(s)): idmap.update(s[i].idmap)
            if globalmrbh: seqid_table = read_MultiRBH_gene_families(gmrbhf)
            else: seqid_table = read_MultiRBH_gene_families(focusname)
            #for fam in seqid_table:
                #for seq in fam: safeid = idmap.get(seq)
            seq_cds = {}
            seq_pro = {}
            for i in range(len(s)):
                seq_cds.update(s[i].cds_sequence)
                seq_pro.update(s[i].pro_sequence)
            rbhgfdirname = outdir + '/' + 'MRBH_GF_FASTA' + '/'
            os.mkdir(rbhgfdirname)
            Parallel(n_jobs=nthreads)(delayed(getfastaf)(i,fam,rbhgfdirname,seq_pro,idmap,seq_cds) for i, fam in enumerate(seqid_table))
           # for i, fam in enumerate(seqid_table):
           #     for seqs in fam:
           #         fname = os.path.join(rbhgfdirname, 'GF{:0>5}'.format(i+1) + ".pep")
           #         with open(fname,'a') as f:
           #             Record = seq_pro.get(idmap.get(seqs))
           #             f.write(">{}\n{}\n".format(seqs, Record))
           #         fname2 = os.path.join(rbhgfdirname, 'GF{:0>5}'.format(i+1) + ".cds")
           #         with open(fname2,'a') as f:
           #             Record = seq_cds.get(idmap.get(seqs))
           #             f.write(">{}\n{}\n".format(seqs, Record))
            if not anchorpoints is None:
                seqid_table = read_MultiRBH_gene_families(focusapname)
                rbhgfapdirname = outdir + '/' + 'MRBH_AP_GF_FASTA' + '/'
                os.mkdir(rbhgfapdirname)
                Parallel(n_jobs=nthreads)(delayed(getfastaf)(i,fam,rbhgfapdirname,seq_pro,idmap,seq_cds) for i, fam in enumerate(seqid_table))
                #for i, fam in enumerate(seqid_table):
                #    for seqs in fam:
                #        fname = os.path.join(rbhgfapdirname, 'GF{:0>5}'.format(i+1) + ".pep")
                #        with open(fname,'a') as f:
                #            Record = seq_pro.get(idmap.get(seqs))
                #            f.write(">{}\n{}\n".format(seqs, Record))
                #        fname2 = os.path.join(rbhgfapdirname, 'GF{:0>5}'.format(i+1) + ".cds")
                #        with open(fname2,'a') as f:
                #            Record = seq_cds.get(idmap.get(seqs))
                #            f.write(">{}\n{}\n".format(seqs, Record))

def get_MultipRBH_gene_families(seqs, fams, tree_method, treeset, outdir,nthreads, option="--auto", **kwargs):
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
    cds_fastaf = []
    for i in range(len(seqs)):
        seq_cds.update(seqs[i].cds_sequence)
        seq_pro.update(seqs[i].pro_sequence)
        idmap.update(seqs[i].idmap)
    fnamecalns, fnamepalns = {},{}
    Parallel(n_jobs=nthreads)(delayed(getseqmetaln)(i,fam,outdir,idmap,seq_pro,seq_cds,option) for i, fam in enumerate(fams))
    for i in range(len(fams)):
        add2table(i,outdir,cds_fastaf,palnfs,pro_alns,calnfs,calnfs_length,cds_alns,fnamecalns,fnamepalns)
    x = lambda i : "GF{:0>8}".format(i+1)
    if tree_method == "mrbayes":
        Parallel(n_jobs=nthreads)(delayed(mrbayes_run)(outdir,x(i),fnamepalns[x(i)],pro_alns[x(i)],treeset) for i in range(len(fams)))
        for i in range(len(fams)): addmbtree(outdir,tree_fams,tree_famsf,i=i,concat=False)
    if tree_method == "iqtree":
        Parallel(n_jobs=nthreads)(delayed(iqtree_run)(treeset,fnamecalns[x(i)]) for i in range(len(fams)))
        for i in range(len(fams)): addiqfatree(x(i),tree_fams,fnamecalns[x(i)],tree_famsf,postfix = '.treefile')
    if tree_method == "fasttree":
        Parallel(n_jobs=nthreads)(delayed(fasttree_run)(fnamecalns[x(i)],treeset) for i in range(len(fams)))
        for i in range(len(fams)): addiqfatree(x(i),tree_fams,fnamecalns[x(i)],tree_famsf,postfix = '.fasttree')
    return cds_alns, pro_alns, tree_famsf, calnfs, palnfs, calnfs_length, cds_fastaf, tree_fams

def select_phylogeny(tree_fams,slist):
    tree_fams_phylocorrect = {}
    x = lambda i : "GF{:0>8}".format(i+1)
    wgd_mrca = [sp for sp in slist if sp[-4:] == '_ap1' or sp[-4:] == '_ap2']
    for i in range(len(tree_fams)):
        tree = copy.deepcopy(tree_fams[x(i)])
        tree.root_at_midpoint()
        wgd_node = tree.common_ancestor({"name": wgd_mrca[0]}, {"name": wgd_mrca[1]})
        if wgd_node.count_terminals() == 2:
            tree_fams_phylocorrect[x(i)] = tree_fams[x(i)]
    return tree_fams_phylocorrect

def judgetree(tree,wgd_mrca):
    tree_copy = copy.deepcopy(tree)
    wgd_node = tree_copy.common_ancestor({"name": wgd_mrca[0]}, {"name": wgd_mrca[1]})
    if wgd_node.count_terminals() == 2: return True
    else: return False

#def Test_tree_boots(speciestree,tree_famsf):    
def GetG2SMap(families, outdir):
    df = pd.read_csv(families,header=0,index_col=0,sep='\t')
    G2SMap = os.path.join(outdir, "G2S.Map")
    Slist = []
    yids = lambda i: ', '.join(list(i)).split(', ')
    for i in df.columns:
        Slist.append(i)
        ids = yids(df[i].dropna())
        with open(G2SMap, "a") as f:
            for j in ids: f.write(j + " "+ i + "\n")
    return G2SMap, Slist

def FileRn(cds_alns_rn, pro_alns_rn, calnfs, palnfs):
    famnum = len(pro_alns_rn)
    calnfs_rn = [i + ".rename" for i in calnfs]
    palnfs_rn = [i + ".rename" for i in palnfs]
    for i in range(famnum):
        famid = "GF{:0>8}".format(i+1)
        cds_aln_rn = cds_alns_rn[famid]
        pro_aln_rn = pro_alns_rn[famid]
        for j in range(len(pro_aln_rn)):
            with open(calnfs_rn[i], "a") as f:
                f.write(">{}\n{}\n".format(cds_aln_rn[j].id,cds_aln_rn[j].seq))
                    #if k[0] == pro_aln[j].id:
                        #pro_aln[j].id = k[1]
            with open(palnfs_rn[i], "a") as f:
                f.write(">{}\n{}\n".format(pro_aln_rn[j].id,pro_aln_rn[j].seq))
    return calnfs_rn, palnfs_rn

def Concat(cds_alns, pro_alns, families, tree_method, treeset, outdir):
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
        famid = "GF{:0>8}".format(i+1)
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
    Concat_ctrees, ctree_pths, Concat_ptrees, ptree_pths, famid = {},[],{},[],'Concat'
    if tree_method == 'mrbayes':
        mrbayes_run(outdir,famid,Concat_palnf,Concat_paln,treeset)
        addmbtree(outdir,Concat_ptrees,ptree_pths,i,concat=True)
        # TO DO -- get caln work in mrbayes
        Concat_ctrees, ctree_pths = Concat_ptrees, ptree_pths
    if tree_method == "iqtree":
        iqtree_run(treeset,Concat_calnf)
        addiqfatree(famid,Concat_ctrees,Concat_calnf,ctree_pths,postfix='.treefile')
        iqtree_run(treeset,Concat_palnf)
        addiqfatree(famid,Concat_ptrees,Concat_palnf,ptree_pths,postfix='.treefile')
    if tree_method == "fasttree":
        fasttree_run(Concat_calnf,treeset)
        addiqfatree(famid,Concat_ctrees,Concat_calnf,ctree_pths,postfix='.fasttree')
        fasttree_run(Concat_palnf,treeset)
        addiqfatree(famid,Concat_ptrees,Concat_palnf,ptree_pths,postfix='.fasttree')
    Concat_ctree, ctree_pth, Concat_ptree, ptree_pth = Concat_ctrees[famid], ctree_pths[0], Concat_ptrees[famid], ptree_pths[0]
    return cds_alns_rn, pro_alns_rn, Concat_ctree, Concat_ptree, Concat_calnf, Concat_palnf, ctree_pth, ctree_length, gsmap, Concat_caln, Concat_paln, slist

def _Codon2partition_(alnf, outdir):
    pos_1 = alnf + ".pos1"
    pos_2 = alnf + ".pos2"
    pos_3 = alnf + ".pos3"
    with open(alnf,"r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('>'):
                with open(pos_1,"a") as f1:
                    f1.write(line)
                with open(pos_2,"a") as f2:
                    f2.write(line)
                with open(pos_3,"a") as f3:
                    f3.write(line)
            else:
                Seq = line.strip('\n')
                Seq_1 = Seq[0:-1:3]
                Seq_2 = Seq[1:-1:3]
                Seq_3 = Seq[2:-1:3]
                Seq_3 = Seq_3 + Seq[-1]
                with open(pos_1,"a") as f1:
                    f1.write(Seq_1+'\n')
                with open(pos_2,"a") as f2:
                    f2.write(Seq_2+'\n')
                with open(pos_3,"a") as f3:
                    f3.write(Seq_3+'\n')
    pos_1_aln = AlignIO.read(pos_1, "fasta")
    pos_2_aln = AlignIO.read(pos_2, "fasta")
    pos_3_aln = AlignIO.read(pos_3, "fasta")
    return pos_1_aln, pos_2_aln, pos_3_aln, pos_1, pos_2, pos_3

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
    gsmap = os.path.join(outdir, "G2S.Map")
    if not os.path.isfile(gsmap):
        gsmap, slist = GetG2SMap(families, outdir)
    ASTER_cmd = ["astral-pro", "-i", whole_treef, "-a", gsmap, "-o", coalescence_treef]
    ASTER_cout = sp.run(ASTER_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    coalescence_ctree = Phylo.read(coalescence_treef,'newick')
    return coalescence_ctree, coalescence_treef

def fasta2paml(aln,alnf):
    alnf_paml = alnf + '.paml'
    with open (alnf_paml,'w') as f:
        f.write(' {0} {1}\n'.format(len(aln),aln.get_alignment_length()))
        for i in aln:
            f.write('{0}          {1}\n'.format(i.id,i.seq))
    return alnf_paml

def Getpartitionedpaml(alnf,outdir):
    aln_1, aln_2, aln_3, alnf_1, alnf_2, alnf_3 = _Codon2partition_(alnf,outdir)
    aln_1_paml = fasta2paml(aln_1,alnf_1)
    aln_2_paml = fasta2paml(aln_2,alnf_2)
    aln_3_paml = fasta2paml(aln_3,alnf_3)
    alnfpartitioned_paml = alnf + '.partitioned.paml'
    with open(alnfpartitioned_paml,'w') as f:
        with open(aln_1_paml,'r') as f1:
            data1  = f1.read()
        with open(aln_2_paml,'r') as f2:
            data2  = f2.read()
        with open(aln_3_paml,'r') as f3:
            data3  = f3.read()
        f.write(data1+data2+data3)
    return alnfpartitioned_paml

def get_dates(wgd_mrca,CI_table,PM_table,prefixx):
    Figtree = Phylo.read('FigTree.tre','nexus')
    wgd_node = Figtree.common_ancestor({"name": wgd_mrca[0]}, {"name": wgd_mrca[1]})
    CI = wgd_node.comment.strip('[&95%={').strip('}]').split(', ')
    PM = wgd_node.clades[0].branch_length
    CI_table[prefixx]=[float(i) for i in CI]
    PM_table[prefixx]=PM

def Getback_CIPM(outdir,CI_table,PM_table,wgd_mrca,calnfs_rn,Concat_calnf_paml):
    parent = os.getcwd()
    calnfs_rn_cat = calnfs_rn + [Concat_calnf_paml]
    for i,calnf_rn in enumerate(calnfs_rn_cat):
        prefix = os.path.basename(calnf_rn).replace('.caln','').replace('.rename','').replace('.paml','').replace('.','_')
        folder = os.path.join(outdir, "mcmctree",prefix,"cds")
        os.chdir(folder)
        get_dates(wgd_mrca,CI_table,PM_table,prefix+"_cds")
        os.chdir(parent)
        folder = os.path.join(outdir, "mcmctree",prefix,"pep")
        os.chdir(folder)
        get_dates(wgd_mrca,CI_table,PM_table,prefix+"_pep")
        os.chdir(parent)

def Run_BEAST(Concat_caln, Concat_paln, Concat_calnf, cds_alns_rn, pro_alns_rn, calnfs, tmpdir, outdir, speciestree, datingset, slist, nthreads, beastlgjar, beagle, fossil, chainset, rootheight):
    beasts = []
    famnum = len(calnfs)
    beast_concat = beast(Concat_calnf, Concat_caln, Concat_paln, tmpdir, outdir, speciestree, datingset, slist, fossil, chainset, rootheight)
    beasts.append(beast_concat)
    for fam in range(famnum):
        famid = "GF{:0>8}".format(fam+1)
        cds_aln_rn = cds_alns_rn[famid]
        pro_aln_rn = pro_alns_rn[famid]
        calnf = calnfs[fam]
        beast_i = beast(calnf, cds_aln_rn, pro_aln_rn, tmpdir, outdir, speciestree, datingset, slist, fossil, chainset, rootheight)
        beasts.append(beast_i)
    beast_i.run_beast(beastlgjar,beagle)
    Parallel(n_jobs=nthreads)(delayed(i.run_beast)(beastlgjar,beagle) for i in beasts)

# Run MCMCtree
def Run_MCMCTREE(Concat_caln, Concat_paln, Concat_calnf, Concat_palnf, cds_alns_rn, pro_alns_rn, calnfs, palnfs, tmpdir, outdir, speciestree, datingset, aamodel, partition, slist, nthreads):
    CI_table = {}
    PM_table = {}
    wgd_mrca = [sp for sp in slist if sp[-4:] == '_ap1' or sp[-4:] == '_ap2']
    Concat_calnf_paml = fasta2paml(Concat_caln,Concat_calnf)
    Concat_palnf_paml = fasta2paml(Concat_paln,Concat_palnf)
    McMctrees = []
    if partition:
        #logging.info("Running mcmctree on concatenated codon alignment with partition")
        Concatpospartitioned_paml = Getpartitionedpaml(Concat_calnf, outdir)
        McMctree = mcmctree(Concatpospartitioned_paml, Concat_palnf_paml, tmpdir, outdir, speciestree, datingset, aamodel, partition)
        McMctrees.append(McMctree)
        #McMctree.run_mcmctree(CI_table,PM_table,wgd_mrca)
    #logging.info("Running mcmctree on concatenated codon and peptide alignment without partition")
    if aamodel == 'wag':
        logging.info('Running mcmctree using Hessian matrix of WAG+Gamma for protein model')
    elif aamodel == 'lg':
        logging.info('Running mcmctree using Hessian matrix of LG+Gamma for protein model')
    elif aamodel == 'dayhoff':
        logging.info('Running mcmctree using Hessian matrix of Dayhoff-DCMut for protein model')
    else:
        logging.info('Running mcmctree using Poisson without gamma rates for protein model')
    McMctree = mcmctree(Concat_calnf_paml, Concat_palnf_paml, tmpdir, outdir, speciestree, datingset, aamodel, partition=False)
    McMctrees.append(McMctree)
    #McMctree.run_mcmctree(CI_table,PM_table,wgd_mrca)
    famnum = len(calnfs)
    calnfs_rn, palnfs_rn = FileRn(cds_alns_rn, pro_alns_rn, calnfs, palnfs)
    for fam in range(famnum):
        calnf_rn = calnfs_rn[fam]
        palnf_rn = palnfs_rn[fam]
        if partition:
            #logging.info("Running mcmctree on GF{:0>5} codon alignment with partition".format(fam+1))
            calnfpartitioned_paml = Getpartitionedpaml(calnf_rn, outdir)
            McMctree = mcmctree(calnfpartitioned_paml, palnf_rn, tmpdir, outdir, speciestree, datingset, aamodel, partition)
            McMctrees.append(McMctree)
            #McMctree.run_mcmctree(CI_table,PM_table,wgd_mrca)
        #logging.info("Running mcmctree on GF{:0>5} codon and peptide alignment without partition".format(fam+1))
        McMctree = mcmctree(calnf_rn, palnf_rn, tmpdir, outdir, speciestree, datingset, aamodel, partition=False)
        McMctrees.append(McMctree)
        #McMctree.run_mcmctree(CI_table,PM_table,wgd_mrca)
    Parallel(n_jobs=nthreads)(delayed(McMctree.run_mcmctree)(CI_table,PM_table,wgd_mrca) for McMctree in McMctrees)
    Getback_CIPM(outdir,CI_table,PM_table,wgd_mrca,calnfs_rn,Concat_calnf_paml)
    df_CI = pd.DataFrame.from_dict(CI_table,orient='index',columns=['CI_lower','CI_upper'])
    df_PM = pd.DataFrame.from_dict(PM_table,orient='index',columns=['PM'])
    fname_CI = os.path.join(outdir,'mcmctree','CI.tsv')
    fname_PM = os.path.join(outdir,'mcmctree','PM.tsv')
    df_CI.to_csv(fname_CI,header = True,index=True,sep='\t')
    df_PM.to_csv(fname_PM,header = True,index=True,sep='\t')
    #print(len(CI_table))
    #print(len(PM_table))
    #for items in CI_table.items():
    #    print(items)
#Run r8s
def Reroot(inputtree,outgroup):
    spt = inputtree + '.reroot'
    rr_cmd = ['nw_reroot', inputtree, outgroup]
    rr_out = sp.run(rr_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    logging.info("Rerooting the species tree with outgroup {}".format(outgroup))
    with open (spt,"w") as f: f.write(rr_out.stdout.decode('utf-8'))
    return spt

def Run_r8s(spt, nsites, outdir, datingset):
    treecontent = ""
    with open(spt,"r") as f:
        lines = f.readlines()
        for line in lines:
            treecontent = line.strip('\n').strip('\t').strip(' ')
    prefix = os.path.basename(spt)
    r8s_inf = os.path.join(outdir, prefix + "_r8s_in.txt")
    config = {'#nexus':'','begin trees;':'','tree inputtree = ':'{}'.format(treecontent),'end;':'','begin r8s;':'','blformat ':'lengths=persite nsites={} ultrametric=no round =yes;'.format(nsites),'MRCA':[],'fixage':[],'constrain':[],'set smoothing=':'100;', 'divtime':' method=PL crossv=yes cvstart=0 cvinc=1 cvnum=4;', 'divtime ':'method=PL algorithm=TN;', 'showage;':'', 'describe ':'plot=cladogram;', 'describe':' plot=chrono_description;', 'end;':''}
    for i in datingset:
        i.strip('\t').strip('\n').strip(' ')
        if 'MRCA' in i:
            i = i.strip('MRCA')
            config['MRCA'].append(i)
        if 'fixage' in i:
            i = i.strip('fixage')
            config['fixage'].append(i)
        if 'constrain' in i:
            i = i.strip('constrain')
            config['constrain'].append(i)
        if 'smoothing' in i:
            i = i.replace('set','').replace('smoothing=','').replace(' ','')
            config['set smoothing='] = i
        if 'divtime' in i:
            i = i.strip('divtime').strip(' ')
            config['divtime '] = i
    if len(config['MRCA']) == 0 or len(config['fixage']) + len(config['constrain']) ==0:
        logging.error("Please provide at lease one fixage or constrain information for an interal node in r8s dating")
        exit(0)
    with open(r8s_inf,"w") as f:
        for (k,v) in config.items():
            if type(v) == list:
                if len(v):
                    for i in range(len(v)):
                        f.write('{}'.format(k))
                        f.write('{}'.format(v[i]))
                        f.write('\n')
            else:
                f.write('{0}{1}\n'.format(k,v))
        f.write('end;')
    r8s_outf = os.path.join(outdir, prefix + "_r8s_out.txt")
    #r8s_cmd = ['r8s', '-b', '-f {}'.format(r8s_inf), '> {}'.format(r8s_outf)]
    r8s_bashf = os.path.join(outdir, prefix + "_r8s_bash.txt")
    with open (r8s_bashf,"w") as f: f.write('r8s -b -f {0} > {1}'.format(r8s_inf,r8s_outf))
    r8s_cmd = ['sh', '{}'.format(r8s_bashf)]
    r8s_out = sp.run(r8s_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    #with open (r8s_outf,"w") as f: f.write(r8s_out.stdout.decode('utf-8'))

def pfam_annot(cmd,pfam):
    if pfam == "denovo":
        cmd.append('--pfam_realign')
        cmd.append('denovo')
    if pfam == "realign":
        cmd.append('--pfam_realign')
        cmd.append('realign')
    return cmd

def dmnb_annot(cmd,dmnb):
    if not dmnb is None:
        cmd.append('--dmnd_db')
        cmd.append(os.path.abspath(dmnb))
    return cmd

def eggnog(cds_fastaf,eggnogdata,outdir,pfam,dmnb,evalue,nthreads):
    parent = os.getcwd()
    data_fir = os.path.abspath(eggnogdata)
    os.chdir(outdir)
    annotdir = _mkdir('egg_annotation')
    cmds = []
    for i, cds_fasta in enumerate(cds_fastaf):
        famid = "GF{:0>8}".format(i+1)
        famid = os.path.join(annotdir,famid)
        cmd = ['emapper.py', '-m', 'diamond', '--itype', 'CDS', '--evalue', '{}'.format(evalue), '-i', os.path.basename(cds_fasta), '-o', famid, '--data_dir', data_fir]
        cmd = pfam_annot(cmd,pfam)
        cmd = dmnb_annot(cmd,dmnb)
        cmds.append(cmd)
    Parallel(n_jobs=nthreads)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    #out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    os.chdir(parent)

def hmmer_pfam(cds_fastaf,hmm,outdir,evalue,nthreads):
    parent = os.getcwd()
    hmmdb_dir = os.path.abspath(hmm)
    os.chdir(outdir)
    annotdir = _mkdir('hmmer_pfam_annotation')
    cmds = []
    for i, cds_fasta in enumerate(cds_fastaf):
        famid = "GF{:0>8}".format(i+1)
        famid = os.path.join(annotdir,famid) 
        cmd = ['hmmscan','-o', '{}.txt'.format(famid), '--tblout', '{}.tbl'.format(famid), '--domtblout', '{}.dom'.format(famid), '--pfamtblout', '{}.pfam'.format(famid), '--noali', '-E', '{}'.format(evalue), hmmdb_dir, os.path.basename(cds_fasta)]
        cmds.append(cmd)
        #out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    Parallel(n_jobs=nthreads)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    os.chdir(parent)

def cpgf_interproscan(cds_fastaf,exepath):
    for cds_fasta in cds_fastaf:
        cmd = ['cp',cds_fasta,exepath]
        sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def mvgfback_interproscan(cds_fastaf,out_path):
    for fname in cds_fastaf:
        cmd = ['mv',fname + '.tsv',out_path]
        sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
        cmd = ['rm',fname]
        sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def interproscan(cds_fastaf,exepath,outdir,nthreads):
    cpgf_interproscan(cds_fastaf,exepath)
    parent = os.getcwd()
    os.chdir(outdir)
    annotdir = _mkdir('interproscan_annotation')
    out_path = os.path.join(parent,outdir,annotdir)
    os.chdir(exepath)
    cmds = []
    for i, cds_fasta in enumerate(cds_fastaf):
        famid = "GF{:0>8}".format(i+1)
        cmd = ['./interproscan.sh', '-i', os.path.basename(cds_fasta), '-f', 'tsv', '-dp']
        cmds.append(cmd)
    Parallel(n_jobs=nthreads)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    mvgfback_interproscan(cds_fastaf,out_path)
    os.chdir(parent)

def endt(tmpdir,start,s):
    if tmpdir is None: [x.remove_tmp(prompt=False) for x in s]
    end = timer()
    logging.info("Total run time: {}s".format(int(end-start)))
    logging.info("Done")
    exit()

def concathmm(outdir,df):
    hmmconcatf = os.path.join(outdir,'Full.hmm')
    gids = map(lambda i:os.path.join(outdir,i+'.pep.hmm'),df.index)
    cmd = ['cat'] + [i for i in gids]
    out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    with open(hmmconcatf,'w') as f: f.write(out.stdout.decode('utf-8'))
    return hmmconcatf

def run_hmmerbc(ids,fc,fp,s):
    for i in ids:
        with open(fc,'a') as f: f.write('>{}\n{}\n'.format(i,s.cds_sequence[s.idmap[i]]))
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(i,s.pro_sequence[s.idmap[i]]))
    fpaln,o,fcaln,fhmm = fp + '.aln','--auto',fc + '.aln',fc + '.hmm'
    mafft_cmd(fp,o,fpaln)
    backtrans(fpaln,fcaln,s.idmap,s.cds_sequence)
    cmd = ['hmmbuild'] + [fhmm] + [fcaln]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def run_hmmerbp(ids,fp,s):
    for i in ids:
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(i,s.pro_sequence[s.idmap[i]]))
    fpaln,o,fhmm = fp + '.aln','--auto',fp + '.hmm'
    mafft_cmd(fp,o,fpaln)
    cmd = ['hmmbuild'] + [fhmm] + [fpaln]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def hmmerbuild(df,s,outdir,nthreads):
    yids = lambda i: ', '.join(list(df.loc[i,:].dropna())).split(', ')
    yfnc = lambda i: os.path.join(outdir,'{}.cds'.format(i))
    yfnp = lambda i: os.path.join(outdir,'{}.pep'.format(i))
    #Parallel(n_jobs=nthreads)(delayed(run_hmmerb)(yids(i),yfnc(i),yfnp(i),s) for i in df.index)
    Parallel(n_jobs=nthreads)(delayed(run_hmmerbp)(yids(i),yfnp(i),s) for i in df.index)

def hmmerscan(outdir,querys,hmmf,eval,nthreads):
    cmd = ['hmmpress'] + [hmmf]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    cmds = []
    outs = []
    yprefix = lambda i: os.path.join(outdir,os.path.basename(i).strip('.pep'))
    for s in querys:
        s.orig_profasta()
        pf = yprefix(s.orig_pro_fasta)
        cmd = ['hmmscan','-o', '{}.txt'.format(pf), '--tblout', '{}.tbl'.format(pf), '--domtblout', '{}.dom'.format(pf), '--pfamtblout', '{}.pfam'.format(pf), '--noali', '-E', '{}'.format(eval), hmmf, s.orig_pro_fasta]
        cmds.append(cmd)
        outs.append('{}.tbl'.format(pf))
    Parallel(n_jobs=nthreads)(delayed(sp.run)(cmd,stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    return outs

def modifydf(df,outs,outdir,fam2assign):
    fname = os.path.join(outdir,os.path.basename(fam2assign)+'.assigned')
    yb = lambda i:os.path.basename(i).strip('.tbl')
    outdict = {yb(i):{} for i in outs}
    for out in outs:
        dfo = pd.read_csv(out,header = None, index_col=False,sep ='\t')
        end = dfo.shape[0] - 10
        for i in range(3,end):
            pair = dfo.iloc[i,0].split()
            f = pair[0][:-4]
            g = pair[2]
            if outdict[yb(out)].get(f) == None: outdict[yb(out)].update({f:g})
            else: outdict[yb(out)][f] = ', '.join([outdict[yb(out)][f],g])
    for k,v in outdict.items():
        yf,yg = (lambda v:(list(v.keys()),list(v.values())))(v)
        df.insert(0, k, pd.Series(yg, index=yf))
    df.to_csv(fname,header = True, index = True,sep = '\t')
    return df

def getassignfasta(df,s,querys,outdir):
    yids = lambda i: ', '.join(list(df.loc[i,:].dropna())).split(', ')
    for i in querys: s.merge_seq(i)
    p = _mkdir(os.path.join(outdir,'Orthologues_Sequence_Assigned'))
    pc = _mkdir(os.path.join(p,'cds'))
    pp = _mkdir(os.path.join(p,'pep'))
    for i in df.index:
        fc = os.path.join(pc,i+'.cds')
        fp = os.path.join(pp,i+'.pep')
        for gi in yids(i):
            with open(fc,'a') as f: f.write('>{}\n{}\n'.format(gi,s.cds_sequence[s.idmap[gi]]))
            with open(fp,'a') as f: f.write('>{}\n{}\n'.format(gi,s.pro_sequence[s.idmap[gi]]))

def hmmer4g2f(outdir,s,nthreads,querys,df,eval,fam2assign):
    hmmerbuild(df,s,outdir,nthreads)
    hmmf = concathmm(outdir,df)
    outs = hmmerscan(outdir,querys,hmmf,eval,nthreads)
    df = modifydf(df,outs,outdir,fam2assign)
    getassignfasta(df,s,querys,outdir)

def rmtmp(tmpdir,outdir,querys):
    if tmpdir == None:
        [x.remove_tmp(prompt=False) for x in querys]
        bf = os.path.join(outdir,'rm.sh')
        with open(bf,'w') as f: f.write('rm *.hmm *.dom *.pfam *.pep *.aln *.txt *.tbl Full.hmm*')
        cwd = os.getcwd()
        os.chdir(outdir)
        sp.run(['sh','rm.sh'],stdout=sp.PIPE,stderr=sp.PIPE)
        sp.run(['rm','rm.sh'],stdout=sp.PIPE,stderr=sp.PIPE)
        os.chdir(cwd)
        
def dmd4g2f(outdir,s,nthreads,querys,df):
    print('dmd4')

def genes2fams(assign_method,seq2assign,fam2assign,outdir,s,nthreads,tmpdir,to_stop,cds,cscore,eval,start):
    logging.info("Assigning sequences into given gene families")
    seqs_query = [SequenceData(s, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore) for s in seq2assign]
    df = pd.read_csv(fam2assign,header=0,index_col=0,sep='\t')
    for i in range(1, len(s)): s[0].merge_seq(s[i])
    if assign_method == 'hmmer': hmmer4g2f(outdir,s[0],nthreads,seqs_query,df,eval,fam2assign)
    else: dmd4g2f(outdir,s[0],nthreads,seqs_query,df)
    rmtmp(tmpdir,outdir,seqs_query)
    endt(tmpdir,start,s)

def run_or(i,j,s,eval,orthoinfer):
    s[i].run_diamond(s[j], orthoinfer, eval=eval)

def back_dmdhits(i,j,s,eval):
    ftmp = os.path.join(s[i].tmp_path,'_'.join([s[i].prefix,s[j].prefix])+'.tsv')
    df = pd.read_csv(ftmp, sep="\t", header=None)
    df = df.loc[df[0] != df[1]]
    s[i].dmd_hits[s[j].prefix] = df = df.loc[df[10] <= eval]

def ortho_infer_mul(s,nthreads,eval,inflation,orthoinfer):
    for i in range(len(s)):
        Parallel(n_jobs=nthreads)(delayed(run_or)(i,j,s,eval,orthoinfer) for j in range(i, len(s)))
        for j in range(i, len(s)): back_dmdhits(i,j,s,eval)
        s[i].rndmd_hit()
    for i in range(1, len(s)):
        s[0].merge_dmd_hits(s[i])
        s[0].merge_seq(s[i])
    s[0].get_para_skip_dmd(inflation=inflation, eval=eval)
    prefix = s[0].prefix
    s[0].prefix = 'Orthologues'
    txtf = s[0].write_paranome(True)
    s[0].prefix = prefix
    return s[0],txtf

def concatcdss(sequences,outdir):
    Concat_cdsf = os.path.join(outdir,'Orthologues')
    cmd = ['cat'] + [s for s in sequences]
    out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    with open(Concat_cdsf,'w') as f: f.write(out.stdout.decode('utf-8'))
    return Concat_cdsf

def ortho_infer(s,outdir,tmpdir,to_stop,cds,cscore,inflation,eval,nthreads,getsog,tree_method,treeset,msogcut):
    #Concat_cdsf = concatcdss(sequences,outdir)
    #ss = SequenceData(Concat_cdsf, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore)
    ss,txtf = ortho_infer_mul(s,nthreads,eval,inflation,False)
    #logging.info("tmpdir = {} for {}".format(ss.tmp_path,ss.prefix))
    #ss.get_paranome(inflation=inflation, eval=eval)
    #txtf = ss.write_paranome(True)
    sgmaps = {}
    slist = []
    for seq in s: sgmaps.update(seq.spgenemap())
    for seq in s: slist.append(seq.prefix)
    txt2tsv(txtf,outdir,sgmaps,slist,ss,nthreads,getsog,tree_method,treeset,msogcut)
    #if tmpdir is None: ss.remove_tmp(prompt=False)
    #sp.run(['rm'] + [Concat_cdsf], stdout=sp.PIPE,stderr=sp.PIPE)
    return txtf

def writeogsep(table,seq,fc,fp):
    for v in table.values():
        if type(v) == list: v = v[0]
        if v == '': continue
        cds = seq.cds_sequence[seq.idmap[v]]
        pro = seq.pro_sequence[seq.idmap[v]]
        with open(fc,'a') as f: f.write('>{}\n{}\n'.format(v,cds))
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(v,pro))

def getnestedfasta(fnest,df,ss,nfs_count):
    fc_nest = _mkdir(os.path.join(fnest,'cds'))
    fp_nest = _mkdir(os.path.join(fnest,'pep'))
    ndc = copy.deepcopy(nfs_count)
    for j,rn in enumerate(df.index):
        if nfs_count[rn] == 1:
            fcname = os.path.join(fc_nest,'{}.cds'.format(rn))
            fpname = os.path.join(fp_nest,'{}.pep'.format(rn))
            with open(fcname,'w') as f:
                for i in df.iloc[j,:][:-1]: f.write('>{}\n{}\n'.format(i,ss.cds_sequence[ss.idmap[i]]))
            with open(fpname,'w') as f:
                for i in df.iloc[j,:][:-1]: f.write('>{}\n{}\n'.format(i,ss.pro_sequence[ss.idmap[i]]))
        else:
            t = ndc[rn]
            fcname = os.path.join(fc_nest,'{0}_{1}.cds'.format(rn,t))
            fpname = os.path.join(fp_nest,'{0}_{1}.pep'.format(rn,t))
            with open(fcname,'w') as f:
                for i in df.iloc[j,:][:-1]: f.write('>{}\n{}\n'.format(i,ss.cds_sequence[ss.idmap[i]]))
            with open(fpname,'w') as f:
                for i in df.iloc[j,:][:-1]: f.write('>{}\n{}\n'.format(i,ss.pro_sequence[ss.idmap[i]]))
            ndc[rn] = ndc[rn] - 1

def filternested(sps,msogcut):
    counts_table = {i:sps.count(i) for i in set(sps)}
    return len([i for i in counts_table.values() if i==1])/len(set(sps)) >= msogcut

def getunique(ids,sps,idmap,pros):
    d = {}
    leng = lambda n: len(pros[idmap[n]])
    for i,s in zip(ids,sps):
        if d.get(s) == None: d[s] = i
        elif leng(i) > leng(d[s]): d[s] = i
    d.update({'NestedType':'loose'})
    return d

def label2nest(tree,slist,sgmaps,ss,msogcut):
    dics = []
    treecopy = copy.deepcopy(tree)
    treecopy.root_at_midpoint()
    for i,clade in enumerate(treecopy.get_nonterminals()): clade.name = str(i)
    for clade in treecopy.get_nonterminals():
        if clade.count_terminals() == len(slist):
            cladec = copy.deepcopy(clade)
            cladec.collapse_all()
            ids = [i.name for i in cladec.clades]
            sps = list(map(lambda n: sgmaps[n],ids))
            if set(sps) == set(slist):
                dic = {j:i for i,j in zip(ids,sps)}
                dic.update({'NestedType':'strict'})
                dics.append(dic)
        elif clade.count_terminals() > len(slist):
            cladec = copy.deepcopy(clade)
            cladec.collapse_all()
            ids = [i.name for i in cladec.clades]
            sps = list(map(lambda n: sgmaps[n],ids))
            if set(sps) == set(slist) and filternested(sps,msogcut):
                dic = getunique(ids,sps,ss.idmap,ss.pro_sequence)
                dics.append(dic)
    return dics

def getnestedog(fp,fc,slist,i,outd,tree_method,tree_famsf,tree_fams,sgmaps,nested_dfs,ss,msogcut):
    x = lambda i : "GF{:0>8}".format(i+1)
    fpaln,fcaln = fp + '.aln',fc + '.aln'
    if tree_method == 'fasttree': addiqfatree(x(i),tree_fams,fcaln,tree_famsf,postfix = '.fasttree')
    if tree_method == 'iqtree': addiqfatree(x(i),tree_fams,fcaln,tree_famsf,postfix = '.treefile')
    if tree_method == 'mrbayes': addmbtree(outd,tree_fams,tree_famsf,i=i,concat=False)
    dics = label2nest(tree_fams[x(i)],slist,sgmaps,ss,msogcut)
    if dics:
        for dic in dics:
            dic.update({'NestedSOG':x(i)})
            df = pd.DataFrame.from_dict([dic])
            nested_dfs.append(df)

def aln2tree_sc(fp,fc,ss,tree_method,treeset,outd,i):
    x = lambda i : "GF{:0>8}".format(i+1)
    fpaln,o,fcaln = fp + '.aln','--auto',fc + '.aln'
    mafft_cmd(fp,o,fpaln)
    pro_aln = backtrans(fpaln,fcaln,ss.idmap,ss.cds_sequence)
    if tree_method == "iqtree": iqtree_run(treeset,fcaln)
    if tree_method == "fasttree": fasttree_run(fcaln,treeset)
    if tree_method == "mrbayes": mrbayes_run(outd,x(i),fpaln,pro_aln,treeset)

def sgratio(l):
    t = [i for i in l if i]
    ratio = len(t)/len(l)
    return ratio

def sgdict(gsmap,slist,fams_df,counts_df,reps_df,ss,ftmp,frep,fsog,i,getsog,msogcut):
    fam_table,represent_seqs = {},{}
    count_table = {s:0 for s in slist}
    sumcount = 0
    ct = 'multi-copy'
    exist_sp = set(gsmap.values())
    coverage = len(exist_sp)/len(slist)
    fc = os.path.join(_mkdir(os.path.join(ftmp,'cds')),"GF{:0>8}.cds".format(i+1))
    fp = os.path.join(_mkdir(os.path.join(ftmp,'pep')),"GF{:0>8}.pep".format(i+1))
    fc_rep = os.path.join(_mkdir(os.path.join(frep,'cds')),"GF{:0>8}.cds".format(i+1))
    fp_rep = os.path.join(_mkdir(os.path.join(frep,'pep')),"GF{:0>8}.pep".format(i+1))
    fc_sog = os.path.join(_mkdir(os.path.join(fsog,'cds')),"GF{:0>8}.cds".format(i+1))
    fp_sog = os.path.join(_mkdir(os.path.join(fsog,'pep')),"GF{:0>8}.pep".format(i+1))
    for k,v in gsmap.items():
        cds = ss.cds_sequence[ss.idmap[k]]
        pro = ss.pro_sequence[ss.idmap[k]]
        if fam_table.get(v) == None:
            fam_table[v] = k
            represent_seqs[v] = k
        else:
            fam_table[v] = ", ".join([fam_table[v],k])
            if len(pro) > len(ss.pro_sequence[ss.idmap[represent_seqs[v]]]): represent_seqs[v] = k
        count_table[v] = count_table[v] + 1
        sumcount = sumcount + 1
        with open(fc,'a') as f: f.write('>{}\n{}\n'.format(k,cds))
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(k,pro))
    writeogsep(represent_seqs,ss,fc_rep,fp_rep)
    for ms in set(slist) - set(gsmap.values()): fam_table[ms],represent_seqs[ms] = '',''
    li = [v == 1 for v in count_table.values()]
    if all(li):
        writeogsep(fam_table,ss,fc_sog,fp_sog)
        ct = 'single-copy'
    elif sgratio(li) >= msogcut: ct = 'mostly single-copy'
    fam_df = pd.DataFrame.from_dict([fam_table])
    count_table.update({'Sum':sumcount,'PhylogenyCoverage':coverage,'CopyType':ct})
    count_df = pd.DataFrame.from_dict([count_table])
    rep_df = pd.DataFrame.from_dict([represent_seqs])
    fams_df.append(fam_df)
    counts_df.append(count_df)
    reps_df.append(rep_df)

def seqdict(gsmap,ss,ftmpc,ftmpp,i):
    fc = os.path.join(ftmpc,"GF{:0>8}.cds".format(i+1))
    fp = os.path.join(ftmpp,"GF{:0>8}.pep".format(i+1))
    for k,v in gsmap.items():
        cds = ss.cds_sequence[ss.idmap[k]]
        pro = ss.pro_sequence[ss.idmap[k]]
        with open(fc,'a') as f: f.write('>{}\n{}\n'.format(k,cds))
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(k,pro))

def countdict(gsmap,slist,ss,counts_df):
    count_table = {s:0 for s in slist}
    sumcount = 0
    for k,v in gsmap.items():
        count_table[v] = count_table[v] + 1
        sumcount = sumcount + 1
    count_table.update({'Sum':sumcount})
    count_df = pd.DataFrame.from_dict([count_table])
    counts_df.append(count_df)

def txt2tsv(txtf,outdir,sgmaps,slist,ss,nthreads,getsog,tree_method,treeset,msogcut):
    fname_fam = os.path.join(outdir,'Orthogroups.sp.tsv')
    fname_count = os.path.join(outdir,'Orthogroups.genecount.tsv')
    fname_rep = os.path.join(outdir,'Orthogroups.representives.tsv')
    fname_nest = os.path.join(outdir,'Orthogroups.nested_single_copy.tsv')
    ftmp = _mkdir(os.path.join(outdir,'Orthologues_Sequence'))
    frep = _mkdir(os.path.join(outdir,'Orthologues_Sequence_Representives'))
    fsog = _mkdir(os.path.join(outdir,'Orthologues_Single_Copy'))
    fnest = _mkdir(os.path.join(outdir,'Orthologues_Nested_Single_Copy'))
    txt = pd.read_csv(txtf,header = None,index_col=0,sep='\t')
    y= lambda x: {j:sgmaps[j] for j in x}
    fams_df,counts_df,reps_df = [],[],[]
    for i in range(txt.shape[0]):
        sgdict(y(txt.iloc[i,0].split(', ')),slist,fams_df,counts_df,reps_df,ss,ftmp,frep,fsog,i,getsog,msogcut)
    if getsog:
        tree_famsf,tree_fams,nested_dfs,aln_fam_is = [],{},[],[]
        yc = lambda x: os.path.join(ftmp,'cds',"GF{:0>8}.cds".format(x+1))
        yp = lambda x: os.path.join(ftmp,'pep',"GF{:0>8}.pep".format(x+1))
        yco = lambda x,y: counts_df[x].loc[0,y]
        outd = os.path.join(ftmp,'pep')
        for i in range(txt.shape[0]):
            li = [yco(i,s) for s in slist]
            if all([j > 0 for j in li]) and sum(li) > len(slist): aln_fam_is.append(i)
        Parallel(n_jobs=nthreads)(delayed(aln2tree_sc)(yp(i),yc(i),ss,tree_method,treeset,outd,i) for i in aln_fam_is)
        for i in aln_fam_is: getnestedog(yp(i),yc(i),slist,i,outd,tree_method,tree_famsf,tree_fams,sgmaps,nested_dfs,ss,msogcut)
        if nested_dfs:
            nested_coc = pd.concat(nested_dfs,ignore_index=True).set_index('NestedSOG')
            nested_coc.to_csv(fname_nest,header = True,index =True,sep = '\t')
            logging.info("{} nested single-copy families delineated".format(nested_coc.shape[0]))
            nfs = list(nested_coc.index)
            nfs_count = {i:nfs.count(i) for i in set(nfs)}
            getnestedfasta(fnest,nested_coc,ss,nfs_count)
        else: logging.info("No nested single-copy families delineated")
    #Parallel(n_jobs=nthreads)(delayed(sgdict)(y(txt.iloc[i,0].split(', ')),slist,fams_df,counts_df,ss,ftmpc,ftmpp,i) for i in range(txt.shape[0]))
    #for i in range(txt.shape[0]): sgdict(y(txt.iloc[i,0].split(', ')),slist,fams_df,counts_df)
    #Parallel(n_jobs=nthreads)(delayed(seqdict)(y(txt.iloc[i,0].split(', ')),ss,ftmpc,ftmpp,i) for i in range(txt.shape[0]))
    fams_coc = pd.concat(fams_df,ignore_index=True)
    counts_coc = pd.concat(counts_df,ignore_index=True)
    reps_coc = pd.concat(reps_df,ignore_index=True)
    sogs_coc = pd.concat(fams_df,ignore_index=True)
    _label_families(fams_coc)
    _label_families(counts_coc)
    _label_families(reps_coc)
    fams_coc.to_csv(fname_fam,header = True,index =True,sep = '\t')
    counts_coc.to_csv(fname_count,header = True,index =True,sep = '\t')
    reps_coc.to_csv(fname_rep,header = True,index =True,sep = '\t')
    logging.info("In total {} orthologous families delineated".format(counts_coc.shape[0]))
    mu,mo,sg=(counts_coc[counts_coc['CopyType']==i].shape[0] for i in ['multi-copy','mostly single-copy','single-copy'])
    logging.info("with {0} multi-copy, {1} mostly single-copy, {2} single-copy".format(mu,mo,sg))

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
        self.out = os.path.join(self.tmp_path, "{}_ks.csv".format(gfid))

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
                info = Aligninfo(self.cds_aln)
                d[pair] = {"node": node.name, "family": self.id}
                d[pair].update(info)
        df = pd.DataFrame.from_dict(d, orient="index")
        self.codeml_results = self.codeml_results.join(df)

def get_outlierexcluded(df,cutoff = 5):
    df = df[df['dS']<=cutoff]
    weight_exc = 1/df.groupby(['family', 'node'])['dS'].transform('count')
    weight_exc = weight_exc.to_frame(name='weightoutlierexcluded')
    return weight_exc

def get_outlierincluded(df):
    weight_inc = 1/df.groupby(['family', 'node'])['dS'].transform('count')
    weight_inc = weight_inc.to_frame(name='weightoutlierincluded')
    return weight_inc

def get_nodeaverged_dS_outlierincluded(df):
    node_averaged_dS_inc = df.groupby(["family", "node"])["dS"].mean()
    node_averaged_dS_inc = node_averaged_dS_inc.to_frame(name='node_averaged_dS_outlierincluded')
    return node_averaged_dS_inc

def get_nodeaverged_dS_outlierexcluded(df,cutoff = 5):
    df = df[df['dS']<=cutoff]
    node_averaged_dS_exc = df.groupby(["family", "node"])["dS"].mean()
    node_averaged_dS_exc = node_averaged_dS_exc.to_frame(name='node_averaged_dS_outlierexcluded')
    return node_averaged_dS_exc

def _get_ks(family):
    family.get_ks()
    if family.codeml_results.shape[1] !=3:
        weight_inc = get_outlierincluded(family.codeml_results)
        weight_exc = get_outlierexcluded(family.codeml_results,cutoff = 5)
        node_averaged_dS_inc = get_nodeaverged_dS_outlierincluded(family.codeml_results)
        node_averaged_dS_exc = get_nodeaverged_dS_outlierexcluded(family.codeml_results,cutoff = 5)
        family.codeml_results = family.codeml_results.join(weight_inc)
        family.codeml_results = family.codeml_results.join(weight_exc)
        family.codeml_results = family.codeml_results.merge(node_averaged_dS_inc,on = ['family', 'node'])
        family.codeml_results = family.codeml_results.merge(node_averaged_dS_exc,on = ['family', 'node'],how = 'left')
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

