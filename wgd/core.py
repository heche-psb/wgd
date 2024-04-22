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
from concurrent.futures import ProcessPoolExecutor
from wgd.codeml import Codeml
from wgd.cluster import cluster_ks #fastcluster is not compatible with some versions of numpy which causes problems for python3.8 and higher
from wgd.mcmctree import mcmctree
from wgd.beast import beast
from timeit import default_timer as timer
import copy
import psutil
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from tqdm import trange
# Reconsider the renaming, more a pain than helpful?

# helper functions
def memory_reporter_initial():
    logging.info("Checking cores and threads...")
    logging.info("The number of logical CPUs/Hyper Threading in the system: {}".format(int(psutil.cpu_count())))
    logging.info("The number of physical cores in the system: {}".format(int(psutil.cpu_count(logical=False))))
    logging.info("The number of actually usable CPUs in the system: {}".format(len(psutil.Process().cpu_affinity())))
    d = psutil.virtual_memory()
    logging.info("Checking memory...")
    logging.info("Total physical memory: {:.4f} GB".format(d.total/(1024 ** 3)))
    logging.info("Available memory: {:.4f} GB".format(d.available/(1024 ** 3)))
    logging.info("Free memory: {:.4f} GB".format(d.free/(1024 ** 3)))

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

def demension(l):
    x = []
    for i in l: x+= i.split()
    return x

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

def _process_unrooted_tree(treefile, gfid, fformat="newick"):
    tree = Phylo.read(treefile, fformat)
    try:
        tree.root_at_midpoint()
    except UnboundLocalError as e:
        logging.warning("Branch length all zero for family {}".format(gfid))
    _label_internals(tree)
    return tree

def linearfit(df):
    bit_score = np.array(df[11])
    gene_length = np.array(df[12])

def fit_linregress(df):
    cutoff = np.percentile(df[11], 95)
    #here I used top 10% bit-score for fitting
    filtered_bsb = df[df[11]>=cutoff]
    #filtered_bsb = df
    slope, intercept, r, p, se = stats.linregress(np.log10(filtered_bsb[12]), np.log10(filtered_bsb[11]))
    normalized = [j/(pow(10, intercept)*(l**slope)) for j,l in zip(df[11],df[12])]
    #for j,l in zip(df[11],df[12]): print(j/(pow(10, intercept)*(l**slope)))
    return normalized

def fit_linear(df,orig_df):
    slope, intercept, r, p, se = stats.linregress(np.log10(df[12]), np.log10(df[11]))
    normalized = [j/(pow(10, intercept)*(l**slope)) for j,l in zip(orig_df[11],orig_df[12])]
    return normalized

def genelengthpercentile5(df,hitper = 5):
    cut = int(100-hitper)
    cutoff = np.percentile(df[11], cut)
    df = df[df[11]>=cutoff]
    return df

def checkdupnamessp1sp2(dic1,dic2):
    l_or,l_sa = len(dic1),len(dic2)
    if len({v:k for k,v in {**dic1,**dic2}.items()}) == l_or + l_sa:
        return False
    else:
        return True

def handledupnamessp1sp2(dic1,dic2):
    """
    Deal with identical gene ids occurred in two cds files
    By creating a new dic for one of the cds file
    """
    new_keys_dic1,new_keys_dic2 = {k:k+'_sp1' for k in dic1.keys()}, {k:k+'_sp2' for k in dic2.keys()}
    new_keys_dic_re = {**{v:k for k,v in new_keys_dic1.items()}, **{v:k for k,v in new_keys_dic2.items()}}
    new_dic1,new_dic2 = {new_keys_dic1[k]:v for k,v in dic1.items()}, {new_keys_dic2[k]:v for k,v in dic2.items()}
    return new_keys_dic_re,new_dic1,new_dic2

def normalizebitscore(gene_length,df,outpath,sgidmaps=None,idmap=None,seqmap=None,hicluster=False,nonbins = False,allbins = True,bins = 100, hitper = 5):
    y = lambda x : gene_length[x[0]] * gene_length[x[1]]
    df[12] = [y(df.loc[i,0:1]) for i in df.index]
    df = df.sort_values(12,ascending=False).reset_index(drop=True)
    if hicluster:
        X = [[i] for i in df[12]]
        df[13] = AgglomerativeClustering(n_clusters=bins).fit(X).labels_
        index = []
        normalized_df = []
        for i in range(bins):
            df_w = df[df[13]==i].loc[:,11:12]
            normalized = fit_linregress(df_w)
            index = index + list(df_w.index)
            normalized_df = normalized_df + normalized
        df_index_norm = pd.DataFrame(data=normalized_df,index=index,columns=[14])
        df = df.join(df_index_norm)
        df = df.drop(columns=[13]).rename(columns={14:13})
        df.to_csv(outpath,sep="\t", header=False, index=False)
    elif nonbins:
        df[13] = fit_linregress(df.loc[:,11:12])
    else:
        if len(df)%bins == 0: bin_size = len(df)/bins
        elif len(df) < 100:
            logging.info("The number of hits is less than 100, will aggregate all the hits in one bin")
            bin_size,bins = len(df),1
        else: bin_size = (len(df)-(len(df)%bins))/bins
        if allbins:
            if sgidmaps is None:
                data_per_bin = []
                for i in range(bins):
                    if i != bins-1: bit_score_bins = df.loc[int((i*bin_size)):int(((i+1)*bin_size-1)),11:12].copy()
                    else: bit_score_bins = df.loc[int((i*bin_size)):,11:12].copy()
                    data_per_bin.append(genelengthpercentile5(bit_score_bins,hitper = hitper))
                merged_data = pd.concat(data_per_bin)
                df.loc[:,13] = fit_linear(merged_data,df)
                if checkdupnamessp1sp2(idmap,seqmap):
                    change_map,new_idmap,new_seqmap = handledupnamessp1sp2(idmap,seqmap)
                    combinedidmaps = {v:change_map[k] for k,v in {**new_idmap,**new_seqmap}.items()}
                else: combinedidmaps = {v:k for k,v in {**idmap,**seqmap}.items()} # if the idmap and seqmap have the same gene name, errors might occur, for instance two identical protein input files
                df.loc[:,14] = df[0].apply(lambda x:combinedidmaps[x])
                df.loc[:,15] = df[1].apply(lambda x:combinedidmaps[x])
                df.to_csv(outpath+'_withoriglabel',sep="\t", header=False, index=False)
                df = df.drop(columns=[14,15])
            else:
                gids_records = {v:k for k,v in idmap.items()}
                df.loc[:,13] = df[0].apply(lambda x:sgidmaps[gids_records[x]])
                df.loc[:,14] = df[1].apply(lambda x:sgidmaps[gids_records[x]])
                #print(df[13])
                #print(df[14])
                species_list = list(set(sgidmaps.values()))
                dfs = []
                #print(species_list)
                for i in range(len(species_list)):
                    for j in range(int(i),len(species_list)):
                        m,n = species_list[int(i)],species_list[int(j)]
                        df_spair = df.loc[(df[13]==m) & (df[14]==n)].copy()
                        if len(df_spair)%bins == 0: bin_size = len(df_spair)/bins
                        else: bin_size = (len(df_spair)-(len(df_spair)%bins))/bins
                        data_per_bin = []
                        #a=list(df_spair.loc[:,13])
                        #b=list(df_spair.loc[:,14])
                        logging.info("Normalization between {0} and {1}".format(m,n))
                        #print(a[1])
                        #print(b[1])
                        df_spair = df_spair.reset_index(drop=True)
                        if bin_size == 0:
                            logging.info('number of hits are less than bins, will use one bin of all hits for normalization')
                            bit_score  = df_spair.loc[:,11:12].copy()
                            data_per_bin.append(genelengthpercentile5(bit_score,hitper = hitper))
                        #a=list(df_spair.loc[:,13])
                        #b=list(df_spair.loc[:,14])
                        #print(a[1])
                        #print(b[1])
                        #df_spair = df_spair.reset_index(drop=True)
                        else:
                            logging.info("binsize is {}".format(bin_size))
                            for k in range(bins):
                                if k != bins-1: bit_score_bins = df_spair.loc[int((k*bin_size)):int(((k+1)*bin_size-1)),11:12].copy()
                                else: bit_score_bins = df_spair.loc[int((k*bin_size)):,11:12].copy()
                                #print(bit_score_bins.shape)
                                data_per_bin.append(genelengthpercentile5(bit_score_bins,hitper = hitper))
                        merged_data = pd.concat(data_per_bin)
                        df_spair.loc[:,15] = fit_linear(merged_data,df_spair)
                        dfs.append(df_spair)
                        #print(df_spair.shape)
                df = pd.concat(dfs,ignore_index=True)
                df.loc[:,13] = df[0].apply(lambda x:gids_records[x])
                df.loc[:,14] = df[1].apply(lambda x:gids_records[x])
                df.to_csv(outpath+'.withorigid',sep="\t", header=False, index=False)
                df = df.drop(columns=[13,14]).rename(columns={15:13})
        else:
            normalized_bitscores = []
            for i in range(bins):
                if i != bins-1: bit_score_bins = df.loc[int((i*bin_size)):int(((i+1)*bin_size-1)),11:12]
                else: bit_score_bins = df.loc[int((i*bin_size)):,11:12]
            #normalized = fit_linregress(bit_score_bins)
        #cutoff = np.percentile(bit_score_bins[11], 95)
        #filtered_bsb = bit_score_bins[bit_score_bins[11]>=cutoff]
        #slope, intercept, r, p, se = stats.linregress(filtered_bsb[12], filtered_bsb[11])
        #normalized = [l**slope*j/pow(10, intercept) for j,l in zip(bit_score_bins[11],bit_score_bins[12])]
                normalized_bitscores = normalized_bitscores + fit_linregress(bit_score_bins)
    #if len(df)%bins != 0:
    #    print(50*bin_size)
    #    bit_score_bins = df.loc[int(50*bin_size):,11:12]
    #    normalized = fit_linregress(bit_score_bins)
    #    normalized_bitscores = normalized_bitscores + normalized
            df[13] = normalized_bitscores
    df.to_csv(outpath,sep="\t", header=False, index=False)
    return df

class SequenceData:
    """
    Sequence data container for Ks distribution computation pipeline. A helper
    class that bundles sequence manipulation methods.
    """
    def __init__(self, cds_fasta,
            tmp_path=None, out_path="wgd_dmd",
            to_stop=True, cds=True, cscore=None,threads = 4, bins = 100, normalizedpercent = 5, nonormalization=False, prot = False):
        if tmp_path == None:
            tmp_path = "wgdtmp_" + str(uuid.uuid4())
        else:
            tmp_path = tmp_path + "/"+ "wgdtmp_" + str(uuid.uuid4())
        self.np = normalizedpercent
        self.tmp_path  = _mkdir(tmp_path)
        self.out_path  = _mkdir(out_path)
        self.cds_fasta = cds_fasta
        self.prot_fasta = cds_fasta
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
        self.gene_length = {}
        self.idmap     = {}  # map from the new safe id to the input seq id
        self.prot = prot
        if prot: self.read_prot()
        else: self.read_cds(to_stop=to_stop, cds=cds)
        self.threads = threads
        self.bins = bins
        self.nonormalization = nonormalization
        _write_fasta(self.pro_fasta, self.pro_seqs)

    def read_prot(self):
        """
        Read a Protein file
        """
        self.idmapop = {'original_id':[],'safe_id':[]}
        for i, record in enumerate(SeqIO.parse(self.prot_fasta, 'fasta')):
            gid = "{0}_{1:0>5}".format(self.prefix, i)
            aa_sequence = record.seq
            self.pro_seqs[gid] = record
            self.pro_sequence[gid] = aa_sequence
            self.idmap[record.id] = gid
            self.idmapop['original_id'].append(record.id)
            self.idmapop['safe_id'].append(gid)
            self.gene_length[gid] = len(aa_sequence)
        df_idmap_tooutput = pd.DataFrame.from_dict(self.idmapop)
        df_idmap_tooutput.to_csv(os.path.join(self.tmp_path, self.prefix+'.original_safe_id'),header=True,index=False,sep='\t')
        return

    def read_cds(self, to_stop=True, cds=True):
        """
        Read a CDS fasta file. We give each input record a unique safe ID, and
        keep the full records in a dict with these IDs. We use the newly assigned
        IDs in further analyses, but can reconvert at any time.
        """
        self.idmapop = {'original_id':[],'safe_id':[]}
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
            self.idmapop['original_id'].append(record.id)
            self.idmapop['safe_id'].append(gid)
            self.gene_length[gid] = len(aa_sequence)
        df_idmap_tooutput = pd.DataFrame.from_dict(self.idmapop)
        df_idmap_tooutput.to_csv(os.path.join(self.tmp_path, self.prefix+'.original_safe_id'),header=True,index=False,sep='\t')
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
        if not self.prot:
            self.cds_seqs.update(other.cds_seqs)
            self.cds_sequence.update(other.cds_sequence)
        self.pro_sequence.update(other.pro_sequence)
        self.pro_seqs.update(other.pro_seqs)
        self.idmap.update(other.idmap)

    def merge_seqs(self,other):
        self.cds_sequence.update(other.cds_sequence)
        self.pro_sequence.update(other.pro_sequence)
        self.idmap.update(other.idmap)

    def merge_gene_length(self,other):
        self.gene_length.update(other.gene_length)

    def merge_dmd_hits(self,other):
        self.dmd_hits.update(other.dmd_hits)

    def make_diamond_db(self):
        if not os.path.isfile(self.pro_db + '.dmnd'):
            cmd = ["diamond", "makedb", "--in", self.pro_fasta, "-d", self.pro_db, "-p", str(self.threads)]
            out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            logging.debug(out.stderr.decode())
            if out.returncode == 1: logging.error(out.stderr.decode())

    def run_diamond(self, seqs, orthoinfer, eval=1e-10, savememory=False, sgidmaps=None):
        self.merge_gene_length(seqs)
        self.make_diamond_db()
        run = "_".join([self.prefix, seqs.prefix + ".tsv"])
        outfile = os.path.join(self.tmp_path, run)
        if not orthoinfer:
            cmd = ["diamond", "blastp", "-d", self.pro_db, "-q", seqs.pro_fasta, "-o", outfile, "-p", str(self.threads)]
            out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            logging.debug(out.stderr.decode())
        if savememory: df = pd.read_csv(outfile, sep="\t", header=None,usecols=[0,1,10,11])
        else: df = pd.read_csv(outfile, sep="\t", header=None)
        df = df.loc[df[0] != df[1]]
        df = df.loc[df[10] <= eval]
        outpath = os.path.join(self.tmp_path, '{0}_{1}_hits_gene_length_normalizedBitscore.tsv'.format(self.prefix,seqs.prefix))
        if not self.nonormalization:
            logging.info("Normalization between {} & {}".format(self.prefix,seqs.prefix))
            logging.info("{} bins & upper {}% hits in linear regression".format(self.bins,self.np))
            df = normalizebitscore(self.gene_length,df,outpath,sgidmaps=sgidmaps,idmap=self.idmap,seqmap=seqs.idmap,bins = self.bins,hitper = self.np).drop(columns=[11,12]).rename(columns={13:11})
        self.dmd_hits[seqs.prefix] = df
        return df

    def get_rbh_orthologs(self, seqs, cscore, orthoinfer, eval=1e-10):
        if self == seqs:
            raise ValueError("RBH orthologs only defined for distinct species")
        df = self.run_diamond(seqs, orthoinfer, eval=eval)
        if cscore == None:
            #df1 = df.sort_values(10).drop_duplicates([0])
            # originally sorted by e-value in ascending, but it flaws due to the e-value has a lower threshold below which it will become 0
            df1 = df.sort_values(11,ascending=False).drop_duplicates([0])
            # now sorted by normalized bit-score in descending which avoids the issue of e-value
            #df2 = df.sort_values(10).drop_duplicates([1])
            df2 = df.sort_values(11,ascending=False).drop_duplicates([1])
            self.rbh[seqs.prefix] = df1.merge(df2)
        else:
            cscore = float(cscore)
            #df_species1_best=df.sort_values(10).drop_duplicates([0])
            df_species1_best=df.sort_values(11,ascending=False).drop_duplicates([0])
            #df_species2_best=df.sort_values(10).drop_duplicates([1])
            df_species2_best=df.sort_values(11,ascending=False).drop_duplicates([1])
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
        #ysave = lambda i:i.iloc[:,[0,1,10]]
        df = pd.concat([v for v in self.dmd_hits.values()])
        df.to_csv(gf, sep="\t", header=False, index=False)
        gf = SequenceSimilarityGraph(gf)
        mcl_out = gf.run_mcl(inflation=inflation)
        with open(mcl_out, "r") as f:
            for i, line in enumerate(f.readlines()): self.mcl[i] = line.strip().split()

    def get_paranome(self, inflation=2.0, eval=1e-10, savememory=False, sgidmaps = None):
        df = self.run_diamond(self, False, eval=eval, savememory=savememory, sgidmaps=sgidmaps)
        gf = self.get_mcl_graph(self.prefix)
        mcl_out = gf.run_mcl(inflation=inflation)
        with open(mcl_out, "r") as f:
            for i, line in enumerate(f.readlines()): self.mcl[i] = line.strip().split()

    def get_mcl_graph(self, *args):
        # args are keys in `self.dmd_hits` to use for building MCL graph
        gf = os.path.join(self.tmp_path, "_".join([self.prefix] + list(args)))
        #ysave = lambda i:i.iloc[:,[0,1,10]]
        df = pd.concat([self.dmd_hits[x] for x in args])
        df = df[[0,1,11]]
        df.to_csv(gf, sep="\t", header=False, index=False)
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
                f.write(", ".join([self.pro_seqs[x].id for x in v]))
                f.write("\n")
        return fname

    def add_singletons_paranome(self):
        xs = set(itertools.chain.from_iterable(self.mcl.values()))  # all genes in families
        gs = set(self.pro_seqs.keys())  # all genes
        ys = gs - xs 
        i = max(self.mcl.keys()) + 1
        for j, y in enumerate(ys):
            self.mcl[i + j] = [y]

    def write_rbh_orthologs(self, seqs, singletons=True, ogformat=False):
        fname = "{}_{}.rbh.tsv".format(self.prefix, seqs.prefix)
        fname = os.path.join(self.out_path, fname)
        df = self.rbh[seqs.prefix]
        df[seqs.prefix] = df[0].apply(lambda x: seqs.pro_seqs[x].id)
        df[self.prefix] = df[1].apply(lambda x: self.pro_seqs[x].id)
        if singletons:  # this must be here, cannot be before renaming, not after labeling fams
            df = pd.concat([df, self.add_singletons_rbh(seqs)]) 
        #_label_families(df)
        if ogformat:
            _label_families(df)
            df.to_csv(fname, columns=[seqs.prefix, self.prefix], sep="\t",index=True)
        else: df.to_csv(fname, columns=[seqs.prefix, self.prefix], sep="\t",index=False)
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
    #Only the column 0,1,10 are in use
    def __init__(self, graph_file):
        self.graph_file = graph_file

    def run_mcl(self, inflation=1.5):
        g1 = self.graph_file
        g2 = g1 + ".tab"
        g3 = g1 + ".mci"
        g4 = g2 + ".I{}".format(inflation*10)
        outfile = g1 + ".mcl"
        #command = ['mcxload', '-abc', g1, '--stream-mirror', '--stream-neg-log10', '-o', g3, '-write-tab', g2]
        #here I removed the '--stream-neg-log10' option to fit the bit-score calculation
        command = ['mcxload', '-abc', g1, '--stream-mirror', '-o', g3, '-write-tab', g2]
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
    orig_ids = []
    for x in family:
        if ids.get(x) is None:
            logging.info("Couldn't find the gene id for {}".format(x))
        else:
            orig_ids.append(ids[x])
    return orig_ids
    #return [ids[x] for x in family]

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
        #if len(seqs) > 2: raise ValueError("More than two sequence data objects?")
        #if len(seqs) == 2: seqs[0].merge(seqs[1])
        #seqs = seqs[0]
        for i,s in enumerate(seqs):
            if i!=0: seqs[0].merge(s)
    return seqs[0]

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
        else: logging.info("Skipping singleton family {}{}".format(fid,family))
    return gene_families

def identity_ratio(aln):
    if aln.get_alignment_length() == 0: return 0
    else:
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

def getaln(famid,seqids,outdir,s,option):
    fnamec =os.path.join(outdir, famid + ".cds")
    fnamep =os.path.join(outdir, famid + ".pep")
    with open(fnamep,'w') as f:
        for seqid in seqids:
            safeid = s.idmap.get(seqid)
            f.write(">{}\n{}\n".format(seqid, s.pro_sequence.get(safeid)))
    with open(fnamec,'w') as f:
        for seqid in seqids:
            safeid = s.idmap.get(seqid)
            f.write(">{}\n{}\n".format(seqid, s.cds_sequence.get(safeid)))
    fnamepaln =os.path.join(outdir, famid + ".paln")
    mafft_cmd(fnamep,option,fnamepaln)
    fnamecaln =os.path.join(outdir, famid + ".caln")
    backtrans(fnamepaln,fnamecaln,s.idmap,s.cds_sequence)
    return fnamecaln

def getseqmetaln(i,fam,outdir,idmap,seq_pro,seq_cds,option):
    famid = "GF{:0>8}".format(i+1)
    fnamep =os.path.join(outdir, famid + ".pep")
    fnamec =os.path.join(outdir, famid + ".cds")
    with open(fnamep,'w') as f:
        for seqid in fam:
            safeid = idmap.get(seqid)
            f.write(">{}\n{}\n".format(seqid, seq_pro.get(safeid)))
    with open(fnamec,'w') as f:
        for seqid in fam:
            safeid = idmap.get(seqid)
            f.write(">{}\n{}\n".format(seqid, seq_cds.get(safeid)))
    fnamepaln =os.path.join(outdir, famid + ".paln")
    mafft_cmd(fnamep,option,fnamepaln)
    fnamecaln =os.path.join(outdir, famid + ".caln")
    backtrans(fnamepaln,fnamecaln,idmap,seq_cds)
    #Note that here the backtranslated codon-alignment will be shorter than the original cds file by a stop codon

def addmbtree(outdir,tree_fams,tree_famsf,i=0,concat=False,Multiplicon=False,mid = ''):
    if not concat: famid = "GF{:0>8}".format(i+1)
    else: famid = 'Concat'
    #if Multiplicon: famid = "Multiplicon{}".format(i)
    if Multiplicon: famid = mid
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

def iqtree_run(treeset,fnamecaln,treeoption=False):
    if not treeset is None:
        if treeoption: treeset = treeset.split(',')
        treesetfull = []
        iq_cmd = ["iqtree", "-s", fnamecaln]
        for i in treeset:
            i = i.strip(" ").split(" ")
            treesetfull = treesetfull + i
        iq_cmd = iq_cmd + treesetfull
    else: iq_cmd = ["iqtree", "-s", fnamecaln] #+ ["-fast"] + ["-st","CODON"] + ["-bb", "1000"] + ["-bnni"]
    sp.run(iq_cmd, stdout=sp.PIPE)

def fasttree_run(fnamecaln,treeset,treeoption=False):
    tree_pth = fnamecaln + ".fasttree"
    if not treeset is None:
        if treeoption: treeset = treeset.split(',')
        treesetfull = []
        ft_cmd = ["FastTree", '-out', tree_pth, fnamecaln]
        for i in treeset:
            i = i.strip(" ").split(" ")
            treesetfull = treesetfull + i
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

def parallelrbh(s,i,j,ogformat,cscore,eval):
    logging.info("{} vs. {}".format(s[i].prefix, s[j].prefix))
    s[i].get_rbh_orthologs(s[j], cscore, False, eval=eval)
    s[i].write_rbh_orthologs(s[j],singletons=False,ogformat=ogformat)

def mrbh(globalmrbh,outdir,s,cscore,eval,keepduplicates,anchorpoints,focus,keepfasta,nthreads):
    if globalmrbh:
        if not s[0].prot: logging.info("Multiple cds files: will compute globalMRBH orthologs or cscore-defined homologs regardless of focal species")
        else: logging.info("Multiple protein files: will compute globalMRBH orthologs or cscore-defined homologs regardless of focal species")
        table = pd.DataFrame()
        gmrbhf = os.path.join(outdir, 'global_MRBH.tsv')
        if nthreads!=(len(s)-1)*len(s)/2: logging.info("Note that setting the number of threads as {} is the most efficient".format(int((len(s)-1)*len(s)/2)))
        pairs = sum(map(lambda i:[(i,j) for j in range(i+1,len(s))],range(len(s)-1)),[])
        Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(get_mrbh)(s[i],s[j],cscore,eval) for i,j in pairs)
        for i in range(len(s)-1):
            tables = []
            for j in range(i+1,len(s)):
                df = getrbhf(s[i],s[j],outdir)
                if table.empty: table = df
                else:
                    table = table.merge(df)
                    if not keepduplicates:
                        for c in table.columns: table.drop_duplicates(subset=[c],inplace=True)
        #gfid = ['GF{:0>8}'.format(str(i+1)) for i in range(table.shape[0])]
        #table.insert(0,'GF', gfid)
        _label_families(table)
        #if not keepduplicates:
        #    for i in table.columns: table.drop_duplicates(subset=[i],inplace=True)
        table.to_csv(gmrbhf, sep="\t",index=True)
    elif not focus is None:
        if not s[0].prot: logging.info("Multiple cds files: will compute RBH orthologs or cscore-defined homologs between focal species and remaining species")
        else: logging.info("Multiple protein files: will compute RBH orthologs or cscore-defined homologs between focal species and remaining species")
        x = 0
        table = pd.DataFrame()
        focusname = os.path.join(outdir, 'merge_focus.tsv')
        for i in range(len(s)):
            if s[i].prefix == focus: x = i
        nonfocal_index = [k for k in range(len(s)) if k!= x]
        if nthreads!=len(s)-1: logging.info("Note that setting the number of threads as {} is the most efficient".format(int(len(s)-1)))
        Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(get_mrbh)(s[x],s[k],cscore,eval) for k in nonfocal_index)
        for k in nonfocal_index:
            df = getrbhf(s[x],s[k],outdir)
            if table.empty: table = df
            else:
                table = table.merge(df)
                if not keepduplicates: table.drop_duplicates([focus])
        table.insert(0, focus, table.pop(focus))
        #gfid = ['GF{:0>8}'.format(str(i+1)) for i in range(table.shape[0])]
        #table.insert(0,'GF', gfid)
        _label_families(table)
        table.to_csv(focusname, sep="\t",index=True)
    if not anchorpoints is None:
        ap = pd.read_csv(anchorpoints,header=0,index_col=False,sep='\t')
        ap = ap.loc[:,'gene_x':'gene_y']
        ap_reverse = ap.rename(columns = {'gene_x' : 'gene_y', 'gene_y' : 'gene_x'})
        ap_combined = pd.concat([ap,ap_reverse])
        focusapname = os.path.join(outdir, 'merge_focus_ap.tsv')
        table.insert(1, focus, table.pop(focus))
        table_ap = table.merge(ap_combined,left_on = focus,right_on = 'gene_x')
        table_ap['gene_xy'] = ["_".join(sorted([x,y])) for x,y in zip(list(table_ap['gene_x']),list(table_ap['gene_y']))]
        table_ap = table_ap.drop_duplicates(subset=['gene_xy']).drop('gene_xy',axis=1).drop('gene_x',axis=1)
        #table_ap.drop('gene_x', inplace=True, axis=1)
        table_ap.insert(2, 'gene_y', table_ap.pop('gene_y'))
        #table_ap.columns = table_ap.columns.str.replace(focus, focus + '_ap1')
        #table_ap.columns = table_ap.columns.str.replace('gene_y', focus + '_ap2')
        table_ap.rename(columns = {focus : focus + '_ap1', 'gene_y' : focus + '_ap2'}, inplace = True)
        #here I rename the GF index
        _label_families(table_ap)
        table_ap.to_csv(focusapname, sep="\t",index=True,header=True)
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
            Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(getfastaf)(i,fam,rbhgfdirname,seq_pro,idmap,seq_cds) for i, fam in enumerate(seqid_table))
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
                Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(getfastaf)(i,fam,rbhgfapdirname,seq_pro,idmap,seq_cds) for i, fam in enumerate(seqid_table))
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

def getproaln(i,fam,outdir,idmap,seq_pro,option):
    famid = "GF{:0>8}".format(i+1)
    fnamep =os.path.join(outdir, famid + ".pep")
    with open(fnamep,'w') as f:
        for seqid in fam:
            safeid = idmap.get(seqid)
            f.write(">{}\n{}\n".format(seqid, seq_pro.get(safeid)))
    fnamepaln =os.path.join(outdir, famid + ".paln")
    mafft_cmd(fnamep,option,fnamepaln)

def addproaln(i,outdir,pro_alns,pro_alnfs):
    famid = "GF{:0>8}".format(i+1)
    fnamepaln =os.path.join(outdir, famid + ".paln")
    pro_alns[famid] = AlignIO.read(fnamepaln, "fasta")
    pro_alnfs[famid] = fnamepaln

def get_only_protaln(seqs,fams,outdir,nthreads,option="--auto"):
    seq_pro,idmap,pro_alns,pro_alnfs = {},{},{},{}
    for i in range(len(seqs)):
        seq_pro.update(seqs[i].pro_sequence)
        idmap.update(seqs[i].idmap)
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(getproaln)(i,fam,outdir,idmap,seq_pro,option) for i, fam in enumerate(fams))
    for i in range(len(fams)): addproaln(i,outdir,pro_alns,pro_alnfs)
    return pro_alns,pro_alnfs

def Concat_prot(pro_alns,families,outdir):
    gsmap, slist = GetG2SMap(families, outdir)
    famnum = len(pro_alns)
    pro_alns_rn = {}
    Concat_palnf = os.path.join(outdir, "Concatenated.paln")
    proseq = {}
    slist_set = set(slist)
    for i in range(famnum):
        famid = "GF{:0>8}".format(i+1)
        pro_aln = pro_alns[famid]
        added_sp = set()
        for j in range(len(pro_aln)):
            with open(gsmap,"r") as f:
                lines = f.readlines()
                for k in lines:
                    k = k.strip('\n').strip(' ').split(' ')
                    if k[0] == pro_aln[j].id:
                        # here the repeated occurance of one ap will lead to errornously multiply that sequence
                        spn = k[1]
                        if spn in added_sp:
                            if spn.endswith('_ap1'): spn=spn[:-1]+'2'
                            else: spn=spn[:-1]+'1'
                        added_sp.add(spn)
                        pro_aln[j].id = spn
                        sequence = pro_aln[j].seq
                        if proseq.get(spn) is None: proseq[spn] = str(sequence)
                        else: proseq[spn] = proseq[spn] + str(sequence)
                        break
        if slist_set != added_sp: logging.info('Error in concatenation process that multiple genes were concatenated to the same species. Please check the input file that if two genes in the same family could be assigned to the same species!')
        pro_alns_rn[famid] = pro_aln
    with open(Concat_palnf,"w") as f:
        for spname in range(len(slist)):
            spn = slist[spname]
            f.write(">{}\n{}\n".format(spn, proseq[spn]))
    Concat_paln = AlignIO.read(Concat_palnf, "fasta")
    return Concat_palnf,Concat_paln,slist

def Run_MCMCTREE_concprot(Concat_paln,Concat_palnf,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads):
    CI_table,PM_table = {},{}
    wgd_mrca = [sp for sp in slist if sp[-4:] == '_ap1' or sp[-4:] == '_ap2']
    Concat_palnf_paml = fasta2paml(Concat_paln,Concat_palnf)
    if aamodel == 'wag':
        logging.info('Running mcmctree using Hessian matrix of WAG+Gamma for protein model')
    elif aamodel == 'lg':
        logging.info('Running mcmctree using Hessian matrix of LG+Gamma for protein model')
    elif aamodel == 'dayhoff':
        logging.info('Running mcmctree using Hessian matrix of Dayhoff-DCMut for protein model')
    else:
        logging.info('Running mcmctree using Poisson without gamma rates for protein model')
    McMctree = mcmctree(None, Concat_palnf_paml, tmpdir, outdir, speciestree, datingset, aamodel, partition=False)
    McMctree.run_mcmctree(CI_table,PM_table,wgd_mrca)

def Parallel_MCMCTREE_Prot(Concat_palnf,Concat_paln,palns,palnfs,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads):
    palns_palnfs = [(palns[key],palnfs[key]) for key in palns.keys()] + [(Concat_paln,Concat_palnf)]
    if aamodel == 'wag':
        logging.info('Running mcmctree using Hessian matrix of WAG+Gamma for protein model')
    elif aamodel == 'lg':
        logging.info('Running mcmctree using Hessian matrix of LG+Gamma for protein model')
    elif aamodel == 'dayhoff':
        logging.info('Running mcmctree using Hessian matrix of Dayhoff-DCMut for protein model')
    else:
        logging.info('Running mcmctree using Poisson without gamma rates for protein model')
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(Run_MCMCTREE_onlyprot)(paln,palnf,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads) for paln,palnf in palns_palnfs)

def Run_MCMCTREE_onlyprot(paln,palnf,tmpdir,outdir,speciestree,datingset,aamodel,slist,nthreads):
    CI_table,PM_table = {},{}
    wgd_mrca = [sp for sp in slist if sp[-4:] == '_ap1' or sp[-4:] == '_ap2']
    palnf_paml = fasta2paml(paln,palnf)
    McMctree = mcmctree(None, palnf_paml, tmpdir, outdir, speciestree, datingset, aamodel, partition=False)
    McMctree.run_mcmctree(CI_table,PM_table,wgd_mrca)

def concatcalnf(calnfs,gsmap,slist,outdir):
    seqs_persp = {sp:'' for sp in slist}
    y = lambda x: [i for i in x if i!='-']
    for calnf in calnfs:
        occured_sp = {}
        longest_sp = {}
        for record in SeqIO.parse(calnf, 'fasta'):
            sp = gsmap[record.id]
            if sp not in occured_sp:
                #seqs_persp[sp] += record.seq
                longest_sp[sp] = record.seq
                occured_sp[sp] = len(y(record.seq))
            elif len(y(record.seq)) > occured_sp[sp]:
                longest_sp[sp] = record.seq
                occured_sp[sp] = len(y(record.seq))
        for sp in seqs_persp.keys(): seqs_persp[sp] += longest_sp[sp]
    concatf = os.path.join(outdir,'Concat.caln')
    with open(concatf,'w') as f:
        for key,value in seqs_persp.items():
            f.write(">{0}\n{1}\n".format(key,value))
    return concatf

def addbackiqfatree(calnf,tree_method):
    postfix = ".treefile" if tree_method == "iqtree" else ".fasttree"
    treef = calnf + postfix
    treef_c = treef+"clean"
    tree = Phylo.read(treef,'newick')
    if not tree.rooted: tree.root_at_midpoint()
    for i in tree.get_nonterminals():
        i.branch_length = None
        i.comment = None
        i.confidence = None
    for i in tree.get_terminals():
        i.branch_length = None
        i.comment = None
        i.confidence = None
    Phylo.write(tree,treef_c,format='newick')
    with open(treef_c,'r') as f: content = f.read().replace(':0.00000','')
    with open(treef_c,'w') as f: f.write(content)
    return treef_c

def caln2tree(calnf,tree_method,tree_options):
    if tree_method == "iqtree": iqtree_run(tree_options,calnf,treeoption=True)
    if tree_method == "fasttree": fasttree_run(calnf,tree_options,treeoption=True)
    treef = addbackiqfatree(calnf,tree_method)
    return treef

def getconcataln(seqs, families, nthreads, outdir, sptree, spgenemap, onlyconcatkstree, tree_options, option="--auto", tree_method="fasttree"):
    kstree_dir,katree_dir,wtree_dir = _mkdir(os.path.join(outdir,"dStree")), _mkdir(os.path.join(outdir,"dNtree")), _mkdir(os.path.join(outdir,"wtree"))
    s = seqs[0]
    for i in seqs[1:]: s.merge_seqs(i)
    fams = read_MultiRBH_gene_families(families)
    df = pd.read_csv(families,header=0,index_col=0,sep='\t')
    slist = list(df.columns)
    fam_ids = list(df.index)
    calnfs = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(getaln)(fam_ids[i],fam,s.tmp_path,s,option) for i, fam in enumerate(fams))
    calnfs = [i for i in calnfs]
    concatf = concatcalnf(calnfs,spgenemap,slist,outdir)
    caln = AlignIO.read(concatf,'fasta')
    caln_tmppath = _mkdir(os.path.join(outdir,"dStree_tmp"))
    if not onlyconcatkstree:
        treefs = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(caln2tree)(calnfs[i],tree_method,tree_options) for i in range(len(fams)))
        treefs = [i for i in treefs]
        calns = [AlignIO.read(i,'fasta') for i in calnfs]
        caln_tmppaths = [_mkdir(os.path.join(caln_tmppath,i)) for i in fam_ids]
    #getalnks(caln,s.tmp_path,sptree,kstree_dir,katree_dir,wtree_dir)
        calns.append(caln)
        caln_tmppaths.append(_mkdir(os.path.join(caln_tmppath,"Concatenated")))
        fam_ids.append("Concatenated")
        treefs.append(sptree)
        Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(getalnks)(calns[i],caln_tmppaths[i],treefs[i],kstree_dir,katree_dir,wtree_dir,fam_ids[i]) for i in range(len(calns)))
    else:
        getalnks(caln,_mkdir(os.path.join(caln_tmppath,"Concatenated")),sptree,kstree_dir,katree_dir,wtree_dir,"Concatenated")
    out = sp.run(["rm", "-r", caln_tmppath], stdout=sp.PIPE, stderr=sp.PIPE)
    out = sp.run(["rm", "-r", concatf], stdout=sp.PIPE, stderr=sp.PIPE)

def getalnks(caln,tmp_path,sptree,kstree_dir,katree_dir,wtree_dir,gfid):
    #codeml = Codeml(caln, exe="codeml", tmp=tmp_path, prefix="Concatenated",treefile=os.path.abspath(sptree))
    codeml = Codeml(caln, exe="codeml", tmp=tmp_path, prefix=gfid, treefile=os.path.abspath(sptree))
    result = codeml.run_codeml(preserve=True, times=1, kstree_dir=kstree_dir,katree_dir=katree_dir,wtree_dir=wtree_dir)

def get_MultipRBH_gene_families(seqs, fams, tree_method, treeset, outdir,nthreads, option="--auto", runtree=False, **kwargs):
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
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(getseqmetaln)(i,fam,outdir,idmap,seq_pro,seq_cds,option) for i, fam in enumerate(fams))
    for i in range(len(fams)): add2table(i,outdir,cds_fastaf,palnfs,pro_alns,calnfs,calnfs_length,cds_alns,fnamecalns,fnamepalns)
    if runtree:
        x = lambda i : "GF{:0>8}".format(i+1)
        if tree_method == "mrbayes":
            Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(mrbayes_run)(outdir,x(i),fnamepalns[x(i)],pro_alns[x(i)],treeset) for i in range(len(fams)))
            for i in range(len(fams)): addmbtree(outdir,tree_fams,tree_famsf,i=i,concat=False)
        if tree_method == "iqtree":
            Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(iqtree_run)(treeset,fnamecalns[x(i)]) for i in range(len(fams)))
            for i in range(len(fams)): addiqfatree(x(i),tree_fams,fnamecalns[x(i)],tree_famsf,postfix = '.treefile')
        if tree_method == "fasttree":
            Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(fasttree_run)(fnamecalns[x(i)],treeset) for i in range(len(fams)))
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

def Concat(cds_alns, pro_alns, families, tree_method, treeset, outdir, infer_tree=False):
    gsmap, slist = GetG2SMap(families, outdir)
    famnum = len(pro_alns)
    cds_alns_rn = {}
    pro_alns_rn = {}
    Concat_calnf = os.path.join(outdir, "Concatenated.caln")
    Concat_palnf = os.path.join(outdir, "Concatenated.paln")
    cdsseq = {}
    proseq = {}
    ctree_length = 0
    slist_set = set(slist)
    Gs_dict = {}
    with open(gsmap,"r") as f:
        lines = f.readlines()
        for k in lines:
            k = k.strip('\n').strip(' ').split(' ')
            Gs_dict[k[0]] = k[1]
    for i in range(famnum):
        famid = "GF{:0>8}".format(i+1)
        cds_aln = cds_alns[famid]
        pro_aln = pro_alns[famid]
        added_sp = set()
        for j in range(len(pro_aln)):
            #with open(gsmap,"r") as f:
            #    lines = f.readlines()
            #    for k in lines:
            #        k = k.strip('\n').strip(' ').split(' ')
                    #if k[0] == cds_aln[j].id:
                        # here the repeated occurance of one ap will lead to errornously multiply that sequence
                        #spn = k[1]
                        #if spn in added_sp:
                        #    if spn.endswith('_ap1'): spn=spn[:-1]+'2'
                        #    else: spn=spn[:-1]+'1'
                        #added_sp.add(spn)
                        #cds_aln[j].id = spn
                        #sequence = cds_aln[j].seq
                        #if cdsseq.get(spn) is None: cdsseq[spn] = str(sequence)
                        #else: cdsseq[spn] = cdsseq[spn] + str(sequence)
                    #if k[0] == pro_aln[j].id:
                    #    spn = k[1]
                        #pro_aln[j].id = spn
                        #sequence = pro_aln[j].seq
                        #if proseq.get(spn) is None: proseq[spn] = str(sequence)
                        #else: proseq[spn] = proseq[spn] + str(sequence)
                        #break
            spn = Gs_dict[cds_aln[j].id]
            if spn in added_sp:
                if spn.endswith('_ap1'): spn=spn[:-1]+'2'
                else: spn=spn[:-1]+'1'
            added_sp.add(spn)
            cds_aln[j].id = spn
            sequence = cds_aln[j].seq
            if cdsseq.get(spn) is None: cdsseq[spn] = str(sequence)
            else: cdsseq[spn] = cdsseq[spn] + str(sequence)
            pro_aln[j].id = spn
            sequence = pro_aln[j].seq
            if proseq.get(spn) is None: proseq[spn] = str(sequence)
            else: proseq[spn] = proseq[spn] + str(sequence)
        if slist_set != added_sp: logging.info('Error in concatenation process that multiple genes were concatenated to the same species. Please check the input file that if two genes in the same family could be assigned to the same species!')
        cds_alns_rn[famid] = cds_aln
        pro_alns_rn[famid] = pro_aln
    with open(Concat_palnf,"w") as f:
        for spname in range(len(slist)):
            spn = slist[spname]
            sequence = proseq[spn]
            f.write(">{}\n{}\n".format(spn, sequence))
    with open(Concat_calnf,"w") as f:
        for spname in range(len(slist)):
            spn = slist[spname]
            sequence = cdsseq[spn]
            f.write(">{}\n{}\n".format(spn, sequence))
    Concat_caln = AlignIO.read(Concat_calnf, "fasta")
    ctree_length = Concat_caln.get_alignment_length()
    Concat_paln = AlignIO.read(Concat_palnf, "fasta")
    Concat_ctrees, ctree_pths, Concat_ptrees, ptree_pths, famid = {},[],{},[],'Concat'
    if infer_tree:
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
    else:
        return cds_alns_rn, pro_alns_rn, Concat_calnf, Concat_palnf, ctree_length, gsmap, Concat_caln, Concat_paln, slist

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
    CI = wgd_node.comment.strip('[&95%HPD={').strip('[&95%={').strip('}]').split(', ')
    PM = wgd_node.clades[0].branch_length
    CI_table[prefixx]=[float(i) for i in CI]
    PM_table[prefixx]=PM

def Getback_CIPM(outdir,CI_table,PM_table,wgd_mrca,calnfs_rn,Concat_calnf_paml,partition):
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
        if partition:
            folder = os.path.join(outdir, "mcmctree",prefix+'_partitioned',"cds")
            os.chdir(folder)
            get_dates(wgd_mrca,CI_table,PM_table,prefix+"_partitioned")
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
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(i.run_beast)(beastlgjar,beagle) for i in beasts)

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
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(McMctree.run_mcmctree)(CI_table,PM_table,wgd_mrca) for McMctree in McMctrees)
    Getback_CIPM(outdir,CI_table,PM_table,wgd_mrca,calnfs_rn,Concat_calnf_paml,partition)
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
        if pfam != 'none': cmd = pfam_annot(cmd,pfam)
        cmd = dmnb_annot(cmd,dmnb)
        cmds.append(cmd)
    #Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=10)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
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
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
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
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=100)(delayed(sp.run)(cmd, stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    mvgfback_interproscan(cds_fastaf,out_path)
    os.chdir(parent)

def endt(tmpdir,start,s):
    if tmpdir is None: [x.remove_tmp(prompt=False) for x in s]
    end = timer()
    logging.info("Total run time: {:.2f} minutes".format((end-start)/60))
    logging.info("Done")
    exit()

def endtime(start):
    end = timer()
    logging.info("Total run time: {:.2f} minutes".format((end-start)/60))
    logging.info("Done")
    exit()

def writevervangencat(files,fname):
    with open (fname,'w') as f:
        for i in files:
            with open(i,'r') as ff:
                f.write(ff.read()+'\n')

def concathmm(outdir,df):
    hmmconcatf = os.path.join(outdir,'Full.hmm')
    gids = map(lambda i:os.path.join(outdir,i+'.pep.hmm'),df.index)
    writevervangencat(gids,hmmconcatf)
    #cmd = ['cat'] + [i for i in gids]
    #out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    #with open(hmmconcatf,'w') as f: f.write(out.stdout.decode('utf-8'))
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
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=10)(delayed(run_hmmerbp)(yids(i),yfnp(i),s) for i in df.index)

def writetmpseq(s,df,outdir):
    fname_whole = os.path.join(outdir,'reference.pep')
    for fid in df.index:
        #fname = os.path.join(outdir,fid)
        #tmp = df.loc[[fid],:].dropna(axis=1)
        genes_genes = df.loc[[fid],:].dropna(axis=1).loc[fid,:]
        for genes in genes_genes:
            with open(fname_whole,'a') as f:
                for gene in genes.split(', '): f.write('>{}\n{}\n'.format(gene,s.pro_sequence[s.idmap[gene]]))
    return fname_whole

def reference_hmmscan(df,s,hmmf,outdir,eval):
    refer_fp = writetmpseq(s,df,outdir)
    out = scanrefer(refer_fp,hmmf,outdir,eval)
    f_g_score = {fid:{} for fid in df.index}
    score_per_f = {fid:[] for fid in df.index}
    cutoff_per_f = {fid:0 for fid in df.index}
    dfo = pd.read_csv(out,header = None, index_col=False,sep ='\t')
    end = dfo.shape[0] - 10
    for i in range(3,end):
        pair = dfo.iloc[i,0].split()
        f,g,score = pair[0][:-4],pair[2],float(pair[5])
        f_g_score[f][g] = score
    for fid in df.index:
        genes_genes = df.loc[[fid],:].dropna(axis=1).loc[fid,:]
        for genes in genes_genes:
            for gene in genes.split(', '):
                if f_g_score[fid].get(gene) == None:
                    logging.info("{0} in {1} has no hits, please double check this possibly misassigned gene".format(gene,fid))
                    continue
                score_per_f[fid].append(f_g_score[fid][gene])
    for fid,vs in score_per_f.items():
        if len(vs) == 0:
            logging.info("Genes in {} all have no hits, please double check the authenticity of this family".format(fid))
            cutoff_per_f[fid] = 0
        else: cutoff_per_f[fid] = min(vs)*0.9
    for fid,cutoff in cutoff_per_f.items(): logging.info('The cutoff score for family {} is {:.2f}'.format(fid,cutoff))
    return cutoff_per_f

def scanrefer(refer_fp,hmmf,outdir,eval):
    cmd = ['hmmpress'] + [hmmf]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    #pf = os.path.join(outdir,os.path.basename(refer_fp).strip('.pep'))
    pf = os.path.join(outdir,os.path.basename(refer_fp)[:-4])
    cmd = ['hmmscan','-o', '{}.txt'.format(pf), '--tblout', '{}.tbl'.format(pf), '--domtblout', '{}.dom'.format(pf), '--pfamtblout', '{}.pfam'.format(pf), '--noali', '-E', '{}'.format(eval), hmmf, refer_fp]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    out = '{}.tbl'.format(pf)
    return out

def hmmerscan(outdir,querys,hmmf,eval,nthreads,skipress=False):
    if not skipress:
        cmd = ['hmmpress'] + [hmmf]
        sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    cmds = []
    outs = []
    #yprefix = lambda i: os.path.join(outdir,os.path.basename(i).strip('.pep'))
    yprefix = lambda i: os.path.join(outdir,os.path.basename(i)[:-4])
    for s in querys:
        s.orig_profasta()
        pf = yprefix(s.orig_pro_fasta)
        cmd = ['hmmscan','-o', '{}.txt'.format(pf), '--tblout', '{}.tbl'.format(pf), '--domtblout', '{}.dom'.format(pf), '--pfamtblout', '{}.pfam'.format(pf), '--noali', '-E', '{}'.format(eval), hmmf, s.orig_pro_fasta]
        cmds.append(cmd)
        outs.append('{}.tbl'.format(pf))
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(sp.run)(cmd,stdout=sp.PIPE,stderr=sp.PIPE) for cmd in cmds)
    return outs

def buildbfam(fname,outdict):
    indexs = []
    for v in outdict.values():
        for f in v.keys():
            indexs.append(f)
    indexs = set(indexs)
    dic = {indice:['' for i in range(len(outdict))] for indice in indexs}
    df = pd.DataFrame.from_dict(dic, orient='index',columns=list(outdict.keys()))
    for f in df.index:
        for sp in df.columns:
            if f in list(outdict[sp].keys()):
                df.loc[f,sp] = outdict[sp][f][0]
    df.to_csv(fname,header = True, index = True,sep = '\t')
    return df

def modifydf(df,outs,outdir,fam2assign,sogtest = False, bhmm = False, cutoff = None, use_cf = None):
    if cutoff != None: ctf = pd.read_csv(cutoff, header = None,index_col = 0,sep='\t')
    if not bhmm: fname = os.path.join(outdir,os.path.basename(fam2assign)+'.assigned')
    #yb = lambda i:os.path.basename(i).strip('.tbl')
    yb = lambda i:os.path.basename(i)[:-4]
    #if sogtest: yb = lambda i:os.path.basename(i).strip('.tbl') + '_assigned'
    if sogtest: yb = lambda i:os.path.basename(i)[:-4] + '_assigned'
    outdict = {yb(i):{} for i in outs}
    if not bhmm: f_g_score = {fid:{} for fid in df.index}
    for out in outs:
        #print(yb(out))
        dfo = pd.read_csv(out,header = None, index_col=False,sep ='\t')
        end = dfo.shape[0] - 10
        for i in range(3,end):
            pair = dfo.iloc[i,0].split()
            if bhmm:
                f,g,score = pair[0],pair[2],float(pair[5])
                ctf_v = ctf.loc[f,1]
                if score < ctf_v:
                    continue
                if outdict[yb(out)].get(f) == None: outdict[yb(out)].update({f:(g,score)})
                elif outdict[yb(out)][f][1] < score: outdict[yb(out)][f] = (g,score)
            else:
                f,g,score = pair[0][:-4],pair[2],float(pair[5])
                f_g_score[f][g] = score
                #print((f,g,score))
                if use_cf != None:
                    if score >= use_cf[f]:
                        if outdict[yb(out)].get(f) == None: outdict[yb(out)][f]=g
                        else: outdict[yb(out)][f] = ', '.join([outdict[yb(out)][f],g])
                else:
                    if outdict[yb(out)].get(f) == None: outdict[yb(out)][f]=g
                    else: outdict[yb(out)][f] = ', '.join([outdict[yb(out)][f],g])
    if bhmm: df = buildbfam(df,outdict)
    else:
        if sogtest:
            #print(outdict)
            #ks = outdict.keys()
            #new_outdict = {yb(i):{} for i in outs}
            #for k in ks:
            #    v = outdict[k]
            cutoff_fs = {}
            for fid in df.index: cutoff_fs[fid] = min([f_g_score[fid][gene] for gene in list(df.loc[fid,:])])*0.9
            for f,v in cutoff_fs.items():
                logging.info('The cutoff score for single-copy gene family {} is {:.2f}'.format(f,v))
            for k,v in outdict.items():
                for f in v.keys():
                    #orig_gene = df.loc[f,k[:-9]]
                    #highest_score = g_score[orig_gene]
                    #cutoff_score = highest_score*0.9
                    #logging.info('The cutoff score for single-copy gene family {} is {:.2f}'.format(f,cutoff_fs[f]))
                    genes_prefilter = outdict[k][f].split(', ')
                    genes_afterfilter = ''
                    for gene in genes_prefilter:
                        ge_score = f_g_score[f][gene]
                        if ge_score >= cutoff_fs[f]:
                            genes_afterfilter = ', '.join([genes_afterfilter,gene]) if genes_afterfilter != '' else gene
                    outdict[k][f] = genes_afterfilter
                    #new_outdict[k][f] = genes_afterfilter
            #outdict = new_outdict
        for k,v in outdict.items():
            yf,yg = (lambda v:(list(v.keys()),list(v.values())))(v)
            df.insert(0, k, pd.Series(yg, index=yf))
        if sogtest:
            l = int(len(df.columns)/2)
            df2 = df.iloc[:,:l]
            df2.to_csv(fname,header = True, index = True,sep = '\t')
        else: df.to_csv(fname,header = True, index = True,sep = '\t')
    return df

def getassignfasta(df,s,querys,outdir,second=False,third=False):
    yids = lambda i: ', '.join(list(df.loc[i,:].dropna())).split(', ')
    for i in querys: s.merge_seq(i)
    if second:
        p = _mkdir(os.path.join(outdir,'Orthologues_Sequence_Assigned_Furtherscorefiltered'))
    elif third:
        p = _mkdir(os.path.join(outdir,'Orthologues_Sequence_Assigned_Furtherscorefiltered_Tree-based'))
    else:
        p = _mkdir(os.path.join(outdir,'Orthologues_Sequence_Assigned'))
    if not s.prot: pc = _mkdir(os.path.join(p,'cds'))
    pp = _mkdir(os.path.join(p,'pep'))
    pps,glength = [],{}
    for i in df.index:
        if not s.prot: fc = os.path.join(pc,i+'.cds')
        fp = os.path.join(pp,i+'.pep')
        pps.append(fp)
        if not s.prot:
            with open(fc,'w') as f:
                for gi in yids(i):
                    if gi == '':
                        continue
                    f.write('>{}\n{}\n'.format(gi,s.cds_sequence[s.idmap[gi]]))
        with open(fp,'w') as f:
            for gi in yids(i):
                if gi == '':
                    continue
                f.write('>{}\n{}\n'.format(gi,s.pro_sequence[s.idmap[gi]]))
                glength[gi] = len(s.pro_sequence[s.idmap[gi]])
        #for gi in yids(i):
        #    with open(fc,'a') as f: f.write('>{}\n{}\n'.format(gi,s.cds_sequence[s.idmap[gi]]))
        #    with open(fp,'a') as f: f.write('>{}\n{}\n'.format(gi,s.pro_sequence[s.idmap[gi]]))
    return pps,glength

def hmmer4g2f(outdir,s,nthreads,querys,df,eval,fam2assign,Noldsp,Nnewsp,gsmap,tmpdir,tree_method,treeset):
    hmmerbuild(df,s,outdir,nthreads)
    hmmf = concathmm(outdir,df)
    c_f = reference_hmmscan(df,s,hmmf,outdir,eval)
    outs = hmmerscan(outdir,querys,hmmf,eval,nthreads,skipress=True)
    df = modifydf(df,outs,outdir,fam2assign,use_cf=c_f)
    fromgene2count(df,outdir,fam2assign)
    pps,glength = getassignfasta(df,s,querys,outdir)
    df = postrbhcutoff(df,nthreads,eval,outdir,pps,glength,Noldsp,Nnewsp,gsmap,fam2assign,tmpdir)
    pps,glength = getassignfasta(df,s,querys,outdir,second=True)
    logging.info("Inferring gene trees using {}".format(tree_method))
    treepaths = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(finalinfertree)(fp,tree_method,treeset) for fp in pps)
    logging.info("The path of tree files is at {}".format(os.path.join(outdir,'Orthologues_Sequence_Assigned_Furtherscorefiltered','pep')))
    logging.info("Pruning the tree")
    df = filtersp(treepaths,gsmap,Nnewsp,df)
    fname = os.path.join(outdir,os.path.basename(fam2assign)+".assigned.furtherscorefiltered.tree-based")
    df.to_csv(fname,header=True,index=True,sep='\t')
    fromgene2count(df,outdir,fam2assign,third=True)
    getassignfasta(df,s,querys,outdir,third=True)

def prunebranch(allchildren,newsps,df,gsmaps,fam):
    for sp in newsps: df.loc[fam,sp] = ''
    for gene in allchildren:
        if gsmaps[gene] not in newsps:
            continue
        df.loc[fam,gsmaps[gene]] = gene if df.loc[fam,gsmaps[gene]] == '' else ", ".join([df.loc[fam,gsmaps[gene]],gene])
    return df

def findsubfamily(tree,oldsps,newsps,df,fam,gsmaps):
    tree.root_at_midpoint()
    oldseqs = []
    for sp in df.columns:
        if sp in oldsps:
            content = df.loc[fam,sp]
            if type(content) == float or type(content) == np.float64 or content == '':
                continue
            oldseqs = oldseqs + content.split(", ")
    mrca = tree.common_ancestor(*oldseqs)
    tips = mrca.get_terminals()
    allchildren = [tip.name for tip in tips]
    df = prunebranch(allchildren,newsps,df,gsmaps,fam)
    return df

def filtersp(treepaths,gsmap,Nnewsp,df):
    Noldsp = [i for i in df.columns if i not in Nnewsp]
    treepaths = {fam:tp for tp,fam in zip(treepaths,df.index)}
    for fam,tp in treepaths.items():
        tree = Phylo.read(tp,'newick')
        df = findsubfamily(tree,Noldsp,Nnewsp,df,fam,gsmap)
    return df

def waln(fpep,faln):
    cmd = ["mafft"] + ["--amino", fpep]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open(faln, 'w') as f: f.write(out.stdout.decode('utf-8'))

def runiqree(faln,treeset):
    if not treeset is None:
        treesetfull = []
        cmd = ["iqtree", "-s", faln]
        for i in treeset:
            i = i.strip(" ").split(" ")
            treesetfull = treesetfull + i
        cmd = cmd + treesetfull
    else:
        cmd = ["iqtree", "-s", faln]
    sp.run(cmd, stdout=sp.PIPE)

def runfastree(faln,treeset):
    if not treeset is None:
        treesetfull = []
        cmd = ["FastTree", "-out", faln+".fasttree", faln]
        for i in treeset:
            i = i.strip(" ").split(" ")
            treesetfull = treesetfull + i
        cmd = cmd[:1] + treesetfull + cmd[1:]
    else:
        cmd = ["FastTree", "-out", faln+".fasttree", faln]
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

def wgenetree(faln,tree_method,treeset):
    if tree_method == 'iqtree':
        runiqree(faln,treeset)
        return faln+".treefile"
    if tree_method == 'fasttree':
        runfastree(faln,treeset)
        return faln+".fasttree"

def finalinfertree(fp,tree_method,treeset):
    waln(fp,fp+".aln")
    treepath = wgenetree(fp+".aln",tree_method,treeset)
    return treepath


def postrbhcutoff(df,nthreads,eval,outdir,pps,glength,Noldsp,Nnewsp,gsmap,fam2assign,tmpdir):
    outfiles = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(runselfdiamond)(fnamep,eval,nthreads,fam) for fam,fnamep in zip(df.index,pps))
    dfs = normalizedout(outfiles,glength,gsmap)
    df = filterbymin(dfs,df,Noldsp,Nnewsp)
    fname = os.path.join(outdir,os.path.basename(fam2assign)+".assigned.furtherscorefiltered")
    df.to_csv(fname,header=True,index=True,sep='\t')
    fromgene2count(df,outdir,fam2assign,second=True)
    if tmpdir ==None: rmtmpp(outdir)
    return df

def rmtmpp(outdir):
    parent = os.getcwd()
    target = os.path.join(outdir,'Orthologues_Sequence_Assigned','pep')
    os.chdir(target)
    with open('rm.sh','w') as f: f.write('rm *.dmnd *.tsv *.tsv_normalized')
    sp.run(['sh','rm.sh'],stdout=sp.PIPE,stderr=sp.PIPE)
    sp.run(['rm','rm.sh'],stdout=sp.PIPE,stderr=sp.PIPE)
    os.chdir(parent)

def filterbymin(dfs,df,Noldsp,Nnewsp):
    fams = list(df.index)
    for d,fam in zip(dfs,fams):
        cutoff = getreferencecutoff(d,Noldsp)
        logging.info("The normalized bit-score cutoff for {0} is {1:.2f}".format(fam,cutoff))
        retainednewseqs = realfilter(d,cutoff,Noldsp,Nnewsp)
        df = filterdf(df,retainednewseqs,fam,Nnewsp)
    return df

def filterdf(df,retainednewseqs,fam,Nnewsp):
    for sp in Nnewsp: df.loc[fam,sp] = ''
    for gene, sp in retainednewseqs:
        if df.loc[fam,sp] == '':
            df.loc[fam,sp] = gene
        else:
            df.loc[fam,sp] = ", ".join([df.loc[fam,sp],gene])
    return df

def realfilter(d,cutoff,Noldsp,Nnewsp):
    retainednewseqs = []
    for i in d.index:
        sp1,sp2 = d.loc[i,14],d.loc[i,15]
        if sp1 in Noldsp and sp2 in Nnewsp:
            if d.loc[i,13] >= cutoff:
                retainednewseqs.append((d.loc[i,1],sp2))
        if sp1 in Nnewsp and sp2 in Noldsp:
            if d.loc[i,13] >= cutoff:
                retainednewseqs.append((d.loc[i,0],sp1))
    retainednewseqs = [i for i in set(retainednewseqs)]
    return retainednewseqs

def getreferencecutoff(d,Noldsp):
    scores_olds = []
    for i in d.index:
        sp1,sp2 = d.loc[i,14],d.loc[i,15]
        if sp1 in Noldsp and sp2 in Noldsp:
            scores_olds.append(d.loc[i,13])
    cutoff = min(scores_olds)
    return cutoff

def normalizedout(outfiles,glength,gsmap):
    dfs = []
    for out in outfiles:
        df = pd.read_csv(out,header=None,index_col=None,sep='\t',usecols=[0,1,11])
        df = addgleng(df,glength)
        df = fitall_linear(df)
        df = addspn(df,gsmap)
        fname = out + "_normalized"
        df.to_csv(fname,header=False,index=False,sep='\t')
        dfs.append(df)
    return dfs

def addspn(df,gsmap):
    df[14] = df[0].apply(lambda x:gsmap[x])
    df[15] = df[1].apply(lambda x:gsmap[x])
    return df

def fitall_linear(df):
    slope, intercept, r, p, se = stats.linregress(np.log10(df[12]), np.log10(df[11]))
    df[13] = [j/(pow(10, intercept)*(l**slope)) for j,l in zip(df[11],df[12])]
    return df

def addgleng(df,glength):
    df[12] = df[0].apply(lambda x:glength[x])
    df[13] = df[1].apply(lambda x:glength[x])
    df[14] = [g1*g2 for g1,g2 in zip(df[12],df[13])]
    df = df.drop(columns=[12, 13]).rename(columns={14: 12})
    return df

def runselfdiamond(fnamep,eval,nthreads,fam):
    if not os.path.isfile(fnamep[:-4] + '.dmnd'):
        cmd = ["diamond", "makedb", "--in", fnamep, "-d", fnamep[:-4], "-p", str(nthreads)]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        logging.debug(out.stderr.decode())
        if out.returncode == 1: logging.error(out.stderr.decode())
    outfile = "_".join([fnamep[:-4],fam+".tsv"])
    cmd = ["diamond", "blastp", "-d", fnamep[:-4] + '.dmnd', "-q", fnamep, "-e", str(eval), "-o", outfile, "-p", str(nthreads)]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    logging.debug(out.stderr.decode())
    return outfile


def fromgene2count(df,outdir,fam2assign,second = False,third=False):
    fname = os.path.join(outdir,os.path.basename(fam2assign)+'.assigned.genecount')
    if second: fname = os.path.join(outdir,os.path.basename(fam2assign)+'.assigned.furtherscorefiltered.genecount')
    if third: fname = os.path.join(outdir,os.path.basename(fam2assign)+'.assigned.furtherscorefiltered.tree-based.genecount')
    Index = df.index
    columns = {c:[] for c in df.columns}
    for indice in df.index:
        for c in df.columns:
            if type(df.loc[indice,c]) != str:
                columns[c].append(0)
            elif df.loc[indice,c] == '':
                columns[c].append(0)
            else:
                columns[c].append(len(df.loc[indice,c].split(", ")))
    df_out = pd.DataFrame.from_dict(columns)
    df_out.index = Index
    df_out.to_csv(fname,header=True,index=True,sep='\t')

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
    return None

def genes2fams(seq2assign,fam2assign,outdir,s,nthreads,tmpdir,to_stop,cds,cscore,eval,start,normalizedpercent,tree_method,treeset,assign_method='hmmer',prot=False):
    Noldsp = [i.prefix for i in s]
    gsmap = {}
    for seq in s: gsmap.update({gid:seq.prefix for gid in seq.idmap.keys()})
    logging.info("Assigning sequences into given gene families")
    seqs_query = [SequenceData(s, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore, threads=nthreads, normalizedpercent=normalizedpercent,prot=prot) for s in seq2assign]
    for seq in seqs_query: gsmap.update({gid:seq.prefix for gid in seq.idmap.keys()})
    Nnewsp = [i.prefix for i in seqs_query]
    df = pd.read_csv(fam2assign,header=0,index_col=0,sep='\t')
    for i in range(1, len(s)): s[0].merge_seq(s[i])
    if assign_method == 'hmmer': hmmer4g2f(outdir,s[0],nthreads,seqs_query,df,eval,fam2assign,Noldsp,Nnewsp,gsmap,tmpdir,tree_method,treeset)
    else: dmd4g2f(outdir,s[0],nthreads,seqs_query,df)
    rmtmp(tmpdir,outdir,seqs_query)
    endt(tmpdir,start,s)

def run_or(i,j,s,eval,orthoinfer):
    s[i].run_diamond(s[j], orthoinfer, eval=eval, savememory=True)

def back_dmdhits(i,j,s,eval):
    ftmp = os.path.join(s[i].tmp_path,'_'.join([s[i].prefix,s[j].prefix])+'.tsv')
    df = pd.read_csv(ftmp, sep="\t", header=None)
    df = df.loc[df[0] != df[1]]
    s[i].dmd_hits[s[j].prefix] = df = df.loc[df[10] <= eval]

def ortho_infer_mul(s,nthreads,eval,inflation,orthoinfer):
    pairs = sum(map(lambda i:[(i,j) for j in range(i,len(s))],range(len(s))),[])
    if nthreads!=(len(s)+1)*len(s)/2: logging.info("Note that setting the number of threads as {} is the most efficient".format(int((len(s)+1)*len(s)/2)))
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(run_or)(i,j,s,eval,orthoinfer) for i,j in pairs)
    for i in range(len(s)):
        #Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(run_or)(i,j,s,eval,orthoinfer) for j in range(i, len(s)))
        #res = zip(*r)
        #for e in res: print(e.shape)
        #for j,e in zip(range(i, len(s)),res):
        #    print(type(e))
        #    print(e)
        #    s[i].dmd_hits[s[j].prefix] = e
        for j in range(i, len(s)): back_dmdhits(i,j,s,eval)
        s[i].rndmd_hit()
    #memory_reporter()
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
    writevervangencat(sequences,Concat_cdsf)
    #cmd = ['cat'] + [s for s in sequences]
    #out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    #with open(Concat_cdsf,'w') as f: f.write(out.stdout.decode('utf-8'))
    return Concat_cdsf

def ortho_infer(sequences,s,outdir,tmpdir,to_stop,cds,cscore,inflation,eval,nthreads,getsog,tree_method,treeset,msogcut,concat,testsog,normalizedpercent,bins=100,nonormalization=False):
    s0_orig = copy.deepcopy(s[0])
    if concat:
        sgidmaps = {}
        for x in s : sgidmaps.update(x.spgenemap())
        Concat_cdsf = concatcdss(sequences,outdir)
        ss = SequenceData(Concat_cdsf, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore, threads=nthreads, bins=bins, normalizedpercent=normalizedpercent,nonormalization=nonormalization)
        logging.info("tmpdir = {} for {}".format(ss.tmp_path,ss.prefix))
        #memory_reporter()
        ss.get_paranome(inflation=inflation, eval=eval, savememory = True,sgidmaps = sgidmaps)
        txtf = ss.write_paranome(True)
    #Concat_cdsf = concatcdss(sequences,outdir)
    #ss = SequenceData(Concat_cdsf, out_path=outdir, tmp_path=tmpdir, to_stop=to_stop, cds=cds, cscore=cscore)
    else: ss,txtf = ortho_infer_mul(s,nthreads,eval,inflation,False)
    #logging.info("tmpdir = {} for {}".format(ss.tmp_path,ss.prefix))
    #ss.get_paranome(inflation=inflation, eval=eval)
    #txtf = ss.write_paranome(True)
    sgmaps = {}
    slist = []
    for seq in s: sgmaps.update(seq.spgenemap())
    for seq in s: slist.append(seq.prefix)
    txt2tsv(txtf,outdir,sgmaps,slist,ss,nthreads,getsog,tree_method,treeset,msogcut,s,testsog,eval,s0_orig)
    if concat:
        if tmpdir is None: ss.remove_tmp(prompt=False)
        sp.run(['rm'] + [Concat_cdsf], stdout=sp.PIPE,stderr=sp.PIPE)
    return txtf

def writeogsep(table,seq,fc,fp):
    for v in table.values():
        if type(v) == list: v = v[0]
        if v == '': continue
        if not seq.prot: cds = seq.cds_sequence[seq.idmap[v]]
        pro = seq.pro_sequence[seq.idmap[v]]
        if not seq.prot:
            with open(fc,'a') as f: f.write('>{}\n{}\n'.format(v,cds))
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(v,pro))

def getnestedfasta(fnest,df,ss,nfs_count):
    fc_nest = _mkdir(os.path.join(fnest,'cds'))
    fp_nest = _mkdir(os.path.join(fnest,'pep'))
    ndc = copy.deepcopy(nfs_count)
    for j,rn in enumerate(df.index):
        if nfs_count[rn] == 1:
            if not ss.prot: fcname = os.path.join(fc_nest,'{}.cds'.format(rn))
            fpname = os.path.join(fp_nest,'{}.pep'.format(rn))
            if not ss.prot:
                with open(fcname,'w') as f:
                    for i in df.iloc[j,:][:-1]: f.write('>{}\n{}\n'.format(i,ss.cds_sequence[ss.idmap[i]]))
            with open(fpname,'w') as f:
                for i in df.iloc[j,:][:-1]: f.write('>{}\n{}\n'.format(i,ss.pro_sequence[ss.idmap[i]]))
        else:
            t = ndc[rn]
            if not ss.prot: fcname = os.path.join(fc_nest,'{0}_{1}.cds'.format(rn,t))
            fpname = os.path.join(fp_nest,'{0}_{1}.pep'.format(rn,t))
            if not ss.prot:
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
    d.update({'NestedType':'mostly single-copy'})
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
                dic.update({'NestedType':'single-copy'})
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
    if tree_method == 'fasttree': addiqfatree(x(i),tree_fams,fpaln,tree_famsf,postfix = '.fasttree')
    if tree_method == 'iqtree': addiqfatree(x(i),tree_fams,fpaln,tree_famsf,postfix = '.treefile')
    if tree_method == 'mrbayes': addmbtree(outd,tree_fams,tree_famsf,i=i,concat=False)
    dics = label2nest(tree_fams[x(i)],slist,sgmaps,ss,msogcut)
    if dics:
        for dic in dics:
            dic.update({'NestedSOG':x(i)})
            df = pd.DataFrame.from_dict([dic])
            nested_dfs.append(df)

def aln2tree_sc(fp,fc,ss,tree_method,treeset,outd,i,Multiplicon=False):
    if Multiplicon: x = lambda i : "Multiplicon{}".format(i)
    else: x = lambda i : "GF{:0>8}".format(i+1)
    fpaln,o,fcaln = fp + '.aln','--auto',fc + '.aln'
    mafft_cmd(fp,o,fpaln)
    #if not ss.prot: backtrans(fpaln,fcaln,ss.idmap,ss.cds_sequence)
    if tree_method == "iqtree": iqtree_run(treeset,fpaln)
    if tree_method == "fasttree": fasttree_run(fpaln,treeset)
    if tree_method == "mrbayes": mrbayes_run(outd,x(i),fpaln,AlignIO.read(fpaln, "fasta"),treeset)

def sgratio(l):
    t = [i for i in l if i]
    ratio = len(t)/len(l)
    return ratio

def first_tsv_genecounts(gsmap,slist,ss,msogcut):
    fam_table,represent_seqs = {},{}
    count_table = {s:0 for s in slist}
    sumcount = 0
    ct = 'multi-copy'
    exist_sp = set(gsmap.values())
    coverage = len(exist_sp)/len(slist)
    for k,v in gsmap.items():
        pro = ss.pro_sequence[ss.idmap[k]]
        if fam_table.get(v) == None:
            fam_table[v] = k
            represent_seqs[v] = k
        else:
            fam_table[v] = ", ".join([fam_table[v],k])
            if len(pro) > len(ss.pro_sequence[ss.idmap[represent_seqs[v]]]): represent_seqs[v] = k
        count_table[v] = count_table[v] + 1
        sumcount = sumcount + 1
    for ms in set(slist) - set(gsmap.values()): fam_table[ms],represent_seqs[ms] = '',''
    li = [v == 1 for v in count_table.values()]
    if all(li): ct = 'single-copy'
    elif sgratio(li) >= msogcut: ct = 'mostly single-copy'
    fam_df = pd.DataFrame.from_dict([fam_table])
    count_table.update({'Sum':sumcount,'PhylogenyCoverage':coverage,'CopyType':ct})
    count_df = pd.DataFrame.from_dict([count_table])
    rep_df = pd.DataFrame.from_dict([represent_seqs])
    return fam_df,count_df,rep_df

def sgdict(gsmap,slist,ss,ftmp,frep,fsog,i,msogcut):
    fam_table,represent_seqs = {},{}
    count_table = {s:0 for s in slist}
    #sumcount = 0
    #ct = 'multi-copy'
    #exist_sp = set(gsmap.values())
    #coverage = len(exist_sp)/len(slist)
    if not ss.prot:
        fc = os.path.join(_mkdir(os.path.join(ftmp,'cds')),"GF{:0>8}.cds".format(i+1))
    fc_rep = os.path.join(_mkdir(os.path.join(frep,'cds')),"GF{:0>8}.cds".format(i+1))
    fc_sog = os.path.join(_mkdir(os.path.join(fsog,'cds')),"GF{:0>8}.cds".format(i+1))
    fp = os.path.join(_mkdir(os.path.join(ftmp,'pep')),"GF{:0>8}.pep".format(i+1))
    fp_rep = os.path.join(_mkdir(os.path.join(frep,'pep')),"GF{:0>8}.pep".format(i+1))
    fp_sog = os.path.join(_mkdir(os.path.join(fsog,'pep')),"GF{:0>8}.pep".format(i+1))
    for k,v in gsmap.items():
        if not ss.prot: cds = ss.cds_sequence[ss.idmap[k]]
        pro = ss.pro_sequence[ss.idmap[k]]
        if fam_table.get(v) == None:
            fam_table[v] = k
            represent_seqs[v] = k
        else:
            fam_table[v] = ", ".join([fam_table[v],k])
            if len(pro) > len(ss.pro_sequence[ss.idmap[represent_seqs[v]]]): represent_seqs[v] = k
        count_table[v] = count_table[v] + 1
        #sumcount = sumcount + 1
        if not ss.prot:
            with open(fc,'a') as f: f.write('>{}\n{}\n'.format(k,cds))
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(k,pro))
    writeogsep(represent_seqs,ss,fc_rep,fp_rep)
    for ms in set(slist) - set(gsmap.values()): fam_table[ms],represent_seqs[ms] = '',''
    li = [v == 1 for v in count_table.values()]
    if all(li):
        writeogsep(fam_table,ss,fc_sog,fp_sog)
    #    ct = 'single-copy'
    #elif sgratio(li) >= msogcut: ct = 'mostly single-copy'
    #fam_df = pd.DataFrame.from_dict([fam_table])
    #count_table.update({'Sum':sumcount,'PhylogenyCoverage':coverage,'CopyType':ct})
    #count_df = pd.DataFrame.from_dict([count_table])
    #rep_df = pd.DataFrame.from_dict([represent_seqs])
    #fam_df.to_csv(os.path.join(ftmp,'GF{:0>8}.tsv'.format(i)),header=True,index=True,sep='\t')
    #count_df.to_csv(os.path.join(ftmp,'GF{:0>8}_count.tsv'.format(i)),header=True,index=True,sep='\t')
    #rep_df.to_csv(os.path.join(ftmp,'GF{:0>8}_rep.tsv'.format(i)),header=True,index=True,sep='\t')
    #return fam_df,count_df,rep_df
    #fams_df.append(fam_df)
    #counts_df.append(count_df)
    #reps_df.append(rep_df)

def seqdict(gsmap,ss,i,ftmp):
    fc = os.path.join(_mkdir(os.path.join(ftmp,'cds')),"GF{:0>8}.cds".format(i+1))
    fp = os.path.join(_mkdir(os.path.join(ftmp,'pep')),"GF{:0>8}.pep".format(i+1))
    with open(fc,'w') as f:
        for k in gsmap.keys(): f.write('>{}\n{}\n'.format(k,ss.cds_sequence[ss.idmap[k]]))
    with open(fp,'w') as f:
        for k in gsmap.keys(): f.write('>{}\n{}\n'.format(k,ss.pro_sequence[ss.idmap[k]]))
    #for k in gsmap.keys():
    #    cds = ss.cds_sequence[ss.idmap[k]]
    #    pro = ss.pro_sequence[ss.idmap[k]]
    #    with open(fc,'a') as f: f.write('>{}\n{}\n'.format(k,cds))
    #    with open(fp,'a') as f: f.write('>{}\n{}\n'.format(k,pro))

def countdict(gsmap,slist,ss,counts_df):
    count_table = {s:0 for s in slist}
    sumcount = 0
    for k,v in gsmap.items():
        count_table[v] = count_table[v] + 1
        sumcount = sumcount + 1
    count_table.update({'Sum':sumcount})
    count_df = pd.DataFrame.from_dict([count_table])
    counts_df.append(count_df)

def rmtsv(ftmp):
    cwd = os.getcwd()
    os.chdir(ftmp)
    with open('rm.sh','w') as f: f.write('rm *.tsv')
    sp.run(['sh','rm.sh'],stdout=sp.PIPE, stderr=sp.PIPE)
    sp.run(['rm','rm.sh'],stdout=sp.PIPE, stderr=sp.PIPE)
    os.chdir(cwd)

def phmmbuild(fp):
    fpaln,o,fhmm = fp + '.aln','--auto',fp + '.hmm'
    mafft_cmd(fp,o,fpaln)
    cmd = ['hmmbuild'] + [fhmm] + [fpaln]
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def concatehmm(outd,famids):
    hmmconcatf = os.path.join(outd,'Full.hmm')
    gids = map(lambda i:os.path.join(outd,i+'.pep.hmm'),famids)
    writevervangencat(gids,hmmconcatf)
    #cmd = ['cat'] + [i for i in gids]
    #out = sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    #with open(hmmconcatf,'w') as f: f.write(out.stdout.decode('utf-8'))
    return hmmconcatf

def testassign(df):
    col_assign = [i for i in df.columns if i.endswith('_assigned')]
    for i in df.index:
        assigned_genes = map(lambda j:df.loc[i,j].split(', '),col_assign)
        if all([len(j) == 1 for j in assigned_genes]):
            logging.info("{} passed the strict single-copy test".format(i))
        else:
            logging.info("{} failed the strict single-copy test".format(i))

def mvassignf(pepf,bhmm = False):
    cwd = os.getcwd()
    os.chdir(pepf)
    if bhmm:
        with open('rmv.sh','w') as f: f.write('rm *.dom *.pfam *.tbl *.txt')
    else:
        with open('rmv.sh','w') as f: f.write('rm *.dom *.pfam *.tbl *.txt *.hmm* *.aln;mv Orthogroups_single_copy.tsv.assigned ../..')
    cmd = ['sh'] + ['rmv.sh']
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)
    cmd = ['rm'] + ['rmv.sh']
    sp.run(cmd, stdout=sp.PIPE,stderr=sp.PIPE)

def unbiasedsog(fsog,fams_coc,counts_coc,nthreads,s,outdir,eval,s0_orig):
    sog_famids = counts_coc[counts_coc['CopyType'] == 'single-copy'].index
    if len(sog_famids) == 0: logging.info("No single-copy gene families delineated, skipping unbiased test")
    else:
        sog_fams = fams_coc[fams_coc.index.isin(sog_famids)]
        fn_sog = os.path.join(outdir,'Orthogroups_single_copy.tsv')
        sog_fams.to_csv(fn_sog,header = True,index =True,sep = '\t')
        pepf = os.path.join(fsog,'pep')
        yp = lambda i: os.path.join(pepf,'{}.pep'.format(i))
        Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=10)(delayed(phmmbuild)(yp(i)) for i in sog_famids)
        hmmf = concatehmm(pepf,sog_famids)
        outs = hmmerscan(pepf,[s0_orig]+s[1:],hmmf,eval,nthreads)
        df = modifydf(sog_fams,outs,pepf,fn_sog,sogtest = True)
        testassign(df)
        mvassignf(pepf)

def txt2tsv(txtf,outdir,sgmaps,slist,ss,nthreads,getsog,tree_method,treeset,msogcut,s,testsog,eval,s0_orig):
    fname_fam = os.path.join(outdir,'Orthogroups.sp.tsv')
    fname_count = os.path.join(outdir,'Orthogroups.genecount.tsv')
    fname_rep = os.path.join(outdir,'Orthogroups.representives.tsv')
    fname_nest = os.path.join(outdir,'Orthogroups.nested_single_copy.tsv')
    ftmp = _mkdir(os.path.join(outdir,'Orthologues_Sequence'))
    frep = _mkdir(os.path.join(outdir,'Orthologues_Sequence_Representives'))
    fsog = _mkdir(os.path.join(outdir,'Orthologues_Single_Copy'))
    txt = pd.read_csv(txtf,header = None,index_col=0,sep='\t')
    y= lambda x: {j:sgmaps[j] for j in x}
    fams_df,counts_df,reps_df = [],[],[]
    sh = txt.shape[0]
    gsmaps = [y(txt.iloc[i,0].split(', ')) for i in range(sh)]
    #memory_reporter()
    #Parallel(n_jobs=nthreads)(delayed(seqdict)(gsmaps[i],ss,i,ftmp) for i in range(sh))
    #for i in range(sh):
    #r = Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=1000)(delayed(first_tsv_genecounts)(gsmaps[i],slist,ss,msogcut) for i in range(sh))
    for i in range(sh):
        fam_df,count_df,rep_df = first_tsv_genecounts(gsmaps[i],slist,ss,msogcut)
        fams_df.append(fam_df)
        counts_df.append(count_df)
        reps_df.append(rep_df)
    fams_coc = pd.concat([i for i in fams_df],ignore_index=True)
    counts_coc = pd.concat([i for i in counts_df],ignore_index=True)
    reps_coc = pd.concat([i for i in reps_df],ignore_index=True)
    #Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=1000)(delayed(sgdict)(gsmaps[i],slist,ss,ftmp,frep,fsog,i,msogcut) for i in range(sh))
    #fam_dfs,count_dfs,rep_dfs=zip(*r)
    #    sgdict(gsmaps[i],slist,fams_df,counts_df,reps_df,ss,ftmp,frep,fsog,i,msogcut)
    #Parallel(n_jobs=nthreads)(delayed(sgdict)(y(txt.iloc[i,0].split(', ')),slist,fams_df,counts_df,ss,ftmpc,ftmpp,i) for i in range(txt.shape[0]))
    #for i in range(txt.shape[0]): sgdict(y(txt.iloc[i,0].split(', ')),slist,fams_df,counts_df)
    #Parallel(n_jobs=nthreads)(delayed(seqdict)(y(txt.iloc[i,0].split(', ')),ss,ftmpc,ftmpp,i) for i in range(txt.shape[0]))
    #fams_coc = pd.concat(fams_df,ignore_index=True)
    #counts_coc = pd.concat(counts_df,ignore_index=True)
    #reps_coc = pd.concat(reps_df,ignore_index=True)
    #fams_coc = pd.concat([i for i in fams_df],ignore_index=True)
    #counts_coc = pd.concat([i for i in counts_df],ignore_index=True)
    #reps_coc = pd.concat([i for i in reps_df],ignore_index=True)
    #fams_coc = pd.concat([pd.read_csv(os.path.join(ftmp,'GF{:0>8}.tsv'.format(i)),header=0,index_col=0,sep='\t') for i in range(sh)],ignore_index=True)
    #counts_coc = pd.concat([pd.read_csv(os.path.join(ftmp,'GF{:0>8}_count.tsv'.format(i)),header=0,index_col=0,sep='\t') for i in range(sh)],ignore_index=True)
    #reps_coc = pd.concat([pd.read_csv(os.path.join(ftmp,'GF{:0>8}_rep.tsv'.format(i)),header=0,index_col=0,sep='\t') for i in range(sh)],ignore_index=True)
    _label_families(fams_coc)
    _label_families(counts_coc)
    _label_families(reps_coc)
    #rmtsv(ftmp)
    fams_coc.to_csv(fname_fam,header = True,index =True,sep = '\t')
    counts_coc.to_csv(fname_count,header = True,index =True,sep = '\t')
    reps_coc.to_csv(fname_rep,header = True,index =True,sep = '\t')
    logging.info("In total {} orthologous families delineated".format(counts_coc.shape[0]))
    mu,mo,sg=(counts_coc[counts_coc['CopyType']==i].shape[0] for i in ['multi-copy','mostly single-copy','single-copy'])
    logging.info("With {0} multi-copy, {1} mostly single-copy, {2} single-copy\nWriting sequences for each gene family (might take a while..)".format(mu,mo,sg))
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=1000)(delayed(sgdict)(gsmaps[i],slist,ss,ftmp,frep,fsog,i,msogcut) for i in trange(sh))
    #memory_reporter()
    if getsog:
        fnest = _mkdir(os.path.join(outdir,'Orthologues_Nested_Single_Copy'))
        tree_famsf,tree_fams,nested_dfs,aln_fam_is = [],{},[],[]
        yc = lambda x: os.path.join(ftmp,'cds',"GF{:0>8}.cds".format(x+1))
        yp = lambda x: os.path.join(ftmp,'pep',"GF{:0>8}.pep".format(x+1))
        #yco = lambda x,y: counts_df[x].loc[0,y]
        yco = lambda x,y: counts_coc.loc["GF{:0>8}".format(x+1),y]
        outd = os.path.join(ftmp,'pep')
        for i in range(sh):
            li = [yco(i,s) for s in slist]
            if all([j > 0 for j in li]) and sum(li) > len(slist): aln_fam_is.append(i)
        logging.info("Inferring gene tree on candidate gene families")
        Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=100)(delayed(aln2tree_sc)(yp(i),yc(i),ss,tree_method,treeset,outd,i) for i in aln_fam_is)
        for i in aln_fam_is: getnestedog(yp(i),yc(i),slist,i,outd,tree_method,tree_famsf,tree_fams,sgmaps,nested_dfs,ss,msogcut)
        if nested_dfs:
            nested_coc = pd.concat(nested_dfs,ignore_index=True).set_index('NestedSOG')
            nested_coc.to_csv(fname_nest,header = True,index =True,sep = '\t')
            logging.info("{} nested single-copy families delineated".format(nested_coc.shape[0]))
            nfs = list(nested_coc.index)
            nfs_count = {i:nfs.count(i) for i in set(nfs)}
            getnestedfasta(fnest,nested_coc,ss,nfs_count)
        else: logging.info("No nested single-copy families delineated")
        #memory_reporter()
    if testsog: unbiasedsog(fsog,fams_coc,counts_coc,nthreads,s,outdir,eval,s0_orig)

def get_sog_multiplicons(df,species_num):
    sp_counted = df.groupby(["multiplicon"])["genome"].aggregate(lambda x: len(set(x)))
    level = df.groupby(["multiplicon"])["genome"].aggregate(lambda x: len(x))
    FullCoverage_Multiplicons = sp_counted[sp_counted==species_num]
    RightLevel_Multiplicons = level[level==species_num]
    FullCoverage_Multiplicons = FullCoverage_Multiplicons.to_frame()
    RightLevel_Multiplicons = RightLevel_Multiplicons.to_frame()
    SOG_Multiplicons = FullCoverage_Multiplicons.merge(RightLevel_Multiplicons,left_index=True,right_index=True)
    SOG_Multiplicons_ids = list(SOG_Multiplicons.index)
    return SOG_Multiplicons_ids

def Allratio(profile,Ratios):
    sps = profile.columns
    text = ":".join(sps)
    ratios = []
    levels = []
    Multiplicons_matrix = []
    scale = lambda x: 1 if x > 1 else x
    for i in range(profile.shape[0]):
        levels.append(sum([int(j) for j in profile.iloc[i,:]]))
        ratio = ":".join([str(int(j)) for j in profile.iloc[i,:]])
        Multiplicon_matrix = [scale(int(j)) for j in profile.iloc[i,:]]
        ratios.append(ratio)
        Multiplicons_matrix.append(Multiplicon_matrix)
        if Ratios.get(ratio) == None: Ratios[ratio] = 1
        else: Ratios[ratio] = Ratios[ratio] + 1
    profile['ratio'] = ratios
    profile[text] = ratios
    profile['level'] = levels
    return text,Multiplicons_matrix,profile

def multipliconid2aps(Multiplicons_ids,anchorpoints):
    df = pd.read_csv(anchorpoints, sep="\t", index_col=0, header=0)
    df.set_index("multiplicon")

#def getWGDderivedmultiplicons(profile,WGDingroup,outgroup):

def Poisson_fit(x,observations,loc=0):
    Sum = 0
    Sum_o = 0 
    for i,o in zip(x,observations):
        Sum = Sum + i*o
        Sum_o = Sum_o + o
    u = Sum/Sum_o
    y = poisson.pmf(x, mu=u, loc = loc)
    return x,y

def processap(segments,sequences,msogcut):
    Ratios = {}
    species_num = len(sequences)
    df = pd.read_csv(segments, sep="\t", index_col=0)
    #SOG_Multiplicons_ids = get_sog_multiplicons(df,species_num)
    df["segment"] = df.index
    counted = df.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    profile = counted.unstack(level=-1).fillna(0)
    Allratio(profile,Ratios)
    SOG_symbol = ":".join([str(1) for i in range(species_num)])
    SOG_Multiplicons_ids = list(profile[profile['ratio']==SOG_symbol].index)
    y = lambda x : [i.count('1')/species_num >= msogcut for i in x]
    MSOG_Multiplicons_ids = list(profile["ratio"].where(y(profile["ratio"])).dropna().index)
    x,y = Poisson_fit(np.linspace(0, 100, num=len(Ratios)),Ratios.values(),loc=0)

def orderap(aps,order):
    anchorpairs = aps.split(';')
    order_infos = []
    for ap in anchorpairs:
        genes = ap.split(', ')
        order_info = []
        for gene in genes:
            for i,gxy in enumerate(order):
                if gene in gxy:
                    order_info.append(i)
                    break
        order_info = ", ".join([str(i) for i in order_info])
        order_infos.append(order_info)
    order_infos = ";".join(order_infos)
    return order_infos

def msap2mf(df_msap,order):
    M = list(set(df_msap['multiplicon']))
    slist = set(df_msap['genome'])
    MFs,MFs_order = [],[]
    for m in M:
        MF,MF_order = {sp:'' for sp in slist},{sp:'' for sp in slist}
        df = df_msap[df_msap['multiplicon']==m]
        for seg,genes,genome in zip(df.index,df['anchors'],df['genome']):
            if MF.get(genome) == '':
                MF[genome],MF_order[genome] = genes,order.loc[seg,'position']
            else:
                MF[genome],MF_order[genome] = ";".join([MF[genome],genes]),";".join([MF_order[genome],order.loc[seg,'position']])
        MF.update({'multiplicon':m})
        MF_order.update({'multiplicon':m})
        MF_df = pd.DataFrame.from_dict([MF]).set_index('multiplicon')
        MF_order_df = pd.DataFrame.from_dict([MF_order]).set_index('multiplicon')
        MFs.append(MF_df)
        MFs_order.append(MF_order_df)
    MFs_coc = pd.concat(MFs)
    MFs_order_coc = pd.concat(MFs_order)
    return MFs_coc,MFs_order_coc

def genes2Concat(genes,order,seqs,cds=True):
    genes,order = genes.split(', '),[int(i) for i in order.split(', ')]
    genes = [x for _, x in sorted(zip(order, genes), key=lambda y: y[0])]
    sequence = ''
    for i in genes:
        seq = seqs.cds_sequence[seqs.idmap[i]] if cds else seqs.pro_sequence[seqs.idmap[i]]
        sequence = sequence + seq
    return sequence

def getseq(i,seqs,genes_list,fp):
    fpaln = fp + '.aln'
    for sp, genes in genes_list:
        real_geneid = genes[i]
        seq = seqs.pro_sequence[seqs.idmap[real_geneid]]
        with open(fp,'a') as f: f.write('>{}\n{}\n'.format(sp,seq))
    mafft_cmd(fp,'--auto',fpaln)
    #fpaln_object = AlignIO.read(fpaln, "fasta")

def concat_seq(fname_mid,fps,sps):
    sp_seq = {sp:'' for sp in sps}
    for fp in fps:
        fpaln = fp + '.aln'
        fpaln_object = AlignIO.read(fpaln, "fasta")
        for i in fpaln_object: sp_seq[i.id] = sp_seq[i.id] + i.seq
            #if sp_seq.get(i.id) == '': sp_seq[i.id] = i.seq
            #else: sp_seq[i.id] = sp_seq[i.id] + i.seq
    with open(fname_mid,'w') as f:
        for sp,seq in sp_seq.items(): f.write('>{}\n{}\n'.format(sp,seq))

def writeseq_per_aps(mid,genes_sorted,seqs,f,nthreads):
    sp_list = list(genes_sorted.keys())
    #tmp = genes_sorted[sp_list[0]].split(';')
    #tmp = genes_sorted[sp_list[0]]
    num_aps = len(genes_sorted[sp_list[0]])
    #genes_list = list(map(lambda x:(x[0],x[1].split(', ')) ,genes_sorted.items()))
    genes_list = list(map(lambda x:(x[0],x[1]) ,genes_sorted.items()))
    fname = _mkdir(os.path.join(f, "pep"))
    fname_mid = os.path.join(fname,'Multiplicon{}.pep.aln'.format(mid))
    yf = lambda i: os.path.join(fname,'{0}_anchorpoint{1}.pep'.format(mid,i+1))
    fps = [yf(i) for i in range(num_aps)]
    Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=20,verbose=11)(delayed(getseq)(i,seqs,genes_list,fps[i]) for i in range(num_aps))
    concat_seq(fname_mid,fps,sp_list)
    return fname_mid

def Concat_by_order(MFs,MFs_order,seqs,f,nthreads):
    gsmap = {i:i for i in MFs.columns}
    findex,num_sp = [],len(gsmap)
    #print(num_sp)
    for i in MFs.index:
        genes_allsp = list(MFs.loc[i,:])
        cutoff = 1
        #print(([j !='' for j in genes_allsp],sum([j !='' for j in genes_allsp])))
        if sum([j !='' for j in genes_allsp])/num_sp < cutoff:
            continue
        #print('Multi{}'.format(i))
        genes_per_sp_sorted = {}
        order_allsp = list(MFs_order.loc[i,:])
        for sp,genes,orders in zip(MFs.columns,genes_allsp,order_allsp):
            if genes == '':
                continue
            gene,order = genes.split(';'),orders.split(';')
            le = len(gene)
            for d,ge,od in zip(range(le),gene,order):
                ge = [x for _, x in sorted(zip([int(i) for i in od.split(', ')], ge.split(', ')), key=lambda y: y[0])]
                if le == 1: genes_per_sp_sorted[sp] = ge
                else:
                    genes_per_sp_sorted['{0}_{1}'.format(sp,d+1)] = ge
                    gsmap['{0}_{1}'.format(sp,d+1)] = sp
        fname_mid = writeseq_per_aps(i,genes_per_sp_sorted,seqs,f,nthreads)
        findex.append(fname_mid)
    return findex,gsmap

def backtrans_Multi(fpaln,fcaln,seq_cds):
    aln = {}
    pro_aln = AlignIO.read(fpaln, "fasta")
    for i, s in enumerate(pro_aln):
        cds_aln = ""
        cds_seq = seq_cds[s.id]
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

def aln_2tree(fp,fc,seqs,tree_method,treeset,outd,i,CDS):
    x = lambda i : "Multiplicon{}".format(i)
    fpaln,o,fcaln = fp + '.aln','--auto',fc + '.aln'
    mafft_cmd(fp,o,fpaln)
    #pro_aln = backtrans_Multi(fpaln,fcaln,CDS)
    #if tree_method == "iqtree": iqtree_run(treeset,fcaln)
    if tree_method == "iqtree": iqtree_run(treeset,fpaln)
    if tree_method == "fasttree": fasttree_run(fpaln,treeset)
    #if tree_method == "fasttree": fasttree_run(fcaln,treeset)
    #if tree_method == "mrbayes": mrbayes_run(outd,x(i),fpaln,pro_aln,treeset)

def run_tree_msc(fpaln,tree_method,treeset):
    if tree_method == "iqtree": iqtree_run(treeset,fpaln)
    if tree_method == "fasttree": fasttree_run(fpaln,treeset)
    if tree_method == "mrbayes": mrbayes_run(os.path.join(fname_seq,'pep'),fpaln.strip('.pep.aln'),fpaln,AlignIO.read(fpaln, "fasta"),treeset)

#def getMultipliconstrees(fp,fc,i,outd,tree_method,tree_famsf,tree_fams):
def getMultipliconstrees(findex,tree_method,tree_famsf,tree_fams):
    y = lambda x:os.path.basename(x)
    #x = lambda i : "Multiplicon{}".format(i)
    #fpaln,fcaln = fp + '.aln',fc + '.aln'
    #if tree_method == 'fasttree': addiqfatree(x(i),tree_fams,fcaln,tree_famsf,postfix = '.fasttree')
    #if tree_method == 'fasttree': addiqfatree(x(i),tree_fams,fpaln,tree_famsf,postfix = '.fasttree')
    if tree_method == 'fasttree': addiqfatree(y(findex),tree_fams,findex,tree_famsf,postfix = '.fasttree')
    if tree_method == 'iqtree': addiqfatree(y(findex),tree_fams,findex,tree_famsf,postfix = '.treefile')
    #if tree_method == 'iqtree': addiqfatree(x(i),tree_fams,fpaln,tree_famsf,postfix = '.treefile')
    #if tree_method == 'iqtree': addiqfatree(x(i),tree_fams,fcaln,tree_famsf,postfix = '.treefile')
    #if tree_method == 'mrbayes': addmbtree(outd,tree_fams,tree_famsf,i=i,concat=False,Multiplicon=True)

def getmsctree(fpaln,fid,tree_method,tree_famsf,tree_fams,outd = ''):
    if tree_method == 'fasttree': addiqfatree(fid,tree_fams,fpaln,tree_famsf,postfix = '.fasttree')
    if tree_method == 'iqtree': addiqfatree(fid,tree_fams,fpaln,tree_famsf,postfix = '.treefile')
    if tree_method == 'mrbayes': addmbtree(outd,tree_fams,tree_famsf,i=0,concat=False,Multiplicon=True,mid = fid)

def Astral_infer(tree_famsf,gsmapf,outdir):
    whole_tree = ""
    whole_treef = os.path.join(outdir, "Whole.ctree")
    coalescence_treef = os.path.join(outdir, "Coalescence.ctree")
    for tree in tree_famsf:
        with open(tree,"r") as f:
            tree_content = f.readlines()
            for i in tree_content: whole_tree = whole_tree + i
    with open(whole_treef,"w") as f: f.write(whole_tree)
    ASTER_cmd = ["astral-pro", "-i", whole_treef, "-a", gsmapf, "-o", coalescence_treef]
    ASTER_cout = sp.run(ASTER_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    coalescence_ctree = Phylo.read(coalescence_treef,'newick')
    return coalescence_ctree, coalescence_treef

def writegsmap(gsmap,gsmapf):
    with open(gsmapf,'w') as f:
        for k,v in gsmap.items(): f.write("{0} {1}\n".format(k,v))

def hierarchy_dendrogram(X,labels,outdir,label=True):
    fname = os.path.join(outdir, "Hierarchical_Clustering_Multiplicons_Index.pdf")
    if label: fname = os.path.join(outdir, "Hierarchical_Clustering_Multiplicons_SpeciesID.pdf")
    Z = linkage(X, 'ward')
    fig, ax = plt.subplots()
    plt.rcParams['lines.linewidth'] = 10
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram',fontdict={'fontsize': 28})
    plt.xlabel('Species',fontsize = 24)
    plt.ylabel('Distance',fontsize = 24)
    plt.yticks(fontsize = 20)
    if label: dendrogram(Z,labels=labels,leaf_font_size=20,orientation='top')
    else: dendrogram(Z,leaf_font_size=20,orientation='top')
    plt.savefig(fname,format ='pdf')

def search_shared_aps(ap_filtered,num_sp,gene_sp_gl,cutoff):
    Mul_groups = list(ap_filtered.groupby('multiplicon'))
    Mids = list(map(lambda x:x[0],Mul_groups))
    Aps_per_Mul = list(map(lambda x:x[1].loc[:,['sp_x','gl_x','gene_x','gene_y','sp_y','gl_y','multiplicon','level']],Mul_groups))
    num = 0
    dfs_container = []
    minimum_sp = num_sp*cutoff
    if minimum_sp-int(minimum_sp) != 0: logging.info('Only consider multiplcons containing at least {} species'.format(int(minimum_sp)+1))
    else: logging.info('Only consider multiplcons containing at least {} species'.format(int(minimum_sp)))
    for mid,df in zip(Mids,Aps_per_Mul):
        df = df.drop_duplicates(subset=['gene_x', 'gene_y']).copy()
        level = list(df['level'])[0]
        # Here I used the 'gene_y' column as grouping proxy because the structure of anchorpoints.txt is that the 'gene_y' column will be fixed as only one segment, while the 'gene_x' column is variable in segments. (For instance there is 10 segments in a multiplicon, the 'gene_y' column will always be the genes from segment 1, while the 'gene_x' column will be genes from the other 9(or 10 if self-collinearity) segments)
        occurs_gl = df.groupby('gene_y')[['gl_x','gene_x']].aggregate(lambda x:list(x))
        for gy in occurs_gl.index: occurs_gl.loc[gy,'gene_x'] = occurs_gl.loc[gy,'gene_x'] + [gy]
        occurs_gl = occurs_gl.rename(columns={"gl_x": "gl_x_y",'gene_x':'gene_xy'})
        for gy in occurs_gl.index: occurs_gl.loc[gy,'gl_x_y'] = occurs_gl.loc[gy,'gl_x_y'] + [gene_sp_gl[gy][1]]
        occurs_gl['num_gl_x_y'] = [len(i) for i in occurs_gl['gl_x_y']]
        occurs_sp = df.groupby('gene_y')['sp_x'].aggregate(lambda x:set(x))
        occurs_sp.name = 'sp_x_y'
        occurs_sp = occurs_sp.to_frame()
        for gy in occurs_sp.index: occurs_sp.loc[gy,'sp_x_y'] = occurs_sp.loc[gy,'sp_x_y'] | set([gene_sp_gl[gy][0]])
        occurs_sp['num_sp_x_y'] = [len(i) for i in occurs_sp['sp_x_y']]
        df = df.drop_duplicates(subset=['gene_y']).merge(occurs_gl.reset_index(),on='gene_y').merge(occurs_sp.reset_index(),on='gene_y')
        df = df[(df['num_gl_x_y']>=level) & (df['num_sp_x_y']>=minimum_sp)]
        #for x,y in zip(df['gene_xy'],df['num_gl_x_y']): print((x,y))
        if len(df) ==0: logging.info('Skip multiplicon {} due to no intersection of anchor pairs across all levels'.format(mid))
        else:
            dfs_container.append(df)
            num = num + 1
            logging.info('Multiplicon {} satisfying required intersection is to be analyzed'.format(mid))
    logging.info('In total {} multiplicons are to be considered'.format(num))
    dfs = pd.concat(dfs_container)
    #for glxy,leve,mid,i in zip(dfs['gl_x_y'],dfs['level'],dfs['multiplicon'],dfs['gene_xy']):
    #    print((mid,leve,glxy,i))
    return dfs

def concat_per_mlt(ind,df,fname_seq_pep,tree_method,treeset,mlt_ap,gene_sp_gl,gene_start,mlts_seq,sg_glsmap):
    df_tmp = df.loc[[ind],:].copy().dropna(axis=1)
    #sp_gl_level = {sp_gl:level for sp_gl,level in map(lambda x:(x,len(df_tmp.loc[ind,x].split(', '))),df_tmp.columns)}
    sp_gl_genes = {sp_gl:df_tmp.loc[ind,sp_gl].split(', ') for sp_gl in df_tmp.columns}
    sp_gl_level = {sp_gl:len(df_tmp.loc[ind,sp_gl].split(', ')) for sp_gl in df_tmp.columns}
    fname_aln = os.path.join(fname_seq_pep,ind+'.aln')
    aln_object = AlignIO.read(fname_aln,'fasta')
    gene_start_order = {seq.id:gene_start[seq.id] for seq in aln_object}
    aln_gene = {sequ.id:sequ.seq for sequ in aln_object}
    sorted_sg_gl = sorted(df_tmp.columns)
    #gene_id_order = []
    #for concat_order,sg_gl in enumerate(sorted_sg_gl):
    for sg_gl in sorted_sg_gl:
        #ap_id = '{0}_{1}'.format(concat_order+1,sg_gl)
        ap_id = 'Multiplicon{0}_{1}'.format(mlt_ap[ind],sg_gl)
        genes = sp_gl_genes[sg_gl]
        if len(genes) > 1: genes = [x for _,x in sorted(zip([gene_start_order[g] for g in genes],genes),key=lambda y:y[0])]
        for go,gene in enumerate(genes):
            gene_id = '{0}_{1}'.format(ap_id,go+1)
            #gene_id_order.append(gene_id)
            sg_glsmap[gene_id] = gene_sp_gl[gene][0]
            if mlts_seq[mlt_ap[ind]].get(gene_id) == None: mlts_seq[mlt_ap[ind]][gene_id] = aln_gene[gene]
            else: mlts_seq[mlt_ap[ind]][gene_id] = mlts_seq[mlt_ap[ind]][gene_id] + aln_gene[gene]
    #mlt_gid_order[mlt_ap[ind]] = gene_id_order
    #text = ' '.join(gene_id_order)
    #logging.info('The concatenation order of {0} is {1}'.format(ind,text))
            #with open(,'a') as f: f.write('>{}\n{}\n'.format(gene_id,aln_gene[gene]))
    #for seq in aln_object:
    #    sp_gl = gene_sp_gl[seq.id][1]
    #    level = sp_gl_level[sp_gl]
    #    if level > 1:
def getseq_para(ind,df,s,fname_seq_pep,tree_method,treeset):
    fname = os.path.join(fname_seq_pep,ind)
    fname_aln = fname + '.aln'
    df_tmp = df.loc[[ind],:].copy()
    df_tmp = df_tmp.dropna(axis=1)
    for sp_gl in df_tmp.columns:
        genes = df_tmp.loc[ind,sp_gl].split(', ')
        for gene in genes:
            with open(fname,'a') as f: f.write('>{}\n{}\n'.format(gene,s.pro_sequence[s.idmap[gene]]))
    mafft_cmd(fname,'--auto',fname_aln)
    run_tree_msc(fname_aln,tree_method,treeset)

def writeconcatseq(mlt,seq_per_ap,f,tree_method,tree_famsf,tree_fams,treeset):
    fn = os.path.join(f,'Multiplicon'+str(mlt)+'_concatenation.aln')
    with open(fn,'w') as f:
        for gene_id,seq in seq_per_ap.items(): f.write('>{}\n{}\n'.format(gene_id,seq))
    run_tree_msc(fn,tree_method,treeset)
    getmsctree(fn,'Multiplicon'+str(mlt),tree_method,tree_famsf,tree_fams,outd = f)

def writemscseq(df,fname_seq,s,nthreads,tree_method,treeset,gsmapf,outdir,mlt_ap,gene_sp_gl,gene_start):
    fname_seq_pep = _mkdir(os.path.join(fname_seq,'pep'))
    fname_seq_pep_concat = _mkdir(os.path.join(fname_seq,'pep_concatenation_per_multiplicon'))
    mlts_seq = {mlt_ap[ind]:{} for ind in df.index}
    tree_famsf,tree_fams,sg_glsmap=[],{},{}
    #Parallel(n_jobs=nthreads,backend='multiprocessing',batch_size=5)(delayed(getseq_para)(ind,df,s,fname_seq_pep,tree_method,treeset) for ind in df.index)
    for ind in df.index:
        getseq_para(ind,df,s,fname_seq_pep,tree_method,treeset)
        getmsctree(os.path.join(fname_seq_pep,ind+'.aln'),ind,tree_method,tree_famsf,tree_fams,outd = fname_seq_pep)
    #Astral_infer(tree_famsf,gsmapf,fname_seq_pep)
    tree_famsf_concat,tree_fams_concat=[],{}
    for ind in df.index: concat_per_mlt(ind,df,fname_seq_pep,tree_method,treeset,mlt_ap,gene_sp_gl,gene_start,mlts_seq,sg_glsmap)
    sg_glsmapf = os.path.join(outdir,'gl_sp.map')
    writegsmap(sg_glsmap,sg_glsmapf)
    for mlt,seq_per_ap in mlts_seq.items(): writeconcatseq(mlt,seq_per_ap,fname_seq_pep_concat,tree_method,tree_famsf_concat,tree_fams_concat,treeset)
    Astral_infer(tree_famsf,gsmapf,fname_seq_pep)
    Astral_infer(tree_famsf_concat,sg_glsmapf,fname_seq_pep_concat)

def writeog(df,outdir,sp_name,gene_sp_gl):
    ap_per_mlt_counts = df.pivot_table(index = ['multiplicon'], aggfunc ='size').to_frame(name = 'num_ap')
    aps_mlts = {}
    num_ap_total = 0
    ap_mul_dfs = []
    mlt_ap = {}
    for mlt in ap_per_mlt_counts.index:
        num = ap_per_mlt_counts.loc[mlt,'num_ap']
        for i in range(num):
            num_ap_total = num_ap_total + 1
            label = "Multiplicon{0}_Ap{1}".format(mlt,i+1)
            mlt_ap[label] = mlt
            aps_mlts.update({label:{}})
    logging.info('In total {} anchor pairs are to be considered'.format(num_ap_total))
    group_mul = list(df.groupby('multiplicon'))
    mul_id = list(map(lambda x:x[0],group_mul))
    per_mul = list(map(lambda x:x[1],group_mul))
    for mid,df_mul in zip(mul_id,per_mul):
        for ap_ind,ind in enumerate(df_mul.index):
            genes = df_mul.loc[ind,'gene_xy']
            for gene in genes:
                sp_gl = gene_sp_gl[gene][1]
                if aps_mlts["Multiplicon{0}_Ap{1}".format(mid,ap_ind+1)].get(sp_gl) == None:
                    aps_mlts["Multiplicon{0}_Ap{1}".format(mid,ap_ind+1)][sp_gl] = gene
                else:
                    aps_mlts["Multiplicon{0}_Ap{1}".format(mid,ap_ind+1)][sp_gl] = ', '.join([aps_mlts["Multiplicon{0}_Ap{1}".format(mid,ap_ind+1)][sp_gl],gene])
    for ap_mul,genes in aps_mlts.items():
        ap_tmp = genes.copy()
        ap_tmp.update({'Multiplicon_Ap':[ap_mul]})
        ap_mul_df = pd.DataFrame.from_dict(ap_tmp).set_index('Multiplicon_Ap')
        ap_mul_dfs.append(ap_mul_df)
    assembled_df = pd.concat(ap_mul_dfs)
    fname = os.path.join(outdir,'Multiplicon_Ap.tsv')
    assembled_df.to_csv(fname,header=True,index=True,sep='\t')
    return assembled_df,mlt_ap
    #for leve,mid,genes in zip(df['level'],df['multiplicon'],df['gene_xy']):
    #    ap_num = ap_per_mlt_counts.loc[mid,'num_ap']
    #    table = {'multiplicon{0}_ap{1}'.format(mid,)}
    #    for gene in genes:
    #        sp = gene_sp_gl[gene][0]
    #        if table.get(sp) == None: table[sp] = 

def segmentsaps(genetable,listsegments,anchorpoints,segments,outdir,seqs,nthreads,tree_method,treeset,minimum_portion):
    Ratios={}
    num_sp = len(seqs)
    sp_name = []
    gsmap = {}
    for s in seqs: gsmap.update({gid:s.prefix for gid in s.idmap.keys()})
    gsmapf = os.path.join(outdir, "Gene_Species.Map")
    writegsmap(gsmap,gsmapf)
    for s in seqs: sp_name.append(s.prefix)
    for s in seqs[1:]: seqs[0].merge_seqs(s)
    seqs = seqs[0]
    ap = pd.read_csv(anchorpoints, sep="\t", index_col=None, header=0)
    ap = ap.loc[:,['multiplicon','gene_x','gene_y']]
    genetable = pd.read_csv(genetable, sep=",", index_col=None, header=0)
    gene_sp_gl = {gene:(sp,'{0}_{1}'.format(sp,gl)) for gene,sp,gl in zip(genetable['gene'],genetable['species'],genetable['scaffold'])}
    gene_start = {gene:start for gene,start in zip(genetable['gene'],genetable['start'])}
    ap['sp_x'] = [gene_sp_gl[gx][0] for gx in ap['gene_x']]
    ap['gl_x'] = [gene_sp_gl[gx][1] for gx in ap['gene_x']]
    ap['sp_y'] = [gene_sp_gl[gy][0] for gy in ap['gene_y']]
    ap['gl_y'] = [gene_sp_gl[gy][1] for gy in ap['gene_y']]
    #ap_genelist = ap.merge(genetable.loc[:,['gene','species','scaffold']],left_on = 'gene_x',right_on = 'gene')
    #g_x_y_inorder = [[x,y] for x,y in zip(df['gene_x'],df['gene_y'])]
    #gxy = list(set(list(df['gene_x']) + list(df['gene_y'])))
    #df_gxy = pd.DataFrame(gxy, columns = ['anchors'])
    #le = pd.read_csv(listsegments, sep="\t", index_col=0)
    #le = le.merge(df_gxy,left_on = 'gene',right_on = 'anchors')
    #segs_anchors = le.groupby('segment')['anchors'].aggregate(lambda x: ", ".join([i for i in x])).to_frame()
    #segs_orders = le.groupby('segment')['position'].aggregate(lambda x: ", ".join([str(i) for i in x])).to_frame()
    df = pd.read_csv(segments, sep="\t", index_col=0)
    df['segment'] = df.index
    counted = df.groupby(["multiplicon", "genome"])["segment"].aggregate(lambda x: len(set(x)))
    df = df.loc[:,['multiplicon','segment','genome']]
    profile = counted.unstack(level=-1).fillna(0)
    text,MP_matrix,profile = Allratio(profile,Ratios)
    MP_matrix_array = np.transpose(MP_matrix)
    hierarchy_dendrogram(MP_matrix_array,text.split(':'),outdir)
    hierarchy_dendrogram(MP_matrix_array,text.split(':'),outdir,label=False)
    #profile = profile.loc[:,text]
    #minimum_portion = 0.75
    species_ratio = []
    for i in profile.index: species_ratio.append(sum(map(lambda x:int(x)!=0,profile.loc[i,profile.columns[:-3]]))/num_sp)
    profile.loc[:,'species_ratio'] = species_ratio
    #print(profile['species_ratio'])
    profile = profile[profile['species_ratio']>=minimum_portion]
    #for sp in profile.columns[:-3]: profile = profile[profile[sp].astype('int')>0]
    ap_filtered = ap.merge(profile.reset_index(),on='multiplicon')
    dfs_coc = search_shared_aps(ap_filtered,num_sp,gene_sp_gl,minimum_portion)
    fname_seq = _mkdir(os.path.join(outdir, "Multiplicons_Sequences"))
    assembled_df,mlt_ap = writeog(dfs_coc,outdir,sp_name,gene_sp_gl)
    writemscseq(assembled_df,fname_seq,seqs,nthreads,tree_method,treeset,gsmapf,outdir,mlt_ap,gene_sp_gl,gene_start)
    mlts_segs_anchors = segs_anchors.join(df.set_index('segment'))
    mlts_segs_anchors_ratios = mlts_segs_anchors.reset_index().set_index('multiplicon').join(profile).reset_index().set_index('segment')
    fname_msar = os.path.join(outdir, "Mlts_Segs_Ancs_Ratios.tsv")
    mlts_segs_anchors_ratios.to_csv(fname_msar,header = True,index =True,sep = '\t')
    MFs_coc,MFs_order_coc = msap2mf(mlts_segs_anchors,g_x_y_inorder)
    #MFs_coc,MFs_order_coc = msap2mf(mlts_segs_anchors,segs_orders)
    MF_fname = os.path.join(outdir, "Multiplicon_Families.tsv")
    MFs_coc.to_csv(MF_fname,header = True,index =True,sep = '\t')
    MF_order_fname = os.path.join(outdir, "Multiplicon_Families_Order.tsv")
    MFs_order_coc.to_csv(MF_order_fname,header = True,index =True,sep = '\t')
    fname_seq = _mkdir(os.path.join(outdir, "Multiplicons_Sequences"))
    #fcs,fps,findex,gsmap,CDS = Concat_by_order(MFs_coc,MFs_order_coc,seqs,fname_seq)
    findex,gsmap = Concat_by_order(MFs_coc,MFs_order_coc,seqs,fname_seq,nthreads)
    gsmapf = os.path.join(outdir, "Genomes_Species.Map")
    writegsmap(gsmap,gsmapf)
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(run_tree_msc)(fpaln,tree_method,treeset) for fpaln in findex)
    #outd = _mkdir(os.path.join(fname_seq, "pep"))
    #Parallel(n_jobs=nthreads,backend='multiprocessing',verbose=11,batch_size=1000)(delayed(aln_2tree)(fps[i],fcs[i],seqs,tree_method,treeset,outd,findex[i],CDS) for i in range(len(fcs)))
    #for i in range(len(fcs)):
    #    aln_2tree(fps[i],fcs[i],seqs,tree_method,treeset,outd,findex[i],CDS)
    tree_famsf,tree_fams=[],{}
    #map(getMultipliconstrees(fps[i],fcs[i],findex[i],outd,tree_method,tree_famsf,tree_fams),range(len(fcs)))
    #for i in range(len(fcs)): getMultipliconstrees(fps[i],fcs[i],findex[i],outd,tree_method,tree_famsf,tree_fams)
    for i in range(len(findex)): getMultipliconstrees(findex[i],tree_method,tree_famsf,tree_fams)
    Astral_infer(tree_famsf,gsmapf,outdir)
    # segs_anchors indexed by segment, only one column as anchors

def bget_seq(s, fid, gene, tmp_pathc, tmp_pathp):
    if not s.prot: fc = os.path.join(tmp_pathc,'{}.cds'.format(fid))
    fp = os.path.join(tmp_pathp,'{}.pep'.format(fid))
    for ge in gene:
        if ge != '':
            if not s.prot:
                with open (fc,'a') as f: f.write(">{0}\n{1}\n".format(ge,s.cds_sequence[s.idmap[ge]]))
            with open (fp,'a') as f: f.write(">{0}\n{1}\n".format(ge,s.pro_sequence[s.idmap[ge]]))

def bgetseq(df,outdir,s,nthreads):
    for i,seq in enumerate(s):
        if i != 0: s[0].merge_seqs(seq)
    tmp_path = _mkdir(os.path.join(outdir,'Sequences'))
    if not s[0].prot: tmp_pathc = _mkdir(os.path.join(tmp_path,'cds'))
    else: tmp_pathc = os.path.join(tmp_path,'cds')
    tmp_pathp = _mkdir(os.path.join(tmp_path,'pep'))
    genes = map(lambda x:list(df.loc[x,:]),df.index)
    Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(bget_seq)(s[0], fid, gene, tmp_pathc, tmp_pathp) for fid, gene in zip(df.index,genes))

def bsog(s,buscohmm,outdir,eval,nthreads,buscocutoff):
    outs = hmmerscan(outdir,s,buscohmm,eval,nthreads)
    fn = os.path.join(outdir,'BUSCO_fam.tsv')
    df = modifydf(fn,outs,outdir,'',bhmm = True, cutoff = buscocutoff)
    bgetseq(df,outdir,s,nthreads)
    mvassignf(outdir,bhmm = True)

#def calculatekstree(fams,s,spgenemap):


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
        if self.codeml_results is not None: self.codeml_results = pd.concat([self.codeml_results, self.no_codeml_results])
        else: self.codeml_results = self.no_codeml_results
    
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
        elif self.aligner == "muscle":
            self.run_muscle(options=self.aln_options)
        elif self.aligner == "prank":
            self.run_prank(options=self.aln_options)
        else:
            logging.error("Unsupported aligner {}".format(self.aligner))
        self.get_codon_alignment()

    def run_mafft(self, options="--auto"):
        cmd = ["mafft"] + demension(options.split(',')) + ["--amino", self.pro_fasta]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        with open(self.pro_alnf, 'w') as f: f.write(out.stdout.decode('utf-8'))
        _log_process(out, program="mafft")
        self.pro_aln = AlignIO.read(self.pro_alnf, "fasta")

    def run_muscle(self, options="--auto"):
        if options == "--auto": cmd = ["muscle"] + ['-in',self.pro_fasta] + ['-out',self.pro_alnf]
        else: cmd = ["muscle"] + ['-in',self.pro_fasta] + ['-out',self.pro_alnf] + demension(options.split(','))
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="muscle")
        self.pro_aln = AlignIO.read(self.pro_alnf, "fasta")

    def run_prank(self, options="--auto"):
        if options == "--auto": cmd = ["prank"] + ['-d='+self.pro_fasta] + ['-o='+self.pro_alnf]
        else: cmd = ["prank"] + ['-d='+self.pro_fasta] + ['-o='+self.pro_alnf] + demension(options.split(','))
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="prank")
        self.pro_alnf = self.pro_alnf+".best.fas"
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
            tree = self.run_fasttree(options=self.tree_options)
        self.tree = tree

    def run_iqtree(self, options="-m LG"):
        if options == None: cmd = ["iqtree", "-s", self.pro_alnf]
        else: cmd = ["iqtree", "-s", self.pro_alnf] + demension(options.split(','))
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="iqtree")
        return _process_unrooted_tree(self.pro_alnf + ".treefile",self.id)

    def run_fasttree(self, options=None):
        tree_pth = self.pro_alnf + ".nw"
        if options == None: cmd = ["FastTree", '-out', tree_pth, self.pro_alnf]
        else: cmd = ["FastTree"] + demension(options) + ['-out', tree_pth, self.pro_alnf]
        out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        _log_process(out, program="fasttree")
        return _process_unrooted_tree(self.pro_alnf + ".nw",self.id)

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
    family.codeml_results.to_csv(family.out,header=True,index=False)

class KsDistributionBuilder:
    def __init__(self, gene_families, seqs, n_threads=4):
        self.families = gene_families
        self.df = None
        self.seqs = seqs
        self.n_threads = n_threads

    def get_distribution(self):
        if self.n_threads < len(self.families): logging.info("{} threads are used for {} gene families\nNote that adding threads can significantly accelerate the Ks estimation process".format(int(self.n_threads),int(len(self.families))))
        Parallel(n_jobs=self.n_threads,backend='multiprocessing')(
            delayed(_get_ks)(family) for family in self.families)
        df = pd.concat([pd.read_csv(x.out, index_col=None) 
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

