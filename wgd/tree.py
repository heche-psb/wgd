import pandas as pd
from Bio import SeqIO
from Bio.Data.CodonTable import TranslationError
import logging
from collections import Counter
import os
from joblib import Parallel, delayed
from tqdm import tqdm,trange
import subprocess as sp
from Bio import AlignIO
from Bio.Alphabet import IUPAC
from Bio import Phylo
from ete3 import PhyloTree

def _mkdir(dirname):
    if not os.path.isdir(dirname) :
        os.mkdir(dirname)
    return dirname

def fetcher(config):
    """
    Return the essential info in dict
    """
    para_dict = {}
    getinfo = lambda x : (x[0],x[1].strip())
    with open(config,'r') as f:
        for line in f.readlines():
            key,value = getinfo(line.split('\t')[0:2])
            para_dict[key] = value
    return para_dict

def reporter(dic):
    """
    Log out the info
    """
    for key,value in dic.items(): logging.info("{0}\t{1}".format(key,value))

def listdir(data,align=False):
    """
    Return the path to sequence files in a list
    """
    parent = os.getcwd()
    os.chdir(data)
    y = lambda x:os.path.join(data,x)
    files_clean = [y(i) for i in os.listdir() if i!="__pycache__"]
    if not align:
        logging.info("In total {} sequence files are found".format(len(files_clean)))
        logging.info(", ".join([os.path.basename(i) for i in files_clean]))
    os.chdir(parent)
    return files_clean

def write_seq(fam,og,seq,seqtype):
    """
    og is a series
    """
    og = og.dropna()
    with open(fam+'.'+seqtype,'w') as f:
        for sp,gids in og.items():
            for gid in gids.split(', '): f.write('>{0}\n{1}\n'.format(gid,seq[sp][gid].seq))

def write_seq_translate(fam,og,seq,to_stop,cds):
    """
    og is a series
    """
    og = og.dropna()
    with open(fam+'.'+'pep','w') as f:
        for sp,gids in og.items():
            for gid in gids.split(', '): f.write('>{0}\n{1}\n'.format(gid,seq[sp][gid].translate(to_stop=to_stop,cds=cds,id=gid).seq))

def write_seq_translate_fammode(fam,fseq,to_stop,cds,pepdir):
    """
    original seq translation
    """
    with open(os.path.join(pepdir,fam+'.'+'pep'),'w') as f:
        for record in SeqIO.parse(fseq, 'fasta'):
            f.write(">{0}\n{1}\n".format(record.id,record.translate(to_stop=to_stop,cds=cds,id=record.id).seq))

def find_singleton(og):
    """
    return True or False
    """
    og = og.dropna()
    num = 0
    for seqs in og.values:
        num += len(seqs.split(', '))
        if num > 1:
            return False
    return True

def deal_options(options):
    Options = []
    for i in options:
        if len(i.split(' '))==1: Options.append(i)
        else: Options+=i.split(' ')
    return Options

def mafft(fseq,options,faln):
    if options == '': cmd = ["mafft"] + [fseq]
    else: cmd = ["mafft"] + options + [fseq]
    out = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open(faln, 'w') as f: f.write(out.stdout.decode('utf-8'))

def muscle(fseq,options,faln):
    if options == '': cmd = ["muscle"] + ['-in',fseq] + ['-out',faln]
    else: cmd = ["muscle"] + ['-in',fseq] + ['-out',faln] + options
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

def prank(fseq,options,faln):
    if options == '': cmd = ["prank"] + ['-d='+fseq] + ['-o='+faln]
    else: cmd = ["prank"] + ['-d='+fseq] + ['-o='+faln] + options
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

def iqtree(faln,options):
    cmd = ["iqtree", "-s", faln] + options
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    if os.path.exists(faln+".treefile"): return faln+".treefile"

def iqtree2(faln,options):
    cmd = ["iqtree2", "-s", faln] + options
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    if os.path.exists(faln+".treefile"): return faln+".treefile"

def fasttree(faln,options):
    cmd = ["FastTree"] + options + ["-out", faln+'.FastTree', faln]
    sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    if os.path.exists(faln+".FastTree"): return faln+'.FastTree'

def mrbayes(faln,options):
    faln_nex = faln+".nexus"
    AlignIO.convert(faln, 'fasta', faln_nex, 'nexus', IUPAC.extended_protein)
    conf =  faln_nex+".config.mb"
    logf = faln_nex+".mb.log"
    bashf = faln_nex+".bash.mb"
    with open(conf,"w") as f:
        if 'set' in options: f.write('set'+' '+' '.join(options['set'])+'\n')
        else: f.write("set autoclose=yes nowarn=yes\n")
        f.write("execute {}\n".format(os.path.basename(faln_nex)))
        if 'prset' in options: f.write('prset'+' '+' '.join(options['prset'])+'\n')
        else: f.write("prset ratepr=variable\n")
        if 'lset' in options: f.write('lset'+' '+' '.join(options['lset'])+'\n')
        else: f.write("lset rates=gamma\n")
        if 'mcmcp' in options: f.write('mcmcp'+' '+' '.join(options['mcmcp'])+'\n')
        else: f.write("mcmcp diagnfreq=100 samplefreq=10\n")
        if 'mcmc' in options: f.write('mcmc'+' '+' '.join(options['mcmc'])+'\n')
        else: f.write("mcmc ngen=1100 savebrlens=yes nchains=1\n")
        f.write("sumt\nsump\nquit\n")
    with open(bashf,"w") as f:
        f.write('mb <{0}> {1}'.format(conf,logf))
    mb_cmd = ["sh", bashf]
    sp.run(mb_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open('rm.sh',"w") as f: f.write("rm *.bash.mb")
    rm_cmd = ['sh', 'rm.sh']
    sp.run(rm_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    rm_cmd = ['rm', 'rm.sh']
    sp.run(rm_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    if os.path.exists(faln+".nexus.con.tre"): return os.path.abspath(faln+".nexus.con.tre")

def labelinternalnodes(tree):
    Tree = Phylo.read(tree,'newick')
    clade_map = {}
    for indice,clade in enumerate(Tree.get_nonterminals()):
        clade.confidence = None
        clade.name = "N" + str(indice)
        clade_map[clade.name] = []
        for tip in clade.get_terminals(): clade_map[clade.name].append(tip.name)
    Phylo.write(Tree,tree+'.labelled','newick')
    with open(tree+".nodemap","w") as f:
        for key,value in clade_map.items(): f.write(key+"\t"+", ".join(value)+"\n")
        for clade in Tree.get_terminals(): f.write(clade.name+"\t"+clade.name+"\n")
    return tree+".nodemap"

def intoffloat(Ixx):
    if all([i.confidence==1 for i in Ixx if i.confidence is not None]):
        return "FT"
    else:
        Float = False
        for i in Ixx:
            if type(i.confidence) is float:
                Float = True
                break
        if Float: return "FT"
        else: return "IQ"

def getbssubtree(tree,threshold):
    Ixx = tree.get_nonterminals()
    treetp = intoffloat(Ixx)
    Threshold = threshold if treetp == "FT" else threshold*100
    for i in Ixx:
        if i.confidence is not None:
            if i.confidence < Threshold:
                for j in i.get_terminals():
                    if tree.get_path(j, terminal=True):
                        tree.prune(j)
    return tree

def prunepolytomy(genetree):
    for i in genetree.get_nonterminals():
        if len(i.clades) > 2:
            for j in i.get_terminals():
                if genetree.get_path(j, terminal=True): genetree.prune(j)
    return genetree

def mb2nw(tree):
    mb_out_content,linenumber = [],0
    with open(tree,"r") as f:
        lines = f.readlines()
        for line in lines:
            linenumber += 1
            mb_out_content.append(line.strip(' ').strip('\t').strip('\n').strip(','))
    linenumber = int((linenumber - 12)/2+3)
    mb_useful = mb_out_content[-linenumber:-1]
    mb_id = mb_useful[:-2]
    mb_tree = mb_useful[-1]
    mb_id_dict = {}
    tree_pth = tree + ".nw"
    for i in mb_id:
        i = i.split("\t")
        mb_id_dict[i[0]]=i[1]
    with open(tree_pth,'w') as f:
        for (k,v) in mb_id_dict.items(): mb_tree = mb_tree.replace('{}[&prob='.format(k),'{}[&prob='.format(v))
        f.write(mb_tree.replace('tree con_50_majrule = [&U]',''))
    Tree = Phylo.read(tree_pth,'newick')
    for i in Tree.get_terminals(): i.comment = None
    for i in Tree.get_nonterminals(): i.comment = None
    Phylo.write(Tree,tree_pth,'newick')
    return tree_pth

def relabeltree_ete3(tree,gsmap,sptree,threshold,ismb):
    """
    gsmap is already dict
    """
    if ismb: tree = mb2nw(tree)
    genetree = Phylo.read(tree,'newick')
    if not genetree.rooted: genetree.root_at_midpoint()
    if threshold is not None: genetree = getbssubtree(genetree,threshold)
    genetree = prunepolytomy(genetree)
    safeID_origID = {}
    y = lambda x:x.strip()
    for indice,clade in enumerate(genetree.get_terminals()):
        if clade.name not in gsmap:
            logging.error("{} is not in the gsmap!".format(clade.name))
            exit(1)
        tmp = gsmap[clade.name]+'_'+str(indice)
        safeID_origID[tmp] = clade.name
        clade.name = tmp
        clade.comment = None
    Phylo.write(genetree,tree+'.labelled','newick')
    return safeID_origID,genetree,tree+'.labelled'

def recon_ete3(tree,gsmap,sptree,threshold,Nodemap,ismb,outdir):
    safeID_origID, Tree, tree_fn = relabeltree_ete3(tree,gsmap,sptree,threshold,ismb)
    y = lambda x: "_".join(x.split("_")[:-1])
    for clade in Tree.get_terminals():
        clade.comment = "&&NHX:S={}:D=N".format(y(clade.name))
    for clade in Tree.get_nonterminals():
        tips = set([y(i.name) for i in clade.get_terminals()])
        candidatenodes = {n:len(s) for n,s in Nodemap.items() if (tips-s) == (s-tips)}
        if len(candidatenodes) == 0:
            candidatenodes = {n:len(s) for n,s in Nodemap.items() if len(s-tips)>0 and len(s) > len(tips)}
        n,s = sorted(candidatenodes.items(),key=lambda x:x[1])[0]
        clade.comment = "&&NHX:S={}".format(n)
    with open(tree_fn,'r') as f: gt_content = f.read().strip("\n").strip().replace(":0.00000","")
    with open(sptree,'r') as f: st_content = f.read().strip("\n").strip()
    genetree = PhyloTree(gt_content)
    sptree = PhyloTree(st_content)
    recon_tree, events = genetree.reconcile(sptree)
    Num_S,Num_D,S_Node,D_Node=0,0,[],[]
    for event in events:
        if event.etype == 'S':
            mrca = Tree.common_ancestor(*[event.inparalogs+event.orthologs])
            S_Node.append(mrca.comment.replace("&&NHX:S=","").replace(":D=N","").replace(":D=Y",""))
            mrca.comment = mrca.comment+":D=N"
            Num_S += 1
        elif event.etype == 'D':
            mrca = Tree.common_ancestor(*[event.inparalogs+event.outparalogs])
            D_Node.append(mrca.comment.replace("&&NHX:S=","").replace(":D=N","").replace(":D=Y",""))
            mrca.comment = mrca.comment+":D=Y"
            Num_D += 1
    Phylo.write(Tree,tree+'.nhx','newick')
    #with open(tree+'_clean.nhx','w') as f:
    #    with open(tree+'.nhx') as ff:
    #        cont = ff.read()
    #        f.write(cont.replace(":0.00000",""))
    for clade in Tree.get_terminals(): clade.name = safeID_origID[clade.name]
    agora_dir = _mkdir(os.path.join(outdir,"AGORA_PREP"))
    yy = lambda x:os.path.join(agora_dir,os.path.basename(x))
    Phylo.write(Tree,yy(tree)+'_originalID.nhx','newick')
    #with open(tree+'_clean_originalID.nhx','w') as f:
    #    with open(tree+'_originalID.nhx') as ff:
    #        cont = ff.read()
    #        f.write(cont.replace(":0.00000",""))
    logging.info("In total {} speciation and {} duplication events were reconciled".format(Num_S,Num_D))
    y = lambda x:[i.replace(":D=N","").replace(":D=Y","") for i in x]
    if len(S_Node)!= 0: logging.info("Speciation event at Node {}".format(", ".join(y(S_Node))))
    if len(D_Node)!= 0: logging.info("Duplication event at Node {}".format(", ".join(y(D_Node))))

def checknode(WGDnodes,genetree,gsmap):
    gt = Phylo.read(genetree,'newick')
    for i in gt.get_terminals(): i.name = gsmap[i.name]
    mrca = gt.common_ancestor(WGDnodes)
    if len(mrca.get_terminals()) > 2:
        return False
    else:
        return True

def checkwholetree(sptree,genetree):
    with open(sptree,'r') as f: sp = f.read().strip()
    with open(genetree,'r') as f: gt = f.read().strip()
    tree1 = PhyloTree(sp)
    tree2 = PhyloTree(gt)
    rf_distance = tree1.robinson_foulds(tree2)
    print(rf_distance[0])

class Config_Hauler:
    """
    Fetch the info in the config file and implement corresponding analysis
    """
    def __init__(self,config):
        logging.info("Fetching infomation from config file")
        self.para = fetcher(config)
        self.outdir = self.para['Outdir:']
        self.OG = pd.read_csv(self.para['Orthogroup path:'],header=0,index_col=0,sep='\t')
        self.threads,self.seqtype = int(self.para['Number of threads:']),self.para['Sequences type:']
        self.data = self.para['Sequences directory:']
        self.sptree = self.para['Species tree:']
        self.wgdcheck = self.para['Check WGDnode:']
        self.ismb = True if self.para['Tree algorithm:'] == 'mrbayes' else False
        reporter(self.para)
        _mkdir(self.outdir)
        if self.para['Sequences form:'] == 'species':
            self.read_seq()
            logging.info("Writing sequences per family")
            self.write_famseq()
        else:
            self.getfam()
        logging.info("Aligning each family")
        self.aligning()
        logging.info("Inferring gene tree per family")
        self.genetree()
        if self.wgdcheck == 'yes':
            self.checkWGDnode()
        if self.para['Reconciliation:'] == 'yes':
            self.reconciliation()

    def read_seq(self):
        seq_paths = listdir(self.data)
        self.gsmap,self.SEQ = {},{}
        for seq in seq_paths:
            self.SEQ[os.path.basename(seq)] = {}
            for record in tqdm(SeqIO.parse(seq, 'fasta'),desc="Reading {}".format(os.path.basename(seq)),unit=" sequences"):
                if not (self.gsmap.get(record.id) is None):
                    logging.error("Duplicated gene id found in {}".format(os.path.basename(seq)))
                    exit(1)
                else:
                    self.gsmap[record.id] = os.path.basename(seq)
                    self.SEQ[os.path.basename(seq)][record.id] = record

    def write_famseq(self):
        OGSEQ_dir = _mkdir(os.path.join(self.outdir,"OG_SEQ"))
        parent = os.getcwd()
        os.chdir(OGSEQ_dir)
        self.Fam_list = Fam_list = list(self.OG.index)
        self.Fam_path = {i:os.path.join(OGSEQ_dir,i+'.'+self.seqtype) for i in Fam_list}
        self.SOGs = list(self.OG.index[[find_singleton(self.OG.loc[fam,:]) for fam in Fam_list]])
        self.NonSOGs = list(set(Fam_list)-set(self.SOGs))
        if len(set(self.OG.index)) != len(Fam_list):
            element_freq = Counter(Fam_list)
            duplicated = [element for element,freq in element_freq.items() if freq != 1]
            logging.error("Duplicated gene family id found for {}".format(", ".join(duplicated)))
            exit(1)
        Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(write_seq)(Fam_list[i],self.OG.loc[Fam_list[i],:],self.SEQ,self.seqtype) for i in trange(len(Fam_list)))
        os.chdir(parent)
        if self.para['Translation:'] == 'yes':
            logging.info("Writing translated sequences per family")
            OGSEQPEP_dir = _mkdir(os.path.join(self.outdir,"OG_SEQ_TRANSLATE"))
            os.chdir(OGSEQPEP_dir)
            to_stop,cds = self.para['Translation parameters:'].split(',') 
            Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(write_seq_translate)(Fam_list[i],self.OG.loc[Fam_list[i],:],self.SEQ,to_stop,cds) for i in trange(len(Fam_list)))
            self.Fam_path = {i:os.path.join(OGSEQPEP_dir,i+'.pep') for i in Fam_list}
            os.chdir(parent)

    def getfam(self):
        self.Fam_list = Fam_list = list(self.OG.index)
        self.SOGs = list(self.OG.index[[find_singleton(self.OG.loc[fam,:]) for fam in Fam_list]])
        self.NonSOGs = list(set(Fam_list)-set(self.SOGs))
        self.Fam_path = {fam:path for fam,path in zip(sorted(Fam_list),listdir(self.data,align=True))}
        if self.para['Translation:'] == 'yes' and self.para['Sequences type:'] == 'nucleotide':
            logging.info("Writing translated sequences per family")
            OGSEQPEP_dir = _mkdir(os.path.join(self.outdir,"OG_SEQ_TRANSLATE"))
            to_stop,cds = self.para['Translation parameters:'].split(',')
            Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(write_seq_translate_fammode)(Fam_list[i],self.Fam_path[Fam_list[i]],to_stop,cds,OGSEQPEP_dir) for i in trange(len(Fam_list)))
            self.Fam_path = {i:os.path.join(OGSEQPEP_dir,i+'.pep') for i in Fam_list}

    def aligning(self):
        if self.para['Aligner parameters:'] == 'default': options = ''
        else:
            y = lambda x:[i.strip() for i in x]
            options = deal_options(y(self.para['Aligner parameters:'].split(',')))
        self.OGALIGN_dir = OGALIGN_dir = _mkdir(os.path.join(self.outdir,"OG_ALIGNMENT_TREE"))
        if self.para['Aligner:'] == 'mafft':
            z = lambda x:os.path.join(OGALIGN_dir,x+".mafft")
            Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(mafft)(self.Fam_path[self.NonSOGs[i]],options,z(self.NonSOGs[i])) for i in trange(len(self.NonSOGs)))
        elif self.para['Aligner:'] == 'muscle':
            z = lambda x:os.path.join(OGALIGN_dir,x+".muscle")
            Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(muscle)(self.Fam_path[self.NonSOGs[i]],options,z(self.NonSOGs[i])) for i in trange(len(self.NonSOGs)))
        elif self.para['Aligner:'] == 'prank':
            z = lambda x:os.path.join(OGALIGN_dir,x+".prank")
            Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(prank)(self.Fam_path[self.NonSOGs[i]],options,z(self.NonSOGs[i])) for i in trange(len(self.NonSOGs)))
        self.Aln_Path = {fam:path for fam,path in zip(sorted(self.NonSOGs),listdir(OGALIGN_dir,align=True))}

    def genetree(self):
        y = lambda x:[i.strip() for i in x]
        if self.para['Tree algorithm parameters:'] == 'default': options = []
        elif self.para['Tree algorithm:'] == 'mrbayes':
            options = {}
            for p in y(self.para['Tree algorithm parameters:'].split(',')):
                if options.get(p.split(' ')[0]) is None:
                    options[p.split(' ')[0]] = p.split(' ')[1:]
                else:
                    options[p.split(' ')[0]] = options[p.split(' ')[0]]+p.split(' ')[1:]
        else:
            options = deal_options(y(self.para['Tree algorithm parameters:'].split(',')))
        keys = sorted(self.Aln_Path.keys())
        values = {i:self.Aln_Path[j] for i,j in enumerate(keys)}
        if self.para['Tree algorithm:'] == 'iqtree':
            treepaths = Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(iqtree)(values[i],options) for i in trange(len(keys)))
        if self.para['Tree algorithm:'] == 'iqtree2':
            treepaths = Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(iqtree2)(values[i],options) for i in trange(len(keys)))
        if self.para['Tree algorithm:'] == 'fasttree':
            treepaths = Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(fasttree)(values[i],options) for i in trange(len(keys)))
        if self.para['Tree algorithm:'] == 'mrbayes':
            parent = os.getcwd()
            os.chdir(self.OGALIGN_dir)
            treepaths = Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(mrbayes)(os.path.basename(values[i]),options) for i in trange(len(keys)))
            os.chdir(parent)
        self.treepaths = [i for i in treepaths if not i is None]

    def reconciliation(self):
        if not os.path.exists(self.sptree):
            logging.error("Please provide a species tree for reconciliation!")
            exit(1)
        df = pd.read_csv(labelinternalnodes(self.sptree),header=None,index_col=None,sep='\t')
        if self.para['Branch supportness filter:'] == '0': self.threshold = None
        elif float(self.para['Branch supportness filter:']) > 1: self.threshold = float(self.para['Branch supportness filter:'])/100
        else: self.threshold = float(self.para['Branch supportness filter:'])
        self.Nodemap = {n:set(s.split(", ")) for n,s in zip(df[0],df[1])}
        if self.para['Reconciler:'] == 'ete3':
            Parallel(n_jobs=self.threads,backend='multiprocessing')(delayed(recon_ete3)(self.treepaths[i],self.gsmap,self.sptree,self.threshold,self.Nodemap,self.ismb,self.outdir) for i in trange(len(self.treepaths)))

    def checkWGDnode(self):
        sptree = Phylo.read(self.sptree,'newick')
        WGD_nodes,GoodFam,BadFam = [],[],[]
        for clade in sptree.get_terminals():
            if clade.name.endswith('_ap1') or clade.name.endswith('_ap2'): WGD_nodes.append(clade.name)
        for i in self.treepaths:
            if self.ismb: i = mb2nw(i)
            #checkwholetree(self.para['Species tree:'],i)
            if checknode(WGD_nodes,i,self.gsmap): GoodFam.append(i)
            else: BadFam.append(i)
        logging.info("In total {} families passed the topology test while {} failed".format(len(GoodFam),len(BadFam)))
        Passed_path = os.path.join(self.outdir,os.path.basename(self.para['Orthogroup path:'])+'.Passed')
        Failed_path = os.path.join(self.outdir,os.path.basename(self.para['Orthogroup path:'])+'.Failed')
        y = lambda x: os.path.basename(x).replace('.treefile','').replace('.FastTree','').replace('.nexus.con.tre.nw','').replace('.muscle','').replace('.mafft','').replace('.prank','').replace('.nexus.con.tre','')
        self.OG.loc[[y(i) for i in GoodFam]].to_csv(Passed_path,header=True,index=True,sep='\t')
        self.OG.loc[[y(i) for i in BadFam]].to_csv(Failed_path,header=True,index=True,sep='\t')


