import os
import logging
import tempfile
import pandas as pd
import numpy as np
import subprocess as sp
from collections import defaultdict
from operator import itemgetter
from wgd.peak import formatv2
from wgd.viz import apply_filters

gff_header = ["gene", "scaffold", "start", "orientation"] 

# we construct first a table with each row a gene containing it's family,
# scaffold, location and orientation
def make_gene_table(gffs, families, feature, attribute, additionalgffinfo):
    """
    Construct a table from a bunch of gff files and gene families such that all
    information for synteny and co-linearity related analyses is in one place.
    
    :param gffs: a list of GFF *file names*
    :param families: a pandas data frame with the gene families
    :param feature: feature tag for the GFF files
    :param attribute: attribute tag for parsing out gene IDs from last column
    
    Note: currently we assume the same attribute/feature is required for all
    GFF files, and the burden is on the user to make sure this is the case. We
    may wish, for flexibility, to allow a list of feature/attribute arguments
    in the future, one for each GFF file.
    """
    if not (additionalgffinfo is None) and len(additionalgffinfo) > 0:
        # ignore the info in the original feature and attribute
        features, attributes = [i.split(";")[0].strip() for i in additionalgffinfo], [i.split(";")[1].strip() for i in additionalgffinfo]
        if len(features) != len(attributes):
            logging.error("Please give the additional feature and attribute info of gff in the format of (feature;attribute)")
            exit(1)
        if len(features) != len(gffs):
            logging.error("Please give the same number and order of gff files and its associated additional feature and attribute info")
            exit(1)
        gfftables = [gff2table(gff, f, a) for gff,f,a in zip(gffs,features,attributes)]
    else:
        gfftables = [gff2table(gff, feature, attribute) for gff in gffs]
    familytable = gene2family(families)
    df = pd.concat(gfftables)
    df = familytable.join(df)
    return df

def getattr(s, attribute):
    for x in s.split(";"):
        y = x.split("=")
        if y[0] == attribute:
            return y[1].strip()
    return ""

def gff2table(gff, feature, attribute):
    """
    Read a GFF file to a pandas data frame, from a filename.
    """
    rows = []
    with open(gff, "r") as f:
        for l in f.readlines():
            if l.startswith("#") or l.strip("\n").strip()=='':
                continue
            x = l.strip("\n").strip("\t").split("\t")
            #Note here the empty lines from input will make error
            if x[2] == feature:
                a = getattr(x[-1], attribute)
                rows.append({"gene": a, "scaffold": x[0], "start": int(x[3]), "or": x[6]})
    df = pd.DataFrame.from_dict(rows).set_index("gene")
    return df

def gene2family(families):
    """
    Get a gene to family ID mapping from a gene families data frame in
    OrthoFinder format.
    """
    rows = []
    for fam in families.index:
        for sp in families.columns:
            x = families.loc[fam, sp]
            if type(x) != str:
                continue
            for gene in x.split(", "):
                rows.append({"gene": gene.strip(), 
                    "species": sp, "family": fam})
    df = pd.DataFrame.from_dict(rows).set_index("gene")
    return df

def configure_adhore(table, outdir, **kwargs):
    """
    Write out all required files for running I-ADHoRe.
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    famfile = os.path.join(outdir, "families.tsv")
    logging.info("Writing families file")
    write_families(famfile, table)
    logging.info("Writing gene lists")
    genelists = write_genelists(outdir, table)
    logging.info("Writing config file")
    config_file = os.path.join(outdir, "iadhore.conf")
    out_path = os.path.abspath(os.path.join(outdir, "iadhore-out"))
    write_config_adhore(config_file, out_path, genelists, os.path.abspath(famfile), **kwargs)
    return config_file, out_path

def write_families(fname, table):
    table["family"].to_csv(fname, sep="\t", header=None)

def write_genelists(outdir, table):
    genomes = {}
    for sp, df in table.groupby("species"):
        gdir = os.path.join(outdir, sp + ".lists")
        if not os.path.exists(gdir):
            os.mkdir(gdir)
        lists = {}
        for scaffold, sdf in df.groupby("scaffold"):
            if len(sdf.index) <= 2: continue
            fname = os.path.join(gdir, scaffold)
            with open(fname, "w") as o:
                if len(list(sdf.sort_values(by=["start"]).index)) != len(set(sdf.sort_values(by=["start"]).index)):
                    logging.error("There are duplicated gene IDs for given feature and attribute")
                    exit(1)
                for g in sdf.sort_values(by=["start"]).index:
                    o.write(g + sdf.loc[g,"or"] + "\n")
            lists[scaffold] = os.path.abspath(fname)
        genomes[sp] = lists
    return genomes

def write_config_adhore(
        config_file, output_path, genelists, families, gap_size=30,
        cluster_gap=35, q_value=0.75, prob_cutoff=0.01, anchor_points=3,
        alignment_method='gg2', level_2_only='false', table_type='family',
        multiple_hypothesis_correction='FDR', visualizeGHM='false',
        visualizeAlignment='false', number_of_threads=4):
    """
    Write out the config file for I-ADHoRe. See I-ADHoRe manual for information
    on parameter settings.

    :param gene_lists: directory with gene lists per chromosome
    :param families: file with gene to family mapping
    :param config_file_name: name for the config file
    :param output_path: output path name
    :param gap_size: see I-ADHoRe 3.0 documentation
    :param cluster_gap: see I-ADHoRe 3.0 documentation
    :param q_value: see I-ADHoRe 3.0 documentation
    :param prob_cutoff: see I-ADHoRe 3.0 documentation
    :param anchor_points: see I-ADHoRe 3.0 documentation
    :param alignment_method: see I-ADHoRe 3.0 documentation
    :param level_2_only: see I-ADHoRe 3.0 documentation
    :param table_type: see I-ADHoRe 3.0 documentation
    :param multiple_hypothesis_correction: see I-ADHoRe 3.0 documentation
    :param visualizeGHM: see I-ADHoRe 3.0 documentation
    :param visualizeAlignment: see I-ADHoRe 3.0 documentation
    :param number_of_threads: see I-ADHoRe 3.0 documentation
    :return: configuration file see I-ADHoRe 3.0 documentation
    """
    with open(config_file, 'w') as o:
        for k,v in genelists.items():
            o.write('genome= {}\n'.format(k))
            for scaffold, fname in v.items():
                o.write("{} {}\n".format(scaffold, fname))
            o.write("\n")

        o.write('blast_table= {}\n'.format(families))
        o.write('output_path= {}\n'.format(output_path))
        o.write('gap_size= {}\n'.format(gap_size))
        o.write('q_value= {}\n'.format(q_value))
        o.write('cluster_gap= {}\n'.format(cluster_gap))
        o.write('prob_cutoff= {}\n'.format(prob_cutoff))
        o.write('anchor_points= {}\n'.format(anchor_points))
        o.write('alignment_method= {}\n'.format(alignment_method))
        o.write('level_2_only= {}\n'.format(level_2_only))
        o.write('table_type= {}\n'.format(table_type))
        o.write('multiple_hypothesis_correction= {}\n'.format(
                multiple_hypothesis_correction))
        o.write('visualizeGHM= {}\n'.format(visualizeGHM))
        o.write('visualizeAlignment= {}\n'.format(visualizeAlignment))
        o.write('number_of_threads= {}\n'.format(number_of_threads))
    return os.path.abspath(output_path)

def run_adhore(config_file):
    """
    Run I-ADHoRe for a given config file

    :param config_file: path to I-ADHoRe configuration file
    """
    completed = sp.run(['i-adhore', config_file], stderr=sp.PIPE, stdout=sp.PIPE)
    logging.warning(completed.stderr.decode('utf-8'))
    logging.info(completed.stdout.decode('utf-8'))
    return

def get_anchors(out_path,userdf=None):
    if userdf!=None: anchors = pd.read_csv(userdf, sep="\t", index_col=0)
    else: anchors = pd.read_csv(os.path.join(out_path, "anchorpoints.txt"), sep="\t", index_col=0)
    if len(anchors) == 0:
        return None, None
    orig_anchors = anchors.copy()
    anchors["pair"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[0], x[1]])), axis=1)
    df = anchors[["pair", "multiplicon"]].drop_duplicates("pair").set_index("pair")
    #anchors["pair_reverse"] = anchors[["gene_x", "gene_y"]].apply(lambda x: "__".join(sorted([x[1], x[0]])), axis=1)
    #df_reverse = anchors[["pair_reverse", "multiplicon"]].drop_duplicates("pair_reverse").set_index("pair_reverse")
    #df_reverse.index.set_names('pair',inplace=True)
    # there are duplicates, due to anchors being in multiple multiplicons
    return df, orig_anchors

def get_multi(out_path,userdf2=None):
    if userdf2!=None: multi = pd.read_csv(userdf2, sep="\t", index_col=None,header = 0)
    else: multi = pd.read_csv(os.path.join(out_path, "multiplicons.txt"), sep="\t", index_col=None,header = 0)
    return multi

def get_anchor_ksd(ks_distribution, anchors):
    return ks_distribution.join(anchors).dropna()

def get_segments_profile(multi,keepredun,out_path,userdf3=None):
    if userdf3!=None: segs = pd.read_csv(userdf3, sep="\t", index_col=0)
    else: segs = pd.read_csv(os.path.join(out_path, "segments.txt"), sep="\t", index_col=0)
    if not keepredun:
        I3 = []
        Mul_to_rm = list(multi[multi['is_redundant']==-1].loc[:,'id'])
        Segs_to_rm = segs.loc[segs['multiplicon'].isin(Mul_to_rm),:]
        for i in Segs_to_rm.index: I3.append(i)
        segs = segs.drop(I3)
    segs["segment"] = segs.index
    return segs

def get_chrom_gene(table,outdir):
    df = table.reset_index()
    gdf = list(df.groupby("species"))
    scafs_genes,ordered_dfs,gene_orders = {},{},{}
    for sp, df in gdf:
        scafs_genes[sp] = {}
        for i in set(df['scaffold']): scafs_genes[sp][i] = []
        tmp = list(df.groupby(['scaffold'])[['gene','start']])
        for i in tmp:
            df_tmp = i[1].sort_values(by=['start'])
            for indice,j in enumerate(df_tmp['gene']):
                scafs_genes[sp][i[0]].append(j)
                if gene_orders.get(j) == None:
                    gene_orders[j] = indice+1 # since we want the gene starting from 1 instead of 0
        max_length = max(len(v) for v in scafs_genes[sp].values())
        for key in scafs_genes[sp].keys():
            if len(scafs_genes[sp][key]) < max_length:
                scafs_genes[sp][key].extend([None] * (max_length - len(scafs_genes[sp][key])))
        df_output = pd.DataFrame(scafs_genes[sp])
        ordered_df = df_output[sorted(df_output.columns,key=lambda y: len(df_output[y].dropna()),reverse=True)]
        fname = os.path.join(outdir, "{}_gene_order_perchrom.tsv".format(sp))
        ordered_df.insert(0, "Coordinates", np.arange(1,max_length+1))
        ordered_df.fillna('').to_csv(fname,header=True,index=False,sep='\t')
        ordered_dfs[sp] = ordered_df
    return ordered_dfs, gene_orders

def getunitMP(MP,gene_scaf):
    MP['scaffold_x'] = MP['gene_x'].apply(lambda x:gene_scaf[x])
    MP['scaffold_y'] = MP['gene_y'].apply(lambda x:gene_scaf[x])
    units = []
    for indice, sx, sy in zip(range(len(MP)),MP['scaffold_x'],MP['scaffold_y']):
        if indice == 0:
            unit = 0
            units.append(unit)
            sxy = (sx,sy)
            continue
        if (sx,sy) == sxy:
            units.append(unit)
            continue
        unit = unit + 1
        sxy = (sx,sy)
        units.append(unit)
    MP['unit'] = units
    return MP

def getorien(MP):
    orien = []
    for unit, df in list(MP.groupby('unit')):
        if len(df) == 1:
            orien.append("x")
            continue
        posix,negax,posiy,negay = 0,0,0,0
        for indice, corx, cory in zip(range(len(df)),df['coordinate_x'],df['coordinate_y']):
            if indice == 0:
                cx,cy = corx,cory
                continue
            if corx - cx > 0:
                posix = posix + 1
            if corx - cx < 0:
                negax = negax + 1
            if cory - cy > 0:
                posiy = posiy + 1
            if cory - cy < 0:
                negay = negay + 1
            cx,cy = corx,cory
        Judge = (posix - negax) * (posiy - negay)
        for i in range(len(df)):
            if Judge > 0:
                orien.append("+")
            if Judge < 0:
                orien.append("-")
            if Judge == 0:
                orien.append("x")
    MP['orientation'] = orien
    return MP

def get_mp_geneorder(gene_orders,out_path,outdir,table,userdf4=None):
    gene_scaf = {g:s for g,s in zip(table.index,table['scaffold'])}
    if userdf4 != None: MP = pd.read_csv(userdf4, sep="\t", index_col=0)
    else: MP = pd.read_csv(os.path.join(out_path, "multiplicon_pairs.txt"), sep="\t", index_col=0)
    MP = MP.rename(columns={"gene_y": "code"})
    MP = MP.rename(columns={"Unnamed: 2": "gene_x", "gene_x": "gene_y"})
    MP['coordinate_x'] = MP['gene_x'].apply(lambda x:gene_orders[x])
    MP['coordinate_y'] = MP['gene_y'].apply(lambda x:gene_orders[x])
    MP = getunitMP(MP,gene_scaf)
    MP = getorien(MP)
    fname = os.path.join(outdir, "multiplicon_pairs_coordinates_unit_orien.tsv")
    MP.to_csv(fname,header=True,index=True,sep='\t')
    return MP

def transformunit(segs,ordered_genes_perchrom_allsp,outdir):
    gene_order_dict_allsp = {}
    seg = segs.copy()
    splist = set(seg['genome'])
    for sp in splist:
        gene_order_dict_allsp[sp] = {}
        ordered_genes_perchrom = ordered_genes_perchrom_allsp[sp]
        ordered_genes_perchrom = ordered_genes_perchrom.set_index('Coordinates')
        for i in ordered_genes_perchrom.index:
            for scaf in ordered_genes_perchrom.columns:
                genename = ordered_genes_perchrom.loc[i,scaf]
                gene_order_dict_allsp[sp][genename] = i
    first_coordinate,last_coordinate = [],[]
    for i,j in zip(seg['genome'],seg['first']): first_coordinate.append(gene_order_dict_allsp[i][j])
    for i,j in zip(seg['genome'],seg['last']): last_coordinate.append(gene_order_dict_allsp[i][j])
    seg['first_coordinate'],seg['last_coordinate'] = first_coordinate,last_coordinate
    fname = os.path.join(outdir, "segments_coordinates.tsv")
    seg.to_csv(fname,header=True,index=True,sep='\t')
    return seg, gene_order_dict_allsp

def getsegks(segs_gene_unit,ks_distribution,ordered_genes_perchrom_allsp):
    ksdb_df = pd.read_csv(ks_distribution,header=0,index_col=0,sep='\t')
    ksdb_df = formatv2(ksdb_df)
    df = apply_filters(ksdb_df, [("dS", 0., 5.)])
    for indice in segs_gene_unit.index:
        sp,gl = segs_gene_unit[indice,'genome'],segs_gene_unit[indice,'list']
        coor_s, coor_l = segs_gene_unit[indice,'first_coordinate'],segs_gene_unit[indice,'last_coordinate']
        df_sp = ordered_genes_perchrom_allsp[sp]


def annotatelist(anchorpoints,table):
    ap_with_list = anchorpoints.copy()
    gene_list = {gene:li for gene,li in zip(table.index,table['scaffold'])}
    ap_with_list['list_x'] = ap_with_list['gene_x'].apply(lambda x:gene_list[x])
    ap_with_list['list_y'] = ap_with_list['gene_y'].apply(lambda x:gene_list[x])
    return ap_with_list

def annotatesegy(ap_with_list,segs):
    tmp = list(ap_with_list.groupby['multiplicon'])
    unique_segid = []
    for mlt, df in tmp:
        df = df.sort_values(by=['coord_y'])
        gl = list(df['list_y'])[0]
        starty,endy = list(df['gene_y'])[0], list(df['gene_y'])[-1] # ap should be inside the segment for sure
        seg_tmp = segs.loc[(segs['multiplicon']==mlt) & (segs['list']==gl),:]
        if len(seg_tmp) == 1:
            unique_segid.append(seg_tmp['segment'])
            continue

def getsegid(starty,endy,segs,gene_list,mlt,gene_orders):
    gls,gle = gene_list[starty],gene_list[endy]
    assert gls == gle
    tmp = segs.loc[(segs['multiplicon']==mlt) & (segs['list']==gls),:]
    if len(tmp) == 1:
        return list(tmp['segment'])[0]
    else:
        for segid,first,last in zip(tmp['segment'],tmp['first'],tmp['last']):
            f_coor,l_coor = gene_orders[first], gene_orders[last]
            if gene_orders[starty] >= f_coor and gene_orders[endy] <= l_coor:
                return segid

def getsegidx(df,segs,gene_list,mlt,gene_orders):
    df_tmp = df.copy()
    df_tmp['list_x'] = df_tmp['gene_x'].apply(lambda x:gene_list[x])
    df_tmp = df_tmp.sort_values(by=['list_x','coord_x'])

def annotatelist_full(anchorpoints,table,segs,gene_orders):
    ap_with_list = anchorpoints.copy()
    list_x,list_y = [],[]
    tmp = list(ap_with_list.groupby('multiplicon'))
    gene_list = {gene:li for gene,li in zip(table.index,table['scaffold'])}
    for mlt,df in tmp:
        df = df.sort_values(by=['coord_y'])
        starty,endy = list(df['gene_y'])[0], list(df['gene_y'])[-1]
        belonged_segidy = getsegid(starty,endy,segs,gene_list,mlt,gene_orders)
        for i in range(df.shape[0]): list_y.append(belonged_segidy)
        belonged_segidx = getsegidx(df,segs,gene_list,mlt,gene_orders)

def get_segmentpair_order(anchorpoints,segs,table,gene_orders):
    ap_with_list = annotatelist_full(anchorpoints,table,segs,gene_orders)
    ap_with_list = annotatelist(anchorpoints,table)
    ap_with_list_addsegy = annotatesegy(ap_with_list,segs)

def getmltorder(anchorpoints,multi,gene_orders):
    ap_order_permlt = {i:'' for i in set(anchorpoints['multiplicon'])}
    tmp = list(anchorpoints.groupby('multiplicon'))
    for mlt, df in tmp:
        genexs,geneys = list(df['gene_x'])[:3],list(df['gene_y'])[:3]
        genexs_coor, geneys_coor = [gene_orders[i] for i in genexs], [gene_orders[i] for i in geneys]
        consistent_order = []
        for i in range(2):
            tmpx = genexs_coor[i+1]-genexs_coor[i]
            tmpy = geneys_coor[i+1]-geneys_coor[i]
            if tmpx > 0: pos_neg = "+"
            else: pos_neg = "-"
            consistent_order.append(tmpx*tmpy)
        for i in consistent_order:
            assert i > 0
        ap_order_permlt[mlt] = pos_neg
    return ap_order_permlt
