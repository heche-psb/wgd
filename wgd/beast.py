import pandas as pd
import numpy as np
import subprocess as sp
import logging
import os
import re
from Bio.Align import MultipleSeqAlignment
from Bio import Phylo

def getree(treef):
    content = ""
    with open(treef,"r") as f:
        content = f.read()
    content = content.strip("\n").strip("\t").strip(" ")
    return content

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

def cleanlist(l):
    for i in range(len(l)):
        l[i] = l[i].strip('\t').strip('\n').strip(' ')
    return l

def _run_beast(beastlgjar,xml,beagle):
    """
    Run beast
    """
    fname = xml + ".beastLog"
    if beagle:
        cmd = ['java','-server','-Xms64m','-Xmx1024m','-XX:ParallelGCThreads=1','-XX:+PrintCommandLineFlags','-jar',beastlgjar,'-beagle','-beagle_CPU','-beagle_SSE','-warnings','-overwrite',xml]
    else:
        cmd = ['java','-server','-Xms64m','-Xmx1024m','-XX:ParallelGCThreads=1','-XX:+PrintCommandLineFlags','-jar',beastlgjar,'-java','-warnings','-overwrite',xml]
    log = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    with open(fname, 'w') as f: f.write(log.stdout.decode('utf-8'))

class beast:
    """
    Implementation of beast provided a MRBH family for phylogenetic dating
    """
    def __init__(self, calnf, cds_aln_rn, pro_aln_rn, tmpdir, outdir, speciestree, datingset, slist, fossil, chainset, rootheight):
        self.slist = slist
        self.chainl = chainset[0]
        self.chainf = chainset[1]
        self.tree = speciestree
        self.prefix = os.path.basename(calnf).replace('.caln','').replace('.','_')
        if tmpdir == None:
            tmp_path = os.path.join(outdir, "beast", self.prefix)
        else:
            tmp_path = os.path.join(tmpdir, "beast", self.prefix)
        _mkdir(os.path.join(outdir, "beast"))
        self.tmp_path = _mkdir(tmp_path)
        tmpc_path = os.path.join(tmp_path, "cds")
        self.tmpc_path = _mkdir(tmpc_path)
        self.caln = cds_aln_rn
        self.xmlc = os.path.join(tmpc_path, '{}.xml'.format(self.prefix))
        tmpp_path = os.path.join(tmp_path, "pep")
        self.tmpp_path = _mkdir(tmpp_path)
        self.paln = pro_aln_rn
        #_cp2tmp(self.palnf_rn,self.tree,self.tmpp_path)
        self.xmlp = os.path.join(tmpp_path, '{}.xml'.format(self.prefix))
        self.id,self.taxa,self.mean,self.std,self.offset  = [cleanlist(i.split(';')) for i in fossil]
        print(fossil)
        print(self.id)
        print(self.taxa)
        print(self.mean)
        print(self.std)
        print(self.offset)
        #ids,taxas,means,stds,offsets = [i.split(';').strip('\t').strip(' ').strip('\n') for i in fossil.split(';;').strip('\t').strip(' ').strip('\n')]
        #self.id,self.taxa,self.mean,self.std,self.offset = ids,taxas,means,stds,offsets
        #self.id = [f[0] for f in fossil]
        #self.taxa = [f[1] for f in fossil]
        #self.mean = [f[2] for f in fossil]
        #self.std = [f[3] for f in fossil]
        #self.offset = [f[4] for f in fossil]
        self.peak = [float(offset)+np.exp(float(mean)-float(std)**2) for mean,std,offset in zip(self.mean,self.std,self.offset)]
        self.rooth ={key:i for key,i in zip(['mean','std','offset'],rootheight)}
        self.rootp = self.rooth['offset'] + np.exp(self.rooth['mean']-self.rooth['std']**2)
        self.wgd_mrca = [sp for sp in slist if sp[-4:] == '_ap1' or sp[-4:] == '_ap2']
        #if not datingset is None:
        #    for i in datingset:
        #        i.strip('\t').strip('\n').strip(' ')
        #        for key in self.controlc.keys():
        #            if key in i:
        #                self.controlc[key] = i.replace(key,'').replace('=','').strip(' ')
        #                if not self.partition:
        #                    self.controlp[key] = i.replace(key,'').replace('=','').strip(' ')
    def write_xml(self):
        with open(self.xmlc, "w") as f:
            f.write("<?xml version='1.0' encoding='UTF-8'?>\n<beast>\n")
            f.write("\n".join(["\t","\t\n"]))
            f.write("\t<!-- Define all the taxagroups -->\n")
            f.write("\t<taxa id=\"taxa\">\n")
            f.write("\n".join(["\t\t<taxon id=\"{}\"/>".format(sp) for sp in self.slist]))
            f.write("\n\t</taxa>\n")
            for i,taxa in zip(self.id,self.taxa):
                f.write("\t<taxa id=\"Cal_{}\">\n".format(i))
                taxas = [t.strip(' ').strip('\t').strip('\n') for t in taxa.split(',')]
                f.write("\n".join(["\t\t<taxon idref=\"{}\"/>".format(taxa) for taxa in taxas]+["\t</taxa>\n"]))
            f.write("\t<taxa id=\"WGD\">\n")
            f.write("".join(["\t\t<taxon idref=\"{}\"/>\n".format(i) for i in self.wgd_mrca]+["\t</taxa>\n"]))
            f.write("\n\t\n\t<!-- Define the alignment -->\n")
            f.write("\t<alignment id=\"alignment\" dataType=\"nucleotide\">\n")
            f.write("".join(["\t\t<sequence>\n\t\t\t<taxon idref=\"{}\"/>\n\t\t\t{}\n\t\t</sequence>\n".format(i.id,i.seq) for i in self.caln]+["\t</alignment>\n"]))
            f.write("\t\n\t<!-- Define the data partitions -->\n")
            f.write("\t<patterns id=\"CP1.patterns\" from=\"1\" every=\"3\" strip=\"false\">\n\t\t<alignment idref=\"alignment\"/>\n\t</patterns>\n")
            f.write("\t<patterns id=\"CP2.patterns\" from=\"2\" every=\"3\" strip=\"false\">\n\t\t<alignment idref=\"alignment\"/>\n\t</patterns>\n")
            f.write("\t<patterns id=\"CP3.patterns\" from=\"3\" every=\"3\" strip=\"false\">\n\t\t<alignment idref=\"alignment\"/>\n\t</patterns>\n")
            f.write("\n\t\n\t<!-- Define the Yule prior, starting tree, and treemodel -->\n")
            f.write("\t<yuleModel id=\"yule\" units=\"substitutions\">\n\t\t<birthRate>\n\t\t\t<parameter id=\"yule.birthRate\" value=\"0.1\"/>\n\t\t</birthRate>\n\t</yuleModel>\n")
            f.write("\n\t<newick id=\"startingTree\" usingDates=\"false\">\n\t\t{}\n\t</newick>\n".format(getree(self.tree)))
            f.write("\n\t<treeModel id=\"treeModel\">\n")
            f.write("\t\t<tree idref=\"startingTree\"/>\n")
            f.write("\t\t<rootHeight>\n\t\t\t<parameter id=\"treeModel.rootHeight\"/>\n\t\t</rootHeight>\n")
            f.write("\t\t<nodeHeights internalNodes=\"true\">\n\t\t\t<parameter id=\"treeModel.internalNodeHeights\"/>\n\t\t</nodeHeights>\n")
            f.write("\t\t<nodeHeights internalNodes=\"true\" rootNode=\"true\">\n\t\t\t<parameter id=\"treeModel.allInternalNodeHeights\"/>\n\t\t</nodeHeights>\n")
            f.write("\t</treeModel>\n")
            f.write("\n\t\n\t<!-- Define the calibration points -->\n")
            for i in self.id: f.write("\t<tmrcaStatistic id=\"tmrca(Cal_{})\" includeStem=\"false\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"Cal_{}\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</tmrcaStatistic>\n\t<monophylyStatistic id=\"monophyly(Cal_{})\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"Cal_{}\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</monophylyStatistic>\n\t\n".format(i,i,i,i))
            f.write("\t<tmrcaStatistic id=\"tmrca(WGD)\" includeStem=\"false\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"WGD\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</tmrcaStatistic>\n\t<monophylyStatistic id=\"monophyly(WGD)\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"WGD\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</monophylyStatistic>\n\t\n")
            f.write("\t\n\t\n\t<!-- Define the speciation prior -->\n")
            f.write("\t<speciationLikelihood id=\"speciation\">\n\t\t<model>\n\t\t\t<yuleModel idref=\"yule\"/>\n\t\t</model>\n\t\t<speciesTree>\n\t\t\t<treeModel idref=\"treeModel\"/>\n\t\t</speciesTree>\n\t</speciationLikelihood>\n")
            f.write("\n\t\n\t<!-- Define the UCLD clock model -->\n")
            f.write("\t<discretizedBranchRates id=\"branchRates\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<distribution>\n\t\t\t<logNormalDistributionModel meanInRealSpace=\"true\">\n\t\t\t\t<mean>\n\t\t\t\t\t<parameter id=\"ucld.mean\" value=\"0.003\"/>\n\t\t\t\t</mean>\n\t\t\t\t<stdev>\n\t\t\t\t\t<parameter id=\"ucld.stdev\" value=\"0.3333333333333333\"/>\n\t\t\t\t</stdev>\n\t\t\t</logNormalDistributionModel>\n\t\t</distribution>\n\t\t<rateCategories>\n\t\t\t<parameter id=\"branchRates.categories\"/>\n\t\t</rateCategories>\n\t</discretizedBranchRates>\n\t<rateStatistic id=\"meanRate\" name=\"meanRate\" mode=\"mean\" internal=\"true\" external=\"true\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</rateStatistic>\n\t<rateStatistic id=\"coefficientOfVariation\" name=\"coefficientOfVariation\" mode=\"coefficientOfVariation\" internal=\"true\" external=\"true\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</rateStatistic>\n\t<rateCovarianceStatistic id=\"covariance\" name=\"covariance\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</rateCovarianceStatistic>\n")
            f.write("\n\t\n\t<!-- Define the HKY+G substitution model -->\n")
            for i in [1,2,3]: f.write("\t<HKYModel id=\"CP{}.hky\">\n\t\t<frequencies>\n\t\t\t<frequencyModel dataType=\"nucleotide\">\n\t\t\t\t<patterns idref=\"CP{}.patterns\"/>\n\t\t\t\t<frequencies>\n\t\t\t\t\t<parameter id=\"CP{}.frequencies\" dimension=\"4\"/>\n\t\t\t\t</frequencies>\n\t\t\t</frequencyModel>\n\t\t</frequencies>\n\t\t<kappa>\n\t\t\t<parameter id=\"CP{}.kappa\" value=\"2.0\"/>\n\t\t</kappa>\n\t</HKYModel>\n\t\n".format(i,i,i,i))
            for i in [1,2,3]: f.write("\t<siteModel id=\"CP{}.siteModel\">\n\t\t<substitutionModel>\n\t\t\t<HKYModel idref=\"CP{}.hky\"/>\n\t\t</substitutionModel>\n\t\t<relativeRate>\n\t\t\t<parameter id=\"CP{}.mu\" value=\"1.0\"/>\n\t\t</relativeRate>\n\t\t<gammaShape gammaCategories=\"4\">\n\t\t\t<parameter id=\"CP{}.alpha\" value=\"0.5\"/>\n\t\t</gammaShape>\n\t</siteModel>\n\t\n".format(i,i,i,i))
            f.write("\t<compoundParameter id=\"allMus\">\n\t\t<parameter idref=\"CP1.mu\"/>\n\t\t<parameter idref=\"CP2.mu\"/>\n\t\t<parameter idref=\"CP3.mu\"/>\n\t</compoundParameter>\n\t\n")
            f.write("\t<!-- Define the tree likelihoods -->\n")
            for i in [1,2,3]: f.write("\t<treeLikelihood id=\"CP{}.treeLikelihood\" useAmbiguities=\"false\">\n\t\t<patterns idref=\"CP{}.patterns\"/>\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<siteModel idref=\"CP{}.siteModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</treeLikelihood>\n\t\n".format(i,i,i))
            f.write("\t\n\t<!-- Define the operator list -->\n")
            f.write("\t<operators id=\"operators\">\n")
            f.write("".join(["\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"1\">\n\t\t\t<parameter idref=\"CP{}.kappa\"/>\n\t\t</scaleOperator>\n".format(i) for i in [1,2,3]]))
            f.write("".join(["\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"1\">\n\t\t\t<parameter idref=\"CP{}.alpha\"/>\n\t\t</scaleOperator>\n".format(i) for i in [1,2,3]]))
            f.write("\t\t<deltaExchange delta=\"0.75\" parameterWeights=\"604 604 604\" weight=\"3\">\n\t\t\t<parameter idref=\"allMus\"/>\n\t\t</deltaExchange>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"10\">\n\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t</scaleOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"5\">\n\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t</scaleOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"3\">\n\t\t\t<parameter idref=\"treeModel.rootHeight\"/>\n\t\t</scaleOperator>\n\t\t<uniformOperator weight=\"30\">\n\t\t\t<parameter idref=\"treeModel.internalNodeHeights\"/>\n\t\t</uniformOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"3\">\n\t\t\t<parameter idref=\"yule.birthRate\"/>\n\t\t</scaleOperator>\n\t\t<upDownOperator scaleFactor=\"0.75\" weight=\"3\">\n\t\t\t<up>\n\t\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t</up>\n\t\t\t<down>\n\t\t\t\t<parameter idref=\"treeModel.allInternalNodeHeights\"/>\n\t\t\t</down>\n\t\t</upDownOperator>\n\t\t<swapOperator size=\"1\" weight=\"10\" autoOptimize=\"false\">\n\t\t\t<parameter idref=\"branchRates.categories\"/>\n\t\t</swapOperator>\n\t\t<uniformIntegerOperator weight=\"10\">\n\t\t\t<parameter idref=\"branchRates.categories\"/>\n\t\t</uniformIntegerOperator>\n\t</operators>\n")
            f.write("\n\t\n\t<!-- Define the MCMCM and all the priors on the parameters -->\n")
            f.write("\t<mcmc id=\"mcmc\" chainLength=\"{0}\" autoOptimize=\"true\" operatorAnalysis=\"{1}.ops\">\n\t\t<posterior id=\"posterior\">\n\t\t\t<prior id=\"prior\">\n\t\t\t\t<booleanLikelihood>\n".format(self.chainl,self.prefix))
            f.write("".join(["\t\t\t\t\t<monophylyStatistic idref=\"monophyly(Cal_{})\"/>\n".format(i) for i in self.id]))
            f.write("\t\t\t\t\t<monophylyStatistic idref=\"monophyly(WGD)\"/>\n")
            f.write("\t\t\t\t</booleanLikelihood>\n")
            for peak,mean,std,offset,i in zip(self.peak,self.mean,self.std,self.offset,self.id): f.write("\t\t\t\t<!-- Corresponds to peak at {0} and 97.5% at higher -->\n\t\t\t\t<logNormalPrior mean=\"{1}\" stdev=\"{2}\" offset=\"{3}\" meanInRealSpace=\"false\">\n\t\t\t\t\t<statistic idref=\"tmrca(Cal_{4})\"/>\n\t\t\t\t</logNormalPrior>\n".format(peak,mean,std,offset,i))
            for i in [1,2,3]: f.write("\t\t\t\t<logNormalPrior mean=\"1.0\" stdev=\"1.25\" offset=\"0.0\" meanInRealSpace=\"false\">\n\t\t\t\t\t<parameter idref=\"CP{}.kappa\"/>\n\t\t\t\t</logNormalPrior>\n".format(i))
            for i in [1,2,3]: f.write("\t\t\t\t<logNormalPrior mean=\"0.0\" stdev=\"1.0\" offset=\"0.0\" meanInRealSpace=\"false\">\n\t\t\t\t\t<parameter idref=\"CP{}.mu\"/>\n\t\t\t\t</logNormalPrior>\n".format(i))
            for i in [1,2,3]: f.write("\t\t\t\t<exponentialPrior mean=\"0.5\" offset=\"0.0\">\n\t\t\t\t\t<parameter idref=\"CP{}.alpha\"/>\n\t\t\t\t</exponentialPrior>\n".format(i))
            f.write("\t\t\t\t<exponentialPrior mean=\"0.3333333333333333\" offset=\"0.0\">\n\t\t\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t\t\t</exponentialPrior>\n\t\t\t\t<gammaPrior shape=\"0.001\" scale=\"1000\" offset=\"0.0\">\n\t\t\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t\t</gammaPrior>\n")
            f.write("\t\t\t\t<!-- Corresponds to peak at {0} and 97.5% at higher -->\n\t\t\t\t<logNormalPrior mean=\"{1}\" stdev=\"{2}\" offset=\"{3}\" meanInRealSpace=\"false\">\n\t\t\t\t\t<statistic idref=\"treeModel.rootHeight\"/>\n\t\t\t\t</logNormalPrior>\n\t\t\t\t<uniformPrior lower=\"0.0\" upper=\"1.0E100\">\n\t\t\t\t\t<parameter idref=\"yule.birthRate\"/>\n\t\t\t\t</uniformPrior>\n\t\t\t\t<speciationLikelihood idref=\"speciation\"/>\n\t\t\t</prior>\n\t\t\t<likelihood id=\"likelihood\">\n".format(self.rootp,self.rooth['mean'],self.rooth['std'],self.rooth['offset']))
            for i in [1,2,3]: f.write("\t\t\t\t<treeLikelihood idref=\"CP{}.treeLikelihood\"/>\n".format(i))
            f.write("\t\t\t</likelihood>\n\t\t</posterior>\n\t\t<operators idref=\"operators\"/>\n\n\t\t\n")
            f.write("\t\t<!-- Define the logging to screen and logfiles -->\n")
            f.write("\t\t<log id=\"screenLog\" logEvery=\"{}\">\n\t\t\t<column label=\"Posterior\" dp=\"4\" width=\"12\">\n\t\t\t\t<posterior idref=\"posterior\"/>\n\t\t\t</column>\n\t\t\t<column label=\"Prior\" dp=\"4\" width=\"12\">\n\t\t\t\t<prior idref=\"prior\"/>\n\t\t\t</column>\n\t\t\t<column label=\"Likelihood\" dp=\"4\" width=\"12\">\n\t\t\t\t<likelihood idref=\"likelihood\"/>\n\t\t\t</column>\n\t\t\t<column label=\"rootHeight\" sf=\"6\" width=\"12\">\n\t\t\t\t<parameter idref=\"treeModel.rootHeight\"/>\n\t\t\t</column>\n\t\t\t<column label=\"ucld.mean\" sf=\"6\" width=\"12\">\n\t\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t</column>\n\t\t\t<column label=\"ucld.stdev\" sf=\"6\" width=\"12\">\n\t\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t\t</column>\n\t\t\t<column label=\"WGD\" sf=\"6\" width=\"12\">\n\t\t\t\t<tmrcaStatistic idref=\"tmrca(WGD)\"/>\n\t\t\t</column>\n\t\t</log>\n".format(self.chainf))
            f.write("\n\t\t<log id=\"fileLog\" logEvery=\"{0}\" fileName=\"{1}.log\" overwrite=\"false\">\n\t\t\t<posterior idref=\"posterior\"/>\n\t\t\t<prior idref=\"prior\"/>\n\t\t\t<likelihood idref=\"likelihood\"/>\n\t\t\t<parameter idref=\"treeModel.rootHeight\"/>\n".format(self.chainf,self.prefix))
            f.write("".join(["\t\t\t<tmrcaStatistic idref=\"tmrca(Cal_{})\"/>\n".format(i) for i in self.id]+["\t\t\t<tmrcaStatistic idref=\"tmrca(WGD)\"/>\n"]))
            f.write("\t\t\t<parameter idref=\"yule.birthRate\"/>\n")
            for i in [1,2,3]: f.write("\t\t\t<parameter idref=\"CP{}.kappa\"/>\n".format(i))
            for i in [1,2,3]: f.write("\t\t\t<parameter idref=\"CP{}.mu\"/>\n".format(i))
            for i in [1,2,3]: f.write("\t\t\t<parameter idref=\"CP{}.alpha\"/>\n".format(i))
            f.write("\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t\t<rateStatistic idref=\"meanRate\"/>\n\t\t\t<rateStatistic idref=\"coefficientOfVariation\"/>\n\t\t\t<rateCovarianceStatistic idref=\"covariance\"/>\n")
            for i in [1,2,3]: f.write("\t\t\t<treeLikelihood idref=\"CP{}.treeLikelihood\"/>\n".format(i))
            f.write("\t\t\t<speciationLikelihood idref=\"speciation\"/>\n\t\t</log>\n\n\t\t<logTree id=\"treeFileLog\" logEvery=\"{0}\" nexusFormat=\"true\" fileName=\"{1}.trees\" sortTranslationTable=\"true\">\n\t\t\t<treeModel idref=\"treeModel\"/>\n\t\t\t<trait name=\"rate\" tag=\"rate\">\n\t\t\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t\t\t</trait>\n\t\t\t<posterior idref=\"posterior\"/>\n\t\t</logTree>\n\t</mcmc>\n\t\n\t\n\t<!-- Bye Bye -->\n\t<report>\n\t\t<property name=\"timer\">\n\t\t\t<mcmc idref=\"mcmc\"/>\n\t\t</property>\n\t</report>\n\t\n</beast>\n".format(self.chainf,self.prefix))
        with open(self.xmlp, "w") as f:
            f.write("<?xml version='1.0' encoding='UTF-8'?>\n<beast>\n")
            f.write("\n".join(["\t","\t\n"]))
            f.write("\t<!-- Define all the taxagroups -->\n")
            f.write("\t<taxa id=\"taxa\">\n")
            f.write("\n".join(["\t\t<taxon id=\"{}\"/>".format(sp) for sp in self.slist]))
            f.write("\n\t</taxa>\n")
            for i,taxa in zip(self.id,self.taxa):
                f.write("\t<taxa id=\"Cal_{}\">\n".format(i))
                taxas = [t.strip(' ').strip('\t').strip('\n') for t in taxa.split(',')]
                f.write("\n".join(["\t\t<taxon idref=\"{}\"/>".format(taxa) for taxa in taxas]+["\t</taxa>\n"]))
            f.write("\t<taxa id=\"WGD\">\n")
            f.write("".join(["\t\t<taxon idref=\"{}\"/>\n".format(i) for i in self.wgd_mrca]+["\t</taxa>\n"]))
            f.write("\n\t\n\t<!-- Define the alignment -->\n")
            f.write("\t<alignment id=\"alignment\" dataType=\"amino acid\">\n")
            f.write("".join(["\t\t<sequence>\n\t\t\t<taxon idref=\"{}\"/>\n\t\t\t{}\n\t\t</sequence>\n".format(i.id,i.seq) for i in self.paln]+["\t</alignment>\n"]))
            f.write("\t\n\t<!-- Define the data partitions -->\n")
            f.write("\t<patterns id=\"patterns\" from=\"1\" strip=\"false\">\n\t\t<alignment idref=\"alignment\"/>\n\t</patterns>\n")
            f.write("\n\t\n\t<!-- Define the Yule prior, starting tree, and treemodel -->\n")
            f.write("\t<yuleModel id=\"yule\" units=\"substitutions\">\n\t\t<birthRate>\n\t\t\t<parameter id=\"yule.birthRate\" value=\"0.1\"/>\n\t\t</birthRate>\n\t</yuleModel>\n")
            f.write("\t\n\t<newick id=\"startingTree\" usingDates=\"false\">\n\t\t{}\n\t</newick>\n".format(getree(self.tree)))
            f.write("\n\t<treeModel id=\"treeModel\">\n")
            f.write("\t\t<tree idref=\"startingTree\"/>\n")
            f.write("\t\t<rootHeight>\n\t\t\t<parameter id=\"treeModel.rootHeight\"/>\n\t\t</rootHeight>\n")
            f.write("\t\t<nodeHeights internalNodes=\"true\">\n\t\t\t<parameter id=\"treeModel.internalNodeHeights\"/>\n\t\t</nodeHeights>\n")
            f.write("\t\t<nodeHeights internalNodes=\"true\" rootNode=\"true\">\n\t\t\t<parameter id=\"treeModel.allInternalNodeHeights\"/>\n\t\t</nodeHeights>\n")
            f.write("\t</treeModel>\n")
            f.write("\n\t\n\t<!-- Define the calibration points -->\n")
            for i in self.id: f.write("\t<tmrcaStatistic id=\"tmrca(Cal_{})\" includeStem=\"false\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"Cal_{}\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</tmrcaStatistic>\n\t<monophylyStatistic id=\"monophyly(Cal_{})\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"Cal_{}\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</monophylyStatistic>\n\t\n".format(i,i,i,i))
            f.write("\t<tmrcaStatistic id=\"tmrca(WGD)\" includeStem=\"false\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"WGD\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</tmrcaStatistic>\n\t<monophylyStatistic id=\"monophyly(WGD)\">\n\t\t<mrca>\n\t\t\t<taxa idref=\"WGD\"/>\n\t\t</mrca>\n\t\t<treeModel idref=\"treeModel\"/>\n\t</monophylyStatistic>\n\t\n")
            f.write("\t\n\t\n\t<!-- Define the speciation prior -->\n")
            f.write("\t<speciationLikelihood id=\"speciation\">\n\t\t<model>\n\t\t\t<yuleModel idref=\"yule\"/>\n\t\t</model>\n\t\t<speciesTree>\n\t\t\t<treeModel idref=\"treeModel\"/>\n\t\t</speciesTree>\n\t</speciationLikelihood>\n")
            f.write("\n\t\n\t<!-- Define the UCLD clock model -->\n")
            f.write("\t<discretizedBranchRates id=\"branchRates\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<distribution>\n\t\t\t<logNormalDistributionModel meanInRealSpace=\"true\">\n\t\t\t\t<mean>\n\t\t\t\t\t<parameter id=\"ucld.mean\" value=\"0.003\"/>\n\t\t\t\t</mean>\n\t\t\t\t<stdev>\n\t\t\t\t\t<parameter id=\"ucld.stdev\" value=\"0.3333333333333333\"/>\n\t\t\t\t</stdev>\n\t\t\t</logNormalDistributionModel>\n\t\t</distribution>\n\t\t<rateCategories>\n\t\t\t<parameter id=\"branchRates.categories\"/>\n\t\t</rateCategories>\n\t</discretizedBranchRates>\n\t<rateStatistic id=\"meanRate\" name=\"meanRate\" mode=\"mean\" internal=\"true\" external=\"true\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</rateStatistic>\n\t<rateStatistic id=\"coefficientOfVariation\" name=\"coefficientOfVariation\" mode=\"coefficientOfVariation\" internal=\"true\" external=\"true\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</rateStatistic>\n\t<rateCovarianceStatistic id=\"covariance\" name=\"covariance\">\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</rateCovarianceStatistic>\n")
            f.write("\n\t\n\t<!-- Define the LG+G substitution model -->\n")
            f.write("\t<aminoAcidModel id=\"aa\" type=\"LG\"/>\n\t\n")
            f.write("\t<siteModel id=\"siteModel\">\n\t\t<substitutionModel>\n\t\t\t<aminoAcidModel idref=\"aa\"/>\n\t\t</substitutionModel>\n\t\t<gammaShape gammaCategories=\"4\">\n\t\t\t<parameter id=\"alpha\" value=\"0.5\"/>\n\t\t</gammaShape>\n\t</siteModel>\n")
            f.write("\t\n\t\n\t<!-- Define the tree likelihoods -->\n")
            f.write("\t<treeLikelihood id=\"treeLikelihood\" useAmbiguities=\"false\" stateTagName=\"states\">\n\t\t<patterns idref=\"patterns\"/>\n\t\t<treeModel idref=\"treeModel\"/>\n\t\t<siteModel idref=\"siteModel\"/>\n\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t</treeLikelihood>\n")
            f.write("\n\t\n\t<!-- Define the operator list -->\n")
            f.write("\t<operators id=\"operators\">\n")
            f.write("\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"1\">\n\t\t\t<parameter idref=\"alpha\"/>\n\t\t</scaleOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"10\">\n\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t</scaleOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"5\">\n\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t</scaleOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"3\">\n\t\t\t<parameter idref=\"treeModel.rootHeight\"/>\n\t\t</scaleOperator>\n\t\t<uniformOperator weight=\"30\">\n\t\t\t<parameter idref=\"treeModel.internalNodeHeights\"/>\n\t\t</uniformOperator>\n\t\t<scaleOperator scaleFactor=\"0.75\" weight=\"3\">\n\t\t\t<parameter idref=\"yule.birthRate\"/>\n\t\t</scaleOperator>\n\t\t<upDownOperator scaleFactor=\"0.75\" weight=\"3\">\n\t\t\t<up>\n\t\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t</up>\n\t\t\t<down>\n\t\t\t\t<parameter idref=\"treeModel.allInternalNodeHeights\"/>\n\t\t\t</down>\n\t\t</upDownOperator>\n\t\t<swapOperator size=\"1\" weight=\"10\" autoOptimize=\"false\">\n\t\t\t<parameter idref=\"branchRates.categories\"/>\n\t\t</swapOperator>\n\t\t<uniformIntegerOperator weight=\"10\">\n\t\t\t<parameter idref=\"branchRates.categories\"/>\n\t\t</uniformIntegerOperator>\n\t</operators>\n")
            f.write("\n\t\n\t<!-- Define the MCMCM and all the priors on the parameters -->\n")
            f.write("\t<mcmc id=\"mcmc\" chainLength=\"{0}\" autoOptimize=\"true\" operatorAnalysis=\"{1}.ops\">\n\t\t<posterior id=\"posterior\">\n\t\t\t<prior id=\"prior\">\n\t\t\t\t<booleanLikelihood>\n".format(self.chainl,self.prefix))
            f.write("".join(["\t\t\t\t\t<monophylyStatistic idref=\"monophyly(Cal_{})\"/>\n".format(i) for i in self.id]))
            f.write("\t\t\t\t\t<monophylyStatistic idref=\"monophyly(WGD)\"/>\n")
            f.write("\t\t\t\t</booleanLikelihood>\n")
            for peak,mean,std,offset,i in zip(self.peak,self.mean,self.std,self.offset,self.id): f.write("\t\t\t\t<!-- Corresponds to peak at {0} and 97.5% at higher -->\n\t\t\t\t<logNormalPrior mean=\"{1}\" stdev=\"{2}\" offset=\"{3}\" meanInRealSpace=\"false\">\n\t\t\t\t\t<statistic idref=\"tmrca(Cal_{4})\"/>\n\t\t\t\t</logNormalPrior>\n".format(peak,mean,std,offset,i))
            f.write("\t\t\t\t<exponentialPrior mean=\"0.5\" offset=\"0.0\">\n\t\t\t\t\t<parameter idref=\"alpha\"/>\n\t\t\t\t</exponentialPrior>\n\t\t\t\t<exponentialPrior mean=\"0.3333333333333333\" offset=\"0.0\">\n\t\t\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t\t\t</exponentialPrior>\n\t\t\t\t<gammaPrior shape=\"0.001\" scale=\"1000\" offset=\"0.0\">\n\t\t\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t\t</gammaPrior>\n")
            f.write("\t\t\t\t<!-- Corresponds to peak at {0} and 97.5% at higher -->\n\t\t\t\t<logNormalPrior mean=\"{1}\" stdev=\"{2}\" offset=\"{3}\" meanInRealSpace=\"false\">\n\t\t\t\t\t<statistic idref=\"treeModel.rootHeight\"/>\n\t\t\t\t</logNormalPrior>\n\t\t\t\t<uniformPrior lower=\"0.0\" upper=\"1.0E100\">\n\t\t\t\t\t<parameter idref=\"yule.birthRate\"/>\n\t\t\t\t</uniformPrior>\n\t\t\t\t<speciationLikelihood idref=\"speciation\"/>\n\t\t\t</prior>\n\t\t\t<likelihood id=\"likelihood\">\n\t\t\t\t<treeLikelihood idref=\"treeLikelihood\"/>\n\t\t\t</likelihood>\n\t\t</posterior>\n\t\t<operators idref=\"operators\"/>\n\n\t\t\n".format(self.rootp,self.rooth['mean'],self.rooth['std'],self.rooth['offset']))
            f.write("\t\t<!-- Define the logging to screen and logfiles -->\n")
            f.write("\t\t<log id=\"screenLog\" logEvery=\"{}\">\n\t\t\t<column label=\"Posterior\" dp=\"4\" width=\"12\">\n\t\t\t\t<posterior idref=\"posterior\"/>\n\t\t\t</column>\n\t\t\t<column label=\"Prior\" dp=\"4\" width=\"12\">\n\t\t\t\t<prior idref=\"prior\"/>\n\t\t\t</column>\n\t\t\t<column label=\"Likelihood\" dp=\"4\" width=\"12\">\n\t\t\t\t<likelihood idref=\"likelihood\"/>\n\t\t\t</column>\n\t\t\t<column label=\"rootHeight\" sf=\"6\" width=\"12\">\n\t\t\t\t<parameter idref=\"treeModel.rootHeight\"/>\n\t\t\t</column>\n\t\t\t<column label=\"ucld.mean\" sf=\"6\" width=\"12\">\n\t\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t</column>\n\t\t\t<column label=\"ucld.stdev\" sf=\"6\" width=\"12\">\n\t\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t\t</column>\n\t\t\t<column label=\"WGD\" sf=\"6\" width=\"12\">\n\t\t\t\t<tmrcaStatistic idref=\"tmrca(WGD)\"/>\n\t\t\t</column>\n\t\t</log>\n".format(self.chainf))
            f.write("\n\t\t<log id=\"fileLog\" logEvery=\"{0}\" fileName=\"{1}.log\" overwrite=\"false\">\n\t\t\t<posterior idref=\"posterior\"/>\n\t\t\t<prior idref=\"prior\"/>\n\t\t\t<likelihood idref=\"likelihood\"/>\n\t\t\t<parameter idref=\"treeModel.rootHeight\"/>\n".format(self.chainf,self.prefix))
            f.write("".join(["\t\t\t<tmrcaStatistic idref=\"tmrca(Cal_{})\"/>\n".format(i) for i in self.id]+["\t\t\t<tmrcaStatistic idref=\"tmrca(WGD)\"/>\n"]))
            f.write("\t\t\t<parameter idref=\"yule.birthRate\"/>\n")
            f.write("\t\t\t<parameter idref=\"alpha\"/>\n")
            f.write("\t\t\t<parameter idref=\"ucld.mean\"/>\n\t\t\t<parameter idref=\"ucld.stdev\"/>\n\t\t\t<rateStatistic idref=\"meanRate\"/>\n\t\t\t<rateStatistic idref=\"coefficientOfVariation\"/>\n\t\t\t<rateCovarianceStatistic idref=\"covariance\"/>\n\t\t\t<treeLikelihood idref=\"treeLikelihood\"/>\n")
            f.write("\t\t\t<speciationLikelihood idref=\"speciation\"/>\n\t\t</log>\n\n\t\t<logTree id=\"treeFileLog\" logEvery=\"{0}\" nexusFormat=\"true\" fileName=\"{1}.trees\" sortTranslationTable=\"true\">\n\t\t\t<treeModel idref=\"treeModel\"/>\n\t\t\t<trait name=\"rate\" tag=\"rate\">\n\t\t\t\t<discretizedBranchRates idref=\"branchRates\"/>\n\t\t\t</trait>\n\t\t\t<posterior idref=\"posterior\"/>\n\t\t</logTree>\n\t</mcmc>\n\t\n\t\n\t<!-- Bye Bye -->\n\t<report>\n\t\t<property name=\"timer\">\n\t\t\t<mcmc idref=\"mcmc\"/>\n\t\t</property>\n\t</report>\n\t\n</beast>\n".format(self.chainf,self.prefix))

    def run_beast(self,beastlgjar,beagle):
        """
        Run beast on the codon and peptide alignment.
        """
        parentdir = os.getcwd()
        self.write_xml()
        os.chdir(self.tmpc_path)
        #_run_beast(beastlgjar,os.path.basename(self.xmlc),beagle)
        os.chdir(parentdir)
        os.chdir(self.tmpp_path)
        #_run_beast(beastlgjar,os.path.basename(self.xmlp),beagle)
        os.chdir(parentdir)
    #def get_dates(self,wgd_mrca,cds=True):
    #    Figtree = Phylo.read('FigTree.tre','nexus')
    #    wgd_node = Figtree.common_ancestor({"name": wgd_mrca[0]}, {"name": wgd_mrca[1]})
    #    self.CI = wgd_node.comment.strip('[&95%={').strip('}]').split(', ')
    #    self.PM = wgd_node.clades[0].branch_length
    #    if cds:
    #        logging.info("Posterior mean for the ages of wgd is {0} billion years from {1} codon alignment and 95% credibility intervals (CI) is {2}-{3} billion years".format(self.PM,self.prefix,self.CI[0],self.CI[1]))
    #    else:
    #        logging.info("Posterior mean for the ages of wgd is {0} billion years from {1} peptide alignment and 95% credibility intervals (CI) is {2}-{3} billion years".format(self.PM,self.prefix,self.CI[0],self.CI[1]))
    #    return self.CI, self.PM
