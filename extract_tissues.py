import os
import gzip

# === data directories
vcf_file = "/compbioscratch2/yhua295/dbGap-17031/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v7.p2.c1.GRU/GenotypeFiles/phg000830.v1.GTEx_WGS.genotype-calls-vcf.c1/GTEx_Analysis_2016-01-15_v7_WholeGenomeSeq_635Ind_PASS_AB02_GQ20_HETX_MISS15_PLINKQC.vcf.gz"
eQTL_lung_sig = "/compbioscratch2/wma36/projects/cs584_ge_snp/data/GTEx_Analysis_v7_eQTL/Lung.v7.signif_variant_gene_pairs.txt.gz"
attributes = " /compbioscratch2/yhua295/dbGap-17031/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v7.p2.c1.GRU/PhenotypeFiles/phs000424.v7.pht002743.v7.p2.c1.GTEx_Sample_Attributes.GRU.txt.gz"

import pandas as pd
import numpy as np

''' === sort lung tissue variants
eQTL_lung_sig_dict = {}
# === get top 100 eQTLs
with gzip.open(eQTL_lung_sig, 'rb') as f:
    line = f.readline().decode('utf-8').strip()
    res = line.split('\t')
    eQTL_lung_sig_dict[res[0]] = [] # variant ID
    eQTL_lung_sig_dict[res[1]] = [] # gene
    eQTL_lung_sig_dict[res[6]] = [] # p-value

    for line in f:
        line = line.decode('utf-8').strip()
        res = line.split('\t')
        eQTL_lung_sig_dict['variant_id'].append(res[0])
        eQTL_lung_sig_dict['gene_id'].append(res[1])
        eQTL_lung_sig_dict['pval_nominal'].append(res[6])


eQTL_lung_sig_dict['pval_nominal'] = pd.to_numeric(eQTL_lung_sig_dict['pval_nominal'])
eQTL_lung_sig_df = pd.DataFrame.from_dict(eQTL_lung_sig_dict)
eQTL_lung_sig_df = eQTL_lung_sig_df.sort_values(by=['pval_nominal'])

# === write to file
eQTL_lung_sig_df.to_csv('eQTL_lung_asc.csv', index=False)
'''

eQTL_lung_sig_df = pd.read_csv('eQTL_lung_asc.csv')
eQTLs = list(eQTL_lung_sig_df['variant_id'][:1000]) # top 1000 eQTLs

vcf_eQTL_f = open('vcf_filtered_by_eQTLs.tsv', 'w')
with gzip.open(vcf_file, 'rb') as f:
    for line in f:
        line = line.decode("utf-8")
        if line.startswith("##"):
            if line.startswith("##Sample"): # count samples
                print(len(line.split(';')))
            continue

        if line.startswith('#'): # deal with header
            vcf_eQTL_f.write(line[1:])
            continue

        res = line.split('\t')
        if res[6] != 'PASS' or res[2] not in eQTLs: # pass quanlity filter and in significant eQTLs
            continue
        vcf_eQTL_f.write(line)
        vcf_eQTL_f.flush()
vcf_eQTL_f.close()

