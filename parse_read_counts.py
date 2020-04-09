import os
import gzip
import pandas as pd

rc_file = "/compbioscratch2/yhua295/dbGap-17031/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v7.p2.c1.GRU/ExpressionFiles/phe000020.v1.GTEx_RNAseq.expression-data-matrixfmt.c1/GTEx_Data_20160115_v7_RNAseq_RNASeQCv1.1.8_gene_reads.gct.gz"
lung_samples = "lung_samples.tsv"

lung_sample_ids = []
with open(lung_samples, 'r') as fopen:
    for line in fopen:
        res = line.split('\t')
        id_split = res[1].split('-')
        sample_id = id_split[0] + '-' + id_split[1]
        lung_sample_ids.append(sample_id)

vcf_filtered = "vcf_filtered_by_eQTLs.tsv.tmp"
with open(vcf_filtered, 'r') as fopen:
    header = fopen.readline().strip().split('\t')

intersected_sampes = set.intersection(set(lung_sample_ids), set(header))
print(intersected_sampes)
print(len(intersected_sampes))

exit()

eQTL_lung_sig_df = pd.read_csv('eQTL_lung_asc.csv')
eQTL_genes = set(eQTL_lung_sig_df['gene_id']) # all genes from lung eQTL

f_rc = open('lung_rc.tsv', 'w')
with gzip.open(rc_file, 'rb') as fopen:
    # skip first two lines
    fopen.readline()
    fopen.readline()

    header = fopen.readline().decode('utf-8').strip()
    header_list = header.split('\t')
    
    index_list = [0, 1]
    for lung_sample in lung_sample_ids:
        if lung_sample in header_list:
            index_list.append(header_list.index(lung_sample))
    print(len(index_list))

    header_line = ""
    for index in index_list:
        header_line += header_list[index]
        header_line += '\t'
    header_line += '\n'
    f_rc.write(header_line)
    f_rc.flush()

    for line in fopen:
        line = line.decode('utf-8').strip()
        res = line.split('\t')
        if res[0] not in eQTL_genes:
            continue
        new_line = ""
        for index in index_list:
            new_line += res[index]
            new_line += '\t'
        new_line += '\n'
        f_rc.write(new_line)
        f_rc.flush()
f_rc.close()
