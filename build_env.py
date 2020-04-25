import os
import gzip
import yaml

import pandas as pd
import numpy as np

# === my packages
import data

def get_sample_ids(tissue):
    '''extract sample IDs from certain tissue
    '''

    sample_ids = []
    with gzip.open(data.ATTRIBUTES, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            if tissue in line:
                res = line.split('\t')
                sample_ids.append(res[1])
    print("=== sample IDs length:", len(sample_ids))
    return sample_ids

def get_genes(gene_file):
    genes = []
    with open(gene_file, 'r') as f:
        f.readline() # skip heads
        for line in f:
            genes.append(line.strip().split('\t')[0])
    print("=== genes num:", len(genes))
    return genes

def get_rc_matrix(sample_ids, genes, result_rc_file): 
    '''get read count matrix of gene*sample_ids

    genes: list - gene symbols
           None - without any restrictions
    '''

    frc = open(result_rc_file, 'w')

    with gzip.open(data.RC_DIR, 'rb') as f:
        # skip first two information lines
        f.readline()
        f.readline()

        header = f.readline().decode('utf-8').strip()
        header_list = header.split('\t')
        index_list = [0, 1]

        header_line = header_list[0] + '\t' + header_list[1] + '\t'
        for sample in sample_ids:
            if sample in header_list:
                index_list.append(header_list.index(sample))
                header_line += sample + '\t'
        header_line = header_line.rstrip('\t') # remove the last '\t'
        header_line += '\n'
        frc.write(header_line)
        print("=== length of index: ", len(index_list))

        for line in f:
            line = line.decode('utf-8').strip()
            res = line.split('\t')
            if genes is None or (genes is not None and res[1] in genes):
                selected_rcs = [res[index] for index in index_list]
                new_line = '\t'.join(str(rc) for rc in selected_rcs)+'\n'
                frc.write(new_line)
                frc.flush()
    frc.close()


if __name__ == '__main__':
    tissue = "Whole Blood"

    print("get sample IDs...")
    sample_ids = get_sample_ids(tissue)
    
    print("get genes...")
    gene_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_genes.txt')
    genes = get_genes(gene_file)
    
    print("filter gene read counts by genes and sample IDs...")
    # filtered_rc_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_genes_samples_rc.tsv')
    filtered_rc_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_all_genes_samples_rc.tsv')
    get_rc_matrix(sample_ids, None, filtered_rc_file)
