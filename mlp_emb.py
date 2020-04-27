import os, random
import pandas as pd
import numpy as np

# === my own package
import data

def transform_data(rc_data_file, snps_labels_file, embeddings):
    ''' transform the data into proper ML data
    '''

    snps_data = pd.read_csv(snps_labels_file, header=0, sep=',')
    rc_data = pd.read_csv(rc_data_file, header=0, sep='\t')

    snps_data.drop(['PHENOTYPE', 'IID'], axis=1, inplace=True)
    snps_samples = list(snps_data.loc[:, 'NEW_SAMPID'])

    # extract read counts according to snps samples
    rc_samples = list(rc_data.columns)[2:]
    rc_samples_extracted = []
    for rc_sample in rc_samples:
        if '-'.join(rc_sample.split('-')[:2]) in snps_samples:
            rc_samples_extracted.append(rc_sample)
    rc_samples_index = [list(rc_data.columns).index(_) for _ in rc_samples_extracted]
    rc_samples_index = [0, 1] + rc_samples_index # add gene name and transcripts into the index
    rc_data_filtered = rc_data.iloc[:, rc_samples_index] # select certain samples

    # filter gene with embeddings genes
    embedding_genes_indexes = [i for i in range(len(rc_data_filtered.Description)) if rc_data_filtered.Description[i] in embeddings.index]
    rc_data_filtered = rc_data_filtered.iloc[embedding_genes_indexes, :]
    
    embedding_list = []
    for i_col in range(2, len(rc_data_filtered.columns)):
        genes = rc_data_filtered.Description
        gene_embedding_vector = embeddings.loc[genes,:].to_numpy().astype(np.float)
        rc_vector = rc_data_filtered.iloc[:, i_col].to_numpy().astype(np.float)
        rc_vector_len = len(rc_vector)
        rc_vector_reshape = rc_vector.reshape((1, rc_vector_len))
        res = np.dot(rc_vector_reshape, gene_embedding_vector)
        embedding_list.append(res.tolist())
    embedding_array = np.array(embedding_list).reshape((len(rc_data_filtered.columns)-2, 200))

    rc_data_filtered.drop(columns=['Description'], axis=1, inplace=True)

    rc_data_trans = rc_data_filtered.set_index('Name').T # change into normal dataset
    rc_data_trans = rc_data_trans.loc[:, (rc_data_trans != 0).any(axis=0)] # remove 0s
    rc_data_trans = rc_data_trans.loc[:, rc_data_trans.std(axis=0) > 0.1] # remove std < 0.1


    rc_data_rownames = ['-'.join(_.split('-')[:2]) for _ in list(rc_data_trans.index)]
    snps_data = snps_data.set_index('NEW_SAMPID')
    snps_data_filtered = snps_data.loc[rc_data_rownames, :]

    # random select 100 SNPs
    random.seed(2020)
    random_indexes = random.sample(range(snps_data_filtered.shape[1]), 100)
    snps_data_filtered = snps_data_filtered.iloc[:, random_indexes]
    for snp in snps_data_filtered.columns:
        snps_data_filtered[snp] = snps_data_filtered[snp].astype(int)
    return embedding_array, snps_data_filtered


def get_embeddings(embedding_file):
    # === embedding size 200
    embedding_df = pd.read_csv(embedding_file, header=None, sep='\t')
    embedding_df.columns = ['gene', 'embeddings']

    embeddings = pd.concat([embedding_df.gene, embedding_df.embeddings.str.split(expand=True)], axis=1)
    embeddings = embeddings.set_index('gene')
    return embeddings


if __name__ == '__main__':
    embedding_file = "embeddings/gene2vec_dim_200_iter_9.txt"
    embeddings = get_embeddings(embedding_file) # get embeddings

    # transform data according to embeddings
    rc_data_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_genes_samples_rc.tsv')
    snps_labels_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_SNP.csv')
    rc_data_filtered, snps_data_filtered = transform_data(rc_data_file, snps_labels_file, embeddings)

    print(rc_data_filtered.shape)
    print(snps_data_filtered.shape)
 
