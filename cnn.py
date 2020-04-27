import os, time, datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
import yaml
import random
from datetime import datetime

# === my own package
import data
import mlp_emb
import model

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
    embedding_genes_index = []
    for i in range(len(rc_data_filtered.Description)):
        gene_name = rc_data_filtered.Description[i]
        if gene_name in embeddings.index:
            embeddings_index = list(embeddings.index).index(gene_name)
            embedding_genes_index.append((i, embeddings_index))
    embedding_genes_index.sort(key=lambda x:x[1])
    embedding_genes_indexes = [_[0] for _ in embedding_genes_index]
    rc_data_filtered = rc_data_filtered.iloc[embedding_genes_indexes, :]
    
    embedding_list = []
    for i_col in range(2, len(rc_data_filtered.columns)):
        genes = rc_data_filtered.Description
        gene_embedding_vector = embeddings.loc[genes,:].to_numpy().astype(np.float)
        rc_vector = rc_data_filtered.iloc[:, i_col].to_numpy().astype(np.float)
        rc_vector_len = len(rc_vector)
        rc_vector_reshape = rc_vector.reshape((rc_vector_len, 1))
        res = np.multiply(rc_vector_reshape, gene_embedding_vector)
        embedding_list.append(res.tolist())
    embedding_array = np.array(embedding_list)

    rc_data_filtered.drop(columns=['Description'], axis=1, inplace=True)

    rc_data_trans = rc_data_filtered.set_index('Name').T # change into normal dataset
    rc_data_trans = rc_data_trans.loc[:, (rc_data_trans != 0).any(axis=0)] # remove 0s
    rc_data_trans = rc_data_trans.loc[:, rc_data_trans.std(axis=0) > 0.3] # remove std < 0.3

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


# ====== CNN model part
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GeneCNN(nn.Module):
    '''fixed embeddings (pre-trained by 784 datasets in paper)
    '''
    def __init__(self, emb_dim, out_channels=4):
        super(GeneCNN, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, out_channels, (3, emb_dim),
                               stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv2d(1, out_channels, (5, emb_dim),
                               stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        # FC layer
        self.fc = nn.Linear(out_channels*2, 1)

    def forward(self, X):
        out1 = F.relu(self.conv1(X))
        out1 = self.maxpool(self.bn1(out1.squeeze(3)))

        out2 = F.relu(self.conv2(X))
        out2 = self.maxpool(self.bn2(out2.squeeze(3)))

        fusion = torch.cat((out1.squeeze(2), out2.squeeze(2)), 1)

        out = self.fc(fusion)
        return torch.sigmoid(out)

def evaluation_metrics(snp, y_true, y_pred_prob, y_pred):
    # print("\n\n ==== Evaluation Metrics for {} ==== ".format(snp))
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    eval_dict = {}
    # === AUC score
    if y_pred_prob is not None:
        auc = roc_auc_score(y_true, y_pred_prob)
        # print("AUC score:", auc)
        eval_dict['AUC'] = auc

    # === F1 score
    f1 = f1_score(y_true, y_pred)
    eval_dict['F1'] = f1

    # === Accuracy score
    acc = accuracy_score(y_true, y_pred)
    eval_dict['Acc'] = acc
    return eval_dict

def cross_validation(X, y):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, random_state=2020) # 5-fold 

    snp_result = {}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, ], X[test_index, ]
        y_train, y_test = y.iloc[train_index, ], y.iloc[test_index, ]
        
        now = datetime.now()
        # data normalization
        print("\n\n=== Data Normalization starting at", now.strftime("%H:%M:%S"))
        X_scaled_mean = X_train.mean(axis=0)
        X_scaled_std = X_train.std(axis=0)
        X_scaled_std[X_scaled_std == 0] = 1 # if standard deviation is 0

        X_train_scaled = (X_train-X_scaled_mean)/X_scaled_std
        X_test_scaled = (X_test-X_scaled_mean)/X_scaled_std
        print("=== Data Normalization finish....")

        for snp in y_train.columns:
            print("Deal with %s... " % snp)
            y_snp_train = y_train.loc[:, snp].to_numpy()
            y_snp_test = y_test.loc[:, snp].to_numpy()

            if snp not in snp_result:
                snp_result[snp] = {}
                snp_result[snp]['CNN'] = []

            # CNN model result
            cnn_model = do_train(snp, X_train_scaled, y_snp_train)
            cnn_prob, cnn_pred = do_infer(cnn_model, X_test_scaled)
            snp_result[snp]['CNN'].append(evaluation_metrics(snp, y_snp_test, cnn_prob, cnn_pred))
        print(snp_result)


# === hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

def do_train(snp, X_train, y_train):
    X = torch.from_numpy(X_train).type(torch.FloatTensor)
    y = torch.from_numpy(y_train).type(torch.FloatTensor)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    gene_cnn = GeneCNN(emb_dim=200).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(gene_cnn.parameters(), lr=LEARNING_RATE)
    
    for i_epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = []
        for i_batch, sample_batched in enumerate(data_loader):
            gene_embedding, train_labels = sample_batched
            pred = gene_cnn(gene_embedding[:, None].to(device))
            loss = criterion(pred.squeeze(), train_labels.to(device))
            gene_cnn.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
        if i_epoch % 10 == 0:
            print('Epoch [%d/%d], Loss: %.6f, SNP: %s' % (i_epoch, NUM_EPOCHS,
                sum(epoch_loss)/len(epoch_loss), snp))
    return gene_cnn

def do_infer(train_model, X_test):
    all_preds_prob, all_preds_labels = [], []
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    test_data_loader = DataLoader(TensorDataset(X_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    train_model.eval()
    for i_batch, sample_batched in enumerate(test_data_loader):
        test_data = sample_batched[0] # get test_data
        pred = train_model(test_data[:, None].to(device))
        all_preds_prob.extend(pred.squeeze().cpu().detach().tolist())
    all_preds_prob = np.array(all_preds_prob)
    all_preds_labels = np.where(all_preds_prob >= 0.5, 1, 0)
    return all_preds_prob, all_preds_labels


if __name__ == '__main__':
    embedding_file = "embeddings/gene2vec_dim_200_iter_9.txt"
    embeddings = mlp_emb.get_embeddings(embedding_file) # get embeddings

    # transform data according to embeddings
    rc_data_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_genes_samples_rc.tsv')
    snps_labels_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_SNP.csv')
    rc_data_embedding, snps_data_filtered = transform_data(rc_data_file, snps_labels_file, embeddings)

    print(rc_data_embedding.shape)
    print(snps_data_filtered.shape)

    cross_validation(rc_data_embedding, snps_data_filtered)
