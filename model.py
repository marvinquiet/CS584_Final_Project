import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

# === my own package
import data

def transform_data(rc_data_file, snps_labels_file):
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
    rc_samples_index = [0] + rc_samples_index # add gene name into the index
    rc_data_filtered = rc_data.iloc[:, rc_samples_index] # select certain samples
    rc_data_trans = rc_data_filtered.set_index('Name').T # change into normal dataset
    rc_data_trans = rc_data_trans.loc[:, (rc_data_trans != 0).any(axis=0)] # remove 0s

    rc_data_rownames = ['-'.join(_.split('-')[:2]) for _ in list(rc_data_trans.index)]
    snps_data = snps_data.set_index('NEW_SAMPID')
    snps_data_filtered = snps_data.loc[rc_data_rownames, :]
    return rc_data_trans, snps_data_filtered

def evaluation_metrics(snp, y_true, y_pred):
    print("\n\n ==== Evaluation Metrics for {} ==== ".format(snp))
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred)
    print("AUC score:", auc)


def model(rc_dat, snps_label):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10) # 5-fold 

    for train_index, test_index in kf.split(rc_dat):
        X_train, X_test = rc_dat.iloc[train_index, ].to_numpy(), rc_dat.iloc[test_index, ].to_numpy()
        y_train, y_test = snps_label.iloc[train_index, ], snps_label.iloc[test_index, ]

        # data normalization
        print("\n\n=== Data Normalization... ")
        X_scaled_mean = X_train.mean(axis=0)
        X_scaled_std = X_train.std(axis=0)
        X_train_scaled = preprocessing.scale(X_train)
        X_test_scaled = (X_test-X_scaled_mean)/X_scaled_std
        # print(X_scaled_std)
        # print(X_test_scaled)

        for snp in y_train.columns:
            y_snp_train = y_train.loc[:, snp].to_numpy()
            y_snp_test = y_test.loc[:, snp].to_numpy()

            # model result
            lr = logistic_regression(X_train_scaled, y_snp_train)
            y_lr_pred = lr.predict_proba(X_test_scaled)[:, 1]
            evaluation_metrics(snp, y_snp_test, y_lr_pred)

def logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    clf = LogisticRegressionCV(max_iter=5000, cv=5, random_state=2020, solver='saga',
            penalty='elasticnet', l1_ratios=[.1, .5, .7, .9, .95, .99, 1]).fit(X, y)

    # clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
    #         max_iter=5000, random_state=2020, solver='saga').fit(X, y)

    return clf


def mlp(X, y):
    pass


if __name__ == '__main__':
    rc_data_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_genes_samples_rc.tsv')
    snps_labels_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_SNP.csv')

    rc_data_filtered, snps_data_filtered = transform_data(rc_data_file, snps_labels_file)
    model(rc_data_filtered, snps_data_filtered)

