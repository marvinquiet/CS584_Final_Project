import os, time
import pandas as pd
import numpy as np
from sklearn import preprocessing
import yaml
import random
from datetime import datetime

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

    return rc_data_trans, snps_data_filtered

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


def model(rc_dat, snps_label):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5) # 5-fold 

    snp_result = {}
    for train_index, test_index in kf.split(rc_dat):
        X_train, X_test = rc_dat.iloc[train_index, ].to_numpy(), rc_dat.iloc[test_index, ].to_numpy()
        y_train, y_test = snps_label.iloc[train_index, ], snps_label.iloc[test_index, ]
        
        now = datetime.now()
        # data normalization
        print("\n\n=== Data Normalization starting at", now.strftime("%H:%M:%S"))
        X_scaled_mean = X_train.mean(axis=0)
        X_scaled_std = X_train.std(axis=0)
        X_train_scaled = (X_train-X_scaled_mean)/X_scaled_std
        X_test_scaled = (X_test-X_scaled_mean)/X_scaled_std

        # print(X_scaled_std)
        # print(X_test_scaled)
        # print("Any NAs in train?", sum(np.isnan(X_train_scaled)))
        # print("Any NAs in test?", sum(np.isnan(X_test_scaled)))

        for snp in y_train.columns:
            y_snp_train = y_train.loc[:, snp].to_numpy()
            y_snp_test = y_test.loc[:, snp].to_numpy()

            if snp not in snp_result:
                snp_result[snp] = {}
                # snp_result[snp]['LR'] = []
                # snp_result[snp]['RF'] = []
                # snp_result[snp]['SVM'] = []
                snp_result[snp]['MLP'] = []
                snp_result[snp]['CNN'] = []

            '''
            # LR model result
            y_lr_pred_prob, y_lr_pred = logistic_regression(X_train_scaled, y_snp_train, X_test_scaled)
            snp_result[snp]['LR'].append(evaluation_metrics(snp, y_snp_test, y_lr_pred_prob, y_lr_pred))

            # RF model result
            y_rf_pred_prob, y_rf_pred = random_forest(X_train_scaled, y_snp_train, X_test_scaled)
            snp_result[snp]['RF'].append(evaluation_metrics(snp, y_snp_test, y_rf_pred_prob, y_rf_pred))

            # SVM model result
            y_svm_pred = svm(X_train_scaled, y_snp_train, X_test_scaled)
            snp_result[snp]['SVM'].append(evaluation_metrics(snp, y_snp_test, None, y_svm_pred))
            '''

            y_mlp_pred_prob, y_mlp_pred = mlp(snp, X_train_scaled, y_snp_train, X_test_scaled)
            snp_result[snp]['MLP'].append(evaluation_metrics(snp, y_snp_test, y_mlp_pred_prob, y_mlp_pred))
        print(snp_result)

def svm(X_train, y_train, X_test):
    from sklearn.svm import SVC
    clf = SVC(gamma='auto', random_state=2020).fit(X_train, y_train)
    return clf.predict(X_test)

def random_forest(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=2020).fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1], clf.predict(X_test)

def logistic_regression(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    # clf = LogisticRegressionCV(max_iter=5000, cv=5, random_state=2020, solver='saga',
    #         penalty='elasticnet', l1_ratios=[.1, .5, .7, .9, .95, .99, 1]).fit(X, y)

    clf = LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
            max_iter=5000, random_state=2020, solver='saga').fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1], clf.predict(X_test)

def mlp(snp, X_train, y_train, X_test):
    import mlp # import my own MLP package
    # === do train
    mlp_train = mlp.do_train(snp, X_train, y_train)
    # === do reference
    return mlp.do_infer(mlp_train, X_test)

if __name__ == '__main__':
    rc_data_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_genes_samples_rc.tsv')
    snps_labels_file = os.path.join(data.config_data['processed_prefix'], 'chr22/chr22_SNP.csv')

    rc_data_filtered, snps_data_filtered = transform_data(rc_data_file, snps_labels_file)
    print(rc_data_filtered.shape)
    print(snps_data_filtered.shape)
    model(rc_data_filtered, snps_data_filtered)

