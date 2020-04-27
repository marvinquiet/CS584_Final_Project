import os
import json

import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

def load_result(result_file):
    f = open(result_file, 'r')
    result = json.load(f)
    f.close()

    df = pd.DataFrame.from_dict(result)

    res_dict = {}
    res_dict['method'] = []
    res_dict['f1'] = []
    res_dict['snp'] = []
    for method in df.index:
        for snp in df.columns:
            f1s = []
            for evals in df.loc[method, snp]:
                if 'F1' in evals:
                    f1s.append(evals['F1'])
                else:
                    continue
            if len(f1s) > 0:
                res_dict['f1'].append(sum(f1s)/len(f1s))
                res_dict['snp'].append(snp)
                res_dict['method'].append(method)
    res_df = pd.DataFrame.from_dict(res_dict)

    plt.figure(figsize=(20,20))
    plt.ylim(-0.05, 1.0)
    sns_plot = sns.lineplot(x="snp", y="f1", hue="method", data=res_df)
    plt.xticks(rotation=90)

    sns_plot.set_title('Average F1 score on 100 SNPs (ML and MLP with embedding)')
    sns_plot.set_ylabel("Average F1 score")
    sns_plot.set_xlabel("SNPs")

    fig = sns_plot.get_figure()
    fig.tight_layout()
    fig.savefig("results/ML_MLP_emb_result.png")

    return res_df

def load_ml_and_mlp_result(ml_result_file, mlp_result_file):
    f = open(ml_result_file, 'r')
    ml_result = json.load(f)
    f.close()

    f = open(mlp_result_file, 'r')
    mlp_result = json.load(f)
    f.close()

    res_dict = {}
    res_dict['method'] = []
    res_dict['f1'] = []
    res_dict['snp'] = []

    df = pd.DataFrame.from_dict(ml_result)
    for method in df.index:
        for snp in df.columns:
            f1s = []
            for evals in df.loc[method, snp]:
                if 'F1' in evals:
                    f1s.append(evals['F1'])
                else:
                    continue
            if len(f1s) > 0:
                res_dict['f1'].append(sum(f1s)/len(f1s))
                res_dict['snp'].append(snp)
                res_dict['method'].append(method)
    
    df = pd.DataFrame.from_dict(mlp_result)
    for method in df.index:
        for snp in df.columns:
            f1s = []
            for evals in df.loc[method, snp]:
                if 'F1' in evals:
                    f1s.append(evals['F1'])
                else:
                    continue
            if len(f1s) > 0:
                res_dict['f1'].append(sum(f1s)/len(f1s))
                res_dict['snp'].append(snp)
                res_dict['method'].append(method)

    res_df = pd.DataFrame.from_dict(res_dict)

    plt.figure(figsize=(20,20))
    plt.ylim(-0.05, 1.0)
    sns_plot = sns.lineplot(x="snp", y="f1", hue="method", data=res_df)
    plt.xticks(rotation=90)

    sns_plot.set_title('Average F1 score on 100 SNPs (ML and MLP without embeddings)')
    sns_plot.set_ylabel("Average F1 score")
    sns_plot.set_xlabel("SNPs")

    fig = sns_plot.get_figure()
    fig.tight_layout()
    fig.savefig("results/ML_MLP_noemb_result.png")

    return res_df


def res_correlation(res_df, snp_sig_df):
    '''Get correlation between noemb df and snp sig df
    '''
    for method in set(res_df.method):
        method_df = res_df[res_df["method"] == method]
        snp_sig_list = []
        for method_snp in method_df.snp:
            for i in range(len(snp_sig_df.iloc[:, 0])):
                snp_sig = snp_sig_df.iloc[i, 0]
                if method_snp.startswith(snp_sig):
                    snp_sig_list.append(snp_sig_df.iloc[i, 1])
                    break
        print(method, spearmanr(snp_sig_list, list(method_df.f1)))


if __name__ == '__main__':
    ml_mlp_noemb_df = load_ml_and_mlp_result("results/ml.result", "results/mlp.result")
    ml_mlp_emb_df = load_result("results/mlp_emb.result")

    snp_sig_file = "data/chr22/whole_blood_chr22_eQTLs.tsv"
    snp_sig_df = pd.read_csv(snp_sig_file, header=None, sep='\t') 
    snp_sig_df = snp_sig_df.iloc[:1000, [0, 6]]

    print("--- correlation with no embedings")
    res_correlation(ml_mlp_noemb_df, snp_sig_df)

    print("--- correlation with embeddings")
    res_correlation(ml_mlp_emb_df, snp_sig_df)
