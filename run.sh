#!/bin/bash
#$ -cwd
#$ -q short.q
#$ -o out.log
#$ -e err.log

source ~/.bashrc
conda activate phege_proj

# --- get vcf according to variants name
vcftools --gzvcf /compbioscratch2/yhua295/dbGap-17031/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v7.p2.c1.GRU/GenotypeFiles/phg000830.v1.GTEx_WGS.genotype-calls-vcf.c1/GTEx_Analysis_2016-01-15_v7_WholeGenomeSeq_635Ind_PASS_AB02_GQ20_HETX_MISS15_PLINKQC.vcf.gz --snps data/whole_blood_chr22_top1000_eQTLs.txt --recode --recode-INFO-all --out data/whole_blood_chr22_top1000_eQTLs

# --- recode vcf
plink --bfile data/whole_blood_chr22_top1000_eQTLs --recodeAD --out data/whole_blood_chr22_top1000_eQTLs --mind 0.2 --geno 0.2 --hwe 0.05 --hwe-all

# --- generate
plink --vcf data/whole_blood_chr22_top1000_eQTLs.recode.vcf --make-bed --out data/whole_blood_chr22_top1000_eQTLs
