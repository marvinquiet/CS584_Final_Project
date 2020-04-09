import os
import gzip

attributes = "/compbioscratch2/yhua295/dbGap-17031/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v7.p2.c1.GRU/PhenotypeFiles/phs000424.v7.pht002743.v7.p2.c1.GTEx_Sample_Attributes.GRU.txt.gz"

import pandas as pd
import numpy as np

lung_f = open("lung_samples.tsv", 'w')

with gzip.open(attributes, 'rb') as f:
    for line in f:
        line = line.decode('utf-8')
        if 'Lung' in line:
            lung_f.write(line)
lung_f.close()
