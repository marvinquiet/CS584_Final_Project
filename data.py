import os
import gzip
import yaml

import numpy as np
import pandas as pd

CONFIG = "CONFIG.yaml"
config_data = None
with open(CONFIG, 'r') as fopen:
    try:
        config_data = yaml.safe_load(fopen)
    except yaml.YAMLError as exc:
        print(exc)

VCF = os.path.join(config_data['ori_prefix'], config_data['ori_files']['vcfs'])
ATTRIBUTES = os.path.join(config_data['ori_prefix'], config_data['ori_files']['attributes'])
eQTL_DIR = os.path.join(config_data['processed_prefix'], config_data['processed_files']['eQTLs'])
RC_DIR = os.path.join(config_data['ori_prefix'], config_data['ori_files']['read_counts'])
