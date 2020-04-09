import os
import gzip
import yaml

import numpy as np
import pandas as pd

def load_config(config_file):
    config_data = None
    with open(config_file, 'r') as fopen:
        try:
            config_data = yaml.safe_load(fopen)
        except yaml.YAMLError as exc:
            print(exc)
    return config_data



def load_gzip_data(path):
    ''' load gzip data, convert to pandas dataframe
    '''



if __name__ == '__main__':
    config_data = load_config("CONFIG.yaml")
    base_dir = config_data['data_path']
    attribute_file = config_data['files']['attributes']


    read_counts_file = config_data['files']['read_counts']
