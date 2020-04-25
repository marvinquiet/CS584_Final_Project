#!/bin/bash
#$ -cwd
#$ -q short.q
#$ -o out.log
#$ -e err.log

source ~/.bashrc

conda activate phege_proj
# python build_env.py
python model.py
