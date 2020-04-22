#!/bin/bash
#$ -cwd
#$ -q short.q
#$ -o out.log
#$ -e err.log

source ~/.bashrc
python build_env.py
