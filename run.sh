#!/bin/bash
#$ -cwd
#$ -q short.q
#$ -o out.log
#$ -e err.log

source ~/.bashrc
python extract_tissues.py
