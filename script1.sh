#!/bin/bash
#SBATCH -p qsu,zihuai
#SBATCH --mem=128G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time=05:00:00
#SBATCH -e error2.txt

ml python/3.6.1

#auto load public softwares
python3 main.py
