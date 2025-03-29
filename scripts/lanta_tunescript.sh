#!/bin/bash -l
#SBATCH -p memory                 
#SBATCH -N 32                      
#SBATCH --cpus-per-task=32        
#SBATCH -t 20:00:00               
#SBATCH --mail-user=kittitattuntisak@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -J tune_graphsage         
#SBATCH -A pc701039               

module load Mamba/23.11.0-0
conda activate microw-boosting-graphsage

cd /project/pc701039-microw/ktuntisak/Boosting-GraphSAGE

python -u hypertune.py
