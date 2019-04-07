#!/bin/bash
#SBATCH --mail-type=END
##SBATCH --mail-user=ebc327@nyu.edu
#SBATCH --mem=32GB
#SBATCH -t12:00:00

module load anaconda3/5.3.0
. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh
conda activate HPCenv
cd MLproject
srun fasttext print-word-vectors /scratch/ebc327/cc.tr.300.bin <  myfile.txt > word2vec.txt
