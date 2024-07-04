#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate nina_bep
cd

#RNN
lang="hi"
seed=2342
python train_modified.py --language $lang --seed ${seed} --delay 1 --min_frequency 24 --hidden_dim 250 \
--embedding_dim 100 --lr 0.01 \
--input_data "$(realpath ../u802098/${lang}_data.txt)"
