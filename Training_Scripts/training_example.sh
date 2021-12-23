#!/usr/bin/env bash
#SBATCH --partition=elec.gpu.q
#SBATCH --gres=gpu:1
python Basis.py --config_custom=Example_Config.cfg &
wait
