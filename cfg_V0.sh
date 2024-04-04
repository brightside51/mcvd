#!/bin/bash
#
#SBATCH --partition=gpu_min11GB                                      # Partition (check with "$sinfo")
#SBATCH --output=../MetaBreast/logs/mcvd/V0/output.out           # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=../MetaBreast/logs/mcvd/V0/error.out             # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=mcvd                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min11GB                                            # (Optional) 01.ctm-deep-05

#Commands / scripts to run (e.g., python3 train.py)
#python main.py --config configs/cfg_V0.yml --data_path ../../../datasets/private/METABREST/T1W_Breast/video_data --exp mcvd_V0 --ni
python main.py --config configs/smmnist_DDPM_big5_lite.yml --data_path ../../../datasets/private/METABREST/T1W_Breast/video_data --exp mcvd_V0 --ni
