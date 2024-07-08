#!/bin/bash
#
#SBATCH --partition=gpu_min8GB                                      # Partition (check with "$sinfo")
#SBATCH --output=../MetaBreast/logs/mcvd/V1/output_sample.out           # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=../MetaBreast/logs/mcvd/V1/error_sample.out             # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=mcvd                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min8GB                                            # (Optional) 01.ctm-deep-05

#Commands / scripts to run (e.g., python3 train.py)
#python main.py --config configs/cfg_V1.yml --data_path ../../../datasets/private/METABREST/T1W_Breast/video_data --exp mcvd_V1 --ni
python main.py --config configs/cfg_V1_sample.yml --data_path ../../../datasets/private/METABREST/T1W_Breast/video_data --exp mcvd_V1 --ni --config_mod sampling.max_data_iter=1000 sampling.num_frames_pred=29 sampling.preds_per_test=10 sampling.subsample=100 model.version=DDPM --ckpt 105000 --video_gen -v mcvd_V1_100k
