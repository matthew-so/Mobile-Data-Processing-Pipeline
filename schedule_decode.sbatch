#!/bin/bash

#SBATCH --job-name=Mobile_Data_Processing
#SBATCH --account=gts-ts133
#SBATCH --nodes=1 --ntasks-per-node=24 --mem=0
#SBATCH --time=2:00:00
#SBATCH --qos=inferno
#SBATCH --output=Mobile_Data_Processing-%a.out
#SBATCH --mail-type=NONE
#SBATCH --array=1-40

cd $SLURM_SUBMIT_DIR
singularity exec --nv --bind /storage/home/hcoda1/9/gdeshpande9/p-ts133-0/Mobile-Data-Processing-Pipeline:/Mobile-Data-Processing-Pipeline --bind /storage/home/hcoda1/9/gdeshpande9/p-ts133-0/sign_language_videos:/data/sign_language_videos /storage/coda1/p-ts133/0/containers/copycat-gpu-cuda10.2-cudnn7.sif sh run_decode_slurm.sh
