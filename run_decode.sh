#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/compat 
cd /Mobile-Data-Processing-Pipeline
python3 create_batches.py --num_batches 40
python3 decode.py --job_array_num ${PBS_ARRAYID} --ffmpeg_loglevel info --backup_dir /data/sign_language_videos/raw --dest_dir /data/sign_language_videos/split --num_threads 24