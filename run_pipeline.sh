#!/bin/bash

# Add raw videos from ASL recorder to raw/. Then run this script.

python decode.py --backup_dir raw/ --dest_dir split/ --num_threads 32
python mediapipe_convert.py --inputDirectory split/ --outputDirectory mediapipe/ --noMark
(cd hmm/main/projects/sign_recognition && ./prepare_htk_data.sh)

