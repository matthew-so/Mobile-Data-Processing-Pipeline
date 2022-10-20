#!/bin/bash

# Add raw videos from ASL recorder to raw/. Then run this script.

python decode.py --backup_dir $1 --dest_dir $2 --num_threads 32
python mediapipe_convert.py --inputDirectory $2 --outputDirectory $3 --noMark
(cd hmm/main/projects/sign_recognition && ./prepare_htk_data.sh)

