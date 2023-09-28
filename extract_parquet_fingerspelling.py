import argparse
import os
import json
import shutil
import random

import pandas as pd
pd.set_option('display.max_rows', None)

from glob import glob
from tqdm import tqdm

PQ_FILES = 'supplemental_landmarks'
PQ_METADATA = 'supplemental_metadata.csv'

NUM_LANDMARKS = 21
COORDINATES = ['x','y','z']
# NULL_SEQ_THRESH = [0.85, 0.9, 0.95, 0.97, 0.99]
# NULL_SEQ_THRESH = [0.5, 0.6, 0.7, 0.8, 0.9]
# NULL_SEQ_THRESH = [0.8]

# Global Counter
skipped_sequences = 0

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--parquet-path', '-pp', type=str, default='/root/kaggle', help='Path to parquet files.')
    parser.add_argument('--fill-na', '-fna', type=float, default=0.0, choices=[-1.0, 0.0], help='Fill NA values with this value.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Set threshold of null rows for filtering.')
    parser.add_argument('--min-frame-len', type=int, default=1, help='Set minimum number of frames for a sequence.')
    parser.add_argument('--run-debug', '-rd', action='store_true', help='Run Debug Sequence instead of main script.')

    return parser.parse_args()

def get_hand_cols(parquet_file):
    df = pd.read_parquet(parquet_file)
    
    hand_cols = ['frame']
    for col in df.columns:
        if "_hand_" in col:
            hand_cols.append(col)
    
    return df, hand_cols

def get_handedness(frames):
    rh_count = 0
    lh_count = 0
    for _,frame in frames.iterrows():
        if not _frame_is_null(frame, 'right'):
            rh_count += 1
        elif not _frame_is_null(frame, 'left'):
            lh_count += 1

    return 'right' if rh_count > lh_count else 'left'

def _skip_sequence(frames, handedness, args):
    
    if len(frames) <= args.min_frame_len:
        return True

    null_frames = 0
    for _,frame in frames.iterrows():
        null_frames += int(_frame_is_null(frame, handedness))
    
    if null_frames / len(frames) > args.threshold:
        return True

def get_parquet_data_from_file(df, metadata, args):
    sequence_ids = list(set(df.index.tolist()))
    null_sequences = 0
    for sequence_id in tqdm(sequence_ids, position=1, leave=False, desc='Sequences'):
    # for sequence_id in tqdm(sequence_ids[:10], position=1, leave=False, desc='Sequences'):
        seq_id_key = str(sequence_id)
        data_dict = {}
        
        frames = df.loc[[sequence_id]]
        handedness = get_handedness(frames)
        
        if _skip_sequence(frames, handedness, args):
            global skipped_sequences
            skipped_sequences += 1
            continue

        for _,frame in frames.iterrows():
            frame_key = str(int(frame['frame']))
            data_dict[frame_key] = {'landmarks': {'0':{}, '1':{}}}

            lm_key = str(int(handedness == 'left'))
            landmarks_dict = data_dict[frame_key]['landmarks'][lm_key]
            is_null = _frame_is_null(frame, handedness)

            for i in range(NUM_LANDMARKS):
                hand = []
                for coord in COORDINATES:
                    if is_null:
                        val = args.fill_na
                    else:
                        val = frame['_'.join([coord, handedness, 'hand', str(i)])].item()
                    hand.append(val)
                
                landmarks_dict[str(i)] = hand
        
        write_data_file(data_dict, metadata, sequence_id)

def _frame_is_null(frame, handedness):
    frame_key = '_'.join(['x', handedness, 'hand', '0'])
    return pd.isna(frame[frame_key])

def _print_frame_status(all_null, frame):
    if all_null:
        print("All Frames Null")
    else:
        if _frame_is_null(frame, 'right'):
            print("RH Null")
        elif _frame_is_null(frame, 'left'):
            print("LH Null")
        else:
            print("RH/LH Both Non-Null")
            print("Frame: ", list(frame))

def _debug_frames(frames, handedness, threshold=0.9):
    count_null_frames = 0
    keep_cols = []
    colnames = list(frames.columns)

    if handedness is not None:
        for i,colname in enumerate(colnames):
            if handedness in colname:
                keep_cols.append(i)
    else:
        keep_cols = list(range(1, len(colnames)))

    for _,frame in frames.iterrows():
        all_null = frame[keep_cols].isnull().all()
        # _print_frame_status(all_null, frame)
        if all_null:
            count_null_frames += 1
    
    pct_null = count_null_frames / len(frames)
    if pct_null > threshold:
        # print(frames)
        # print()
        return 1
    return 0

def _run_debug(df, metadata, threshold):
    sequence_ids = list(set(df.index.tolist()))
    random.shuffle(sequence_ids)
    subset = 100

    null_sequences = 0
    # for sequence_id in tqdm(sequence_ids, position=2, leave=False, desc='Sequences'):
    for sequence_id in tqdm(sequence_ids[:subset], position=2, leave=False, desc='Sequences'):
        seq_id_key = str(sequence_id)
        data_dict = {}
        frames = df.loc[[sequence_id]]
        
        if len(frames) == 1:
            continue

        handedness = get_handedness(frames)

        null_sequences += _debug_frames(frames, handedness, threshold=threshold)
    
    pct_null_sequences = null_sequences / subset
    return pct_null_sequences

def _interpret_debug_stats(debug_stats):
    for threshold in NULL_SEQ_THRESH:
        avg_pct_null = sum(debug_stats[threshold]) / len(debug_stats[threshold])
        print(f"Threshold: {threshold} | Pct Null Seq: {avg_pct_null}")

def run_parquet_proc(
    parquet_files,
    metadata,
    args
):
    _, hand_cols = get_hand_cols(parquet_files[0])
    
    if args.run_debug:
        print("Running Debug")
        debug_stats = {threshold: [] for threshold in NULL_SEQ_THRESH}
    else:
        print("Extracting Parquet Data from Files")

    for parquet_file in tqdm(parquet_files, position=0, leave=False, desc='Parquet Files'):
    # for parquet_file in tqdm(parquet_files[:5], position=0, leave=False, desc='Parquet Files'):
        df = pd.read_parquet(parquet_file)
        df = df[hand_cols]

        if args.run_debug:
            for threshold in tqdm(NULL_SEQ_THRESH, position=1, leave=False, desc='Thresholds'):
                debug_stats[threshold].append(_run_debug(df, metadata, threshold))
        else:
            get_parquet_data_from_file(df, metadata, args)
    
    if args.run_debug:
        _interpret_debug_stats(debug_stats)

def write_data_file(data_dict, metadata, sequence_id):
    row = metadata.loc[sequence_id]
    
    user_id = str(row.participant_id)
    phrase = row.phrase.replace(' ', '_')
    uniq_id = str(sequence_id)
    
    filename = '-'.join([user_id, phrase, uniq_id, '0']) + '.data'
    filedir = os.path.join('mediapipe', 'fingerspelling', user_id)

    if not os.path.exists(filedir):
        os.makedirs(filedir)
    
    filepath = os.path.join(filedir, filename)

    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    print(args)

    parquet_pattern = os.path.join(args.parquet_path, PQ_FILES, "*.parquet")
    metadata_file = os.path.join(args.parquet_path, PQ_METADATA)
    
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.set_index('sequence_id')
    
    if os.path.exists('mediapipe/fingerspelling/'):
        shutil.rmtree('mediapipe/fingerspelling/')
        os.mkdir('mediapipe/fingerspelling/')

    parquet_files = glob(parquet_pattern)
    run_parquet_proc(
        parquet_files,
        metadata,
        args
    )
    
    print("Skipped Sequences: ", skipped_sequences)

