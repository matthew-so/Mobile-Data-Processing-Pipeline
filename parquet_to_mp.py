import pandas as pd
import os
import random
import math
import json
import argparse
import pickle

from glob import glob
from tqdm import tqdm

# Consts
SEED = 5627
PADDING_DIGITS = 8
FIVE_SIGNS = ['yellow', 'who', 'listen', 'bye', 'cat']

# Globals
data_dict = {}

def add_metadata(participant, sign):
    if participant not in data_dict:
        data_dict[participant] = {}

    if sign not in data_dict[participant]:
        data_dict[participant][sign] = []
    
    data_dict[participant][sign].append({})
    
def ingest_landmark(row, min_frame, participant, sign):
    lm_type = row.type
    if row.type == "face":
        return

    if row.type == "right_hand":
        lm_type = "landmarks"
        hand = "0"
    elif row.type == "left_hand":
        lm_type = "landmarks"
        hand = "1"

    lm_index = str(row.landmark_index)
    frame = str(row.frame - min_frame)
    landmark_dict = data_dict[participant][sign][-1]

    if frame not in landmark_dict:
        landmark_dict[frame] = {}

    if lm_type not in landmark_dict[frame]:
        landmark_dict[frame][lm_type] = {}
    
    if lm_type == "landmarks":
        if hand not in landmark_dict[frame][lm_type]:
            landmark_dict[frame][lm_type][hand] = {}
        
        if math.isnan(row.x) or math.isnan(row.y) or math.isnan(row.z):
            return
        landmark_dict[frame][lm_type][hand][lm_index] = [row.x, row.y, row.z]
    else:
        if math.isnan(row.x) or math.isnan(row.y) or math.isnan(row.z):
            return
        landmark_dict[frame][lm_type][lm_index] = [row.x, row.y, row.z]

def add_landmark_data(participant, sign, abs_path):
    df = pd.read_parquet(abs_path, engine='pyarrow')
    min_frame = min(df.frame)
    df.apply(ingest_landmark, axis=1, args=(min_frame, participant, sign))

def ingest_parquet_files(df, parquet_dir):
    paths = df['path'].tolist()
    participants = df['participant_id'].tolist()
    signs = df['sign'].tolist()
    
    for row in tqdm(zip(paths, participants, signs), total=len(paths)):
        path = row[0]
        participant = str(row[1])
        sign = row[2]

        abs_path = os.path.join(parquet_dir, row[0])
        add_metadata(participant, sign)
        add_landmark_data(participant, sign, abs_path)
    
def get_train_csv(parquet_dir):
    train_csv_file = os.path.join(parquet_dir, 'train.csv')
    train_csv = pd.read_csv(train_csv_file)
    return train_csv

def get_train_data(parquet_dir):
    train_csv = get_train_csv(parquet_dir)
    signs = list(set(train_csv.sign.tolist()))
    signs.sort()

    print("Total Train Set Size: ", len(train_csv))
    # train_df = train_csv.loc[train_csv['sign'].isin(FIVE_SIGNS)]
    train_df = train_csv[['path', 'participant_id', 'sign']]
    print("Filtered Train Set Size: ", len(train_df))
    return train_df

def create_mp_files(mp_root):
    def pad(num):
        existing_len = len(str(num))
        return f'{(PADDING_DIGITS - existing_len) * "0"}{num}'
    
    for participant in data_dict:
        for sign in data_dict[participant]:
            for i, attempt in enumerate(data_dict[participant][sign]):
                attempt_count_str = pad(i + 1)
                
                filename_components = [participant, sign, 'singlesign', attempt_count_str, 'data']
                filename = '.'.join(filename_components)
                
                mp_dir = os.path.join(mp_root, participant + '-singlesign', sign)
                if not os.path.exists(mp_dir):
                    os.makedirs(mp_dir)
                
                mp_filepath = os.path.join(mp_dir, filename)
                with open(mp_filepath, 'w') as f:
                    json.dump(attempt, f, indent=4)

def add_user_sign_count(row, count_dict):
    if row.participant_id not in count_dict:
        count_dict[row.participant_id] = {}

    if row.sign not in count_dict[row.participant_id]:
        count_dict[row.participant_id][row.sign] = 0

    count_dict[row.participant_id][row.sign] += 1

def add_sign_count(row, count_dict):
    if row.sign not in count_dict:
        count_dict[row.sign] = 0

    count_dict[row.sign] += 1

def print_user_sign_counts(parquet_dir):
    df = get_train_csv(parquet_dir)
    count_dict = {}
    df.apply(add_user_sign_count, axis=1, args=(count_dict,))
    
    for participant in count_dict:
        print(f'Participant: {participant}')
        for sign in count_dict[participant]:
            print(f'\t{sign}: {count_dict[participant][sign]}')

def print_sign_counts(parquet_dir):
    df = get_train_csv(parquet_dir)
    count_dict = {}
    df.apply(add_sign_count, axis=1, args=(count_dict,))
    
    for participant in count_dict:
        print(f'Participant: {participant}')
        for sign in count_dict[participant]:
            print(f'\t{sign}: {count_dict[participant][sign]}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_pickle_file', action='store_true',
                        help='If True, will look for data_dict.pickle to load the ' + \
                                'parquet data. Useful for debugging.')
    parser.add_argument('--create_pickle_file', action='store_true',
                        help='If True will create data_dict.pickle from the processed ' + \
                                'parquet data. Useful for debugging.')
    parser.add_argument('--parquet_dir', type=str, default='kaggle_parquet',
                        help='Points to the dir containing train_landmark_files.')
    parser.add_argument('--dest_dir', type=str, default='mediapipe_parquet',
                        help='The root directory for the mediapipe JSON file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.use_pickle_file:
        with open('data_dict.pickle', 'rb') as f:
            data_dict = pickle.load(f)
    else:
        train_df = get_train_data(args.parquet_dir)
        ingest_parquet_files(train_df, args.parquet_dir)
        
        # total_signs = 0
        # for participant in data_dict:
        #     print(f'Participant {participant}')
        #     for sign in data_dict[participant]:
        #         print(f'\t{sign}: {len(data_dict[participant][sign])}')
        #         total_signs += len(data_dict[participant][sign])
        # print(f'Total signs: {total_signs}')
    
    create_mp_files(args.dest_dir)
    
    if args.create_pickle_file:
        with open('data_dict.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

