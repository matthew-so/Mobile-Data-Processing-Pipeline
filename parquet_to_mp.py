import pandas as pd
import os
import random
import math
import json

from glob import glob
from tqdm import tqdm

# Consts
SEED = 5627
PADDING_DIGITS = 8
PARQUET_DIR = '/data/kaggle-data/competition-data-train/'
TRAIN_CSV = '/data/kaggle-data/competition-data-train/train.csv'

# Globals
data_dict = {}

def add_metadata(participant, sign):
    if participant not in data_dict:
        data_dict[participant] = {}

    if sign not in data_dict[participant]:
        data_dict[participant][sign] = {}

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
    landmark_dict = data_dict[participant][sign]

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

def ingest_parquet_files(df):
    paths = df['path'].tolist()
    participants = df['participant_id'].tolist()
    signs = df['sign'].tolist()
    
    for row in tqdm(zip(paths, participants, signs), total=len(paths)):
        path = row[0]
        participant = str(row[1])
        sign = row[2]

        abs_path = os.path.join(PARQUET_DIR, row[0])
        add_metadata(participant, sign)
        add_landmark_data(participant, sign, abs_path)
    
def get_train_data():
    train_csv = pd.read_csv(TRAIN_CSV)
    signs = list(set(train_csv.sign.tolist()))
    signs.sort()

    random.seed(SEED)
    five_signs = random.sample(signs, 5)
    
    train_df = train_csv.loc[train_csv['sign'].isin(five_signs)]
    train_df = train_df[['path', 'participant_id', 'sign']]
    return train_df

def create_mp_files():
    def pad(num):
        existing_len = len(str(num))
        return f'{(PADDING_DIGITS - existing_len) * "0"}{num}'
    
    attempt_counts = {}
    for participant in data_dict:
        if participant not in attempt_counts:
            attempt_counts[participant] = {}

        for sign in data_dict[participant]:
            if sign not in attempt_counts[participant]:
                attempt_counts[participant][sign] = 0
            
            attempt_counts[participant][sign] += 1
            attempt_count_str = pad(attempt_counts[participant][sign])

            filename_components = [participant, sign, 'singlesign', attempt_count_str, 'data']
            filename = '.'.join(filename_components)
            
            mp_dir = os.path.join('mediapipe_parquet', participant + '-singlesign', sign)
            if not os.path.exists(mp_dir):
                os.makedirs(mp_dir)
            
            mp_filepath = os.path.join(mp_dir, filename)
            dump_dict = data_dict[participant][sign]
            with open(mp_filepath, 'w') as f:
                json.dump(dump_dict, f, indent=4)

if __name__ == "__main__":
    train_df = get_train_data()
    ingest_parquet_files(train_df)
    create_mp_files()

    # participant = list(data_dict.keys())[0]
    # sign = list(data_dict[participant].keys())[0]
    # print(data_dict[participant][sign])

