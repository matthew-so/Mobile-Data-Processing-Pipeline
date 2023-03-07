import os
import json
import argparse

from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt

def get_file_data(mediapipe_files):
    ts_keys = {}
    landmarks = {}
    pose = {}

    for mp_file in tqdm(mediapipe_files):
        with open(mp_file, 'r') as f:
            mp_json = json.load(f)

        keys = list(mp_json.keys())
        
        ts_keys[mp_file] = keys
        landmarks[mp_file] = {}
        pose[mp_file] = {}

        for key in keys:
            landmarks[mp_file][key] = mp_json[key]['landmarks']
            pose[mp_file][key] = mp_json[key]['pose']
    
    return ts_keys, landmarks, pose

def check_keys(ts_keys):
    missing_keys = {'yes': 0, 'no': 0}
    frame_dist = []
    for mp_file in ts_keys:
        keys = ts_keys[mp_file]
        
        int_keys = [int(key) for key in keys]
        int_keys.sort()
        
        if len(int_keys) - 1 == int_keys[-1]:
            frame_dist.append(len(int_keys))
            missing = 'no'
        else:
            missing = 'yes'
        missing = 'yes' if len(int_keys) - 1 != int_keys[-1] else 'no'
        missing_keys[missing] += 1
        
    return missing_keys, frame_dist

def classify_landmark(landmark):
    if len(landmark) == 0:
        return 'missing_all'
    elif len(landmark) == 21:
        return 'missing_none'
    else:
        return 'missing_some'

def classify_pose(pose_data):
    if len(pose_data) == 0:
        return 'missing_all'
    elif len(pose_data) == 33:
        return 'missing_none'
    else:
        return 'missing_some'

def check_landmarks(landmarks):
    missing_landmarks_0 = {'missing_all': 0, 'missing_some': 0, 'missing_none': 0}
    missing_landmarks_1 = {'missing_all': 0, 'missing_some': 0, 'missing_none': 0}

    for mp_file in landmarks:
        for ts in landmarks[mp_file]:
            landmark_0 = landmarks[mp_file][ts]['0']
            landmark_1 = landmarks[mp_file][ts]['1']
            
            missing_landmarks_0[classify_landmark(landmark_0)] += 1
            missing_landmarks_1[classify_landmark(landmark_1)] += 1
            
    return missing_landmarks_0, missing_landmarks_1

def check_pose(pose):
    missing_pose = {'missing_all': 0, 'missing_some': 0, 'missing_none': 0}

    for mp_file in pose:
        for ts in pose[mp_file]:
            pose_data = pose[mp_file][ts]
            missing_pose[classify_pose(pose_data)] += 1
    
    return missing_pose

def print_distribution(header, count_dict):
    print(header)

    total = sum(count_dict.values())
    for key in sorted(count_dict.keys()):
        pct = count_dict[key] / total
        print(f'{key}: {pct}')

    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mediapipe_dir', help='Directory with mediapipe files')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    glob_path = os.path.join(args.mediapipe_dir, '*', '*', '*.data')

    mediapipe_files = glob(glob_path)
    ts_keys, landmarks, pose = get_file_data(mediapipe_files)
    
    missing_keys, frame_dist = check_keys(ts_keys)
    missing_landmarks_0, missing_landmarks_1 = check_landmarks(landmarks)
    missing_pose = check_pose(pose)
    
    bins = [
        0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,
        100,120,140,160,180,200,240,280,320,360,400,
        450,500
    ]
    plt.hist(frame_dist, bins=bins)
    plt.savefig('frame_dist.png')
    # print_distribution("Missing Keys", missing_keys)
    # print_distribution("Missing Landmarks (0)", missing_landmarks_0)
    # print_distribution("Missing Landmarks (1)", missing_landmarks_1)
    # print_distribution("Missing Pose", missing_pose)

