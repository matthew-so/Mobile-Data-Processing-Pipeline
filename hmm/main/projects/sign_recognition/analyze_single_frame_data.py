import argparse
import os

from glob import glob
from tqdm import tqdm
from collections import defaultdict

ALL_DATA_DIR = 'data/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_1_1_1_1/')
    parser.add_argument('--log_file', type=str, default='logs/train.log')

    return parser.parse_args()

def get_frame_distribution(filepaths):
    frame_distribution = defaultdict(int)

    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            frame_distribution[len(lines)] += 1
    
    return frame_distribution

def print_frame_distribution(dist):
    keys = list(dist.keys())
    keys.sort()
    for key in keys:
        print(f"Frame Count: {key}, Data Files: {dist[key]}")

if __name__ == "__main__":
    args = parse_args()
    
    ## Analyze Train Log
    # single_frame_files = set()
    # with open(args.log_file, 'r') as f:
    #     lines = f.readlines()
    #     for i,line in enumerate(lines):
    #         if 'Processing Data: ' in line:
    #             if 'Unable to traverse 9 states in 1 frames' in lines[i + 1]:
    #                 line_arr = line.split(' ')
    #                 single_frame_files.add(line_arr[3][:-5])
    
    # single_frame_files = list(single_frame_files)
    # print("Number of files with one frame: ", len(single_frame_files))
    # print("Examples: ", single_frame_files[:10])
    # print("")

    ## Analyze Single Frame Files
    # single_frame_filepaths = [os.path.join(args.data_dir, 'ark', filename + '.ark') for filename in single_frame_files]
    # frame_distribution = get_frame_distribution(single_frame_filepaths)

    # print("Frame Distribution of Single Frame Files")
    # print_frame_distribution(frame_distribution)
    # print("")

    all_data_filepaths = glob(os.path.join(ALL_DATA_DIR, 'ark', '*.ark'))
    all_data_frame_distribution = get_frame_distribution(all_data_filepaths)
    
    print("Frame Distribution of All Data Files")
    print_frame_distribution(all_data_frame_distribution)
    print("")
