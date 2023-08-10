import os
import argparse
import random
import shutil

from glob import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_data_path', type=str, default='data')
    parser.add_argument('--seed', type=int, default=10)

    return parser.parse_args()

def copy_files(users, curr_dirname, tgt_dirname, ext):
    tgt_dir = os.path.join(tgt_dirname, ext)
    print("Working on: ", tgt_dir)

    if os.path.exists(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)
    
    for user in tqdm(users):
        file_pattern = os.path.join(curr_dirname, ext, user + '-*.' + ext)
        files = glob(file_pattern)
        
        for src in files:
            tgt = os.path.join(tgt_dir, os.path.split(src)[-1])
            shutil.copyfile(src, tgt)
    
    print()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    random.seed(args.seed)
    curr_dirname = args.current_data_path.strip("/")

    ark_files = glob(os.path.join(curr_dirname, 'ark', '*.ark'))
    htk_files = glob(os.path.join(curr_dirname, 'htk', '*.htk'))

    users = list(set([os.path.split(filen.split('-')[0])[-1] for filen in htk_files]))
    users = random.sample(users, k=len(users))
    
    split_pt = int(len(users) / 2)
    users_1 = users[:split_pt]
    users_2 = users[split_pt:]
    
    copy_files(users_1, curr_dirname, curr_dirname + '_1', 'ark')
    copy_files(users_1, curr_dirname, curr_dirname + '_1', 'htk')
    
    copy_files(users_2, curr_dirname, curr_dirname + '_2', 'ark')
    copy_files(users_2, curr_dirname, curr_dirname + '_2', 'htk')

