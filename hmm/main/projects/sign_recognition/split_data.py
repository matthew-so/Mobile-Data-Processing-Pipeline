import os
import sys
import argparse
import random
import shutil

sys.path.append('../../src/prepare_data')

from glob import glob
from tqdm import tqdm
from generate_text_files import generate_text_files


USER_IDX = 0
SIGN_IDX = 1

helpdict = {
    'split_by': 'Choose whether to split by \'sign\' or \'user\'.',
    'split_list': 'Path to a custom split list.',
    'current_data_path': 'Path to folder containing existing ark/htk data.',
    'seed': 'Seed to be used when randomly splitting (by split_by).',
    'is_fingerspelling': 'Is this data fingerspelling data (if it is, assumes it is phrase level bigram)',
}

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     
    parser.add_argument(
        '--current-data-path', '-cdp',
        type=str,
        default='data',
        help=helpdict['current_data_path']
    )
    
    parser.add_argument(
        '--seed', '-sd',
        type=int,
        default=10,
        help=helpdict['seed']
    )
    
    parser.add_argument(
        '--split-by', '-sb',
        type=str,
        choices=['sign', 'user'],
        required=True,
        help=helpdict['split_by']
    )
    
    parser.add_argument(
        '--split-list', '-sl',
        type=str,
        help=helpdict['split_list']
    )

    parser.add_argument(
        '--is-fingerspelling', '-fs',
        action='store_true',
        help=helpdict['is_fingerspelling']
    )

    return parser.parse_args()

def copy_files(entities, curr_dirname, tgt_dirname, ext, split_by='user'):
    tgt_dir = os.path.join(curr_dirname, tgt_dirname, ext)
    print("Working on: ", tgt_dir)

    if os.path.exists(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)
    
    for entity in tqdm(entities):
        if split_by == 'user':
            entity_pattern = entity + '-*.'
        else:
            entity_pattern = '*-' + entity + '-*.'

        file_pattern = os.path.join(curr_dirname, ext, entity_pattern + ext)
        # print("File Pattern: ", file_pattern)
        files = glob(file_pattern)
        
        for src in files:
            tgt = os.path.join(tgt_dir, os.path.split(src)[-1])
            shutil.copyfile(src, tgt)
    

def get_entity_list(curr_dirname, idx, is_fingerspelling=False):
    htk_files = glob(os.path.join(curr_dirname, 'htk', '*.htk'))

    entities = list(set([os.path.split(filen.split('-')[idx])[-1] for filen in htk_files]))
    
    if is_fingerspelling:
        entity_list = []
        for entity in entities:
            entity_list.extend(entity.split('_'))
        entities = list(set(entity_list))
    
    entities = random.sample(entities, k=len(entities))

    return entities

def prepare_files(args, entities, curr_dirname, tgt_dirname):
    print(tgt_dirname)
    data_path = os.path.join(curr_dirname, tgt_dirname)
    
    copy_files(entities, curr_dirname, tgt_dirname, 'ark', args.split_by)
    copy_files(entities, curr_dirname, tgt_dirname, 'htk', args.split_by)
    
    if args.is_fingerspelling:
        is_fingerspelling = True
        is_single_word = False
        is_bigram = True
    else:
        is_fingerspelling = False
        is_single_word = True
        is_bigram = False

    if args.split_by == 'user':
        unique_words = get_entity_list(curr_dirname, 1, args.is_fingerspelling)
        generate_text_files(
            features_dir = '.',
            isFingerspelling = is_fingerspelling,
            isSingleWord = is_single_word,
            isBigram = is_bigram,
            unique_words = sorted(unique_words),
            data_path = data_path,
        )
    elif args.split_by == 'sign':
        generate_text_files(
            features_dir = '.',
            isFingerspelling = is_fingerspelling,
            isSingleWord = is_single_word,
            isBigram = is_bigram,
            unique_words = sorted(entities),
            data_path = data_path,
        )
        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    random.seed(args.seed)
    curr_dirname = args.current_data_path.strip("/")
    entities = get_entity_list(curr_dirname, int(args.split_by != 'user'), args.is_fingerspelling)
    
    if args.split_list is None:
        split_pt = int(len(entities) / 2)
        
        entities_1 = entities[:split_pt]
        entities_2 = entities[split_pt:]
        
        tgt_dirname_1 = '_'.join([args.split_by, '1'])
        tgt_dirname_2 = '_'.join([args.split_by, '2'])
    else:
        with open(args.split_list, 'r') as f:
            entities_1 = f.readlines()
            entities_1 = [entity.strip() for entity in entities_1]
        
        entities_1_set = set(entities_1)
        entities_set = set(entities)
        
        if not(entities_1_set.issubset(entities_set)):
            raise ValueError("Split list should be a subset of the sign/user list.")
        
        entities_2 = list(entities_set - entities_1_set)
        split_basename = os.path.splitext(os.path.basename(args.split_list))[0]

        tgt_dirname_1 = '_'.join([split_basename, '1'])
        tgt_dirname_2 = '_'.join([split_basename, '2'])
        
    prepare_files(args, entities_1, curr_dirname, tgt_dirname_1)
    prepare_files(args, entities_2, curr_dirname, tgt_dirname_2)

