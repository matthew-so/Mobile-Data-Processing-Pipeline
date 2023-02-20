import argparse
import sys
import os
import itertools
from datetime import datetime

sys.path.append('../../')
from src.utils import dump_json

PROTOTYPE_FILE = 'configs/prototypes.json'
RESULTS_DIR = 'hresults'
STATE_IDX = 0
FOLD_IDX = 1
HMM_TYPE_IDX = 2
VAR_IDX = 3
GMM_MIX_IDX = 4
GMM_PATT_IDX = 5

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--is_fingerspelling', action='store_true')
    parser.add_argument('--gmm_mixes', nargs="*", type=int, default=[None])
    parser.add_argument('--gmm_patterns', nargs="*", type=str, choices=['all', 'middle'], default=['all'])
    parser.add_argument('--n_states', nargs='*', type=int, default=[8])
    parser.add_argument('--n_folds', nargs='*', type=int, default=[5]) # Called folds in driver.py/train.py/main.py
    parser.add_argument('--variances', nargs='*', type=float, default=[1e-5])
    parser.add_argument('--command_file', type=str, default='train_recognizer.sh')
    parser.add_argument(
        '--hmm_step_types',
        nargs='*',
        type=str,
        default=['single'],
        choices=['single', 'double', 'triple', 'start_stack', 'end_stack'],
    )
    return parser.parse_args()

def get_alphabet():
    return list(map(chr, range(97, 123))) + ['sil0','sil1']

def get_base_command(command_file):
    with open(command_file, 'r') as f:
        base_command = f.read()
    base_command = base_command.rstrip('\n')
    return base_command

def get_results_filepath(args):
    n_states = tup[STATE_IDX]
    n_folds = tup[FOLD_IDX]
    hmm_step_type = tup[HMM_TYPE_IDX]
    gmm_mix = tup[GMM_MIX_IDX]
    gmm_pattern = tup[GMM_PATT_IDX]
    
    with open('wordList', 'r') as f:
        lines = f.readlines()
        n_signs = len(lines) - 2  # Remove sil0/sil1
    
    n_signs_dir = 'n_signs_{n}'.format(n=n_signs)
    n_states_dir = 'n_states_{n}'.format(n=n_states)
    n_folds_dir = 'n_folds_{n}'.format(n=n_folds)
    hmm_type_dir = 'hmm_type_{s}'.format(s=hmm_step_type)
    gmm_mix_dir = 'gmm_mix_{n}'.format(n=gmm_mix)
    gmm_pattern_dir = 'gmm_pattern_{s}'.format(s=gmm_pattern)

    results_dir = os.path.join(
        'hresults',
        n_signs_dir,
        n_states_dir,
        n_folds_dir,
        hmm_type_dir,
        gmm_mix_dir,
        gmm_pattern_dir
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_filepath = os.path.join(results_dir, 'results.txt')
    return results_filepath

def run_command(base_command, results_filepath, args, tup):
    n_states = tup[STATE_IDX]
    n_folds = tup[FOLD_IDX]
    hmm_step_type = tup[HMM_TYPE_IDX]
    variance = tup[VAR_IDX]
    gmm_mix = tup[GMM_MIX_IDX]
    gmm_pattern = tup[GMM_PATT_IDX]
    
    with open(results_filepath, 'a') as f:
        f.write(
            'Num States: {n} | Num Folds: {f} | HMM Step Type: {h} | Variance: {v} | GMM Num Mixes: {gm} | GMM Pattern: {gp}\n'.format(
                n=n_states,
                f=n_folds,
                h=hmm_step_type,
                v=variance,
                gm=gmm_mix,
                gp=gmm_pattern
            )
        )
        f.write('=====================\n')
    
    command = base_command
    # command += ' \\\n\t--grid_results_file ' + results_filepath
    # command += ' \\\n\t--n_splits {n}'.format(n=n_folds)
    command += ' \\\n\t--hmm_step_type {s}'.format(s=hmm_step_type)
    command += ' \\\n\t--variance {f}'.format(f=variance)
    if gmm_mix is not None:
        command += ' \\\n\t--gmm_mix {n}'.format(n=gmm_mix)
    command += ' \\\n\t--gmm_pattern {s}'.format(s=gmm_pattern)
    
    command = command.expandtabs(4)
    
    print('Running Command: ' + command)
    os.system(command)

if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)
    
    if args.is_fingerspelling:
        tokens = get_alphabet()
    else:
        with open('wordList', 'r') as f:
            tokens = [line.rstrip() for line in f.readlines()]

    grid_search_list = list(
        itertools.product(
            args.n_states,
            args.n_folds,
            args.hmm_step_types,
            args.variances,
            args.gmm_mixes,
            args.gmm_patterns
        )
    )
    # for n in args.n_states:
    for tup in grid_search_list:
        prototype_config = {str(tup[0]): tokens}
        dump_json(PROTOTYPE_FILE, prototype_config)
        
        # for hmm_step_type in args.hmm_step_types:
        results_filepath = get_results_filepath(tup)
        base_command = get_base_command(args.command_file)
        run_command(base_command, results_filepath, args, tup)

