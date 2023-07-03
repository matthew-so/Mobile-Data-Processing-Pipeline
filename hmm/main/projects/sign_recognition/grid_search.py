import sys
import os
import itertools

from datetime import datetime
sys.path.append('../../')
from src.utils import parse_grid_search_args

PROTOTYPE_FILE = 'configs/prototypes.json'
RESULTS_DIR = 'hresults'
STATE_IDX = 0
FOLD_IDX = 1
HMM_TYPE_IDX = 2
GMM_MIX_IDX = 3

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
    
    with open('wordList', 'r') as f:
        lines = f.readlines()
        n_signs = len(lines) - 2  # Remove sil0/sil1
    
    n_signs_dir = 'n_signs_{n}'.format(n=n_signs)
    n_states_dir = 'n_states_{n}'.format(n=n_states)
    n_folds_dir = 'n_folds_{n}'.format(n=n_folds)
    hmm_type_dir = 'hmm_type_{s}'.format(s=hmm_step_type)
    gmm_mix_dir = 'gmm_mix_{n}'.format(n=gmm_mix)

    results_dir = os.path.join(
        'hresults',
        n_signs_dir,
        n_states_dir,
        n_folds_dir,
        hmm_type_dir,
        gmm_mix_dir,
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_filepath = os.path.join(results_dir, 'results.txt')
    return results_filepath

def run_command(base_command, args, tup, results_filepath=None):
    n_states = tup[STATE_IDX]
    n_folds = tup[FOLD_IDX]
    hmm_step_type = tup[HMM_TYPE_IDX]
    gmm_mix = tup[GMM_MIX_IDX]
    
    if results_filepath:
        with open(results_filepath, 'a') as f:
            f.write(
                'Num States: {n} | Num Folds: {f} | HMM Step Type: {h} | GMM Num Mixes: {gm}\n'.format(
                    n=n_states,
                    f=n_folds,
                    h=hmm_step_type,
                    gm=gmm_mix
                )
            )
            f.write('=====================\n')
    
    command = base_command
    # command += ' \\\n\t--grid_results_file ' + results_filepath
    command += ' \\\n\t--n_states {n}'.format(n=n_states)
    
    if args.test_type == 'cross_val':
        command += ' \\\n\t--n_splits {f}'.format(f=n_folds)
    
    command += ' \\\n\t--hmm_step_type {h}'.format(h=hmm_step_type)
    
    if gmm_mix is not None:
        command += ' \\\n\t--gmm_mix {gm}'.format(gm=gmm_mix)
    
    command = command.expandtabs(4)
    
    print('Running Command: ' + command)
    os.system(command)

if __name__ == "__main__":
    args = parse_grid_search_args()
    print("Args: ", args)
    
    grid_search_list = list(
        itertools.product(
            args.n_states,
            args.n_folds,
            args.hmm_step_types,
            args.gmm_mixes,
        )
    )
    # for n in args.n_states:
    if args.test_type == 'cross_val':
        command_file = 'train_recognizer_cv.sh'
    else:
        command_file = 'train_recognizer_tot.sh'

    base_command = get_base_command(command_file)
    for tup in grid_search_list:
        run_command(base_command, args, tup)

