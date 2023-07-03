"""Defines method to train HMM and parser group to pass arguments to
train method.

Methods
-------
train_cli
train
"""
import os
import sys
import glob
import shutil
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from .gen_init_models_each_word import initialize_models
from .gen_prototype import generate_prototype
from src.utils import load_json

def check_args(n_states: int, hmm_step_type: str):
    if n_states < 3 and hmm_step_type == 'single':
        raise ValueError("HMM with Single Step requires at least 3 states.")
    
    if n_states < 4 and hmm_step_type == 'double':
        raise ValueError("HMM with Double Step requires at least 4 states.")
    
    if n_states < 5 and hmm_step_type == 'triple':
        raise ValueError("HMM with Triple Step requires at least 5 states.")
    
def train(
    train_iters: list,
    mean: float,
    variance: float,
    transition_prob: float,
    num_features: int = None,
    fold: str = "",
    is_triletter: bool = False,
    hmm_step_type: str = 'single',
    gmm_mix: int = None,
    features_file: str = 'configs/features.json',
    prototypes_file: str = 'configs/prototypes.json'
) -> None:
    """Trains the HMM using HTK. Calls HCompV, HRest, HERest, HHEd, and
    HParse. Configuration files for prototypes and increasing mixtures
    are found in configs/. 

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """

    if os.path.exists(f'models/{fold}'):
        shutil.rmtree(f'models/{fold}')

    if os.path.exists(f'logs/{fold}'):
        if os.path.exists(f'logs/{fold}train.log'):
            os.remove(f'logs/{fold}train.log')

    os.makedirs(f'models/{fold}')

    if not os.path.exists(f'logs/{fold}'):
        os.makedirs(f'logs/{fold}')

    #n_models = train_iters[-1] + len(train_iters) - 1
    for i in range(train_iters[-1] + 1):
        hmm_dir = os.path.join('models', f'{fold}hmm{i}')
        if not os.path.exists(hmm_dir):
            os.makedirs(hmm_dir)

    features_config = load_json(features_file)
    
    if num_features is None:
        n_features = len(features_config['selected_features'])
    else:
        n_features = num_features

    print("-------------- Training HMM --------------")

    prototypes_config = load_json(prototypes_file)
    for n_states in prototypes_config:
        n_states_int = int(n_states)
        check_args(n_states_int, hmm_step_type)
        prototype_filepath = f'models/{fold}prototype'
        generate_prototype(
            n_states_int, n_features, prototype_filepath, mean,
            variance, transition_prob, hmm_step_type=hmm_step_type,
            gmm_mix=gmm_mix
        )

        print('Running HCompV...')
        HCompV_command = (f'HCompV -A -T 2 -C configs/hcompv.conf -v 2.0 -f 0.01 '
                          f'-m -S lists/{fold}train.data -M models/{fold}hmm0 '
                          f'{prototype_filepath} >> logs/{fold}train.log')
        os.system(HCompV_command)
        print('HCompV Complete')

        initialize_models(f'models/{fold}hmm0/prototype', prototypes_config[n_states], f'models/{fold}hmm0')
        #initialize_models('models/prototype', 'wordList', 'models/hmm0')

    hmm0_files = set(glob.glob(f'models/{fold}hmm0/*')) - {f'models/{fold}hmm0/vFloors'}
    print("Running HRest...")
    for hmm0_file in tqdm(hmm0_files):

        # print(f'Running HRest for {hmm0_file}...')
        # HRest_command = (f'HRest -A -i 60 -C configs/hrest.conf -v 0.1 -I '
        #                  f'all_labels.mlf -M models/{fold}hmm1 -S lists/{fold}train.data '
        #                  f'{hmm0_file} >> logs/{fold}train.log')
        HRest_command = (f'HRest -A -i 5 -C configs/hrest.conf -v 0.001 -I '
                         f'all_labels.mlf -M models/{fold}hmm1 -S lists/{fold}train.data '
                         f'{hmm0_file} >> logs/{fold}train.log')
        os.system(HRest_command)
    print('HRest Complete')

    print('Running HERest Iteration: 1...')
    HERest_command = (f'HERest -A -d models/{fold}hmm1 -c 500.0 -v 0.0005 -I '
                    f'all_labels.mlf -M models/{fold}hmm2 -S lists/{fold}train.data -T '
                    f'1 wordList >> logs/{fold}train.log')
    os.system(HERest_command)
    
    # if gmm_mix is not None and n_states_int > 6:
    #     with open('configs/hhed_gmm.conf', 'w') as f:
    #         f.write('JO 128 2.0\n')
    #         if n_states_int == 7:
    #             f.write('TI \"mix\" {*.state[4].mix}')
    #         else:
    #             end = n_states_int - 3
    #             f.write(f'TI \"mix\" {{*.state[4-{end}]}}')
    #     
    #     HHed_command = (f'HHEd -A -D -H models/{fold}hmm2/newMacros '
    #                     f'configs/hhed_gmm.conf '
    #                     f'wordList >> logs/{fold}train.log')
    #     os.system(HHed_command)

    if is_triletter:
        print('Converting to triletter')
        with open("mktri.led", "w") as f:
            f.write("WB sp\nWB sil\nTC\n")
        HLEd_command = (f'HLEd -A -D -T 1 -n wordList_triletter -l \'*\' -i all_labels_triletter.mlf '
                        f'mktri.led all_labels.mlf ')
        os.system(HLEd_command)
        MHed_command = (f'julia mktrihed.jl wordList wordList_triletter mktri.hed')
        os.system(MHed_command)
        HHed_command = (f'HHEd -A -D -T 1 -H models/{fold}hmm2/newMacros -M models/{fold}hmm3 mktri.hed wordList ')
        os.system(HHed_command)
        
    if is_triletter:
        start = 3
    else:
        start = 2
    for i, n_iters in enumerate(train_iters):

        for iter_ in tqdm(range(start, n_iters)):

            # print(f'Running HERest Iteration: {iter_}...')
            if is_triletter:
                HERest_command = (f'HERest -A -c 500.0 -v 0.0005 -D -T 1 -H '
                            f'models/{fold}hmm{iter_}/newMacros -I all_labels_triletter.mlf -M '
                            f'models/{fold}hmm{iter_+1} -S lists/{fold}train.data -T 1 wordList_triletter '
                            f'>> logs/{fold}train.log')
            else:
                HERest_command = (f'HERest -A -c 500.0 -v 0.0005 -A -H '
                            f'models/{fold}hmm{iter_}/newMacros -I all_labels.mlf -M '
                            f'models/{fold}hmm{iter_+1} -S lists/{fold}train.data -T 1 wordList '
                            f'>> logs/{fold}train.log')
            os.system(HERest_command)
        print('HERest Complete')

        if n_iters == train_iters[-1] and is_triletter:
            HHed_command = (f'HHEd -A -B -H models/{fold}hmm{n_iters-1}/newMacros -M '
                            f'models/{fold}hmm{n_iters} configs/hhed{i}.conf '
                            f'wordList_triletter')
            os.system(HHed_command)
            print('HHed Complete')

        if n_iters != train_iters[-1]:
            print(f'Running HHed Iteration: {n_iters}...')
            if is_triletter:
                HHed_command = (f'HHEd -A -H models/{fold}hmm{n_iters-1}/newMacros -M '
                            f'models/{fold}hmm{n_iters} configs/hhed{i}.conf '
                            f'wordList_triletter')
            else:
                HHed_command = (f'HHEd -A -H models/{fold}hmm{n_iters-1}/newMacros -M '
                            f'models/{fold}hmm{n_iters} configs/hhed{i}.conf '
                            f'wordList')
            os.system(HHed_command)
            print('HHed Complete')
            start = n_iters

    cmd = 'HParse -A -T 1 grammar.txt wordNet.txt'
    os.system(cmd)
