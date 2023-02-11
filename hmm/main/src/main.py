"""Main file used to prepare training data, train, and test HMMs.
    HMM EX = python3 driver.py --test_type standard --train_iters 25 50 --users Naoki --hmm_insertion_penalty -70
    SBHMM EX = python3 driver.py --test_type standard --users Naoki --train_iters 25 50 --sbhmm_iters 25 50 --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -70 --sbhmm_insertion_penalty -115
"""
"""Main file used to prepare training data, train, and test HMMs.
    HMM EX = python3 driver.py --test_type standard --train_iters 25 50 75 100 --users Prerna Linda | 
    HMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --users 02-22-20_Prerna_Android 04-29-20_Linda_Android 07-24-20_Matthew_4K --cross_val_method stratified --n_splits 10 --cv_parallel --parallel_jobs 10  --hmm_insertion_penalty -80
    SBHMM EX = python3 driver.py --test_type standard --train_iters 25 50 75 --sbhmm_iters 25 50 75 --users Prerna Linda --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -70 --sbhmm_insertion_penalty -115 --neighbors 70
    SBHMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 --sbhmm_iters 25 50 75 --users Linda Prerna --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --parallel_jobs 4 --hmm_insertion_penalty -85 --sbhmm_insertion_penalty -85 --neighbors 70 --cross_val_method kfold --n_splits 10 --beam_threshold 2000.0
    SBHMM CV Parallel = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --sbhmm_iters 25 50 75 100 --users Ishan Matthew David --train_sbhmm --sbhmm_cycles 1  --include_word_level_states --include_word_level_states --parallel_classifier_training --hmm_insertion_penalty -80 --sbhmm_insertion_penalty -80 --neighbors 73 --cross_val_method stratified --n_splits 5 --beam_threshold 3000.0 --cv_parallel --parallel_jobs 10
    Prepare Data = python3 driver.py --test_type none --prepare_data --users Matthew_4 Ishan_4 David_4
    Old SBHMM CV = python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 --sbhmm_iters 25 50 75 100 --users Ishan Matthew David --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --parallel_classifier_training --hmm_insertion_penalty -80 --sbhmm_insertion_penalty -80 --cross_val_method stratified --n_splits 5 --beam_threshold 3000.0 --cv_parallel --parallel_jobs 5 --multiple_classifiers
    SBHMM CV Parallel User Independent =  python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 240 --sbhmm_iters 25 50 75 100 --train_sbhmm --sbhmm_cycles 1 --include_word_level_states --include_word_position --parallel_classifier_training --hmm_insertion_penalty 150 --sbhmm_insertion_penalty 150 --neighbors 200 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 7 --users Linda_4 Kanksha_4 Thad_4 Matthew_4 Prerna_4 David_4 Ishan_4
    HMM CV Parallel User Independent =  python3 driver.py --test_type cross_val --train_iters 25 50 75 100 120 140 160 180 200 220 240 --hmm_insertion_penalty 10 --cross_val_method leave_one_user_out --n_splits 5 --beam_threshold 50000.0 --cv_parallel --parallel_jobs 4
"""

"""Verification commands
    HMM Standard (Dry run) = python3 driver.py --test_type standard --train_iters 10 20 --users Matthew --method verification
    HMM CV = python3 driver.py --test_type cross_val --train_iters 10 --users 07-24-20_Matthew_4KDepth 11-08-20_Colby_4KDepth 11-08-20_Ishan_4KDepth --cross_val_method leave_one_user_out --n_splits 10 --cv_parallel --parallel_jobs 3  --hmm_insertion_penalty -80 --method verification --verification_method logistic_regression
"""
import sys
import glob
import argparse
import os
import shutil
import sys
import random
import numpy as np
import tqdm
import pickle


from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneGroupOut, train_test_split)

sys.path.insert(0, '../../')
from src.prepare_data import prepare_data
from src.prepare_data.generate_text_files import generate_text_files
from src.train import create_data_lists, train, trainSBHMM, get_logisitc_regressor, get_neural_net_classifier
from src.utils import get_results, save_results, load_json, get_arg_groups, get_hresults_data
from src.test import test, testSBHMM, verify_simple, return_average_ll_per_sign, return_ll_per_correct_and_one_off_sign, verify_zahoor, verify_classifier
from joblib import Parallel, delayed
from statistics import mean

def returnUserDependentSplits(unique_users, htk_filepaths, test_size):
    splits = [[[],[]] for i in range(len(unique_users))]
    for htk_idx, curr_file in enumerate(htk_filepaths):
        curr_user = curr_file.split("/")[-1].split(".")[0].split('_')[-2]
        for usr_idx, usr in enumerate(unique_users):
            if usr == curr_user:
                if random.random() > test_size:
                    splits[usr_idx][0].append(htk_idx)
                else:
                    splits[usr_idx][1].append(htk_idx)
    splits = np.array(splits)
    return splits

def copyFiles(fileNames: list, newFolder: str, originalFolder: str, ext: str):
    if os.path.exists(newFolder):
        shutil.rmtree(newFolder)
    os.makedirs(newFolder)

    for currFile in fileNames:
        shutil.copyfile(os.path.join(originalFolder, currFile+ext), os.path.join(newFolder, currFile+ext))

def get_user(filepath):
    return filepath.split('/')[-1].split('.')[0].split('_')[-2]

def get_phrase_len(filepath):
    return len(os.path.basename(filepath).split('.')[1].split('_'))

def get_video(filepath):
    extension = '.' + os.path.basename(filepath).split('.')[-1]
    return os.path.basename(filepath).replace(extension, '')

def crossValVerificationFold(train_data: list, test_data: list, args: object, fold: int):
    print(f"Current split = {str(fold)}. Current Test data Size = {len(test_data)}")
    ogDataFolder = "data"
    currDataFolder = os.path.join("data", str(fold))
    trainFiles = [i.split("/")[-1].replace(".htk", "") for i in train_data]
    testFiles = [i.split("/")[-1].replace(".htk", "") for i in test_data]
    allFiles = trainFiles + testFiles

    copyFiles(allFiles, os.path.join(currDataFolder, "ark"), os.path.join(ogDataFolder, "ark"), ".ark")
    copyFiles(allFiles, os.path.join(currDataFolder, "htk"), os.path.join(ogDataFolder, "htk"), ".htk")
    test_user = get_user(testFiles[0])
    users_in_train = set([get_user(filepath) for filepath in trainFiles])
    average_ll_per_sign = {}
    for user in users_in_train:
        curr_train_files = []
        curr_test_files = []
        for filepath in trainFiles:
            if get_user(filepath) != user:
                curr_train_files.append(filepath)
            else:
                curr_test_files.append(filepath)
        
        create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in curr_train_files], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in curr_test_files], args.phrase_len, fold)

        train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""))
        if args.verification_method == "zahoor":
            curr_average_ll_sign = return_average_ll_per_sign(args.end, args.hmm_insertion_penalty, 
                                                            args.beam_threshold, fold=os.path.join(str(fold), ""))
            if len(average_ll_per_sign) == 0:
                average_ll_per_sign = curr_average_ll_sign
            else:
                for sign in average_ll_per_sign:
                    average_ll_per_sign[sign] = np.concatenate((average_ll_per_sign[sign], curr_average_ll_sign[sign]), axis=None)

        elif args.verification_method == "logistic_regression" or args.verification_method == "neural_net":
            curr_average_ll_sign = return_ll_per_correct_and_one_off_sign(args.end, args.hmm_insertion_penalty, args.verification_method == "logistic_regression",
                                                            args.beam_threshold, fold=os.path.join(str(fold), ""))
            for data_set in curr_average_ll_sign:
                if data_set not in average_ll_per_sign:
                    average_ll_per_sign[data_set] = {}
                for sign in curr_average_ll_sign[data_set]:
                    if sign in average_ll_per_sign:
                        average_ll_per_sign[data_set][sign] = np.concatenate((average_ll_per_sign[data_set][sign], curr_average_ll_sign[data_set][sign]), axis=0)
                    else:
                        average_ll_per_sign[data_set][sign] = np.array(curr_average_ll_sign[data_set][sign])
            print(f"Signs in correct set = {str(len(average_ll_per_sign['correct']))} and signs in incorrect set = {str(len(average_ll_per_sign['incorrect']))}")
        else:
            raise Exception("Please select correct verification method")
    
    # Save user independent log likelihoods
    pickle.dump(average_ll_per_sign, open(os.path.join(currDataFolder, f"{test_user}_UI_loglikelihoods.pkl"), "wb"))
    classifier = {}
    if args.verification_method == "zahoor":
        for sign in average_ll_per_sign:
            classifier[sign] = [np.mean(average_ll_per_sign[sign]), np.std(average_ll_per_sign[sign])]
    elif args.verification_method == "logistic_regression":
        print("Training logistic regression classifier for each sign")
        for sign in tqdm.tqdm(average_ll_per_sign["correct"]):
            classifier[sign] = get_logisitc_regressor(average_ll_per_sign["correct"][sign], average_ll_per_sign["incorrect"][sign], args.random_state)
    elif args.verification_method == "neural_net":
        print("Training neural net classifier for each sign")
        for sign in tqdm.tqdm(average_ll_per_sign["correct"]):
            classifier[sign] = get_neural_net_classifier(average_ll_per_sign["correct"][sign], average_ll_per_sign["incorrect"][sign], args.random_state)
    
    else:
        raise Exception("Please select correct verification method")

    
    create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in trainFiles], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in testFiles], args.phrase_len, fold)
    train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""))
    if args.verification_method == "zahoor":
        positive, negative, false_positive, false_negative, test_log_likelihoods = verify_zahoor(args.end, args.hmm_insertion_penalty, classifier, 
                                                                        args.beam_threshold, fold=os.path.join(str(fold), ""))
    elif args.verification_method == "logistic_regression":
        positive, negative, false_positive, false_negative, test_log_likelihoods = verify_classifier(args.end, args.hmm_insertion_penalty, classifier, True, 
                                                                        args.beam_threshold, fold=os.path.join(str(fold), ""))
    elif args.verification_method == "neural_net":
        positive, negative, false_positive, false_negative, test_log_likelihoods = verify_classifier(args.end, args.hmm_insertion_penalty, classifier, False,
                                                                        args.beam_threshold, fold=os.path.join(str(fold), ""))
    else:
        raise Exception("Please select correct verification method")

    pickle.dump(test_log_likelihoods, open(os.path.join(currDataFolder, f"{test_user}_test_split_loglikelihoods.pkl"), "wb"))

    print(f'Current Positive Rate: {positive/(positive + false_negative)}')
    print(f'Current Negative Rate: {negative/(negative + false_positive)}')
    print(f'Current False Positive Rate: {false_positive/(negative + false_positive)}')
    print(f'Current False Negative Rate: {false_negative/(positive + false_negative)}')

    return positive, negative, false_positive, false_negative
    
def crossValFold(train_data: list, test_data: list, args: object, fold: int, run_train = True):
    train_data = np.array(train_data)
    np.random.seed(args.random_state)
    np.random.shuffle(train_data)

    info_string = (
        f"Current split = {str(fold)}. "
        f"Current Test data Size = {len(test_data)}. "
        f"Current Train data Size = {len(train_data)}."
    )
    print(info_string)
    ogDataFolder = "data"
    currDataFolder = os.path.join("data", str(fold))
    trainFiles = [i.split("/")[-1].replace(".htk", "") for i in train_data]
    testFiles = [i.split("/")[-1].replace(".htk", "") for i in test_data]
    allFiles = trainFiles + testFiles

    copyFiles(allFiles, os.path.join(currDataFolder, "ark"), os.path.join(ogDataFolder, "ark"), ".ark")
    copyFiles(allFiles, os.path.join(currDataFolder, "htk"), os.path.join(ogDataFolder, "htk"), ".htk")
    create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in trainFiles], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in testFiles], args.phrase_len, fold)
    
    if args.train_sbhmm:
        classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold, os.path.join(str(fold), ""))
        testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                args.parallel_jobs, args.parallel_classifier_training, os.path.join(str(fold), ""))
    else:
        if run_train: train(args.train_iters, args.mean, args.variance, args.transition_prob, fold=os.path.join(str(fold), ""),
                hmm_step_type=args.hmm_step_type, gmm_mix=args.gmm_mix, gmm_pattern=args.gmm_pattern)
        test(args.start, args.end, args.method, args.hmm_insertion_penalty, fold=os.path.join(str(fold), ""))

    if args.train_sbhmm:
        hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.sbhmm_iters[-1]-1}.txt'
    else:
        hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.train_iters[-1]-1}.txt'    

    results = get_results(hresults_file)

    print(f'Current Word Error: {results["error"]}')
    print(f'Current Sentence Error: {results["sentence_error"]}')
    print(f'Current Insertion Error: {results["insertions"]}')
    print(f'Current Deletions Error: {results["deletions"]}')

    # test(-1, -1, "alignment", args.hmm_insertion_penalty, beam_threshold=args.beam_threshold, fold=os.path.join(str(fold), ""))

    return [results['error'], results['sentence_error'], results['insertions'], results['deletions']]

    
def print_to_stdout(str_list):
    for s in str_list:
        print(s)

def print_to_file(str_list, filepath):
    with open(filepath, 'a') as f:
        for s in str_list:
            f.write(s + '\n')
        f.write('\n')

def main():
    
    parser = argparse.ArgumentParser()
    ############################## ARGUMENTS #####################################
    #Important
    parser.add_argument('--prepare_data', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_results_file', type=str,
                        default='all_results.json')
    parser.add_argument('--features_file', type=str, default='configs/features.json')
    parser.add_argument('--is_single_word', action='store_true')

    # Arguments for create_data_lists()
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['none', 'test_on_train', 'cross_val', 'standard', 'progressive_user_adaptive', 'user_independent_limited_guess'])
    parser.add_argument('--users', nargs='*', default=None)
    parser.add_argument('--cross_val_method', default='kfold', choices=['kfold',
                                                  'leave_one_phrase_out',
                                                  'stratified',
                                                  'leave_one_user_out',
                                                  'user_dependent',
                                                  ])
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--cv_parallel', action='store_true')
    parser.add_argument('--parallel_jobs', default=4, type=int)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--phrase_len', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=42) #The answer to life, the universe and everything

    #Arguments for training
    parser.add_argument('--train_iters', nargs='*', type=int, default=[20, 50, 80])
    parser.add_argument('--hmm_insertion_penalty', default=-10)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=1.0)
    parser.add_argument('--transition_prob', type=float, default=0.6)
    parser.add_argument(
        '--hmm_step_type',
        type=str,
        choices=['single','double','triple', 'start_stack', 'end_stack'],
        default='single'
    )
    parser.add_argument('--gmm_pattern', type=str, default='middle')    
    parser.add_argument('--gmm_mix', type=int, default=None)    

    #Arguments for SBHMM
    parser.add_argument('--train_sbhmm', action='store_true')    
    parser.add_argument('--sbhmm_iters', nargs='*', type=int, default=[20, 50, 80])
    parser.add_argument('--include_word_position', action='store_true')
    parser.add_argument('--include_word_level_states', action='store_true')
    parser.add_argument('--sbhmm_insertion_penalty', default=-10)
    parser.add_argument('--classifier', type=str, default='knn',
                        choices=['knn', 'adaboost'])
    parser.add_argument('--neighbors', default=50)
    parser.add_argument('--sbhmm_cycles', type=int, default=1)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--pca_components', type=int, default=32)
    parser.add_argument('--multiple_classifiers', action='store_true')
    parser.add_argument('--parallel_classifier_training', action='store_true')
    parser.add_argument('--beam_threshold', default=100000000.0)

    #Arguments for testing
    parser.add_argument('--start', type=int, default=-2)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--method', default='recognition', 
                        choices=['recognition', 'verification'])
    parser.add_argument('--acceptance_threshold', default=-150)
    parser.add_argument('--verification_method', default='zahoor', 
                        choices=['zahoor', 'logistic_regression', 'neural_net'])

    parser.add_argument('--training_type', default='sign', 
                        choices=['sign', 'fingerspelling'])

    parser.add_argument('--model_type', default='uniletter', 
                        choices=['uniletter', 'triletter'])
    parser.add_argument('--grid_results_file', type=str, default=None)
    
    args = parser.parse_args()
    ########################################################################################

    if args.users: args.users = [user.capitalize() for user in args.users]

    cross_val_methods = {'kfold': (KFold, True),
                         'leave_one_phrase_out': (LeaveOneGroupOut(), True),
                         'stratified': (StratifiedKFold, True),
                         'leave_one_user_out': (LeaveOneGroupOut(), True),
                         'user_dependent': (None, False),
                         }
    cvm = args.cross_val_method
    cross_val_method, use_groups = cross_val_methods[args.cross_val_method]

    features_config = load_json(args.features_file)
    all_results = {'features': features_config['selected_features'],
                   'average': {}}
                   
    if args.train_sbhmm:
        hresults_file = f'hresults/res_hmm{args.sbhmm_iters[-1]-1}.txt'
    else:
        hresults_file = f'hresults/res_hmm{args.train_iters[-1]-1}.txt'

    isFingerspelling = False
    
    if args.training_type == "fingerspelling":
        isFingerspelling = True

    if args.prepare_data and not args.test_type == 'progressive_user_adaptive':
        # this will include users in verification
        prepare_data(features_config, args.users, isFingerspelling=isFingerspelling, isSingleWord=args.is_single_word)

    if args.test_type == 'none':
        sys.exit()

    elif args.test_type == 'test_on_train':
        
        if not args.users:
            htk_filepaths = glob.glob('data/htk/*.htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        create_data_lists(htk_filepaths, htk_filepaths, args.phrase_len)
        
        if args.train_sbhmm:
            classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
            testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty,
                    args.parallel_jobs, args.parallel_classifier_training)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob, is_triletter=args.model_type=="triletter")
            if args.method == "recognition":
                test(args.start, args.end, args.method, args.hmm_insertion_penalty, is_triletter=args.model_type=="triletter")
            elif args.method == "verification":
                positive, negative, false_positive, false_negative = verify_simple(args.end, args.insertion_penalty, 
                                                                    args.acceptance_threshold, args.beam_threshold)
        
        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')
        
        if args.method == "verification":
            all_results['average']['positive'] = positive
            all_results['average']['negative'] = negative
            all_results['average']['false_positive'] = false_positive
            all_results['average']['false_negative'] = false_negative

            print('Test on Train Results')
    
    elif args.test_type == 'cross_val' and args.cv_parallel:
        print("You have invoked parallel cross validation. Be prepared for dancing progress bars!")

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*.htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
        
        #for filepath in htk_filepaths:
        #    split_path = filepath.split('/')[-1]
        #    print("Filepath: ", split_path)
        #    print("Split Path [0]", split_path.rsplit(".", 4)[0])
        #    print("Split Path [1]", split_path.rsplit(".", 4)[1])
        #    print()
        
        phrases = [filepath.split('/')[-1].rsplit(".", 4)[0] + " " + ' '.join(filepath.split('/')[-1].rsplit(".", 4)[1].split("_"))
            for filepath
            in htk_filepaths]

        if cvm == 'kfold' or cvm == 'stratified':
            unique_phrases = set(phrases)
            print(len(unique_phrases), len(phrases))
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method(n_splits=args.n_splits)
        elif cvm == 'leave_one_phrase_out':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method
        elif cvm == 'leave_one_user_out':
            users = [get_user(filepath) for filepath in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in users]
            cross_val = cross_val_method
        elif cvm == 'user_dependent':
            users = [get_user(filepath) for filepath in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
        
        if cvm == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, htk_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))
        if args.method == "recognition":
            stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                            (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                            for currFold in range(len(splits)))
            all_results['average']['error'] = mean([i[0] for i in stats])
            all_results['average']['sentence_error'] = mean([i[1] for i in stats])
            all_results['average']['insertions'] = mean([i[2] for i in stats])
            all_results['average']['deletions'] = mean([i[3] for i in stats])

        elif args.method == "verification":
            stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValVerificationFold)
                            (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                            for currFold in range(len(splits)))
            all_results['average']['positive'] = mean(i[0] for i in stats)
            all_results['average']['negative'] = mean(i[1] for i in stats)
            all_results['average']['false_positive'] = mean(i[2] for i in stats)
            all_results['average']['false_negative'] = mean(i[3] for i in stats)
        
        print(stats)

    elif args.test_type == 'cross_val':


        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]
        
        users = [get_user(filepath) for filepath in htk_filepaths]     

        if cvm == 'kfold' or cvm == 'stratified':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method(n_splits=args.n_splits)
        elif cvm == 'leave_one_phrase_out':
            unique_phrases = set(phrases)
            group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
            groups = [group_map[phrase] for phrase in phrases]
            cross_val = cross_val_method
        elif cvm == 'leave_one_user_out':
            unique_users = set(users)
            group_map = {user: i for i, user in enumerate(unique_users)}
            groups = [group_map[user] for user in users]            
            cross_val = cross_val_method
        elif cvm == 'user_dependent':
            users = [get_user(filepath) for filepath in htk_filepaths]
            unique_users = list(set(users))
            unique_users.sort()
            print(unique_users)
        
        if cvm == 'user_dependent':
            splits = returnUserDependentSplits(unique_users, htk_filepaths, args.test_size)
        elif use_groups:
            splits = list(cross_val.split(htk_filepaths, phrases, groups))
        else:
            splits = list(cross_val.split(htk_filepaths, phrases))

        for i, (train_index, test_index) in enumerate(splits):

            print(f'Current split = {i}')
            
            train_data = np.array(htk_filepaths)[train_index]
            test_data = np.array(htk_filepaths)[test_index]

            phrase = np.array(phrases)[test_index][0]
            phrase_len = len(phrase.split(' '))
            phrase_count = len(test_data)
            word_count = phrase_len * phrase_count
            word_counts.append(word_count)
            phrase_counts.append(phrase_count)
            create_data_lists(train_data, test_data, args.phrase_len)

            if args.train_sbhmm:
                classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
                testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                        args.parallel_jobs, args.parallel_classifier_training)
            else:
                train(args.train_iters, args.mean, args.variance, args.transition_prob)
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            
            results = get_results(hresults_file)
            all_results[f'fold_{i}'] = results
            all_results[f'fold_{i}']['phrase'] = phrase
            all_results[f'fold_{i}']['phrase_count'] = phrase_count

            print(f'Current Word Error: {results["error"]}')
            print(f'Current Sentence Error: {results["sentence_error"]}')

            substitutions += (word_count * results['substitutions'] / 100)
            deletions += (word_count * results['deletions'] / 100)
            insertions += (word_count * results['insertions'] / 100)
            sentence_errors += (phrase_count * results['sentence_error'] / 100)

        total_words = sum(word_counts)
        total_phrases = sum(phrase_counts)
        total_errors = substitutions + deletions + insertions
        mean_error = (total_errors / total_words) * 100
        mean_error = np.round(mean_error, 4)
        mean_sentence_error = (sentence_errors / total_phrases) * 100
        mean_sentence_error = np.round(mean_sentence_error, 2)

        all_results['average']['error'] = mean_error
        all_results['average']['sentence_error'] = mean_sentence_error

        print('Cross-Validation Results')

    elif args.test_type == 'standard':

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))
        
        phrases = [' '.join(filepath.split('.')[1].split('_'))
            for filepath
            in htk_filepaths]
        train_data, test_data, _, _ = train_test_split(
            htk_filepaths, phrases, test_size=args.test_size,
            random_state=args.random_state)

        create_data_lists(train_data, test_data, args.phrase_len)
        if args.train_sbhmm:
            classifiers = trainSBHMM(args.sbhmm_cycles, args.train_iters, args.mean, args.variance, args.transition_prob, 
                        args.pca_components, args.sbhmm_iters, args.include_word_level_states, args.include_word_position, args.pca, 
                        args.hmm_insertion_penalty, args.sbhmm_insertion_penalty, args.parallel_jobs, args.parallel_classifier_training,
                        args.multiple_classifiers, args.neighbors, args.classifier, args.beam_threshold)
            testSBHMM(args.start, args.end, args.method, classifiers, args.pca_components, args.pca, args.sbhmm_insertion_penalty, 
                    args.parallel_jobs, args.parallel_classifier_training)
        else:
            train(args.train_iters, args.mean, args.variance, args.transition_prob)
            if args.method == "recognition":
                test(args.start, args.end, args.method, args.hmm_insertion_penalty)
            elif args.method == "verification":
                positive, negative, false_positive, false_negative = verify_simple(args.end, args.hmm_insertion_penalty, args.acceptance_threshold, args.beam_threshold)
        
        if args.method == "recognition":
            all_results['fold_0'] = get_results(hresults_file)
            all_results['average']['error'] = all_results['fold_0']['error']
            all_results['average']['sentence_error'] = all_results['fold_0']['sentence_error']

            print('Test on Train Results')
        
        if args.method == "verification":
            all_results['average']['positive'] = positive
            all_results['average']['negative'] = negative
            all_results['average']['false_positive'] = false_positive
            all_results['average']['false_negative'] = false_negative

            print('Standard Train/Test Split Results')

    elif args.test_type == 'user_independent_limited_guess':
        #### 3-SIGN
        print(args.method)
        generate_text_files([3]) #uncomment

        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0        

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]  

        # set the cross-validation as leave-one-user-out
        cross_val_method, use_groups = (LeaveOneGroupOut(), True)

        users = [get_user(filepath) for filepath in htk_filepaths]
        unique_users = list(set(users))
        unique_users.sort()
        print(unique_users, len(unique_users))
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in users]   
        cross_val = cross_val_method

        splits = list(cross_val.split(htk_filepaths, phrases, groups))
        splits = [list(item) for item in splits]

        # Remove all non-3 phrases from the test (test on only 3-sign phrases with only 3-sign grammar predictions)
        for i, htk_filepath in enumerate(htk_filepaths):
            phrase_len = get_phrase_len(htk_filepath)
            phrase_fold = groups[i]
            if phrase_len != 3: #change back to 3
                splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))

        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                        (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                        for currFold in range(len(splits)) if len(np.array(htk_filepaths)[splits[currFold][1]]) != 0 and "Guru" in np.array(htk_filepaths)[splits[currFold][1]][0]) 
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])

        print(stats)
        print(all_results)

        input()
        #### 4-SIGNS
        generate_text_files([4])

        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]    
                
        cross_val_method, use_groups = (LeaveOneGroupOut(), True)

        users = [get_user(filepath) for filepath in htk_filepaths]
        unique_users = list(set(users))
        unique_users.sort()
        print(unique_users)
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in users]   
        cross_val = cross_val_method

        # remove all 3 ones, add correct 3 ones to train dataset
        splits = list(cross_val.split(htk_filepaths, phrases, groups))
        splits = [list(item) for item in splits]

        # indices_to_remove
        for i, htk_filepath in enumerate(htk_filepaths):
            phrase_len = get_phrase_len(htk_filepath)
            phrase_fold = groups[i]

            if phrase_len != 4:
                splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))

        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                        (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold, False)
                        for currFold in range(len(splits)))
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])

        print(stats)
        print(all_results) 

        input()
        #### 5-SIGN
        generate_text_files([5])
        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]    
                
        cross_val_method, use_groups = (LeaveOneGroupOut(), True)

        users = [get_user(filepath) for filepath in htk_filepaths]
        unique_users = list(set(users))
        unique_users.sort()
        print(unique_users)
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in users]   
        cross_val = cross_val_method
        
        # Goals: 
        # 4 sign: remove all 3-sign items from splits[i][1] and add some of them to splits[i][0]
        # some => all the ones that were predicted correctly + all the incorrect ones that the prediction thought was correct (all the ones the system accepted)
        # fix issue where some videos weren't being added to the testing dataset <-- manually check for each video if its correct
        # remove 3 ones from test and add to train
        
        # remove all 3 ones, add correct 3 ones to train dataset
        splits = list(cross_val.split(htk_filepaths, phrases, groups))
        splits = [list(item) for item in splits]

        # indices_to_remove
        for i, htk_filepath in enumerate(htk_filepaths):
            phrase_len = get_phrase_len(htk_filepath)
            phrase_fold = groups[i]
            if phrase_len != 5:
                splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))

        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                        (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold, False)
                        for currFold in range(len(splits)))
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])

        print(stats)
        print(all_results)        
        

    elif args.test_type == 'progressive_user_adaptive':
        #### 3-SIGN

        ### prepare_data ==> 3-sign
        print(args.method)
        prepare_data(features_config, args.users, phrase_len=[3, 4, 5], prediction_len=[3], isSingleWord=args.isSingleWord) #uncomment
        
        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]         
        
        # set the cross-validation as leave-one-user-out
        cross_val_method, use_groups = (LeaveOneGroupOut(), True)

        users = [get_user(filepath) for filepath in htk_filepaths]
        unique_users = list(set(users))
        unique_users.sort()
        print(unique_users, len(unique_users))
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in users]   
        cross_val = cross_val_method

        splits = list(cross_val.split(htk_filepaths, phrases, groups))
        splits = [list(item) for item in splits]

        for i, htk_filepath in enumerate(htk_filepaths):
            phrase_len = get_phrase_len(htk_filepath)
            phrase_fold = groups[i]
            if phrase_len != 3: #change back to 3
                splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))
    
        # splits[i][0] => list of train data indices for fold i, splits[i][1] => list of test data indices for fold i

        # print(htk_filepaths)
        # print(group_map, groups, cross_val, splits, sep='\n\n')
        # for currFold in range(len(splits)):
        #     if len(np.array(htk_filepaths)[splits[currFold][1]]):
        #         print(np.array(htk_filepaths)[splits[currFold][1]][0])
            # print(np.array(htk_filepaths)[splits[currFold][1]])
        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                        (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                        for currFold in range(len(splits))) # if len(np.array(htk_filepaths)[splits[currFold][1]]) != 0 and "Harley" in np.array(htk_filepaths)[splits[currFold][1]][0]
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])

        print(stats)
        print(all_results)
        # save_results(all_results, args.save_results_file, 'w')

        # user-independent 3-sign recognition (so use the default HMM with whatever cross val)
        # save results
        # results => find data to add
        # pass to 4?
        input()
        prepare_data(features_config, args.users, phrase_len=[3, 4, 5], prediction_len=[4], isSingleWord=args.isSingleWord)

        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]    
                
        cross_val_method, use_groups = (LeaveOneGroupOut(), True)

        users = [get_user(filepath) for filepath in htk_filepaths]
        unique_users = list(set(users))
        unique_users.sort()
        print(unique_users)
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in users]   
        cross_val = cross_val_method
        
        # Goals: 
        # 4 sign: remove all 3-sign items from splits[i][1] and add some of them to splits[i][0]
        # some => all the ones that were predicted correctly + all the incorrect ones that the prediction thought was correct (all the ones the system accepted)
        # fix issue where some videos weren't being added to the testing dataset <-- manually check for each video if its correct
        # remove 3 ones from test and add to train
        
        # remove all 3 ones, add correct 3 ones to train dataset
        splits = list(cross_val.split(htk_filepaths, phrases, groups))
        splits = [list(item) for item in splits]

        accepted_3_signs = set()
        # indices_to_remove
        for i, htk_filepath in enumerate(htk_filepaths):
            phrase_len = get_phrase_len(htk_filepath)
            phrase_fold = groups[i]

            if phrase_len != 4:
                splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))
        
                if phrase_len == 3:
                    hresults_filepath = sorted(glob.glob(f'hresults/{phrase_fold}/*.txt'))[1]
                    hresults_data = get_hresults_data.get_hresults_data(hresults_filepath)
                    if get_video(htk_filepath) not in hresults_data:
                        accepted_3_signs.add(htk_filepath)
                        splits[phrase_fold][0] = np.append(splits[phrase_fold][0], i)
                # print(f'{i}: {splits[phrase_fold]}\n\n')

        # HVITE, HRESULTS, error then do                 if get_video(htk_filepath) not in hresults_data:
                    # splits[phrase_fold][0] = np.append(splits[phrase_fold][0], i)


        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                        (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                        for currFold in range(len(splits)))
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])

        print(stats)
        print(all_results) 
        # save_results(all_results, args.save_results_file, 'a')
        

        #### 4-SIGN
        # retrain HMM with what the system thought was correct and retrain HMM with leave-one-user-out cross val
        # 

        input()

        #### 5-SIGN
        prepare_data(features_config, args.users, phrase_len=[3, 4, 5], prediction_len=[5], isSingleWord=args.isSingleWord)
        word_counts = []
        phrase_counts = []
        substitutions = 0
        deletions = 0
        insertions = 0
        sentence_errors = 0

        if not args.users:
            htk_filepaths = glob.glob('data/htk/*htk')
        else:
            htk_filepaths = []
            for user in args.users:
                htk_filepaths.extend(glob.glob(os.path.join("data/htk", '*{}*.htk'.format(user))))

        phrases = [' '.join(filepath.split('.')[1].split("_"))
            for filepath
            in htk_filepaths]    
                
        cross_val_method, use_groups = (LeaveOneGroupOut(), True)

        users = [get_user(filepath) for filepath in htk_filepaths]
        unique_users = list(set(users))
        unique_users.sort()
        print(unique_users)
        group_map = {user: i for i, user in enumerate(unique_users)}
        groups = [group_map[user] for user in users]   
        cross_val = cross_val_method
        
        # Goals: 
        # 4 sign: remove all 3-sign items from splits[i][1] and add some of them to splits[i][0]
        # some => all the ones that were predicted correctly + all the incorrect ones that the prediction thought was correct (all the ones the system accepted)
        # fix issue where some videos weren't being added to the testing dataset <-- manually check for each video if its correct
        # remove 3 ones from test and add to train
        
        # remove all 3 ones, add correct 3 ones to train dataset
        splits = list(cross_val.split(htk_filepaths, phrases, groups))
        splits = [list(item) for item in splits]

        # indices_to_remove
        for i, htk_filepath in enumerate(htk_filepaths):
            phrase_len = get_phrase_len(htk_filepath)
            phrase_fold = groups[i]
            if phrase_len != 5:
                splits[phrase_fold][1] = np.delete(splits[phrase_fold][1], np.where(splits[phrase_fold][1] == i))

                hresults_filepath = sorted(glob.glob(f'hresults/{phrase_fold}/*.txt'))[1]
                hresults_data = get_hresults_data.get_hresults_data(hresults_filepath)
                if htk_filepath in accepted_3_signs or (phrase_len == 4 and get_video(htk_filepath) not in hresults_data):
                    splits[phrase_fold][0] = np.append(splits[phrase_fold][0], i)
                # print(f'{i}: {splits[phrase_fold]}\n\n')

        for i, split in enumerate(splits):
            splits[i][0] = np.array(splits[i][0])
            splits[i][1] = np.array(splits[i][1])

        print(type(splits), type(splits[0]), type(splits[0][0]), type(splits[0][1]))

        # HVITE, HRESULTS, error then do                 if get_video(htk_filepath) not in hresults_data:
                    # splits[phrase_fold][0] = np.append(splits[phrase_fold][0], i)

        stats = Parallel(n_jobs=args.parallel_jobs)(delayed(crossValFold)
                        (np.array(htk_filepaths)[splits[currFold][0]], np.array(htk_filepaths)[splits[currFold][1]], args, currFold)
                        for currFold in range(len(splits)))
        all_results['average']['error'] = mean([i[0] for i in stats])
        all_results['average']['sentence_error'] = mean([i[1] for i in stats])
        all_results['average']['insertions'] = mean([i[2] for i in stats])
        all_results['average']['deletions'] = mean([i[3] for i in stats])

        print(stats)
        print(all_results)
        # save_results(all_results, args.save_results_file, 'a')

        

    if args.method == "recognition":
        out_str_list = []

        avg_error = f'Average Error: {all_results["average"]["error"]}'
        avg_sent_error = f'Average Sentence Error: {all_results["average"]["sentence_error"]}'
        out_str_list.extend([avg_error, avg_sent_error])
        
        if args.test_type == 'cross_val' and args.cv_parallel:
            avg_insertions = f'Average Insertions: {all_results["average"]["insertions"]}'
            avg_deletions = f'Average Deletions: {all_results["average"]["deletions"]}'
            out_str_list.extend([avg_insertions, avg_deletions])
        
        if args.grid_results_file is None:
            print_to_stdout(out_str_list)
        else:
            print_to_file(out_str_list, args.grid_results_file)
    
    if args.method == "verification":

        print(f'Positive Pairs: {all_results["average"]["positive"]}')
        print(f'Negative Pairs: {all_results["average"]["negative"]}')
        print(f'False Positive Pairs: {all_results["average"]["false_positive"]}')
        print(f'False Negative Pairs: {all_results["average"]["false_negative"]}')
        percent_correct = (all_results["average"]["positive"] + all_results["average"]["negative"]) \
                            /(all_results["average"]["positive"] + all_results["average"]["negative"] + all_results["average"]["false_positive"] + all_results["average"]["false_negative"])
        print(f'Correct %: {percent_correct*100}')
        print(f'Precision %: {100*all_results["average"]["positive"]/(all_results["average"]["positive"] + all_results["average"]["false_positive"])}')
        print(f'Recall %: {100*all_results["average"]["positive"]/(all_results["average"]["positive"] + all_results["average"]["false_negative"])}')

    # Loads data as new run into pickle
    if args.save_results:
        save_results(all_results, args.save_results_file, 'a')
