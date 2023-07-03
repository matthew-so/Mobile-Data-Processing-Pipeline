import argparse

def parse_main_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ############################## ARGUMENTS #####################################
    #Important
    parser.add_argument('--prepare_data', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_results_file', type=str,
                        default='all_results.json')
    parser.add_argument('--features_file', type=str, default='configs/features.json')
    parser.add_argument('--prototypes_file', type=str, default='configs/prototypes.json')
    parser.add_argument('--wordlist', type=str, default='wordList')
    parser.add_argument('--is_single_word', action='store_true', help='Useful for prepare_data usage only.')

    # Arguments for create_data_lists()
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['none', 'test_on_train', 'cross_val', 'standard', 'progressive_user_adaptive', 'user_independent_limited_guess'])
    parser.add_argument('--users', nargs='*', default=[])
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
    parser.add_argument('--n_states', type=int, default=8)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--variance', type=float, default=1.0)
    parser.add_argument('--transition_prob', type=float, default=0.6)
    parser.add_argument(
        '--hmm_step_type',
        type=str,
        choices=['single','double','triple', 'start_stack', 'end_stack'],
        default='single'
    )
    parser.add_argument('--gmm_mix', type=int, default=None)
    parser.add_argument('--train_type', type=str, default='standard',
                        choices=['standard', 'five_sign'], help='Type of training.')
    parser.add_argument('--signs', type=str, default='standard',help='')

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
    
    return parser.parse_args()

def parse_grid_search_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--is_fingerspelling', action='store_true',
                        help='Whether the data is fingerspelling.')
    parser.add_argument('--gmm_mixes', nargs="*", type=int, default=[None],
                        help='Number of GMM Mixtures to use. Can be passed as a list.')
    parser.add_argument('--n_states', nargs='*', type=int, default=[8],
                        help='Number of states in the HMM. Can be passed as a list.')
    parser.add_argument('--n_folds', nargs='*', type=int, default=[5],
                        help='Number of folds for cross_val. Can be passed as a list.') # Called folds in driver.py/train.py/main.py
    parser.add_argument('--test_type', type=str, default='test_on_train',
                        choices=['test_on_train', 'cross_val'],
                        help='Test Type. Currently supports cross_val and test_on_train.')
    parser.add_argument(
        '--hmm_step_types',
        nargs='*',
        type=str,
        default=['single'],
        choices=['single', 'double', 'triple', 'start_stack', 'end_stack'],
        help='HMM Architecture Styles. Can be passed as a list.'
    )
    return parser.parse_args()

