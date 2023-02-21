"""Generates prototype files used to initalize models. Should be used
with the prototypes JSON in configs. Different words can be initialized with
different prototypes.

Methods
-------
generate_prototype
"""
import sys
import random
from .gen_hmm_utils import *

def generate_prototype(n_states: int, n_features: int, output_filepath: str, 
                       mean: float = 0.0, variance: float = 1.0, 
                       transition_prob: float = 0.6, hmm_step_type: str = 'single',
                       gmm_mix: int = None) -> None:
    """Generates prototype files used to initalize models.

    Parameters
    ----------
    n_states : int
        Number of states each model has.

    n_features : int
        Number of features being used to train each model.

    output_filepath : str
        File path at which to save prototype.

    mean : float, optional, by default 0.0
        Initial value to use as mean of all features.

    variance : float, optional, by default 1.0
        Initial value to use as variance of all features.

    transition_prob : float, optional, by default 0.6
        Initial probability of transition from one state to the next.
    """

    with open(output_filepath, 'w') as f:

        f.write('~o\n')
        f.write('<VecSize> {} <USER>\n'.format(n_features))
        f.write('~h "prototype"\n')
        f.write('<BeginHMM>\n')
        f.write('<NumStates> {}\n'.format(n_states))

        start = 2
        for i in range(start, n_states):
            if gmm_mix is not None:
                generate_gmm_state_space(f, n_states, n_features, gmm_mix, variance, i)
            else:
                generate_state_space(f, n_states, n_features, mean, variance, i)

        f.write('<TransP> {}\n'.format(n_states))
        row = ['0.0'] + ['1.0'] + ['0.0']*(n_states - 2)
        f.write(' '.join(row) + '\n')
        
        if hmm_step_type == 'single':
            generate_single_step_hmm(f, n_states)
        elif hmm_step_type == 'double':
            generate_double_step_hmm(f, n_states)
        elif hmm_step_type == 'triple':
            generate_triple_step_hmm(f, n_states)
        elif hmm_step_type == 'start_stack':
            generate_start_stack_hmm(f, n_states)
        elif hmm_step_type == 'end_stack':
            generate_end_stack_hmm(f, n_states)

        f.write(' '.join(['0.0']*n_states) + '\n')
        f.write('<EndHMM>\n')
