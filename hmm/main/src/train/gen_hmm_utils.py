import random
## File containing HMM utils

##########################################################################################################################
#################################################### HELPER FUNCTIONS ####################################################
##########################################################################################################################

# Helper for out prob computation. Assumes self transition is passed in and uniformity among split.
def get_out_prob(self_transition_prob: float, n_split: int):
    return (1 - self_transition_prob) / n_split

# Generates last Single/Double/Triple step row
def generate_last_single_step(f, n_states: int, transition_prob: float, out_prob: float):
    row = ['0.0']*(n_states - 2) + [str(transition_prob), str(out_prob)]
    f.write(' '.join(row) + '\n')

def generate_last_double_step(f, n_states: int, transition_prob: float, out_prob: float):
    row = ['0.0']*(n_states - 3) + [str(transition_prob), str(out_prob), str(out_prob)]
    f.write(' '.join(row) + '\n')

def generate_last_triple_step(f, n_states: int, transition_prob: float, out_prob: float):
    row = ['0.0']*(n_states - 4) + [str(transition_prob), str(out_prob), str(out_prob), str(out_prob)]
    f.write(' '.join(row) + '\n')

def generate_uniform_row(f, n_states: int, i: int = 1):
    unif_prob = round(1.0 / (n_states - i), 2)
    last_prob = round(1.0 - (unif_prob * (n_states - i - 1)), 2)

    row = ['0.0']*i + [str(unif_prob)]*(n_states - i - 1) + [str(last_prob)]
    f.write(' '.join(row) + '\n')

def generate_gmm_state_space(
    f, n_states: int, n_features: int, n_mixes: int, variance: float, i: int
):
    f.write('<State> {} <NumMixes> {}\n'.format(i, n_mixes))
    for j in range(1, n_mixes + 1):
        f.write('<Mixture> {} {}\n'.format(j, 1.0 / n_mixes))
        f.write('<Mean> {}\n'.format(n_features))
        f.write(' '.join([str(random.random()) for _ in range(n_features)]) + '\n')
        f.write('<Variance> {}\n'.format(n_features))
        f.write(' '.join([str(variance)]*n_features) + '\n')
    

def generate_state_space(
    f, n_states: int, n_features: int, mean: float,
    variance: float, i: int
):
    f.write('<State> {}\n'.format(i))
    f.write('<Mean> {}\n'.format(n_features))
    f.write(' '.join([str(mean)]*n_features) + '\n')
    f.write('<Variance> {}\n'.format(n_features))
    f.write(' '.join([str(variance)]*n_features) + '\n')

###########################################################################################################################
#################################################### HMM GEN FUNCTIONS ####################################################
###########################################################################################################################

def generate_single_step_hmm(f, n_states: int, transition_prob: float = 0.6, start: int = 1) -> None:
    out_prob = get_out_prob(transition_prob, 1)
    
    for i in range(start, n_states-2):
        row = ['0.0']*i + [str(transition_prob), str(out_prob)] + ['0.0']*(n_states - i - 2)
        f.write(' '.join(row) + '\n')
    
    generate_last_single_step(f, n_states, transition_prob, out_prob)

def generate_double_step_hmm(f, n_states: int, transition_prob: float = 0.4, start: int = 1) -> None:
    out_prob = get_out_prob(transition_prob, 2)
    for i in range(start, n_states-3):
        row = ['0.0']*i + [str(transition_prob), str(out_prob), str(out_prob)] + ['0.0']*(n_states - i - 3)
        f.write(' '.join(row) + '\n')

    generate_last_double_step(f, n_states, transition_prob, out_prob)
    generate_last_single_step(f, n_states, 0.6, get_out_prob(0.6, 1))
    
def generate_triple_step_hmm(f, n_states: int, transition_prob: float = 0.28, start: int = 1) -> None:
    out_prob = get_out_prob(transition_prob, 3)
    for i in range(start, n_states-4):
        row = ['0.0']*i + [str(transition_prob), str(out_prob), str(out_prob), str(out_prob)] + ['0.0']*(n_states - i - 4)
        f.write(' '.join(row) + '\n')

    generate_last_triple_step(f, n_states, transition_prob, out_prob)
    generate_last_double_step(f, n_states, 0.4, get_out_prob(0.4, 2))
    generate_last_single_step(f, n_states, 0.6, get_out_prob(0.6, 1))

def generate_start_stack_hmm(f, n_states: int, transition_prob: float=0.4) -> None:
    out_prob = get_out_prob(transition_prob, 2)
    for i in range(1, n_states-3):
        row = ['0.0']*i + [str(transition_prob), str(out_prob)] + ['0.0']*(n_states - i - 3) + [str(out_prob)]
        f.write(' '.join(row) + '\n')

    generate_last_double_step(f, n_states, transition_prob, out_prob)
    generate_last_single_step(f, n_states, 0.6, get_out_prob(0.6, 1))

def generate_end_stack_hmm(f, n_states: int, transition_prob: float=0.6) -> None:
    generate_uniform_row(f, n_states)
    generate_single_step_hmm(f, n_states=n_states, transition_prob=transition_prob, start = 2)

