"""Prepares training data. Creates .ark files, .htk files, wordList,
dict, grammar, and all_labels.mlf.

Methods
-------
prepare_data
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

from . import create_ark_files, create_htk_files
from .generate_text_files import generate_text_files

def prepare_data(
    features_config: dict,
    users: list,
    phrase_len:list=[3,4,5],
    prediction_len:list=[3,4,5],
    isFingerspelling:bool=False,
    isSingleWord:bool=False,
    num_threads:int=32
) -> None:

    """Prepares training data. Creates .ark files, .htk files, wordList,
    dict, grammar, and all_labels.mlf.

    Parameters
    ----------
    features_config : dict
        A dictionary defining which features to use when creating the 
        data files.
    """
    create_ark_files(
        features_config, users, [1], verbose=False, is_select_features=True,
        use_optical_flow=False, num_threads=num_threads
    )
    print('.ark files created')

    print('Creating .htk files')
    create_htk_files(num_threads=num_threads)
    print('.htk files created')

    print('Creating .txt files')
    generate_text_files(features_config["features_dir"], isFingerspelling, isSingleWord)
    print('.txt files created')
    
    # print("Data already generated, skipping data generation")
