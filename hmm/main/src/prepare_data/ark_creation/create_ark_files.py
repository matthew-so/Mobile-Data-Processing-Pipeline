"""Creates .ark files needed as intermediate step to creating .htk files

Methods
-------
_create_ark_file
create_ark_files
"""
import os
import glob
import shutil
import tqdm
import argparse

from .feature_selection import select_features
from .interpolate_feature_data import interpolate_feature_data
from .feature_extraction_kinect import feature_extraction_kinect
from .feature_extraction_alphapose import feature_extraction_alphapose

from collections import defaultdict
from scipy import interpolate
from scipy.spatial.distance import cdist
from scipy import stats
from multiprocess import Pool, Lock

import numpy as np
import pandas as pd

features_rows_lock = Lock()

def _create_ark_file(df: pd.DataFrame, ark_filepath: str, title: str) -> None:
    """Creates a single .ark file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing selected feature.

    ark_filepath : str
        File path at which to save .ark file.

    title : str
        Title containing label needed as header of .ark file.
    """

    with open(ark_filepath, 'w') as out:
        out.write('{} [ '.format(title))

    df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')

    with open(ark_filepath, 'a') as out:
        out.write(']')

def extract_features(features_config: dict, ark_dir: str, verbose: bool, is_select_features: bool,
                use_optical_flow: bool, features_filepath: str, features_rows: defaultdict):
   if verbose:
       print(features_filepath)

   features_filename = features_filepath.split('/')[-1]
   features_extension = features_filename.split('.')[-1]
   features_df = None

   ark_filename = features_filename.replace(features_extension, 'ark')
   ark_filepath = os.path.join(ark_dir, ark_filename)
   title = ark_filename.replace('.ark', "")
   
   if 'alphapose' in features_filename:
       features_df = feature_extraction_alphapose(features_filepath, features_config['selected_features'], scale = 10, drop_na = True)
   elif features_extension == 'json':
       features_df = feature_extraction_kinect(features_filepath, features_config['selected_features'], scale = 10, drop_na = True)
   elif is_select_features:
       features_df = select_features(features_filepath, features_config['selected_features'], center_on_nose = True, scale = 100, square = True, 
                               drop_na = True, do_interpolate = True, use_optical_flow=use_optical_flow)
   else:
       features_df = interpolate_feature_data(features_filepath, features_config['selected_features'], center_on_face = False, is_2d = True, scale = 10, drop_na = True)
   
   if features_df is not None:
       num_rows = features_df.shape[0]
       if num_rows > 0:
           _create_ark_file(features_df, ark_filepath, title)
       
       features_rows_lock.acquire()
       features_rows[num_rows] += 1
       features_rows_lock.release()

def create_ark_files(features_config: dict, users: list, phrase_len: list, verbose: bool, 
                is_select_features: bool, use_optical_flow: bool, num_threads: int) -> None:
    """Creates .ark files needed as intermediate step to creating .htk
    files

    Parameters
    ----------
    features_config : dict
        Contains features_dir and features_to_extract

    verbose : bool, optional, by default False
        Whether to print output during process.
    """

    ark_dir = os.path.join('data', 'ark')
    
    # if os.path.exists(ark_dir):
    #     shutil.rmtree(ark_dir)
    
    if not os.path.exists(ark_dir):
        os.makedirs(ark_dir)

    if not users:
        features_filepaths = glob.glob(os.path.join(features_config['features_dir'], '**', '*.data'), recursive = True)
        features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '**', '*.json'), recursive = True))
    else:
        features_filepaths = []
        print(users)
        for user in users:
            print(os.path.join(features_config['features_dir'], '*{}_*'.format(user), '**', '*.json'))
            features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '*{}_*'.format(user), '**', '*.data'), recursive = True))
            features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '*{}_*'.format(user), '**', '*.json'), recursive = True))
    
    features_filepaths = list(filter(lambda x: len(x.rsplit('.', 4)[1].split('_')) in phrase_len, features_filepaths))
    
    print(features_config['features_dir'])
    if is_select_features:
        print("Generating ark/htk using select_features data model")
    else:
        print("Generating ark/htk using interpolate_features data model")
    
    features_rows = defaultdict(int)
    results = []
    pool = Pool(num_threads)
    for features_filepath in features_filepaths:
        thread_args = (
            features_config,
            ark_dir,
            verbose,
            is_select_features,
            use_optical_flow,
            features_filepath,
            features_rows
        )
        result = pool.apply_async(extract_features, args=thread_args)
        results.append(result)

    for result in tqdm.tqdm(results):
        result.get()
    pool.close()

    print("Features Num Rows Distribution: ", features_rows)
     
