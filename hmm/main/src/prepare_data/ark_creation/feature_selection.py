"""Processes raw features extracted from MediaPipe/Kinect, and selects
the specified features for use during training of HMMs.

Methods
-------
_load_json
_calc_delta
_add_delta_col
select_features
"""
import json
import random
import argparse
import ipdb
import os

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import cdist
from multiprocess import Lock
from json import JSONDecodeError

log_lock = Lock()

def _load_json(json_file: str) -> dict:
    """Load JSON file TODO: Remove and use src.utils.load_json.

    Parameters
    ----------
    json_file : str
        File path of JSON to be loaded.

    Returns
    -------
    data : dict
        Data loaded from JSON.
    """
    
    data = None
    with open(json_file, 'r') as data_file:
        try:
            data = json.loads(data_file.read())
        except JSONDecodeError:
            print("Problem File: ", json_file)
        
    return data


def _calc_delta(col: str) -> np.ndarray:
    """Calculates delta between consecutives rows of a given column.

    Parameters
    ----------
    col : str
        Column for which to calculate delta.

    Returns
    -------
    np.ndarray
        Delta of rows.
    """
    
    return np.concatenate([[0], col[1:].values - col[:-1].values])


def _add_delta_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calculate delta for a column and add it as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features.

    col : str
        Column for which to calculate delta.

    Returns
    -------
    return_val : pd.DataFrame
        DataFrame containing features with new delta column.
    """
    
    df['delta_{}'.format(col)] = _calc_delta(df[col])
    
    return df.copy()

def landmark_box_dist(landmark: list, hand: list) -> float:

    curr_landmark = np.reshape(landmark, (3,21))
    total_dist = 0
    for point in curr_landmark:
        hand_point = [[hand[0], hand[1]]]
        landmark_point = [[point[0], point[1]]]
        total_dist += cdist(hand_point, landmark_point)[0]
    
    return total_dist/(21)


def select_features(input_filepath: str, features_to_extract: list,
                    interpolation_method: str = 'linear', order: int = 3,
                    center_on_nose: bool = False, center_on_pelvis: bool = False, is_2d: bool = True,
                    scale: int = 10, drop_na: bool = True, do_interpolate: bool = False, use_optical_flow = False,
                    square: bool = False) -> pd.DataFrame:
    """Processes raw features extracted from MediaPipe/Kinect, and
    selects the specified features for use during training of HMMs.

    Parameters
    ----------
    input_filepath : str
        File path of raw feature data to be processed and used for
        selection.

    features_to_extract : list
        Names of columns to be selected after processing features.

    interpolation_method : str, optional, by default 'spline'
        Interpolation method used to fill missing values.

    order : int, optional, by default 3
        Hyperparameter needed for certain interpolation methods.

    center_on_nose : bool, optional, by default False
        Whether to center the features on the main face.

    is_2d : bool, optional, by default True
        Whether data is 2-dimensional.

    scale : int, optional, by default 10
        Raw features are scaled from 0-1, which can cause issues with
        very small means and variances in HTK. Used to scale up
        features.

    Returns
    -------
    df : pd.DataFrame
        Selected features
    """
    data = _load_json(input_filepath)
    # print("Input Filepath: ", input_filepath)
    # print("Data Keys: ", data['1'].keys())

    if not data:
        return None
    data = {int(key): value for key, value in data.items()}

    n_frames = len(data)
    hands = np.zeros((2, n_frames, 18))
    landmarks = np.zeros((2, n_frames, 63))
    noses = np.zeros((1, n_frames, 3))
    optical_flow = np.zeros((2, n_frames, 32))
    # print("Features to Extract: ", features_to_extract)

    frames = list(data.keys())
    frames.sort()
    if n_frames - 1 != frames[-1]:
        log_lock.acquire()
        with open('logs/prepare_data.log', 'a') as f:
            f.write(os.path.basename(input_filepath) + '\n')
        log_lock.release()
        
        return None

    for frame in sorted(data.keys()):

        if use_optical_flow and data[frame]['optical_flow'] is not None:

            x_val = data[frame]['optical_flow']["0"]
            y_val = data[frame]['optical_flow']["1"]

            optical_flow[0, frame, :] = np.array(x_val)/10000000.0
            optical_flow[1, frame, :] = np.array(y_val)/10000000.0
        
        # print('File: ', input_filepath)
        # if data[frame]["landmarks"]:
        #     print("Landmarks Available")
        # else:
        #     print("Landmarks Unavailable")

        # for hand in data[frame]["landmarks"]:
        #     print("Hand: ", hand)
        #     print('Data Frame Info: ', len(data[frame]['landmarks'][hand]))
        #     print()

        if 'pose' in data[frame] and data[frame]['pose']:

            right_hand = np.array([data[frame]['pose'][str(i)] for i in range(11,22,2)]).reshape((-1,))
            left_hand = np.array([data[frame]['pose'][str(i)] for i in range(12,23,2)]).reshape((-1,))
            curr_nose = np.array(data[frame]['pose']["0"]).reshape((-1,))
            hands[0, frame, :] = right_hand
            hands[1, frame, :] = left_hand
            noses[0, frame, :] = curr_nose
        
        if data[frame]['landmarks']:
            # if data[frame]['pose'] is None:
            #     raise Exception('Red Alert: Our assumption that landmarks are only provided when we have boxes is incorrect')
            # else:
            visible_landmarks = [data[frame]['landmarks']["0"], data[frame]['landmarks']["1"]]
            if visible_landmarks[0]:
                left_landmark = np.array([visible_landmarks[0][str(i)] for i in range(21)]).reshape((-1,))
                landmarks[1, frame, :] = left_landmark
            if visible_landmarks[1]:
                right_landmark = np.array([visible_landmarks[1][str(i)] for i in range(21)]).reshape((-1,))
                landmarks[0, frame, :] = right_landmark
                    
    select_hands = np.any(['hand' 
                           in feature 
                           for feature 
                           in features_to_extract])
    select_landmarks = np.any(['landmark' 
                               in feature 
                               for feature 
                               in features_to_extract])
    select_nose = np.any(['nose' 
                                in feature
                                for feature
                                in features_to_extract]) 

    select_optical_flow = np.any(['optical_flow'
                                in feature
                                for feature
                                in features_to_extract])

    if select_hands and not np.any(hands):
        return None

    if select_landmarks and not np.any(landmarks):
        return None

    if select_nose and not np.any(noses):
        return None
    
    if use_optical_flow and select_optical_flow and not np.any(optical_flow):
        return None

    hands_ = ['right', 'left']
    landmarks_ = ['shoulder', 'elbow', 'wrist', 'pinky', 'index', 'thumb']
    coordinates = ['x', 'y', 'z']
    hand_cols = [f'{hand}_{landmark}_{coordinate}' 
                for hand 
                in hands_
                for landmark
                in landmarks_
                for coordinate 
                in coordinates]

    hands_ = ['right', 'left']
    landmarks_ = ['landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y', 'z']
    landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in landmarks_ 
                    for coordinate 
                    in coordinates]

    nose_ = ['nose']
    coordinates = ['x', 'y', 'z']
    nose_cols = ['{}_{}'.format(nose, coordinate)
                for nose
                in nose_
                for coordinate
                in coordinates]
    
    coordinates = ['optical_flow_x', 'optical_flow_y']
    optical_flow_cols = ['{}_{}'.format(coordinate, i)
                for i
                in range(32)
                for coordinate
                in coordinates]

    # ipdb.set_trace()
    cols = hand_cols + landmark_cols + nose_cols + optical_flow_cols
    hands = np.concatenate([hands[0], hands[1]], axis=1)
    left_wrist = np.tile(landmarks[0,:,0:3], 21)
    right_wrist = np.tile(landmarks[1,:,0:3], 21)
    landmarks = np.concatenate([landmarks[0], landmarks[1]], axis=1)
    centered_landmarks = landmarks - np.concatenate([left_wrist, right_wrist], axis=1)
    optical_flow = np.concatenate([optical_flow[0], optical_flow[1]], axis=1)
    noses = np.reshape(noses, (-1, 3))
    all_features = np.concatenate([hands, landmarks, noses, optical_flow], axis=1)
    df = pd.DataFrame(all_features, columns=cols)
    df_lm = df.loc[:, df.columns.isin(features_to_extract)]
    # print("DF Shape (1): ", df.shape)
    df = df.replace(0, np.nan)
    df_lm = df_lm.replace(0, np.nan)
    # print("DF Shape (2): ", df.shape)

    if select_hands and do_interpolate:

        try:
            df[hand_cols] = df[hand_cols].interpolate(interpolation_method, order=order)
        except Exception as e:
            print(df[hand_cols])
            print(input_filepath)
            return None

    if select_landmarks and do_interpolate:

        try:
            df[landmark_cols] = df[landmark_cols].interpolate(interpolation_method, order=order)
        except Exception as e:
            print(df[landmark_cols])
            print(e)
            print(input_filepath)
            return None

    if center_on_nose:
        
        x_cols = [column for column in df.columns if '_x' in column]
        y_cols = [column for column in df.columns if '_y' in column]
        z_cols = [column for column in df.columns if '_z' in column]

        # SHAPES:
        # df[x_cols]: (159, 87)
        # noses: (159, 3)
        # noses[0]: (,3)
        # noses[:,0]: (159,)
        
        # this is what we had before
        # df[x_cols] -= np.mean(noses, axis=1)[0]
        # df[y_cols] -= np.mean(noses, axis=1)[1]
        # df[z_cols] -= np.mean(noses, axis=1)[2]

        df[x_cols] = (df[x_cols].transpose() - noses[:,0]).transpose()
        df[y_cols] = (df[y_cols].transpose() - noses[:,1]).transpose()
        df[z_cols] = (df[z_cols].transpose() - noses[:,2]).transpose()

    # print("DF Shape (3): ", df.shape)
    df['horizontal_hand_dist'] = df['right_index_x'] - df['left_index_x']
    df['vertical_hand_dist'] = df['right_index_y'] - df['left_index_y']

    hands_ = ['right', 'left']
    centered_landmarks_ = ['centered_landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y', 'z']
    centered_landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in centered_landmarks_ 
                    for coordinate 
                    in coordinates]
    wrist = ['{}_{}_{}'.format(hand, wrist_landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for wrist_landmark
                    in ['landmark_0' for i in range(21)]
                    for coordinate 
                    in coordinates]
    for old_lm, centered_lm, wrist_coord in zip(landmark_cols, centered_landmark_cols, wrist):
        df[centered_lm] = df[old_lm] - df[wrist_coord]

    for col in df.columns:

        df = _add_delta_col(df, col)

    df = df.loc[:, df.columns.isin(features_to_extract)]
    # ipdb.set_trace()
    # print("DF Shape (4): ", df.shape)
    if drop_na:
        df = df.dropna(axis=0)
    # print("DF Shape (5): ", df.shape)
    df = df * scale
    if square:
        df = pd.DataFrame(df.values*df.abs().values, columns=df.columns, index=df.index)
    df = df.round(6)
    
    # print("DF Shape (Last): ", df.shape)
    # print()
    return df
