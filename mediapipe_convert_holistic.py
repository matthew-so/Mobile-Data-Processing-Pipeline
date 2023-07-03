import cv2 as cv
import mediapipe as mp
import json
import threading
import os
import argparse
import time

from pathlib import Path
from glob import glob

ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.mkv']

# The parameter for min_detection_confidence when constructing our MediaPipe
# recognition object.
MP_DETECT_CONFIDENCE = 0.5

# The parameter for min_tracking_confidence when constructing our MediaPipe
# recognition object.
MP_TRACK_CONFIDENCE = 0.1

# Whether or not the video files from which we have already extracted MediaPipe
# features should be marked as such. Setting this to False will cause the program
# to repeatedly process the same videos in the same folders unless we delete them.
#
# Disable at runtime using the --noMark option.
MARK_EXTRACTED = True

# The suffix to add to the filename when it has been extracted, if MARK_EXTRACTED
# is True. This should not be changed after the initial setup if we want to avoid
# reruns.
MARKING_SUFFIX = '-done'

INPUT_DIRECTORY = './'

# The root folder for the output. By default, this will just be the location where
# the command is executed. Note that we will create a subdirectory structure under
# OUTPUT_DIRECTORY.
#
# Can be specified at runtime via the --outputDirectory option.
OUTPUT_DIRECTORY = './output/'

# The label to use within the output directory structure. For reference, the format
# for output files is

# {OUTPUT_DIRECTORY}/{user_id}-{FILE_LABEL}/{sign}/{attempt_str}/
#     {user_id}_{FILE_LABEL}_{sign}_{attempt_str}.data
#
# Can be specified at runtime via the --fileLabel option.
FILE_LABEL = 'singlesign'

# The total length of the attempt number in the generated files.
# For example, if PADDING_DIGITS = 8, then the number 1 will be output as
# '00000001'. This prevents numbers like '10' from appearing before '2'
# when using ASCII-based sorting.
#
# Can be specified at runtime via the --paddingDigits option.
PADDING_DIGITS = 8


# Detects the features within our videos using MediaPipe. This is
# copied in part from the MediaPipe website and in part from the
# mediaPipeWrapper.py
#
#
def detect_features(video_file, output_file):
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=MP_DETECT_CONFIDENCE,
        min_tracking_confidence=MP_TRACK_CONFIDENCE
    ) as holistic:
        video = cv.VideoCapture(video_file)
        features = dict()
        curr_frame = 0
        
        # print("Video File: ", video_file)
        while video.isOpened():
            success, image = video.read()
            if not success:
                print(f'Frame {curr_frame} of {video_file} was not readable')
                break

            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = holistic.process(image)

            # if results.left_hand_landmarks is not None:
            #     print("Results (Left Hand Landmarks) is not None.")
            # else:
            #     print("Results (Left Hand Landmarks) is None.")
            # 
            # if results.right_hand_landmarks is not None:
            #     print("Results (Right Hand Landmarks) is not None.")
            # else:
            #     print("Results (Right Hand Landmarks) is None.")

            # print()

            # Available features: results.face_landmarks, results.left_hand_landmarks,
            # results.right_hand_landmarks, results.pose_landmarks
            curr_frame_features = {"pose": {}, "landmarks": {0: {}, 1: {}}}
            available_features = [
                results.left_hand_landmarks,
                results.right_hand_landmarks,
                results.pose_landmarks
            ]

            feature_location = [
                curr_frame_features["landmarks"][0],
                curr_frame_features["landmarks"][1],
                curr_frame_features["pose"]
            ]

            for index, curr_feature in enumerate(available_features):
                feature_num = 0
                if curr_feature is None:
                    feature_location[index] = "None"
                else:
                    for curr_point in curr_feature.landmark:
                        feature_location[index][feature_num] = [curr_point.x, curr_point.y, curr_point.z]
                        feature_num += 1

            features[curr_frame] = curr_frame_features
            curr_frame += 1

        video.release()

        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, "w") as outfile:
            json.dump(features, outfile, indent=4)

        #if MARK_EXTRACTED:
        #    new_name = video_file.split('.')
        #    new_name[-2] += MARKING_SUFFIX
        #    os.rename(video_file, '.'.join(new_name))


# Auto-detect number of CPU threads?
THREADS = 32
lock = threading.Semaphore(THREADS)


class FeatureExtractorThread(threading.Thread):
    def __init__(self, input_filename, output_filename):
        super().__init__()
        self.input_filename = input_filename
        self.output_filename = output_filename

    def run(self) -> None:
        detect_features(self.input_filename, self.output_filename)
        lock.release()


def df_multithreaded(input_filenames, output_filenames):
    if len(input_filenames) != len(output_filenames):
        raise RuntimeError('Length of input and output file name arrays is not equal')

    i = 0
    total = len(input_filenames)
    thread_refs = list()

    while i < total and lock.acquire():
        print(f'{i + 1} / {total}: {input_filenames[i]}')
        thread = FeatureExtractorThread(input_filenames[i], output_filenames[i])
        thread.start()
        thread_refs.append(thread)
        i += 1

    for thread in thread_refs:
        thread.join()

    print('Feature extraction complete')


def enumerate_files(input_folder, processTenSign=False):
    input_filenames, output_filenames = list(), list()

    # Keeps track of the attempt number for each user/sign
    attempt_counts = dict()

    # Pads a number out using zeroes, e.g. pad(5) -> "00000008" (assuming PADDING_DIGITS = 8)
    def pad(num):
        existing_len = len(str(num))
        return f'{(PADDING_DIGITS - existing_len) * "0"}{num}'

    all_files = sorted(glob(os.path.join(input_folder, '**', '*.mp4')))
    print(all_files)

    for file in all_files:
        # Validate file extension
        acceptable = False
        for extension in ALLOWED_EXTENSIONS:
            if file.lower().endswith(extension):
                acceptable = True
                break

        if not acceptable:
            continue
        
        suffix = file.rsplit('.', maxsplit=2)[-2]
        if suffix.endswith(MARKING_SUFFIX):
            continue

        # <id>_<session-start-time>_<sign>_<start-time>.mp4
        # <id> = user's ID
        # <sign> = word being signed
        # <start-time> = time the recording started
        split = file.rsplit('.', maxsplit=2)
        print(f'Split: {split}')
        if len(split) < 3:
            print(f'{file}: Skipped due to incorrect filename format')
            continue

        ### Code below is old and doesn't pull the sign or user_id correctly. Redoing below.
        # user_id, session_start = split[0], split[1]
        # sign = '_'.join(split[2].split('_')[:-1])

        # user_id, sign, session_start = split[0].split('-')
        user_id, sign, _ = split[0].split('-')

        if user_id not in attempt_counts:
            attempt_counts[user_id] = dict()

        # if session_start not in attempt_counts[user_id]:
        #     attempt_counts[user_id][session_start] = dict()

        # if sign not in attempt_counts[user_id][session_start]:
        #     attempt_counts[user_id][session_start][sign] = 0

        if sign not in attempt_counts[user_id]:
            attempt_counts[user_id][sign] = 0

        # attempt_counts[user_id][session_start][sign] += 1
        attempt_counts[user_id][sign] += 1

        input_filenames.append(f'{INPUT_DIRECTORY}{file}')

        # attempt_str = pad(attempt_counts[user_id][session_start][sign])
        attempt_str = pad(attempt_counts[user_id][sign])
        
        # <id>-singlesign/<sign>/<attempt>/<id>.singlesign.<sign>.<attempt>.data
        # <id> = user's ID
        # <sign> = word being signed
        # <attempt> = counter, starting at 00000001
        # print(f'Sign: {sign}')
        # print(f'User ID: {user_id}')
        # print(f'Attempt: {attempt_str}')
        # print(f'Session Start: {session_start}')
        # print()
        # output_filenames.append(f'{OUTPUT_DIRECTORY}{user_id}-{FILE_LABEL}/{sign}/'
        #                         f'{session_start}/{user_id}.{sign}.{FILE_LABEL}.{attempt_str}.data')
        output_filenames.append(f'{OUTPUT_DIRECTORY}{user_id}-{FILE_LABEL}/{sign}/'
                                f'{user_id}.{sign}.{FILE_LABEL}.{attempt_str}.data')

    # '1-2022-05-singlesign'
    # ['test_hello_2022.09.10.mp4'], ['./test-singlesign/hello/2022.09.10/test.singlesign.hello.00000001.data']
    return input_filenames, output_filenames


if __name__ == '__main__':
    args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args.add_argument('--noMark', action='store_true',
                      help='If provided, files that are processed will not be renamed to include "-done" '
                           'at the end of their filenames. This means they will be re-processed the next '
                           'time this script is run.')
    args.add_argument('--inputDirectory',
                      help='Specifies the directory containing the video files to be processed. If not '
                           'specified, defaults to the current directory.')
    args.add_argument('--outputDirectory',
                      help='Specifies the location where the output should be created. If not specified, '
                           'defaults to the "output" folder within the current directory. (This folder '
                           'does not need to already exist at runtime, the script can create it for you.)')
    args.add_argument('--fileLabel',
                      help='The tag to include in the processed file names. If not specified, defaults '
                           'to "singlesign".')
    args.add_argument('--paddingDigits', type=int,
                      help='The number of padding digits to use when numbering attempts of a sign. If '
                           'not specified, defaults to 8.')
    args.add_argument('--processTenSign', action='store_true',
                      help='If provided, will assume input directory contains 10 sign videos.')
    parsed = args.parse_args()

    if parsed.noMark:
        MARK_EXTRACTED = False

    if parsed.inputDirectory is not None:
        INPUT_DIRECTORY = parsed.inputDirectory
        if not INPUT_DIRECTORY.endswith('/'):
            INPUT_DIRECTORY += '/'

    if parsed.outputDirectory is not None:
        OUTPUT_DIRECTORY = parsed.outputDirectory
        if not OUTPUT_DIRECTORY.endswith('/'):
            OUTPUT_DIRECTORY += '/'

    if parsed.fileLabel is not None:
        FILE_LABEL = parsed.fileLabel

    if parsed.paddingDigits is not None:
        PADDING_DIGITS = parsed.paddingDigits

    input_files, output_files = enumerate_files(INPUT_DIRECTORY, processTenSign=parsed.processTenSign)
    print(input_files)
    print(output_files)

    start_time = time.time()
    df_multithreaded(input_files, output_files)
    end_time = time.time()

    print(f'Time elapsed = {(end_time - start_time) * 1000} ms')





