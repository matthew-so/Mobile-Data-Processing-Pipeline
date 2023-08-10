import cv2 as cv
import mediapipe as mp
import json
import threading
import os
import argparse
import time
import csv

from pathlib import Path
from glob import glob
from google.protobuf.json_format import MessageToDict

ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.mkv']

# The parameter for min_detection_confidence when constructing our MediaPipe
# recognition object.
MP_DETECT_CONFIDENCE = 0.5

# The parameter for min_tracking_confidence when constructing our MediaPipe
# recognition object.
# MP_TRACK_CONFIDENCE = 0.1
MP_TRACK_CONFIDENCE = 0.5


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

            
# MediaPipe Pose options
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Detects the features within our videos using MediaPipe. This is
# copied in part from the MediaPipe website and in part from the
# mediaPipeWrapper.py
#
#
def detect_features(video_file, output_file):
    video = cv.VideoCapture(video_file)
    fps = video.get(cv.CAP_PROP_FPS)
    features = dict()
    curr_frame = 0
    
    video_status = {
        "video_passed": True,
    }

    
    # print("Video File: ", video_file)
    while video.isOpened():
        success, image = video.read()
        if not success:
            if curr_frame != video.get(cv.CAP_PROP_FRAME_COUNT):
                print(f'Frame {curr_frame} of {video_file} was not readable')
                video_status["video_passed"] = False
                video_status["frame"] = curr_frame
                video_status["error"] = "Frame was not readable"
            break

        image.flags.writeable = False
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
            running_mode=VisionRunningMode.VIDEO
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            # The landmarker is initialized. Use it here.
            # ...
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = landmarker.detect_for_video(mp_image, int(float(curr_frame)/fps*1000))

            # print(results)

            # Available features: results.face_landmarks, results.left_hand_landmarks,
            # results.right_hand_landmarks, results.pose_landmarks
            # multi_hand_landmarks = results.multi_hand_landmarks
            # multi_handedness = results.multi_handedness

            curr_frame_features = {"pose": {}}

            # print(results)

            pose_landmarks = results.pose_landmarks
            world_landmarks = results.pose_world_landmarks

            for idx, (local_landmark, global_landmark) in enumerate(zip(pose_landmarks[0], world_landmarks[0])):
                curr_frame_features["pose"][idx] = [local_landmark.x, local_landmark.y, local_landmark.z]

            features[curr_frame] = curr_frame_features
            curr_frame += 1

    video.release()

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as outfile:
        json.dump(features, outfile, indent=4)

    if MARK_EXTRACTED:
        new_name = video_file.split('.')
        new_name[-2] += MARKING_SUFFIX
        os.rename(video_file, '.'.join(new_name))

    return video_status


# Auto-detect number of CPU threads?
THREADS = os.cpu_count()
# THREADS = 1
lock = threading.Semaphore(THREADS)


class FeatureExtractorThread(threading.Thread):
    def __init__(self, input_filename, output_filename):
        super().__init__()
        self.input_filename = input_filename
        self.output_filename = output_filename

    def run(self) -> None:
        video_status = detect_features(self.input_filename, self.output_filename)
        lock.release()
        return video_status


def df_multithreaded(input_filenames, output_filenames):
    if len(input_filenames) != len(output_filenames):
        raise RuntimeError('Length of input and output file name arrays is not equal')

    i = 0
    total = len(input_filenames)
    thread_refs = []
    video_statuses = []  # Create an empty list to store video statuses

    while i < total and lock.acquire():
        print(f'{i + 1} / {total}: {input_filenames[i]}')
        thread = FeatureExtractorThread(input_filenames[i], output_filenames[i])
        thread.start()
        thread_refs.append(thread)
        i += 1

    for thread in thread_refs:
        thread.join()
        video_statuses.append(thread.run())  # Append the video_status returned by each thread

    print('Feature extraction complete')
    return video_statuses  # Return the list of video statuses



def enumerate_files(input_folder, job_array_num, processTenSign=False):
    input_filenames, output_filenames = list(), list()

    # Keeps track of the attempt number for each user/sign
    attempt_counts = dict()

    # Pads a number out using zeroes, e.g. pad(5) -> "00000008" (assuming PADDING_DIGITS = 8)
    def pad(num):
        existing_len = len(str(num))
        return f'{(PADDING_DIGITS - existing_len) * "0"}{num}'
    
    # Changing the below behavior to find files recursively ~Guru
    # all_files = sorted(os.listdir(input_folder))
    filename = f"/storage/home/hcoda1/9/mso31/scratch/batches/batch_{job_array_num}.txt"
    
    with open(
        filename
    ) as fin:
        all_files = fin.read().splitlines()
    # print(all_files)
    all_files = [file.replace(input_folder, '') for file in all_files]

    for file in all_files:
        if 'review_1' in file:
            curr_output_directory = f"{OUTPUT_DIRECTORY}review_1/"
        if 'review_2' in file:
            curr_output_directory = f"{OUTPUT_DIRECTORY}review_2/"
        if 'review_3' in file:
            curr_output_directory = f"{OUTPUT_DIRECTORY}review_3/"
        if 'review_4' in file:
            curr_output_directory = f"{OUTPUT_DIRECTORY}review_4/"
        else:
            curr_output_directory = OUTPUT_DIRECTORY

        # Validate file extension
        acceptable = False
        for extension in ALLOWED_EXTENSIONS:
            if file.lower().endswith(extension):
                acceptable = True
                break

        if not acceptable:
            continue
        
        suffix = file.rsplit('.', maxsplit=2)[-2]
        # if suffix.endswith(MARKING_SUFFIX):
        #     continue
        
        input_filenames.append(f'{INPUT_DIRECTORY}{file}')
        file = file.split('/')[-1]

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

        # input_filenames.append(f'{INPUT_DIRECTORY}{file}')

        # file = file.split('/')[-1]

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
        
        output_filenames.append(f'{curr_output_directory}{user_id}-{FILE_LABEL}/{sign}/'
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
    args.add_argument('--jobArrayNum',
                      help='Specifies the directory containing the video files to be processed. If not '
                           'specified, defaults to the current directory.')
    args.add_argument('--inputDirectory')
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

    MARK_EXTRACTED = False

    if parsed.inputDirectory is not None:
        INPUT_DIRECTORY = parsed.inputDirectory
        if not INPUT_DIRECTORY.endswith('/'):
            INPUT_DIRECTORY += '/'

    if parsed.outputDirectory is not None:
        OUTPUT_DIRECTORY = parsed.outputDirectory
        if not OUTPUT_DIRECTORY.endswith('/'):
            OUTPUT_DIRECTORY += '/'
    print(OUTPUT_DIRECTORY)
    if parsed.fileLabel is not None:
        FILE_LABEL = parsed.fileLabel

    if parsed.paddingDigits is not None:
        PADDING_DIGITS = parsed.paddingDigits

    input_files, output_files = enumerate_files(INPUT_DIRECTORY, parsed.jobArrayNum, processTenSign=parsed.processTenSign)
    # print(input_files[:10])
    # print(output_files[:10])
    # input_files = input_files[:10]
    # output_files = output_files[:10]
    
    start_time = time.time()
    video_statuses = df_multithreaded(input_files, output_files)
    end_time = time.time()

    print(f'Time elapsed = {(end_time - start_time) * 1000} ms')
    
    # For each video, get the video_statuses that have "video_passed" = False
    failed_videos = [video_status for video_status in video_statuses if not video_status["video_passed"]]
    
    # Save the failed videos list as a json file
    json_filename = f"failed_videos_{parsed.jobArrayNum}_pose.json"
    json_filename = os.path.join('logs', json_filename)
    with open(json_filename, "w") as outfile:
        json.dump(failed_videos, outfile, indent=4)





