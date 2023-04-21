"""Defines method to verify phases using HMM.

Methods
-------
verify
"""
import os
import glob
import shutil
import numpy as np
from string import Template
import tqdm
import argparse
import sys
import subprocess

os.chdir('/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/')

from json_data import load_json

sys.path.insert(0, '/home/aslr/SBHMM-HTK/SequentialClassification/main/src/prepare_data/ark_creation')
from feature_selection import select_features
from interpolate_feature_data import interpolate_feature_data
from feature_extraction_kinect import feature_extraction_kinect
from feature_extraction_alphapose import feature_extraction_alphapose
import numpy as np

def videoToKinectFeatures(video_filepath: str, is_left: bool = False):
    
    video_dirname = os.path.dirname(video_filepath)
    video_name = os.path.basename(video_filepath)
    source = f"\"{video_filepath}\""

    video_extension = video_name.split('.')[-1]

    phrase = video_name.split('.')[0]
    destination_feature_filename = f"user.{phrase}.0000000000.json"
    destination_feature_filepath = os.path.join(video_dirname, destination_feature_filename)
    dest = f"\"{destination_feature_filepath}\""

    print(f"source: {source}, dest: {dest}")

    command = None

    #TODO: add offline_processor_left
    if is_left: command = None
    else: command = f"/home/aslr/SBHMM-HTK/KinectProcessor/offline_processor {source} {dest}"

    try: 
        output = subprocess.check_output(command, shell=True)
    except:
        print(f'{command} errored. please check source {source} and dest {dest}')
    
    return destination_feature_filepath

def videoToMediapipeFeatures(video_filepath: str, is_left: bool = False):
    video_dirname = os.path.dirname(video_filepath)
    destination_feature_filename = f"fingerspelling_game_recording.data"
    destination_feature_filepath = os.path.join(video_dirname, destination_feature_filename)
    dest = f"\"{destination_feature_filepath}\""

    print(f"source: {source}, dest: {dest}")

    print("Generating Mediapipe Data...")

    runMediapipe = 'python3 mediapipePythonWrapper.py \
                                --video_path="{}" \
                                --feature_filepath="{}"'.format(video_filepath, destination_feature_filepath)
    try:
        runOutput = subprocess.check_output(['bash','-c', runMediapipe])
    except:
        print("Done")

    # mediapipe_features(glob.glob(os.path.join(image_directory, '*.mkv'))[0], feature_filepath)
    print("Processed {}".format(video_filepath))


def videoToAlphaposeFeatures(video_filepath: str, is_left: bool = False):
    return None

def return_average_ll(file_path: str):
    total = 0
    num = 0
    with open(file_path) as verification_path:
        verification_path.readline()
        verification_path.readline()
        for line in verification_path:
            numbers = line.split(" ")
            if len(numbers) > 1:
                total += float(numbers[3])
                num += 1
                if len(numbers) > 4:
                    total += float(numbers[5])
                    num += 1
    
    if num > 0:
        return total/num
    else:
        return None

def generate_htk_from_video(video_filepath, features_config_filepath, processor = 'kinect'):
   
    # generate features: TODO
    features_filepath = None
    if 'alphapose' in processor:
        features_filepath = videoToAlphaposeFeatures(video_filepath)
    elif 'kinect' in processor:
        features_filepath = videoToKinectFeatures(video_filepath)
    elif 'mediapipe' in processor:
        features_filepath = videoToMediapipeFeatures(video_filepath)
    else:
        return -1

    if features_filepath is None: #or not os.path.exists(features_filepath):
        print("features_filepath error")
        return -1

    features_config = load_json(features_config_filepath)

    # generate ark
    features_dirname = os.path.dirname(features_filepath)
    features_filename = os.path.basename(features_filepath)
    features_extension = features_filename.split('.')[-1]
    features_df = None

    ark_filename = features_filename.replace(features_extension, 'ark')
    ark_filepath = os.path.join(features_dirname, ark_filename)
    title = ark_filename.replace('.ark', "")
    
    if 'alphapose' in processor:
        features_df = feature_extraction_alphapose(features_filepath, features_config['selected_features'], scale = 10, drop_na = True) #df = 
    elif 'kinect' in processor:
        features_df = feature_extraction_kinect(features_filepath, features_config['selected_features'], scale = 10, drop_na = True)
    elif 'mediapipe' in processor:
        features_df = select_features(features_filepath, features_config['selected_features'], center_on_nose = True, scale = 100, square = True, 
                                    drop_na = True, do_interpolate = True, use_optical_flow = False)
    else:
        return -1

    if features_df is not None:
        with open(ark_filepath, 'w') as out:
            out.write('{} [ '.format(title))
        features_df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')
        with open(ark_filepath, 'a') as out:
            out.write(']')
    
    # generate htk
    htk_dirname = features_dirname
    kaldi_command = (f'~/kaldi/src/featbin/copy-feats-to-htk '
                         f'--output-dir={htk_dirname} '
                         f'--output-ext=htk '
                         f'--sample-period=40000 '
                         f'ark:{ark_filepath}'
                         f'>/dev/null 2>&1')
    os.system(kaldi_command)

    htk_filename = features_filename.replace(features_extension, 'htk')
    htk_filepath = os.path.join(htk_dirname, htk_filename)

    # print(htk_filepath)

    return htk_filepath

def generate_verification_files(htk_filepath):
    htk_dirname = os.path.dirname(htk_filepath) # not needed?
    htk_filename = os.path.basename(htk_filepath) # curr_video

    correct_phrase = htk_filename.split('.')[1]

    label_filepath = "\"*/" + htk_filename.replace(".htk", ".lab\"")

    curr_verification_phrase = os.path.join(htk_dirname, 'curr_verification.data')
    curr_verification_label = os.path.join(htk_dirname, 'curr_verification_label.mlf')

    with open(curr_verification_phrase, "w") as verification_list:
        verification_list.write(htk_filepath+"\n")
    with open(curr_verification_label, "w") as verification_label:
        verification_label.write("#!MLF!#\n")
        verification_label.write(label_filepath+"\n")
        verification_label.write("sil0\n")
        for word in correct_phrase.split("_"):
            verification_label.write(word+"\n")
        verification_label.write("sil1\n")
        verification_label.write(".\n")

    return curr_verification_phrase, curr_verification_label

def verification_cmd(macros_filepath, verification_list, label_file, results_filepath, hresults_filepath, insertion_penalty: int = 0, beam_threshold: int = 2000):

    results_dirname = os.path.dirname(results_filepath)
    verification_log = os.path.join(results_dirname, 'verification_log.data')
    if os.path.exists(verification_log): os.remove(verification_log)
    if os.path.exists(results_filepath): os.remove(results_filepath)
    if os.path.exists(hresults_filepath): os.remove(hresults_filepath)

    # HVite_str = (f'HVite -a -o N -T 1 -H $macros -m -f -S '
    #                  f'{verification_list} -i $results -t {beam_threshold} '
    #                  f'-p {insertion_penalty} -I {label_file} -s 25 dict wordList '
    #                  f'>> {verification_log}')

    # HVite_cmd = Template(HVite_str)
    # os.system(HVite_cmd.substitute(macros=macros_filepath, results=results_filepath))
    # HVite_str = (f'HVite -A -H $macros -m -S verifyint/test.data -i '
    #                  f'{results_filepath} -p {insertion_penalty} -w wordNet.txt -s 25 dict wordList')

    # HVite_cmd = Template(HVite_str)

    # HResults_str = (f'HResults -A -h -e \\?\\?\\? sil0 -e \\?\\?\\? '
    #                     f'sil1 -p -t -I verifyint/all_labels.mlf wordList {alignmentdata}')
    # HResults_cmd = Template(HResults_str)
    print(macros_filepath)
    print(verification_list)
    print(results_filepath)
    print(insertion_penalty)
    print(label_file)
    HVite_str = f'HVite -A -H {macros_filepath} -m -S {verification_list} -i {results_filepath} -p {insertion_penalty} -w wordNet.txt -s 25 dict wordList'
    HResults_str = f'HResults -A -h -e \\?\\?\\? sil0 -e \\?\\?\\? sil1 -p -d 5 -t -I {label_file} wordList {results_filepath} >> {hresults_filepath}'
    os.system(HVite_str)
    os.system(HResults_str)

def verification_outcome(results_filepath: str, acceptance_threshold: int = None):
    
    if acceptance_threshold is None:
        # print('running acceptance')
        # print(open(results_filepath, 'r').readlines())
        # return True
        file = open(results_filepath, 'r').readlines()
        for i in file:
            # print(i)
            if 'Sum/Avg' in i:
                return i.split("|")[3].split(" ")[1] == '100.00'
        # for en, i in enumerate(file):
        #     print(i)
        #     print(en)
        return False
    else:
        curr_average = return_average_ll(results_filepath) # should be mlf file
        
        if curr_average:
            if (curr_average >= acceptance_threshold):
                return True

        return False

def recognition_cmd(macros_filepath, htk_filepath, results_filepath, game_output_filepath, insertion_penalty: int = 0):
    HVite_str = f'HVite -A -H {macros_filepath} -m htk_filepath -i {results_filepath} -p {insertion_penalty} -w wordNet.txt -s 25 dict wordList'
    file = open(results_filepath, 'r').readlines()
    phrase = ""
    add_to_phrase = False
    for i in file:
        if 'sil0' in i:
            add_to_phrase = True
        if 'sil1' in i:
            add_to_phrase = False
        if add_to_phrase:
            phrase += i.split(" ")[0]
            phrase += " "
    f = open(game_output_filepath, 'w')
    f.write(phrase)
    f.close()

def execute_protocol(video_filepath, macros_filepath, features_config_filepath, results_filepath, hresults_filepath, game_output_filepath, acceptance_threshold: int = 0.01):
    htk_filepath = generate_htk_from_video(video_filepath, features_config_filepath)
    verification_phrase, verification_label = generate_verification_files(htk_filepath)
    verification_cmd(macros_filepath, verification_phrase, verification_label, results_filepath, hresults_filepath)
    isAccepted = verification_outcome(hresults_filepath)
    # isAccepted = True
    if isAccepted:
        print(f"video {video_filepath} is accepted: YES")
        # f = open(game_output_filepath, 'w')
        # f = open(game_output_filepath, 'a')
        # f.write(video_filepath + '\n')
        # f.write('ACCEPTED')        
        # f.close()
        return True
    else:
        print(f"video {video_filepath} is accepted: NOT")
        # f = open(game_output_filepath, 'w')
        # f.write('REJECTED')
        # f.close()
        return False

def execute_protocol_classification(video_filepath, macros_filepath, features_config_filepath, results_filepath, hresults_filepath, game_output_filepath, acceptance_threshold: int = 0.01):
    htk_filepath = generate_htk_from_video(video_filepath, features_config_filepath)
    recognition_cmd(macros_filepath, htk_filepath, results_filepath)

def guru_filter_videos(game_output_filepath):
    f = open(game_output_filepath + '.text', 'w')

    info = set()
    with open(game_output_filepath, 'r') as fp:
        for line in fp:
            line = line.strip()
            if line.split('.')[0] in info: continue
            info.add(line.split('.')[0])
            f.write(line + '\n')
    f.close()


def guru_collect_videos(video_directory, macros_filepath, features_config_filepath, results_filepath, hresults_filepath, game_output_filepath, acceptance_threshold: int = 0.01):
    os.remove(game_output_filepath)
    video_filepaths = sorted(glob.glob(os.path.join(video_directory, '*', '*.mkv')))
    print(video_filepaths)
    # print(acceptance_threshold)
    for video_filepath in video_filepaths:
        print(f'Running {video_filepath}')
        execute_protocol(video_filepath, macros_filepath, features_config_filepath, results_filepath, hresults_filepath, game_output_filepath, acceptance_threshold)
        # input()

def verify():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_filepath', type=str, default='/home/aslr/Documents/11-01-20_Ishan_4KDepth.alligator_above_orange_wagon.0000000000.mkv')    
    parser.add_argument('--macros_filepath', type=str, default='/home/aslr/SBHMM-HTK/SequentialClassification/main/src/utils/verifyint/newMacros_ui')
    parser.add_argument('--macros_filepath_pua', type=str, default='/home/aslr/SBHMM-HTK/SequentialClassification/main/src/utils/verifyint/newMacros_pua')
    parser.add_argument('--features_config_filepath', type=str, default='/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/configs/features.json')
    parser.add_argument('--results_filepath', type=str, default='/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/testing_verification/verification_results.json')
    parser.add_argument('--hresults_filepath', type=str, default='/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/Kinect/testing_verification/verifcation_hresults.json')
    parser.add_argument('--game_output_filepath', type=str, default='/home/aslr/SBHMM-HTK/SequentialClassification/main/src/utils/final_game_result.txt')


    parser.add_argument('--video_directory', type=str, default='/home/aslr/Documents/recordings')
    args = parser.parse_args()

    # f = open(args.game_output_filepath, 'w')
    # f.write('UNRECOGNIZABLE')
    # f.close()

    # if args.video_directory:
        # guru_filter_videos(args.game_output_filepath)
       # guru_collect_videos(args.video_directory, args.macros_filepath, args.features_config_filepath, args.results_filepath, args.hresults_filepath, args.game_output_filepath, acceptance_threshold=None)
    # else:
    ui = execute_protocol(args.video_filepath, args.macros_filepath, args.features_config_filepath, args.results_filepath, args.hresults_filepath, args.game_output_filepath, acceptance_threshold=None)
    pua = execute_protocol(args.video_filepath, args.macros_filepath_pua, args.features_config_filepath, args.results_filepath, args.hresults_filepath, args.game_output_filepath, acceptance_threshold=None)
    if (ui and pua) :
        f = open(args.game_output_filepath, 'w')
        f.write('ACCEPTED')        
        f.close()
    else:
        f = open(args.game_output_filepath, 'w')
        f.write('REJECTED')        
        f.close()

print(sys.path)
print("Running")
verify()
