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
import random

def get_video_dict(video_array):
    video_dict = {}
    for curr_video_path in video_array:
        correct_phrase = curr_video_path.split("/")[-1].split(".")[1]
        if correct_phrase in video_dict:
            video_dict[correct_phrase].append(curr_video_path)
        else:
            video_dict[correct_phrase] = [curr_video_path]
    
    return video_dict


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

def return_word_ll(file_path: str):
    ll_list = []
    with open(file_path) as verification_path:
        verification_path.readline()
        verification_path.readline()
        for line in verification_path:
            numbers = line.split(" ")
            if len(numbers) > 4:
                ll_list.append(float(numbers[5]))
    
    if len(ll_list) > 0:
        return ll_list
    else:
        return None

def verification_cmd(model_iter: int, insertion_penalty: int, verification_list: str, label_file: str, 
                    beam_threshold: int = 2000, fold: str = ""):

    verification_log = f'logs/{fold}verification_log.data'

    HVite_str = (f'HVite -a -o N -T 1 -H $macros -m -f -S '
                     f'{verification_list} -i $results -t {beam_threshold} '
                     f'-p {insertion_penalty} -I {label_file} -s 25 dict wordList '
                     f'>> {verification_log}')
    HVite_cmd = Template(HVite_str)
    macros_filepath = f'models/{fold}hmm{model_iter}/newMacros'
    results_filepath = f'results/{fold}res_hmm{model_iter}.mlf'

    os.system(HVite_cmd.substitute(macros=macros_filepath, results=results_filepath))

def get_one_off_phrases(curr_phrase: str, unique_phrases: set):
    # return_two_off = {'alligator_above_bed', 'lion_below_orange_chair', 'orange_snake_below_blue_flowers', 'lion_above_orange_bed', 'orange_monkey_below_grey_flowers', 
    #                     'monkey_in_orange_flowers', 'white_snake_in_blue_flowers', 'orange_monkey_in_grey_box', 'snake_in_flowers', 'blue_monkey_above_grey_box', 
    #                     'orange_monkey_in_grey_box'}
    # return_three_off = {'black_snake_below_blue_chair'}
    one_off_phrases = []
    two_off_phrases = []
    three_off_phrases = []
    curr_phrase_arr = curr_phrase.split("_")
    for phrase in unique_phrases:
        phrase_arr = phrase.split("_")
        dp_table = np.zeros((len(phrase_arr), len(curr_phrase_arr)))
        for idx_1, word_1 in enumerate(phrase_arr):
            for idx_2, word_2 in enumerate(curr_phrase_arr):
                if idx_1 == 0 and idx_2 == 0:
                    dp_table[idx_1, idx_2] = 1 - (word_1 == word_2)
                elif idx_1 == 0:
                    dp_table[idx_1, idx_2] = dp_table[idx_1, idx_2 - 1] + (1 - (word_1 == word_2))
                elif idx_2 == 0:
                    dp_table[idx_1, idx_2] = dp_table[idx_1-1, idx_2] + (1 - (word_1 == word_2))
                else:
                    dp_table[idx_1, idx_2] = min((1 - (word_1 == word_2)) + dp_table[idx_1-1, idx_2-1], 
                                                dp_table[idx_1-1, idx_2] + 1,
                                                dp_table[idx_1, idx_2-1] + 1)
        if dp_table[-1,-1] <= 1:
            one_off_phrases.append(phrase)
        if dp_table[-1, -1] <= 2:
            two_off_phrases.append(phrase)
        if dp_table[-1, -1] <= 3:
            three_off_phrases.append(phrase)
    
    if len(one_off_phrases) != 1:
        return one_off_phrases
    elif len(two_off_phrases) != 1:
        return two_off_phrases
    else:
        return three_off_phrases
    # if curr_phrase in return_two_off:
    #     return two_off_phrases
    # elif curr_phrase in return_three_off:
    #     return three_off_phrases
    # else:
    #     return one_off_phrases

def return_ll_per_correct_and_one_off_sign(model_iter:int, insertion_penalty: int, average: bool,
                            beam_threshold: int = 2000, fold: str = "") -> None:
    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    test_phrases = f'lists/{fold}test.data'
    train_phrases = f'lists/{fold}train.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.
    unique_phrases = set()
    with open(train_phrases) as file:
        for line in file:
            curr_phrase = line.split("/")[-1].split(".")[1]
            unique_phrases.add(curr_phrase)
    
    ll_per_sign = {'incorrect': {}, 'correct': {}}
    total_incorrect_signs = set()

    #perform verification for each video with each possible phrase and get score.
    with open(test_phrases) as file:
        test_videos_array = file.readlines()
        test_video_dict = get_video_dict(test_videos_array)
        for curr_video_path in tqdm.tqdm(test_videos_array):
            curr_video = curr_video_path.split("/")[-1]
            curr_phrase = curr_video.split(".")[1]
            closest_phrases = get_one_off_phrases(curr_phrase, unique_phrases)
            for curr_closest_phrase in closest_phrases:
                if curr_closest_phrase == curr_phrase:
                    curr_closest_vid = curr_video_path
                else:
                    curr_closest_vid = random.choice(test_video_dict[curr_closest_phrase]) #We pick a random video to evaluate. All is too expensive.
                label_file_path = "\"*/" + curr_closest_vid.split("/")[-1].replace(".htk", ".lab\"")
                with open(curr_verification_phrase, "w") as verification_list:
                    verification_list.write(curr_closest_vid+"\n")
                with open(curr_verification_label, "w") as verification_label:
                    verification_label.write("#!MLF!#\n")
                    verification_label.write(label_file_path)
                    verification_label.write("sil0\n")
                    for word in curr_phrase.split("_"):
                        verification_label.write(word+"\n")
                    verification_label.write("sil1\n")
                    verification_label.write(".\n")
                    
                verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                                curr_verification_label, beam_threshold, fold)
                curr_ll = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf') if average \
                                else return_word_ll(f'results/{fold}res_hmm{model_iter}.mlf')
                
                if curr_ll:
                    if curr_closest_phrase == curr_phrase:
                        if curr_phrase in ll_per_sign['correct']:
                            ll_per_sign['correct'][curr_phrase].append(curr_ll)
                        else:
                            ll_per_sign['correct'][curr_phrase] = [curr_ll]
                    else:
                        total_incorrect_signs.add(curr_phrase)
                        if curr_phrase in ll_per_sign['incorrect']:
                            ll_per_sign['incorrect'][curr_phrase].append(curr_ll)
                        else:
                            ll_per_sign['incorrect'][curr_phrase] = [curr_ll]
    return ll_per_sign


def return_average_ll_per_sign(model_iter:int, insertion_penalty: int,
                            beam_threshold: int = 2000, fold: str = "") -> None:
    
    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    test_phrases = f'lists/{fold}test.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.

    average_ll_per_sign = {}

    #perform verification for each video and get score.
    with open(test_phrases) as file:
        for curr_video_path in tqdm.tqdm(file):
            curr_video = curr_video_path.split("/")[-1]
            correct_phrase = curr_video.split(".")[1]
            label_file_path = "\"*/" + curr_video.replace(".htk", ".lab\"")

            with open(curr_verification_phrase, "w") as verification_list:
                verification_list.write(curr_video_path+"\n")
            with open(curr_verification_label, "w") as verification_label:
                verification_label.write("#!MLF!#\n")
                verification_label.write(label_file_path)
                verification_label.write("sil0\n")
                for word in correct_phrase.split("_"):
                    verification_label.write(word+"\n")
                verification_label.write("sil1\n")
                verification_label.write(".\n")
                
            verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                            curr_verification_label, beam_threshold, fold)
            curr_average = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf')
            
            if curr_average:
                if correct_phrase in average_ll_per_sign:
                    average_ll_per_sign[correct_phrase].append(curr_average)
                else:
                    average_ll_per_sign[correct_phrase] = [curr_average]
    
    return average_ll_per_sign


def verify_zahoor(model_iter:int, insertion_penalty: int, average_ll_per_sign: dict, 
                beam_threshold: int = 2000, fold: str = "") -> None:

    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    train_phrases = f'lists/{fold}train.data'
    test_phrases = f'lists/{fold}test.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.
    unique_phrases = set()

    with open(train_phrases) as file:
        for line in file:
            curr_phrase = line.split("/")[-1].split(".")[1]
            unique_phrases.add(curr_phrase)
    
    positive = 0
    false_positive = 0
    false_negative = 0
    negative = 0
    test_log_likelihoods = {"incorrect":{}, "correct":{}}

    #perform verification for each video with each possible phrase and get score.
    with open(test_phrases) as file:
        for curr_video_path in tqdm.tqdm(file):
            curr_video = curr_video_path.split("/")[-1]
            correct_phrase = curr_video.split(".")[1]
            label_file_path = "\"*/" + curr_video.replace(".htk", ".lab\"")
            one_off_phrases = get_one_off_phrases(correct_phrase, unique_phrases)
            for curr_phrase in one_off_phrases:
                with open(curr_verification_phrase, "w") as verification_list:
                    verification_list.write(curr_video_path+"\n")
                with open(curr_verification_label, "w") as verification_label:
                    verification_label.write("#!MLF!#\n")
                    verification_label.write(label_file_path)
                    verification_label.write("sil0\n")
                    for word in curr_phrase.split("_"):
                        verification_label.write(word+"\n")
                    verification_label.write("sil1\n")
                    verification_label.write(".\n")
                
                verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                                curr_verification_label, beam_threshold, fold)
                curr_average = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf')
                threshold = average_ll_per_sign[curr_phrase][0] - average_ll_per_sign[curr_phrase][1]
                if curr_average:
                    if correct_phrase == curr_phrase:

                        if correct_phrase in test_log_likelihoods["correct"]:
                            test_log_likelihoods["correct"][correct_phrase].append(curr_average)
                        else:
                            test_log_likelihoods["correct"][correct_phrase] = [curr_average]

                        if curr_average >= threshold:
                            positive += 1
                        else:
                            false_negative += 1
                        
                    elif correct_phrase != curr_phrase:

                        if correct_phrase in test_log_likelihoods["incorrect"]:
                            test_log_likelihoods["incorrect"][correct_phrase].append(curr_average)
                        else:
                            test_log_likelihoods["incorrect"][correct_phrase] = [curr_average]

                        if curr_average < threshold:
                            negative += 1
                        else:
                            false_positive += 1

    return positive, negative, false_positive, false_negative, test_log_likelihoods

def verify_classifier(model_iter:int, insertion_penalty: int, classifier_per_sign: dict, average: bool,
                beam_threshold: int = 2000, fold: str = "") -> None:

    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    train_phrases = f'lists/{fold}train.data'
    test_phrases = f'lists/{fold}test.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.
    unique_phrases = set()

    with open(train_phrases) as file:
        for line in file:
            curr_phrase = line.split("/")[-1].split(".")[1]
            unique_phrases.add(curr_phrase)
    
    positive = 0
    false_positive = 0
    false_negative = 0
    negative = 0
    test_log_likelihoods = {"incorrect":{}, "correct":{}}

    #perform verification for each video with each possible phrase and get score.
    with open(test_phrases) as file:
        for curr_video_path in tqdm.tqdm(file):
            curr_video = curr_video_path.split("/")[-1]
            correct_phrase = curr_video.split(".")[1]
            label_file_path = "\"*/" + curr_video.replace(".htk", ".lab\"")
            one_off_phrases = get_one_off_phrases(correct_phrase, unique_phrases)
            for curr_phrase in one_off_phrases:
                with open(curr_verification_phrase, "w") as verification_list:
                    verification_list.write(curr_video_path+"\n")
                with open(curr_verification_label, "w") as verification_label:
                    verification_label.write("#!MLF!#\n")
                    verification_label.write(label_file_path)
                    verification_label.write("sil0\n")
                    for word in curr_phrase.split("_"):
                        verification_label.write(word+"\n")
                    verification_label.write("sil1\n")
                    verification_label.write(".\n")
                
                verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                                curr_verification_label, beam_threshold, fold)
                curr_average = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf') if average \
                                else return_word_ll(f'results/{fold}res_hmm{model_iter}.mlf')
                if curr_average:
                    predicted_label = classifier_per_sign[curr_phrase].predict(np.array(curr_average).reshape((1,-1)))[0]
                    if correct_phrase == curr_phrase:
                        if curr_phrase in test_log_likelihoods["correct"]:
                            test_log_likelihoods["correct"][curr_phrase].append(curr_average)
                        else:
                            test_log_likelihoods["correct"][curr_phrase] = [curr_average]

                        if predicted_label == 1:
                            positive += 1
                        else:
                            false_negative += 1
                        
                    else:
                        if curr_phrase in test_log_likelihoods["incorrect"]:
                            test_log_likelihoods["incorrect"][curr_phrase].append(curr_average)
                        else:
                            test_log_likelihoods["incorrect"][curr_phrase] = [curr_average]

                        if predicted_label == 0:
                            negative += 1
                        else:
                            false_positive += 1

    return positive, negative, false_positive, false_negative, test_log_likelihoods


'''
    While evaluating network accuracy:
        For each video, calculate how many times it correctly verifies it. Also calculate how many times
        it incorrectly verifies correctly rejects other phrases. Report #correct_labels/total_labels.

        For calculating log likelihood probability, make a list of the phrase you want to check and 
        an mlf file with the label corresponding to that phase. Change this label to perform alignment
        with other phrases.

    For now, use a threshold on the average log likelihood probability for verifying or rejecting.
'''
def verify_simple(model_iter:int, insertion_penalty: int, acceptance_threshold: int, 
                beam_threshold: int = 2000, fold: str = "") -> None:

    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')

    if os.path.exists(f'hresults/{fold}'):
        shutil.rmtree(f'hresults/{fold}')
    os.makedirs(f'hresults/{fold}')
    
    if model_iter == -1:
        model_iter = len(glob.glob(f'models/{fold}*hmm*')) - 1

    train_phrases = f'lists/{fold}train.data'
    test_phrases = f'lists/{fold}test.data'
    curr_verification_phrase = f'lists/{fold}curr_verification.data'
    curr_verification_label = f'lists/{fold}curr_verification_label.mlf' #I may regret putting label in list later.
    unique_phrases = set()

    with open(train_phrases) as file:
        for line in file:
            curr_phrase = line.split("/")[-1].split(".")[1]
            unique_phrases.add(curr_phrase)
    
    positive = 0
    false_positive = 0
    false_negative = 0
    negative = 0

    #perform verification for each video with each possible phrase and get score.
    with open(test_phrases) as file:
        for curr_video_path in tqdm.tqdm(file):
            curr_video = curr_video_path.split("/")[-1]
            correct_phrase = curr_video.split(".")[1]
            label_file_path = "\"*/" + curr_video.replace(".htk", ".lab\"")

            for curr_phrase in unique_phrases:
                with open(curr_verification_phrase, "w") as verification_list:
                    verification_list.write(curr_video_path+"\n")
                with open(curr_verification_label, "w") as verification_label:
                    verification_label.write("#!MLF!#\n")
                    verification_label.write(label_file_path)
                    verification_label.write("sil0\n")
                    for word in curr_phrase.split("_"):
                        verification_label.write(word+"\n")
                    verification_label.write("sil1\n")
                    verification_label.write(".\n")
                
                verification_cmd(model_iter, insertion_penalty, curr_verification_phrase,
                                curr_verification_label, beam_threshold, fold)
                curr_average = return_average_ll(f'results/{fold}res_hmm{model_iter}.mlf')

                if curr_average:
                    if (correct_phrase == curr_phrase and curr_average >= acceptance_threshold):
                        positive += 1
                    elif (correct_phrase != curr_phrase and curr_average < acceptance_threshold):
                        negative += 1
                    elif (correct_phrase != curr_phrase and curr_average >= acceptance_threshold) :
                        false_positive += 1
                    else :
                        false_negative += 1

    return positive, negative, false_positive, false_negative 

