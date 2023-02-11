import os
from get_confusion_matrix import get_confusion_matrix
from get_hresults_data import get_hresults_data


output_file = open("hresults_improved.txt", "w")
orig_avg_SE = 0
orig_avg_WE = 0
new_avg_SE = 0
new_avg_WE = 0
total_num_in_above_that_turned_wrong = 0
total_num_in_above_that_were_fixed = 0
total_in_above_already_correct = 0
total_in_above_wrong = 0

total_wrong = 0

def compareTwoModels(file1, file2):

    global orig_avg_SE, orig_avg_WE, new_avg_SE, new_avg_WE, total_num_in_above_that_turned_wrong, total_num_in_above_that_were_fixed, total_wrong, total_in_above_already_correct, total_in_above_wrong

    overall_model_data = get_hresults_data(file1) # wrong files
    in_above_model_data = get_hresults_data(file2) # wrong files

    overall_model_cm = get_confusion_matrix(file1) 
    in_above_model_cm = get_confusion_matrix(file2)

    orig_num_sentences_wrong = len(overall_model_data.items())

    orig_num_words_wrong = overall_model_cm['general']['num_words_incorrect']
    total_wrong += orig_num_words_wrong

    best_model = {}
    fixed_sentences = 0

    num_in_above_already_correct_from_incorrect_sentences = 0
    num_in_above_incorrect_from_incorrect_sentences = 0
    num_in_above_that_turned_wrong = 0
    num_in_above_that_were_fixed = 0

    for datafile, data in overall_model_data.items():
        true_data = data['true']
        pred_data = data['pred']

        for i, word in enumerate(true_data):
            canBeFixed = False
            predBefore = ""
            if 'above' in word or 'in' in word: # note, above/in may not be the problem in the sentence, but this will still replace it
                predBefore = pred_data[i]

                if true_data[i] == pred_data[i]: 
                    num_in_above_already_correct_from_incorrect_sentences += 1
                    total_in_above_already_correct += 1
                else:
                    num_in_above_incorrect_from_incorrect_sentences += 1
                    total_in_above_wrong += 1
                    canBeFixed = True

                
                if datafile in in_above_model_data: #exists in specific tuned model, then do some placing stuff
                    pred_data[i] = in_above_model_data[datafile]['pred'][i]
                else: # does not exist, which means we can just replace it
                    pred_data[i] = true_data[i]

                if not canBeFixed and true_data[i] != pred_data[i]: #we replaced a word that was already correct
                    num_in_above_that_turned_wrong += 1

                if canBeFixed and true_data[i] == pred_data[i]:
                    num_in_above_that_were_fixed += 1
                    # if datafile in in_above_model_data:
                    #     print(datafile, data['true'], predBefore, data['pred'], in_above_model_data[datafile]['true'], in_above_model_data[datafile]['pred'])
                    # else:
                    #     print(datafile, data['true'], predBefore, data['pred'], "does not exist", "does not exist")
                    # print()

        if true_data == pred_data:
            fixed_sentences += 1
        else:
            best_model[datafile] = {}
            best_model[datafile]['true'] = true_data
            best_model[datafile]['pred'] = pred_data

    # print(best_model)

    new_num_words_wrong = 0
    for datafile, data in best_model.items():
    	true_data = data['true']
    	pred_data = data['pred']

    	true_size = len(true_data)
    	pred_size = len(pred_data)

    	# print(new_num_words_wrong)

    	for i in range(max(true_size, pred_size)):
    		try:
    			if true_data[i] != pred_data[i]:
    				new_num_words_wrong += 1
    		except:
    			new_num_words_wrong += 1

    	# print(new_num_words_wrong, true_data, pred_data)


    total_sentences = overall_model_cm['general']['num_sentences']
    total_words = overall_model_cm['general']['num_words']
    session = overall_model_cm['user']

    print(f'User = {session}\n', file=output_file)

    # ORIGINAL MODEL
    SE = round(100 * orig_num_sentences_wrong/total_sentences, 2)
    orig_avg_SE += SE
    print(f'Original SE = {orig_num_sentences_wrong} / {total_sentences} = {SE}%', file=output_file)

    new_num_sentences_wrong = orig_num_sentences_wrong - fixed_sentences
    SE = round(100 * new_num_sentences_wrong/total_sentences, 2)
    new_avg_SE += SE
    print(f'Fixed {fixed_sentences} sentences. SE = {new_num_sentences_wrong} / {total_sentences} = {SE}%\n', file=output_file)

    WE = round(100 * orig_num_words_wrong/total_words, 2)
    orig_avg_WE += WE
    print(f'Original WE = {orig_num_words_wrong} / {total_words} = {WE}%', file=output_file)

    fixed_words = orig_num_words_wrong - new_num_words_wrong
    WE = round(100 * new_num_words_wrong/total_words, 2)
    new_avg_WE += WE
    print(f'Fixed {fixed_words} words. WE = {new_num_words_wrong} / {total_words} = {WE}%\n', file=output_file)

    print("num_in_above_that_turned_wrong: ", num_in_above_that_turned_wrong, file=output_file)
    total_num_in_above_that_turned_wrong += num_in_above_that_turned_wrong

    print("num_in_above_that_were_fixed: ", num_in_above_that_were_fixed, file=output_file)
    total_num_in_above_that_were_fixed += num_in_above_that_were_fixed

base = '/home/aslr/SBHMM-HTK/SequentialClassification/main/projects/'

num_files = 12

for i in range(num_files):
	file1 = os.path.join(base, f'Kinect/hresults/{i}/res_hmm220.txt')
	file2 = os.path.join(base, f'Kinect_inabove_features/hresults/{i}/res_hmm220.txt')
	compareTwoModels(file1, file2)
	print('\n********************************\n', file=output_file)

orig_avg_SE /= num_files
orig_avg_WE /= num_files
new_avg_SE /= num_files
new_avg_WE /= num_files

orig_avg_SE = round(orig_avg_SE, 2)
orig_avg_WE = round(orig_avg_WE, 2)
new_avg_SE = round(new_avg_SE, 2)
new_avg_WE = round(new_avg_WE, 2)

print(f'**** AVG ****', file=output_file)
print(f'orig_avg_SE: {orig_avg_SE}', file=output_file)
print(f'new_avg_SE : {new_avg_SE}', file=output_file)
print(f'orig_avg_WE: {orig_avg_WE}', file=output_file)
print(f'new_avg_WE : {new_avg_WE}', file=output_file)
print(f'\ntotal_num_in_above_that_turned_wrong: {total_num_in_above_that_turned_wrong}', file=output_file)
print(f'total_num_in_above_that_were_fixed: {total_num_in_above_that_were_fixed}', file=output_file)
print(f'\ntotal_in_above_already_correct from incorrect sentences: {total_in_above_already_correct}', file=output_file)
print(f'\ntotal_in_above_wrong from incorrect sentences: {total_in_above_wrong}', file=output_file)
print(f'total_wrong_words from incorrect sentences: {total_wrong}', file=output_file)

output_file.close()