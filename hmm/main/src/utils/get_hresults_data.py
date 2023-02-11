import os

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def compare_true_pred(labelTRUEwords, labelTRUE, labelPREDwords, labelPRED):

    # print(labelTRUE)
    # print(labelPRED)

    trueSpaces = findOccurrences(labelTRUE, ' ')
    predSpaces = findOccurrences(labelPRED, ' ')
    spaces = sorted( list( set(trueSpaces) & set(predSpaces) ) + [-1])
    # print(trueSpaces, predSpaces, spaces)

    true_data = []
    true_idx = 0

    pred_data = []
    pred_idx = 0

    # print(spaces)

    for space in spaces:
        # print(f"start of next word is {space + 1}")
        first_letter = space + 1

        # print(labelTRUE[first_letter], labelPRED[first_letter])

        if labelTRUE[first_letter] != ' ': # true has a word
            true_data.append(labelTRUEwords[true_idx])
            true_idx += 1
        else:
            true_data.append(' ')

        if labelPRED[first_letter] != ' ':
            pred_data.append(labelPREDwords[pred_idx])
            pred_idx += 1
        else:
            pred_data.append(' ')

    return true_data, pred_data


def get_hresults_data(results_file):
    results_dict = {} # { '<FILENAME>': { 'true': [word1, word2, ...], 'pred': [word1, word2, ...] } }
    
    labels = {}

    start_index = -1

    with open(results_file, 'r') as lf:
        
        objectFILENAME = ""

        labelTRUEwords = []
        labelTRUE = ""

        labelPREDwords = []
        labelPRED = ""

        for cnt, line in enumerate(lf):
            l = line.rstrip()
            if cnt == 0: continue
            if "--------------" in l: break

            if cnt % 3 == 1:
                file = l.split()[2]
                file = os.path.basename(file)
                extension = '.' + file.split('.')[-1]                
                objectFILENAME = file.replace(extension, '')

            elif cnt % 3 == 2:
                words = l.split()[1:]
                labelTRUEwords = words
                labelTRUE = l.split(": ")[1]
            else:
                words = l.split()[1:]
                labelPREDwords = words
                labelPRED = l.split(": ")[1]

                true_data, pred_data = compare_true_pred(labelTRUEwords, labelTRUE, labelPREDwords, labelPRED)

                results_dict[objectFILENAME] = {}
                results_dict[objectFILENAME]['true'] = true_data
                results_dict[objectFILENAME]['pred'] = pred_data

    return results_dict