from get_confusion_matrix import get_confusion_matrix
import os

def sequence_results(hresults_dir, phrases):
    data_dict = {}

    # Iterate over users
    for fp in os.listdir(hresults_dir):
        user_results = get_confusion_matrix(os.path.join(hresults_dir, fp))
        for phrase in phrases:
            words = phrase.split('_')
            phrase_len = len(words)
            accuracy = 0

            # Get average of word level accuracy
            # Can look at insertions/deletions specifically
            for word in words:
                word_abrev1 = [k for k in user_results['matrix'].keys() if word.startswith(k)][0]
                word_abrev2 = [k for k in user_results['matrix'][word_abrev1].keys() if word.startswith(k)][0]
                accuracy += user_results['matrix'][word_abrev1][word_abrev2] / sum(user_results['matrix'][word_abrev1].values())
            if user_results['user'] in data_dict:
                data_dict[user_results['user']][phrase] = accuracy / phrase_len
            else:
                data_dict[user_results['user']] = {phrase: accuracy / phrase_len}
    print(data_dict)

    # Can implement some sorting function over here
    return data_dict


hd = '/mnt/884b8515-1b2b-45fa-94b2-ec73e4a2e557/SBHMM-HTK/SequentialClassification/main/projects/Mediapipe/hresults/9/'
p = ['in', 'above']
sequence_results(hd, p)