"""Creates all text files needed to train/test HMMs with HTK, including
wordList, dict, grammar, and all_labels.mlf.

Methods
-------
generate_text_files
_get_unique_words
_write_grammar_line
_generate_word_list
_generate_word_dict
_generate_grammar
_generate_mlf_file
"""
import os
import glob
from io import TextIOWrapper

import string

from numpy.lib.arraysetops import unique

ENTER = '!ENTER'
EXIT = '!EXIT'

def generate_text_files(
    features_dir: str = 'configs/features_sign.json',
    isFingerspelling: bool = False,
    isSingleWord: bool = False,
    isBigram: bool = False,
    unique_words: set = None,
    data_path: str = 'data',
) -> None:
    """Creates all text files needed to train/test HMMs with HTK,
    including wordList, dict, grammar, and all_labels.mlf.
    
    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """
    if not unique_words:
        unique_words = _get_unique_words(features_dir)

    _generate_word_list(
        unique_words,
        isFingerspelling,
        isSingleWord,
        os.path.join(data_path, 'wordList'),
        data_path,
    )

    _generate_word_dict(
        unique_words,
        isFingerspelling,
        os.path.join(data_path, 'dict'),
    )

    _generate_mlf_file(
        isFingerspelling,
        isSingleWord,
        os.path.join(data_path, 'all_labels.mlf'),
        data_path,
    )

    _generate_grammar(
        unique_words,
        features_dir,
        data_path,
        os.path.join(data_path, 'grammar.txt'),
        isFingerspelling,
        isSingleWord,
        isBigram,
    )

def _get_basename(filen):
    return os.path.splitext(os.path.basename(filen))[0]

def _get_unique_words(features_dir: str) -> set:
    """Gets all unique words from a data set.

    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.

    Returns
    -------
    unique_words : set
        Set of all words found in the training data.
    """

    unique_words = set()
    features_filepaths = glob.glob(os.path.join(features_dir, '**/*.data'), recursive = True)
    features_filepaths.extend(glob.glob(os.path.join(features_dir, '**/*.json'), recursive = True))
    split_index = 1

    for features_filepath in features_filepaths:
        filename = features_filepath.split('/')[-1]
        phrase = filename.rsplit('-', 3)[split_index].split('_')
        phrase = [word.lower() for word in phrase]
        # if phrase[-1] == "47e2":
        #     print("Mystery File: ", filename)
        unique_words = unique_words.union(phrase)

    unique_words = sorted(unique_words)
    print("Unique Words: ", unique_words)
    return unique_words

def _write_word_list(word_list: list, wordList_file):
    with open(wordList_file, 'w') as f:
        
        for word in word_list[:-1]:
            f.write('{}\n'.format(word))
        f.write('{}'.format(word_list[-1]))
    

def _generate_word_list(
    unique_words: list,
    is_fingerspelling: bool,
    isSingleWord: bool,
    wordList_file: str,
    data_path: str
) -> None:
    """Generates wordList file containing all unique words and silences.

    Parameters
    ----------
    unique_words : set
        Set of all words found in the training data.
    """
    if is_fingerspelling:
        if isSingleWord:
            word_list = list(string.ascii_lowercase)
        else:
            wl_basename = _get_basename(wordList_file)
            
            word_list = list(string.ascii_lowercase)
            wordList_file = os.path.join(data_path, wl_basename + '_letter')
            
            wordList_file_2 = os.path.join(data_path, wl_basename + '_word')
            word_list_2 = list(unique_words)
            word_list_2 += [ENTER, EXIT]
            
            _write_word_list(word_list_2, wordList_file_2)
    else:
        word_list = list(unique_words)
    
    word_list += [ENTER, EXIT]
    _write_word_list(word_list, wordList_file)


def _generate_word_dict(unique_words: list, is_fingerspelling: bool, dict_file: str) -> None:
    """Generates dict file containing key-value pairs of words. In our
    case, the key and value are both the single, unique word.

    Parameters
    ----------
    unique_words : set
        Set of all words found in the training data.
    """
    
    word_list = list(unique_words)
    word_list += [ENTER, EXIT]

    with open(dict_file, 'w') as f:

        f.write(f'SENT-START [] {ENTER}\n')
        f.write(f'SENT-END [] {EXIT}\n')
        
        for word in word_list[:-2]:
            if is_fingerspelling:
                f.write('{} {}\n'.format(word, ' '.join(word)))
            else:
                f.write('{} {}\n'.format(word, word))
        f.write('{} {}\n'.format(word_list[-2], word_list[-2]))
        f.write('{} {}\n'.format(word_list[-1], word_list[-1]))
    
    # if is_fingerspelling:
    #     with open('dict_tri', 'w') as f:
    #         f.write(f'SENT-START [] {ENTER}\n')
    #         f.write(f'SENT-END [] {EXIT}\n')
    #         
    #         for word in word_list[:-2]:
    #             letters = list(word)
    #             letters.insert(0, ENTER)
    #             letters.append(EXIT)
    #             f.write('{} '.format(letters[0]))
    #             for i in range(len(letters)-2):
    #                 f.write('{}+{}-{} '.format(letters[i], letters[i+1], letters[i+2]))
    #             f.write('{}\n'.format(letters[-1]))
    #         f.write('{} {}\n'.format(word_list[-2], word_list[-2]))
    #         f.write('{} {}\n'.format(word_list[-1], word_list[-1]))

def _write_grammar_line(
        f: TextIOWrapper, part_of_speech: str, words: list, n='') -> None:
    """Writes a single line to grammar.txt.

    Parameters
    ----------
    f : TextIOWrapper
        Buffered text stream to write to grammar.txt file.

    part_of_speech : str
        Part of speech being written on line.

    words : list
        List of words to be written to line.

    n : str, optional, by default ''
        If a part of speech can be included more than once in the
        grammar, each one should have a distinct count.
    """

    f.write('${}{} = '.format(part_of_speech, n))
    for word in words[:-1]:
        f.write('{} | '.format(word))
    f.write('{};\n'.format(words[-1]))


def _generate_grammar(
    unique_words: set,
    features_dir: str,
    data_path: str,
    grammar_file: str,
    isFingerspelling :bool,
    isSingleWord: bool,
    isBigram: bool,
) -> None:
    """Creates rule-based grammar depending on the length of the longest
    phrase of the dataset.

    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """
    

    if (isFingerspelling or isSingleWord) and not(isBigram):
        with open(grammar_file, 'w') as f:
            _write_grammar_line(f, 'word', unique_words)
            f.write('\n')
            f.write('(SENT-START $word SENT-END)')
            f.write('\n')
        f.close()
        print("DO")
        return
    
    if isBigram:
        if isFingerspelling and not(isSingleWord):
            all_labels_file = os.path.join(data_path, 'all_labels_word.mlf')
            wordList_file = os.path.join(data_path, 'wordList_word')
        else:
            all_labels_file = os.path.join(data_path, 'all_labels.mlf')
            wordList_file = os.path.join(data_path, 'wordList')
        os.system(f'HLStats -b {grammar_file} -o {wordList_file} {all_labels_file}')
        return

    subjects = set()
    prepositions = set()
    objects = set()
    adjectives = set()
    max_phrase_len = 0
    features_filepaths = glob.glob(os.path.join(features_dir, '**/*.data'), recursive = True)
    features_filepaths.extend(glob.glob(os.path.join(features_dir, '**/*.json'), recursive = True))
    split_index = 1

    for features_filepath in features_filepaths:
        filename = features_filepath.split('/')[-1]
        phrase = filename.rsplit('-', 3)[split_index].split('_')
        phrase = [word.lower() for word in phrase]
        phrase_len = len(phrase)
        max_phrase_len = max(phrase_len, max_phrase_len)

        if phrase_len == 3:

            subject, preposition, object_ = phrase
            subjects.add(subject)
            prepositions.add(preposition)
            objects.add(object_)

        elif phrase_len == 4:

            subject, preposition, adjective, object_ = phrase
            subjects.add(subject)
            prepositions.add(preposition)
            adjectives.add(adjective)
            objects.add(object_)

        elif phrase_len == 5:

            adjective_1, subject, preposition, adjective_2, object_ = phrase
            adjectives.add(adjective_1)
            subjects.add(subject)
            prepositions.add(preposition)
            adjectives.add(adjective_2)
            objects.add(object_)

    subjects = list(subjects)
    prepositions = list(prepositions)
    objects = list(objects)
    adjectives = list(adjectives)

    with open(grammar_file, 'w') as f:
    
        if max_phrase_len == 3:

            _write_grammar_line(f, 'subject', subjects)
            _write_grammar_line(f, 'preposition', prepositions)
            _write_grammar_line(f, 'object', objects)
            f.write('\n')
            f.write('(SENT-START $subject $preposition $object SENT-END)')
            f.write('\n')
            
        elif max_phrase_len == 4:

           _write_grammar_line(f, 'subject', subjects)
           _write_grammar_line(f, 'preposition', prepositions)
           _write_grammar_line(f, 'adjective', adjectives)
           _write_grammar_line(f, 'object', objects)
           f.write('\n')
           f.write('(SENT-START $subject $preposition [$adjective] $object SENT-END)')
           f.write('\n')

        elif max_phrase_len == 5:

            _write_grammar_line(f, 'adjective', adjectives, 1)
            _write_grammar_line(f, 'subject', subjects)
            _write_grammar_line(f, 'preposition', prepositions)
            _write_grammar_line(f, 'adjective', adjectives, 2)
            _write_grammar_line(f, 'object', objects)
            f.write('\n')
            f.write('(SENT-START [$adjective1] $subject $preposition [$adjective2] $object SENT-END)')
            f.write('\n')

    f.close()

def _write_fs_mlf(filenames: list, mlf_file: str, split_index: int):
    with open(mlf_file, 'w') as f:
        
        f.write('#!MLF!#\n')

        for filename in filenames:
            label = filename.split('/')[-1].replace('htk', 'lab')
            phrase = label.rsplit('-', 3)[split_index].split('_')

            f.write('"*/{}"\n'.format(label))
            f.write(f'{ENTER}\n')

            for word in phrase:
                f.write('{}\n'.format('\n'.join(word)))

            f.write(f'{EXIT}\n')
            f.write('.\n')
    

def _write_mlf(filenames: list, mlf_file: str, split_index: int):
    with open(mlf_file, 'w') as f:
        
        f.write('#!MLF!#\n')

        for filename in filenames:
            label = filename.split('/')[-1].replace('htk', 'lab')
            phrase = label.rsplit('-', 3)[split_index].split('_')

            f.write('"*/{}"\n'.format(label))
            f.write(f'{ENTER}\n')

            for word in phrase:
                f.write('{}\n'.format(word))

            f.write(f'{EXIT}\n')
            f.write('.\n')

def _generate_mlf_file(isFingerspelling: bool, isSingleWord: bool, mlf_file: str, data_path: str) -> None:
    """Creates all_labels.mlf file that contains every phrase in the 
    dataset.
    """

    htk_filepaths = os.path.join(data_path, 'htk', '*.htk')
    filenames = glob.glob(htk_filepaths)
    split_index = 1
    
    if not(isFingerspelling):
        _write_mlf(filenames, mlf_File, split_index)
    else:
        if isSingleWord:
            _write_fs_mlf(filenames, mlf_file, split_index)
        else:
            mlf_filename = _get_basename(mlf_file)
            word_mlf_file = os.path.join(
                data_path,
                mlf_filename + '_word.mlf'
            )
            letter_mlf_file = os.path.join(
                data_path,
                mlf_filename + '_letter.mlf'
            )
            _write_mlf(filenames, word_mlf_file, split_index)
            _write_fs_mlf(filenames, letter_mlf_file, split_index)

