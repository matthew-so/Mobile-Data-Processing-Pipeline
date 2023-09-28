"""Defines method to test HMM. Can perform recognition.

Methods
-------
test
"""
import os
import glob
import shutil
from string import Template

ENTER = '!ENTER'
EXIT = '!EXIT'

def test(
    start: int,
    end: int,
    method: str,
    insertion_penalty: int,
    beam_threshold: int = 2000,
    fold: str = "",
    is_triletter: bool = False,
    wordList_file: str = 'data/wordList',
    dict_file: str = 'data/dict',
    all_labels_file: str = 'data/all_labels.mlf',
) -> None:
    """Tests the HMM using HTK. Calls HVite and HResults. Can perform
    either recognition or verification.

    Parameters
    ----------
    test_args : Namespace
        Argument group defined in test_cli() and split from main
        parser.
    """

    if os.path.exists(f'results/{fold}'):
        shutil.rmtree(f'results/{fold}')
    os.makedirs(f'results/{fold}')
    
    if os.path.exists(f'logs/{fold}'):
        if os.path.exists(f'logs/{fold}test.log'):
            os.remove(f'logs/{fold}test.log')

    if method != 'alignment':

        if os.path.exists(f'hresults/{fold}'):
            shutil.rmtree(f'hresults/{fold}')
        os.makedirs(f'hresults/{fold}')
    
    if end == -1:
        end = len(glob.glob(f'models/{fold}*hmm*'))

    if start < 0:
        start = end + start

    print("METHOD:: ", method)

    if method == 'recognition':
        print("1111")

        if is_triletter:
            HVite_str = (f'HVite -A -H $macros -m -S lists/{fold}test.data -i '
                            f'$results -p {insertion_penalty} -w wordNet.txt -s 25 dict_tri wordList_triletter '
                            f'>> logs/{fold}test.log')
        else:
            # HVite_str = (f'HVite -A -H $macros -m -S lists/{fold}test.data -i '
            #          f'$results -p {insertion_penalty} -w wordNet.txt -s 25 {dict_file} {wordList_file}')
            HVite_str = (f'HVite '
                            f'-T 4 -u 100 '
                            f'-A -H $macros -m -S lists/{fold}test.data -i '
                            f'$results -p {insertion_penalty} -w wordNet.txt -s 25 {dict_file} {wordList_file} '
                            f'>> logs/{fold}test.log')

        HVite_cmd = Template(HVite_str)

        if is_triletter:
            HResults_str = (f'HResults -A -h -e \\?\\?\\? {ENTER} -e \\?\\?\\? '
                        f'{EXIT} -p -t -I all_labels_triletter.mlf wordList_triletter $results '
                        f'>> $hresults')
        else:
            HResults_str = (f'HResults -A -h -e \\?\\?\\? {ENTER} -e \\?\\?\\? '
                        f'{EXIT} -t -I {all_labels_file} {wordList_file} $results '
                        f'>> $hresults')
        HResults_cmd = Template(HResults_str)

    elif method == 'alignment':
        print("22222")

        HVite_str = (f'HVite -a -o N -T 1 -H $macros -m -f -S '
                     f'lists/{fold}train.data -i $results -t {beam_threshold} '
                     f'-p {insertion_penalty} -I {all_labels_file} -s 25 {dict_file} {wordList_file} '
                     f'>/dev/null 2>&1')
        HVite_cmd = Template(HVite_str)
        HResults_cmd = Template('')

    for i in range(start, end):

        macros_filepath = f'models/{fold}hmm{i}/newMacros'
        results_filepath = f'results/{fold}res_hmm{i}.mlf'
        hresults_filepath = f'hresults/{fold}res_hmm{i}.txt'

        os.system(HVite_cmd.substitute(macros=macros_filepath,
                                       results=results_filepath))

        os.system(HResults_cmd.substitute(results=results_filepath,
                                          hresults=hresults_filepath))
