"""Converts .ark files to .htk files for use by HTK.

Methods
-------
create_htk_files
"""
import os
import glob
import shutil
import tqdm
import os

from multiprocess import Pool

KALDI_DIR = '/espnet/tools/kaldi'  # MODIFY AS NEEDED

def run_kaldi_command(htk_dir: str, ark_file: str):
    htk_script_file = os.path.join(KALDI_DIR, 'src/featbin/copy-feats-to-htk')
    kaldi_command = (f'{htk_script_file} '
                     f'--output-dir={htk_dir} '
                     f'--output-ext=htk '
                     f'--sample-period=40000 '
                     f'ark:{ark_file}')
                     # f'>/dev/null 2>&1')

    ##last line silences stdout and stderr
    # print(kaldi_command)
    os.system(kaldi_command)

def create_htk_files(
    htk_dir: str = os.path.join('data', 'htk'),
    ark_dir: str = os.path.join('data', 'ark', '*.ark'),
    num_threads:int = 32
) -> None:
    """Converts .ark files to .htk files for use by HTK.
    """
    # if os.path.exists(htk_dir):
    #     shutil.rmtree(htk_dir)
    
    if not os.path.exists(htk_dir):
        os.makedirs(htk_dir)

    ark_files = glob.glob(ark_dir)
    
    pool = Pool(num_threads)
    results = []
    for ark_file in ark_files:
        result = pool.apply_async(run_kaldi_command, args=(htk_dir, ark_file))
        results.append(result)

    for result in tqdm.tqdm(results):
        result.get()
    
    pool.close()

