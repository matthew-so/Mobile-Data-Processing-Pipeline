import os

from glob import glob

mp_root = '/root/Mobile-Data-Processing-Pipeline/mediapipe/hands/'
ark_root = './data/ark/'
htk_root = './data/htk/'

if __name__ == "__main__":
    mp_filepaths = os.path.join(mp_root, '*', '*', '*.data')
    ark_filepaths = os.path.join(ark_root, '*.ark')
    htk_filepaths = os.path.join(htk_root, '*.htk')

    mp_files = glob(mp_filepaths)
    ark_files = glob(ark_filepaths)
    htk_files = glob(htk_filepaths)
    
    print("MP Files: ", mp_files[:20])
    print("")

    print("Ark Files: ", ark_files[:20])
    print("")

    print("Htk Files: ", htk_files[:20])
    print("")
    
