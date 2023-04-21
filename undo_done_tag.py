import os
from glob import glob
from tqdm import tqdm

folder = '/data/sign_language_videos/review_2'

files = glob(f"{folder}/**/*")
for file in tqdm(files):
    if '-done' in file:
        os.rename(file, file.replace('-done', ''))
