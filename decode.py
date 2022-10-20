from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
from tqdm import tqdm
from multiprocess import Pool, Lock
from collections import defaultdict

import re
import os
import argparse

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp

log_lock = Lock()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup_dir', required=True, type=str)
    parser.add_argument('--dest_dir', required=True, type=str)
    parser.add_argument('--video_dim', nargs=2, type=int, default=(1080, 1920))
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--only_print_signs', action='store_true')
    parser.add_argument('--num_threads', type=int, default=5)
    
    args = parser.parse_args()
    return args

def get_image_description(exifdata):
    description = ""
    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        # print(f"{tag:25}: {data}")
        if tag == "ImageDescription":
            description = data
    return description

def get_data_from_description(description):
    subbed = re.sub(r'file=(.*?),', r'"\1",', description)
    subbed = re.sub(r'videoStart=(.*?),', r'"\1",', subbed)
    subbed = re.sub(r'signStart=(.*?),', r'"\1",', subbed)
    subbed = re.sub(r'signEnd=(.*?),', r'"\1",', subbed)
    subbed = re.sub(r'isValid=(.*?)\)', r'"\1")', subbed)
    data = eval(subbed)
    return data

def clean_sign(sign):
    sign = sign.replace(' / ', '')
    sign = sign.replace(' ', '')
    sign = sign.replace('-', '')
    sign = sign.replace('(', '')
    sign = sign.replace(')', '')
    return sign

def extract_clip_from_video(args, uid, sign, recording_idx, recording, videopath):
    filename, video_start_time, sign_start_time, sign_end_time, isValid = recording
    if isValid == "False":
        return

    video_start_time_date = datetime.strptime(video_start_time+"000", '%Y_%m_%d_%H_%M_%S.%f')
    sign_start_time_date = datetime.strptime(sign_start_time+"000", '%Y_%m_%d_%H_%M_%S.%f')
    sign_end_time_date = datetime.strptime(sign_end_time+"000", '%Y_%m_%d_%H_%M_%S.%f')
    sign = clean_sign(sign)
    
    print(sign, filename, video_start_time, sign_start_time, sign_end_time)
    
    start_seconds = sign_start_time_date - video_start_time_date
    end_seconds = sign_end_time_date - video_start_time_date
    
    with mp.VideoFileClip(videopath) as video:
        try:
            video = video.resize(args.video_dim)
            new = video.subclip(start_seconds.seconds + start_seconds.microseconds/1000000.0, end_seconds.seconds + end_seconds.microseconds/1000000.0)
            new.write_videofile(os.path.join(args.dest_dir, f"{uid}-{sign}-{video_start_time}-{recording_idx}.mp4"), verbose=False)
        except Exception as e:
            log_lock.acquire()
            with open(args.log_file, 'a') as f:
                f.write('Error: %s\n' % e)
                f.write('Sign: %s\n' % sign)
                f.write('Video File Path: %s\n' % videopath)
                f.write('\n')
            log_lock.release()

def get_uid(args, filename):
    imagepath = os.path.join(args.backup_dir, filename)
    videopath = re.sub(r'-timestamps.jpg', r'.mp4', imagepath)
    _, videoname = os.path.split(videopath)
    uid = videoname.split('-')[0]
    return uid, videopath

def process_file(args, pool, pbar, filename):
    pbar.set_description('Processing Timestamps File: %s' % filename)
    results = []
    signs = set()

    if filename.endswith("-timestamps.jpg"):
        image = Image.open(os.path.join(args.backup_dir, filename))
        exifdata = image.getexif()

        description = get_image_description(exifdata) 
        data = get_data_from_description(description)
        uid, videopath = get_uid(args, filename)
        
        if os.path.exists(videopath):
            for sign, recording_list in data.items():
                signs.add(clean_sign(sign))
                if not args.only_print_signs:
                    for idx, recording in enumerate(recording_list):
                        thread_args = (args, uid, sign, idx, recording, videopath)
                        result = pool.apply_async(extract_clip_from_video, args = thread_args)
                        results.append(result)
    
    return results, signs

if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)
    
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    if args.log_file is None:
        args.log_file = os.path.join('logs', 'decode_' + datetime.now().strftime('%Y-%m-%d_%H-%M'))
    
    backup_dir = os.fsencode(args.backup_dir)
    pbar = tqdm(os.listdir(backup_dir))
    pool = Pool(args.num_threads)
    results = []
    signs = set()

    for file in pbar:
        filename = os.fsdecode(file)

        if filename.endswith('.zip'):
            continue

        file_results, processed_signs = process_file(args, pool, pbar, filename)
        results.extend(file_results)
        signs = signs.union(processed_signs)
    
    for result in results:
        result.get()

    pool.close()
    print("Signs: ", signs)
