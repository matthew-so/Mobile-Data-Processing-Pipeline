from PIL import Image
from PIL.ExifTags import TAGS

import re

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp

import os

from datetime import datetime

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup_dir', required=True, type=str)
    parser.add_argument('--dest_dir', required=True, type=str)
    parser.add_argument('--video_dim', nargs=2, type=int, default=(1080, 1920))
    
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
    # print(description)

def get_data_from_description(description):
    subbed = re.sub(r'file=(.*?),', r'"\1",', description)
    subbed = re.sub(r'videoStart=(.*?),', r'"\1",', subbed)
    subbed = re.sub(r'signStart=(.*?),', r'"\1",', subbed)
    subbed = re.sub(r'signEnd=(.*?)\)', r'"\1")', subbed)
    # print(subbed)
    
    data = eval(subbed)
    return data

def extract_clip_from_video(args, uid, sign, recording_idx, recording, video):
    filename, video_start_time, sign_start_time, sign_end_time = recording
    video_start_time_date = datetime.strptime(video_start_time+"000", '%Y_%m_%d_%H_%M_%S.%f')
    sign_start_time_date = datetime.strptime(sign_start_time+"000", '%Y_%m_%d_%H_%M_%S.%f')
    sign_end_time_date = datetime.strptime(sign_end_time+"000", '%Y_%m_%d_%H_%M_%S.%f')
    
    # print(video_start_time_date)
    print(sign, filename, video_start_time, sign_start_time, sign_end_time)
    
    start_seconds = sign_start_time_date - video_start_time_date
    end_seconds = sign_end_time_date - video_start_time_date
    
    # print(start_seconds, end_seconds)
    new = video.subclip(start_seconds.seconds + start_seconds.microseconds/1000000.0, end_seconds.seconds + end_seconds.microseconds/1000000.0)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    new.write_videofile(os.path.join(args.dest_dir, f"{uid}-{sign}-{video_start_time}-{recording_idx}.mp4"))
    
def get_uid(args, filename):
    imagepath = os.path.join(args.backup_dir, filename)
    videopath = re.sub(r'-timestamps.jpg', r'.mp4', imagepath)
    _, videoname = os.path.split(videopath)
    uid = videoname.split('-')[0]
    return uid, videopath

if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)

    backup_dir = os.fsencode(args.backup_dir)
    for file in os.listdir(backup_dir):
        filename = os.fsdecode(file)
        if filename.endswith("-timestamps.jpg"):
            image = Image.open(os.path.join(args.backup_dir, filename))
            exifdata = image.getexif()

            description = get_image_description(exifdata) 
            data = get_data_from_description(description)
            uid, videopath = get_uid(args, filename)

            with mp.VideoFileClip(videopath) as video:
                video = video.resize(args.video_dim)
                for sign, recording_list in data.items():
                    for idx, recording in enumerate(recording_list):
                        extract_clip_from_video(args, uid, sign, idx, recording, video)

