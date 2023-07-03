from PIL import Image
from PIL.ExifTags import TAGS
import datetime
from tqdm import tqdm
from multiprocess import Pool, Lock
from collections import defaultdict

import re
import os
import argparse
import json

import subprocess

log_lock = Lock()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_array_num", required=False, type=int)
    parser.add_argument("--backup_dir", required=True, type=str)
    parser.add_argument("--dest_dir", required=True, type=str)
    parser.add_argument("--video_dim", nargs=2, type=int, default=(1080, 1920))
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument(
        "--buffer",
        nargs=2,
        type=float,
        default=(-0.5, 0.5),
        help="Buffer for start/end of video",
    )
    parser.add_argument(
        "--invert", action="store_true", help="Switch start/end timestamps."
    )
    parser.add_argument("--num_threads", type=int, default=5)
    parser.add_argument("--make_structured_dirs", action="store_true")
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument(
        "--ffmpeg_loglevel",
        choices=[
            "quiet",
            "panic",
            "fatal",
            "error",
            "warning",
            "info",
            "verbose",
            "debug",
            "trace",
        ],
        default="fatal",
    )

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

    print("Image Description:", description)
    return description


def get_data_from_description(description):
    is_valid_exists = "isValid" in description

    subbed = re.sub(r"file=(.*?),", r'"\1",', description)
    subbed = re.sub(r"videoStart=(.*?),", r'"\1",', subbed)
    subbed = re.sub(r"signStart=(.*?),", r'"\1",', subbed)

    if is_valid_exists:
        subbed = re.sub(r"signEnd=(.*?),", r'"\1",', subbed)
        subbed = re.sub(r"isValid=(.*?)\)", r"\1)", subbed)
    else:
        subbed = re.sub(r"signEnd=(.*?)\)", r'"\1")', subbed)
    data = eval(subbed)
    return data, is_valid_exists


def clean_sign(sign):
    sign = sign.replace(" / ", "")
    sign = sign.replace(" ", "")
    sign = sign.replace("-", "")
    sign = sign.replace(",", "")
    sign = sign.replace("(", "")
    sign = sign.replace(")", "")
    return sign


def extract_clip_from_video_hold(
        args, uid, sign, recording_idx, recording, videopath, is_valid_exists
):
    with open("config.json") as f:
        config = json.load(f)

    # print("Recording: ", recording)
    if is_valid_exists:
        signName, filename, video_start_time, sign_start_time, sign_end_time, is_valid = recording
    else:
        signName, filename, video_start_time, sign_start_time, sign_end_time = recording
        is_valid = True

    video_start_time_date = datetime.datetime.strptime(
        video_start_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign_start_time_date = datetime.datetime.strptime(
        sign_start_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign_end_time_date = datetime.datetime.strptime(
        sign_end_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign = clean_sign(sign)

    if is_valid:
        print(sign_end_time_date - sign_start_time_date)
    # print(
    #     "Video Info: ", sign, filename, video_start_time, sign_start_time, sign_end_time
    # )

    start_seconds = sign_start_time_date - video_start_time_date
    end_seconds = sign_end_time_date - video_start_time_date

    start_subclip = start_seconds.seconds + start_seconds.microseconds / 1e6
    end_subclip = end_seconds.seconds + end_seconds.microseconds / 1e6

    if uid in config:
        buffer_0 = config[uid]["buffer_start"]
        buffer_1 = config[uid]["buffer_end"]
        invert = config[uid]["invert"]
        if invert:
            start_subclip = end_subclip + buffer_0
            end_subclip += buffer_1
        else:
            start_subclip += buffer_0
            end_subclip += buffer_1

    else:
        # print(f"Could not find {uid} in config.json file")
        if args.invert:
            start_subclip = end_subclip + args.buffer[0]
            end_subclip += args.buffer[1]
        else:
            start_subclip += args.buffer[0]
            end_subclip += args.buffer[1]

    output_dir = args.dest_dir

    if not is_valid:
        return False, filename, signName

    if args.make_structured_dirs:
        video_filename = f"{sign_start_time}-{recording_idx}.mp4"
        output_dir = os.path.join(args.dest_dir, f"{uid}", f"{sign}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.dest_dir
        # print("Filename Struct: ", uid, sign, video_start_time, recording_idx)
        video_filename = f"{uid}-{sign}-{video_start_time}-{recording_idx}.mp4"

    time = end_subclip - start_subclip
    full_filename = os.path.join(output_dir, video_filename)

    if args.use_cuda:
        subprocess.run(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                videopath,
                "-vf",
                f"scale={str(args.video_dim[0])}:{str(args.video_dim[1])}",
                "-ss",
                start_subclip.strftime("%H:%M:%S.%f")[:-3],
                "-t",
                end_subclip.strftime("%H:%M:%S.%f")[:-3],
                "-c:v",
                "hevc_nvenc",
                "-c:a",
                "copy",
                str(full_filename),
                "-loglevel",
                args.ffmpeg_loglevel,
            ]
        )
    else:
        args = (
            f"ffmpeg -y -nostdin -ss {start_subclip:.2f} -i {videopath} "
            f"-t {time:.2f} -c:v libx264 {full_filename}"
        )

        # Call ffmpeg directly
        subprocess.run(args, shell=True, check=True)

    return True, None, None


def extract_clip_from_video(
        args, uid, sign, recording_idx, recording, prevRecording, videopath, is_valid_exists
):
    with open("config.json") as f:
        config = json.load(f)

    print("Recording: ", recording)
    print("Previous Recording: ", prevRecording)

    if is_valid_exists:
        signName, filename, video_start_time, sign_start_time, sign_end_time, is_valid = recording
    else:
        signName, filename, video_start_time, sign_start_time, sign_end_time = recording
        is_valid = True

    if not is_valid:
        return False, filename, signName

    sign_start_time_date = datetime.datetime.strptime(
        sign_start_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign_end_time_date = datetime.datetime.strptime(
        sign_end_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )

    if sign_end_time_date - sign_start_time_date > datetime.timedelta(seconds=1):
        return extract_clip_from_video_hold(args, uid, sign, recording_idx, recording, videopath, is_valid_exists)

    # To split based on taps, we set the current sign's start time to be the previous sign's end time
    if (prevRecording is not None):
        prevSignName, prevFilename, prev_video_start_time, prev_sign_start_time, prev_sign_end_time, prev_is_valid = prevRecording

        temp = sign_start_time
        sign_start_time = prev_sign_end_time
        sign_end_time = temp
    else:
        temp = sign_start_time
        sign_start_time = video_start_time
        sign_end_time = temp

    video_start_time_date = datetime.datetime.strptime(
        video_start_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign_start_time_date = datetime.datetime.strptime(
        sign_start_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign_end_time_date = datetime.datetime.strptime(
        sign_end_time + "000", "%Y_%m_%d_%H_%M_%S.%f"
    )
    sign = clean_sign(sign)

    print(
        "Video Info: ", sign, filename, video_start_time, sign_start_time, sign_end_time
    )

    start_seconds = sign_start_time_date - video_start_time_date
    end_seconds = sign_end_time_date - video_start_time_date

    start_subclip = start_seconds.seconds + start_seconds.microseconds / 1e6
    end_subclip = end_seconds.seconds + end_seconds.microseconds / 1e6

    # if uid in config:
    #     buffer_0 = config[uid]["buffer_start"]
    #     buffer_1 = config[uid]["buffer_end"]
    #     invert = config[uid]["invert"]
    #     if invert:
    #         start_subclip = end_subclip + buffer_0
    #         end_subclip += buffer_1
    #     else:
    #         start_subclip += buffer_0
    #         end_subclip += buffer_1

    # else:
    #     print(f"Could not find {uid} in config.json file")
    #     if args.invert:
    #         start_subclip = end_subclip + args.buffer[0]
    #         end_subclip += args.buffer[1]
    #     else:
    #         start_subclip += args.buffer[0]
    #         end_subclip += args.buffer[1]

    output_dir = args.dest_dir

    if args.make_structured_dirs:
        video_filename = f"{sign_start_time}-{recording_idx}.mp4"
        output_dir = os.path.join(args.dest_dir, f"{uid}", f"{sign}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.dest_dir
        print("Filename Struct: ", uid, sign, video_start_time, recording_idx)
        video_filename = f"{uid}-{sign}-{video_start_time}-{recording_idx}.mp4"

    # Only take the first video for a particular sign because subsequent ones tend to be bad
    if recording_idx != 0:
        output_dir = os.path.join(args.dest_dir, "error")

    # if not is_valid:
    #     output_dir = os.path.join(args.dest_dir, "error")

    time = end_subclip - start_subclip
    full_filename = os.path.join(output_dir, video_filename)

    if args.use_cuda:
        subprocess.run(
            [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                videopath,
                "-vf",
                f"scale={str(args.video_dim[0])}:{str(args.video_dim[1])}",
                "-ss",
                start_subclip.strftime("%H:%M:%S.%f")[:-3],
                "-t",
                end_subclip.strftime("%H:%M:%S.%f")[:-3],
                "-c:v",
                "hevc_nvenc",
                "-c:a",
                "copy",
                str(full_filename),
                "-loglevel",
                args.ffmpeg_loglevel,
            ]
        )
    else:
        args = (
            f"ffmpeg -y -nostdin -ss {start_subclip:.2f} -i {videopath} "
            f"-t {time:.2f} -c:v libx264 {full_filename}"
        )

        # Call ffmpeg directly
        subprocess.run(args, shell=True, check=True)

    return True, None, None


def get_uid(args, filename):
    imagepath = os.path.join(args.backup_dir, filename)
    videopath = re.sub(r"-timestamps.jpg", r".mp4", imagepath)
    _, videoname = os.path.split(videopath)
    uid = videoname.split("-")[0]
    return uid, videopath


def count_recording(sign, recording, is_valid_exists, recording_count):
    if is_valid_exists:
        _, _, _, _, is_valid = recording
        if is_valid:
            recording_count[sign] += 1
    else:
        recording_count[sign] += 1
    return recording_count


def process_file(args, pool, pbar, filename, results, signs, recording_count):
    pbar.set_description("Processing Timestamps File: %s" % filename)
    # results = []
    # signs = set()
    # recording_count = 0

    if filename.endswith("-timestamps.jpg"):
        image = Image.open(os.path.join(args.backup_dir, filename))
        exifdata = image.getexif()

        description = get_image_description(exifdata)
        data, is_valid_exists = get_data_from_description(description)
        uid, videopath = get_uid(args, filename)

        noSigns = open(os.path.join(args.dest_dir, "error/noSigns.txt"), "a")

        if os.path.exists(videopath) and len(data) > 1:  # We want to skip videos that only have 1 sign in them
            prevRecording = None
            # print("Data Items:", list(data.items()))

            newList = []

            for sign, recording_list in data.items():
                for recording in recording_list:
                    listRecording = list(recording)
                    listRecording.insert(0, sign)
                    listRecording = tuple(listRecording)
                    newList.append(listRecording)

            # print("New List: ", newList)

            sortedData = sorted(newList, key=lambda tup: tup[4])
            # print('sortedData:')
            # print(sortedData)

            # index = 0

            results = pool.map(
                lambda i: extract_clip_from_video(
                    args,
                    uid,
                    sortedData[i][0],
                    0,
                    sortedData[i],
                    sortedData[i - 1] if i > 0 else None,
                    videopath,
                    is_valid_exists,
                ),
                range(len(sortedData))
            )

            errorSignsFile = open(os.path.join(args.dest_dir, "error/errorSigns.txt"), "a")
            for isValid, fileName, signName in results:
                if not isValid:
                    errorSignsFile.write(fileName + ', ' + signName + '\n')
            errorSignsFile.close()


        elif len(data) <= 1:
            noSigns.write(filename + "\n")
            noSigns.close()


"""             for sign, recording_list in sortedData:
                signs.add(clean_sign(sign))
                print("Recording List: ", recording_list)
                for idx, recording in enumerate(recording_list):
                    count_recording(sign, recording, is_valid_exists, recording_count)
                    if not args.skip_extraction:
                        thread_args = (
                            args,
                            uid,
                            sign,
                            idx,
                            recording,
		                    prevRecording,
                            videopath,
                            is_valid_exists,
                        )
                        result = extract_clip_from_video(args,
                            uid,
                            sign,
                            idx,
                            recording,
		                    prevRecording,
                            videopath,
                            is_valid_exists,)
                        results.append(result)
                    prevRecording = recording_list[0] """


# return results, signs, recording_count


def make_missing_dirs(args):
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    if not os.path.exists(os.path.join(args.dest_dir, "error")):
        os.makedirs(os.path.join(args.dest_dir, "error"))

    if not os.path.exists("logs"):
        os.mkdir("logs")


if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)

    make_missing_dirs(args)
    if args.log_file is None:
        args.log_file = os.path.join(
            "logs", "decode_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        )

    if args.job_array_num is not None:
        backup_dir = os.fsencode(args.backup_dir)
        with open(
                f"/data/sign_language_videos/batches/batch_{args.job_array_num}.txt"
        ) as fin:
            assigned_videos = fin.read().splitlines()

        pbar = tqdm(assigned_videos)
        pool = Pool(args.num_threads)

    else:
        backup_dir = os.fsencode(args.backup_dir)
        pbar = tqdm(os.listdir(backup_dir))
        pool = Pool(args.num_threads)

    results = []
    signs = set()
    recording_count = defaultdict(int)

    for file in pbar:
        filename = os.fsdecode(file)

        if filename.endswith(".zip"):
            continue

        process_file(args, pool, pbar, filename, results, signs, recording_count)

    pool.close()
    print("Signs: ", signs)
    print("Recording Count (Total): ", sum(recording_count.values()))
    print("Recording Count (by Sign): ", recording_count)
