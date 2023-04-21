import argparse
import os
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', required=True, type=int)
    
    args = parser.parse_args()
    return args

def create_batches(num_batches, raw_video_location='/data/sign_language_videos', batch_output_location='/data/sign_language_videos/batches'):
    if not os.path.exists(batch_output_location):
        os.makedirs(batch_output_location)
    all_videos = glob.glob(f'{raw_video_location}/**/*.jpg', recursive=True)
    batches = np.array_split(np.array(all_videos), num_batches)
    for i, batch in enumerate(batches):
        with open(f'{batch_output_location}/batch_{i+1}.txt', 'w') as f:
            for video in batch: 
                f.write(f'{video}\n')

if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)
    
    print(f"Creating {args.num_batches} batches...")
    create_batches(args.num_batches)
    print(f"Finished creating batches")