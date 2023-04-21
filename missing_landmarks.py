import glob
import json

if __name__ == "__main__":
    mp_files = glob.glob('./mediapipe_parquet/*/*/*.data')
    
    total_files = 0
    pose_available = 0
    rh_landmarks_available = 0
    lh_landmarks_available = 0

    for mp_file in mp_files:
        with open(mp_file, 'r') as f:
            data = json.load(f)
        total_files += 1
        
        update_rhlm = True
        update_lhlm = True
        update_pose = True

        for frame in data:
            if data[frame]["landmarks"]["1"] and update_rhlm:
                rh_landmarks_available += 1
                update_rhlm = False
            if data[frame]["landmarks"]["0"] and update_lhlm:
                lh_landmarks_available += 1
                update_lhlm = False
            if data[frame]["pose"] and update_pose:
                pose_available += 1
                update_pose = False
    
    print(f'Total Files: {total_files}')
    print(f'Pose Available: {pose_available}')
    print(f'Right Hand Landmarks: {rh_landmarks_available}')
    print(f'Left Hand Landmarks: {lh_landmarks_available}')

