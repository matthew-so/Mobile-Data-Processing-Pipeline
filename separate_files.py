import os
import glob
import csv
from pathlib import Path

review_source = 'review_1'

annotation_files = glob.glob('annotation_output/**/annotations.csv')
print(annotation_files)

reviewed_signs = ['elephant']

dest_dir = 'sorted_videos_first_round'

for annotations in annotation_files:
	print(annotations)
	sign = annotations.split('/')[1]
	if sign not in reviewed_signs:
		continue
	labels = ['clean', 'variant', 'unrecognizable']
	with open(annotations, "r") as f:
		input = csv.reader(f)
		for row in input:
			if not row[1].endswith(".mp4"):
				labels = row[2:]
				continue
			videopath = os.path.join(review_source, sign, row[1])
			if os.path.exists(videopath): # If file exists, decide whether to move it
				if 'x' in row[3:]:
					for i in range(3, len(row)):
						if row[i] == 'x':
							output_path = os.path.join(dest_dir, labels[i-2], review_source, sign)
							Path(output_path).mkdir(parents=True, exist_ok=True)
							output_filepath = os.path.join(output_path, videopath.split('/')[-1])
							if not os.path.exists(output_filepath):
								os.link(videopath, output_filepath)
				else:
					output_path = os.path.join(dest_dir, 'clean', review_source, sign)
					Path(output_path).mkdir(parents=True, exist_ok=True)
					output_filepath = os.path.join(output_path, videopath.split('/')[-1])
					if not os.path.exists(output_filepath):
						os.link(videopath, output_filepath)
