import os
import sys
import csv
import pandas

# sign:            the word being signed
# filename:        filename of the video
# good:            video is good to go
# start too early: too much idle time at beginning, or recording of another sign
#                  is visible at the beginning 
# start too late:  person pressed record too late, clipping the 
#                  beginning of the recording
# end too early:   person let go of record too early, clipping the end of the 
#                  recording
# end too late:    too much idle time at end, or recording of another sign is 
#                  visible at the end
# wrong sign:      user signed the wrong interpretation of the English word, 
#                  or signed the wrong word completely
# empty sign:      no sign was visible at all (likely an error in the app)
# cropped:         the user's hand is partially/totally obscured or cropped out of the frame
# other issue:     none of the above (e.g., framerate issues) - write out explanation
COLUMNS = ['sign', 'filename', 'good', 'start too early', 
           'start too late', 'end too early', 'end too late',
           'wrong sign', 'empty sign', 'cropped', 'other issue']

EMPTY_COLUMNS = [''] * (len(COLUMNS) - 2)


def pad(number):
	if number < 10:
		return f'0{number}'
	return str(number)


def generate_csv(folder, output):
	data = list()
	words = sorted(os.listdir(folder), key=lambda file: file.lower())
	word_number = 0
	for word in words:
		recording_dir = f'{folder}/{word}'
		if not os.path.isdir(recording_dir):
			continue

		# Because this CSV file will have some 5000+ rows in it,
		# we will add the column label before each word to prevent
		# the user filling this form from having to repeatedly
		# scroll to the top of the file to see which column is which.
		data.append(COLUMNS)
		word_number += 1

		recordings = sorted(os.listdir(recording_dir),
			                key=lambda file: file.lower())
		file_number = 0
		for recording in recordings:
			if not recording.endswith('.mp4'):
				continue

			file_number += 1

			# Label each video numerically to make filling the form
			# slightly easier for the user (they can just compare
			# the numbers instead of having to compare the entire date)
			path = f'{folder}/{word}/{recording}'
			label = f'{recording}'
			number = pad(file_number)
			if not recording.startswith(f'{number}-'):
				renamed = f'{folder}/{word}/{number}-{recording}'
				label = f'{number}-{recording}'
				os.rename(path, renamed)

			row = [
				# Include the word number on the first row of the new word
				word if file_number > 1 else f'{word} (#{word_number})',
				label
			]

			row.extend(EMPTY_COLUMNS)
			data.append(row)

	data_frame = pandas.DataFrame(data)
	data_frame.to_csv(output, index=False, header=False)
	print(f'CSV saved successfully at {output}')


if __name__ == '__main__':
	folder = sys.argv[0]

	inputs = [folder]
	outputs = [f'{folder}.csv']

	if folder.endswith('/'):
		inputs = [
		    folder2 for folder2 in os.listdir(folder) 
		    if os.path.isfolder(folder2)
		]
		outputs = [
			f'{folder2}'.csv for folder2 in inputs
		]

	for folder_input, output in zip(inputs, outputs):
		generate_csv(folder_input, output)













