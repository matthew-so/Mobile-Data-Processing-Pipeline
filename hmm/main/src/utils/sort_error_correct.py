import os
import sys
import glob

folder = sys.argv[1]
os.chdir(folder)
os.mkdir("error")
for file in glob.glob("*.mkv"):
    if (file.split(".")[1] == "0"):
        os.rename(file, "error/"+file)