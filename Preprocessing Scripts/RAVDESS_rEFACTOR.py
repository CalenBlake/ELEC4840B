"""
# -----------------------------------
# This script is used to change the groupings of the RAVDESS dataset.
# Initially the dataset is grouped by unique actors, however, it should be
# grouped by classes of emotion (like EMODB).
#
#
# Author: Calen Blake
# Date: 16-05-23
# NOTE: DO NOT RUN THIS SCRIPT MORE THAN ONCE!
# WILL CAUSE ISSUES WITH RENAMING OF FILES
# -----------------------------------
"""

# --------------------- Import necessary libraries ---------------------
import numpy as np
# import time
import random
import os
import glob

# RAVDESS Code reference:
# modality - speech/song - emotion - intensity - statement - repetition - actor
# Can afford to get rid of the first two as all identical
# i.e. all audio-only (03) and all speech (01)

# --------------------- 1. Initial Preprocessing ---------------------
data_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/RAVDESS/'
output_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/RAVDESS_Refactored/'
os.chdir(data_dir)
list_of_files = []
modified_files = []

# Loop through folders of actors
for folder in glob.glob("*"):
    # change directory to each sub-folder, then rename and save
    os.chdir(data_dir+folder)
    for file in glob.glob('*.wav'):
        new_file_name = file[6:]
        os.rename(file, output_dir+new_file_name)


