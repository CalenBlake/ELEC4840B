"""
# -----------------------------------
# This script is used to change the groupings of the RAVDESS dataset.
# Initially the dataset is grouped by unique actors, however, it should be
# grouped by classes of emotion (like EMODB).
#
#
# Author: Calen Blake
# Date: 16-05-23
# NOTE:
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

# --------------------- 1. Load Dataset ---------------------
data_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/RAVDESS/'
os.chdir(data_dir)
list_of_files=[]

# Loop through folders of actors
for folder in glob.glob("*"):
    # change directory and append files to list
    os.chdir(data_dir+folder)
    for file in glob.glob('*.wav'):
        print(file)
        list_of_files.append(file)

# --------------------- 2. Refactor ---------------------
# a.) Rename files, removing first 6 characters and append emotion code
# for i in range(len(list_of_files)):
#     Temp=[]
#     print(i)
#     Input_filename=list_of_files[i]
#     print(Input_filename)

# b.)

