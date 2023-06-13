"""
# -----------------------------------
# Script for Generating CyTex Images from a given WAV dataset
# Author: Calen Blake
# Date: 15-03-23
# NOTE: Currently installed and running Librosa version 0.8.1,
# Due to issues experienced with the resample() method in later versions.
# -----------------------------------
"""

# Import necessary libraries
import torch
import librosa
from PIL import Image
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import imageio
import glob
import os
# from astropy.visualization import make_lupton_rgb
from scipy.ndimage import gaussian_filter, laplace
from sklearn import preprocessing
import csv

# os.getcwd()
dataset_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/EMODB Database/'
output_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/EMODB Database/RGB_IMG_noOverlap/'
# dataset_dir = 'C:/Users/Calen Blake/Documents/University Work/2023-Semester1/ELEC4840B/EMODB Database/'
# output_dir = 'C:/Users/Calen Blake/Documents/University Work/2023-Semester1/ELEC4840B/EMODB Database/DemoTestOutput/'

list_of_files=[]
Labels=[]
f_l = []
t = f = a = w = n = e = l = 0

cnt = 0
# Change current working directory
os.chdir(dataset_dir+'wav/')
# glob allows for the use of the "*" wildcard
# read and append all wav files to a list
for file in glob.glob("*.wav"):
    list_of_files.append(file)
Emo_Dict={'W':1,'L':2,'E':3,'A':4,'F':5,'T':6,'N':0}
Speaker_Dict={'03':1,'08':2,'09':3,'10':4,'11':5,'12':6,'13':7,'14':8,'15':9,'16':10}

# major loop for cycling through each of the files
for i in range(len(list_of_files)):
    Temp=[]
    print(i)
    Input_filename=list_of_files[i]
    filename = dataset_dir + '/wav/' + Input_filename

    # load signal via librosa
    signal, sr = librosa.load(filename)
    # resample signal at rate of 16kHz
    new_sr = 16000
    signal = librosa.resample(signal, sr, new_sr)

    f_pitch = []
    f_period = []
    f_mag = []
    IMG = []

    # Divide the input signal into frame lengths
    for jj in range(1, 1 + int(len(signal) / new_sr)):
        step = 16000 * jj
        if jj == 1:
            st = 2
            n_step = 0
        else:
            st = 1
            # overlap = n_step/new_sr = n_step/16000 (%)
            n_step = 0
        # each second has 16000 samples
        m_frame = signal[step - 16000 - n_step: st * step + n_step, ]
        # note that 160 = 16000/100 => 10ms window... change param to change frame width
        for j in range(1, 1 + int(len(m_frame) / 160)):
            step = j * 160
            # With the sampling rate of 16khz, 160 samples represent a 10ms frame
            frame = signal[step - 160: step, ]
            # Used to find fundamental frequency, commonly used for pitch extraction
            pitches, magnitudes = librosa.core.piptrack(y=frame, sr=new_sr, fmin=40.0, fmax=600.0)

            # Convert stereo signal to mono via selecting the maximum of both channels
            mag_m = magnitudes.max(1)
            pitch_m = pitches.max(1)

            # Find the index of the maximum frequency correspond to the magnitudes above a pre-defind thereshold
            index = mag_m.argmax()
            # %%
            mag_max = mag_m[index]
            if mag_max >= 0:
                pitch_m = pitch_m[index]
                if pitch_m < 40:
                    pitch_m = 40
                # the list of pitches for each frames
                f_pitch.append(pitch_m)
                f_period.append(int(new_sr / pitch_m))  # the period of each frame in samples
                f_mag.append(mag_m)
                f_l.append([pitch_m, mag_max, Input_filename[5]])

                # NumRows for each speech frame (10ms) in the final image
                NOR = (math.ceil(160.0 / (new_sr / pitch_m)))
                # Encode start and end pixel
                for itt in range(NOR):
                    start_p = itt * int(new_sr / pitch_m)
                    if itt < max(range(NOR)):
                        end_p = (itt + 1) * int(new_sr / pitch_m)
                    else:
                        end_p = 160
                    # Fill data in between start and end pixels
                    frames = frame[start_p:end_p]
                    IMG.append(frames)

        IMG_temp = np.array(IMG)
        MIN = min([min(sublist) for sublist in IMG])
        sz = []
        IMG_F = []
        for it in range(len(IMG)):
            sz.append(len(IMG[it]))  # size of image rows
            IMG_F.append(400 * [MIN])
            for jt in range(len(IMG[it])):
                IMG_F[it][jt] = IMG[it][jt]

        # convert list to matrix
        IMG_F = np.array(IMG_F)
        # normalize pixel values
        IMG_F = IMG_F - np.min(IMG_F)
        IMG_F = (IMG_F / np.max(IMG_F))
        # adjust contrast via raised index
        IMG_F = np.power(IMG_F, 3)
        img0 = IMG_F

        # fd_img1 = gaussian_filter(img0, sigma=3)
        # sd_img1 = laplace(img0)
        # sd_img = np.gradient(img0, 2)

        # return 1 and 2 order gradients
        fd_img = np.gradient(img0)

        img = np.zeros([len(img0), 400, 3], dtype='uint8')
        # map pixel values to [0, 255]
        img[:, :, 0] = img0 * 255
        img[:, :, 1] = fd_img[0] * 255
        img[:, :, 2] = fd_img[1] * 255

        img = Image.fromarray(img, 'RGB')
        newsize = (400, 400)
        img = img.resize(newsize)
        cnt += 1

        if Input_filename[5] == 'T':
            img.save(output_dir + 'T/image{}.png'.format(t))
            t += 1

        elif Input_filename[5] == 'A':
            img.save(output_dir + 'A/image{}.png'.format(a))
            a += 1
        elif Input_filename[5] == 'W':
            img.save(output_dir + 'W/image{}.png'.format(w))
            w += 1
        elif Input_filename[5] == 'L':
            img.save(output_dir + 'L/image{}.png'.format(l))
            l += 1
        elif Input_filename[5] == 'E':
            img.save(output_dir + 'E/image{}.png'.format(e))
            e += 1
        elif Input_filename[5] == 'F':
            img.save(output_dir + 'F/image{}.png'.format(f))
            f += 1
        else:
            img.save(output_dir + 'N/image{}.png'.format(n))
            n += 1
        plt.close()
