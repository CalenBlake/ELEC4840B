"""
# -----------------------------------
# Script for Generating CyTex Images from the RAVDESS dataset
# Author: Calen Blake
# Date: 17-05-23
# NOTE: Currently installed and running Librosa version 0.8.1,
# Due to issues experienced with the resample() method in later versions.
# -----------------------------------
"""

# --------------------- Import necessary libraries ---------------------
import librosa
import librosa.display
import numpy as np
import glob
import os
import math
from PIL import Image
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter, laplace

# --------------------- 1. CyTex Generation ---------------------
dataset_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/RAVDESS_Refactored/'
output_dir = '/Users/c3283313/PycharmProjects/ELEC4840B-Programming/RAVDESS_Refactored/RGB_IMG/'

list_of_files = []
Labels = []

f_l = []

# Counter for the number of CyTex images generated per emotion
# Utilised for the naming convention of the output images
# neutral, calm, happy, sad (unhappy), angry, fearful, disgust, surprised
n = c = h = u = a = f = d = s = 0

# *** arbitrary counter, what is use other than counting total number of CyTex imgs generated?
cnt = 0

# Don't need to output as csv, can create default histogram
# this is just to test other stats

# *** UNCOMMENT AND EDIT TO PRODUCE HISTOGRAM OF IMG COUNT PER EMOTION
# def save_hist(pitch_hist, emo):
#     with open('/home/ali/pitch_hist_{}.csv'.format(emo), 'w') as f:
#         write = csv.writer(f)
#         write.writerows(pitch_hist)

# Change current working directory
os.chdir(dataset_dir+'wav/')
# read and append all wav files to a list
for file in glob.glob("*.wav"):
    list_of_files.append(file)

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
            # avoid overlap by setting n_step to ZERO!
            # overlap = n_step/new_sr
            n_step = 8000
        # *** CHANGE FRAME LENGTH (SMALLER)
        # each second has 16000 samples
        #
        m_frame = signal[step - 16000 - n_step: st * step + n_step, ]
        # note that 160 = 16000/100 => 10ms window... change parma to change frame width
        for j in range(1, 1 + int(len(m_frame) / 160)):
            step = j * 160
            # With the sampling rate of 16khz, 160 samples represent a 10ms frame
            frame = signal[step - 160: step, ]
            # *** SEARCH PIPTRACK DOCUMENTATION
            # Used to find fundamental frequency, commonly used for pitch extraction
            pitches, magnitudes = librosa.core.piptrack(y=frame, sr=new_sr, fmin=40.0, fmax=600.0)

            # Convert stereo signal to mono via selecting the maximum of both channels
            # *** CHECK THAT RAVDESS IS IN STEREO FORMAT
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
                # *** Original script uses Input_filename[5] which is the position of emotion
                # note that f_l was only used for statistics collection => can remove
                f_l.append([pitch_m, mag_max, Input_filename[1]])

                # Number Of Rows each speech frame construct in the final image
                NOR = (math.ceil(160.0 / (new_sr / pitch_m)))
                for itt in range(NOR):
                    start_p = itt * int(new_sr / pitch_m)
                    if itt < max(range(NOR)):
                        end_p = (itt + 1) * int(new_sr / pitch_m)
                    else:
                        end_p = 160
                    frames = frame[start_p:end_p]  # * (itt+1)
                    IMG.append(frames)

        IMG_temp = np.array(IMG)
        MIN = min([min(sublist) for sublist in IMG])
        sz = []
        IMG_F = []
        for it in range(len(IMG)):
            sz.append(len(IMG[it]))  # size of image rows
            # 400 pixel width in each row of the output image
            IMG_F.append(400 * [MIN])
            for jt in range(len(IMG[it])):
                IMG_F[it][jt] = IMG[it][jt]

        # *** Check what these three lines are implementing!
        # convert list to matrix
        IMG_F = np.array(IMG_F)
        # normalize pixel values
        IMG_F = IMG_F - np.min(IMG_F)
        IMG_F = (IMG_F / np.max(IMG_F))
        # reduce value of small samples and increase value of larger samples
        # akin to increasing image contrast
        IMG_F = np.power(IMG_F, 3)
        img0 = IMG_F

        # *** COMMENTED LINES ARE UNUSED TRANSFORMS IN FINAL OUTPUT
        # fd_img1 = gaussian_filter(img0, sigma=3)
        # sd_img1 = laplace(img0)
        # *** DOUBLE CHECK! returns first and second order gradients
        fd_img = np.gradient(img0)

        # sd_img = np.gradient(img0, 2)

        # Code values of each channel of the image, incorporate the gradients found above
        img = np.zeros([len(img0), 400, 3], dtype='uint8')
        # MAP PIXEL VALUES TO [0, 255]
        img[:, :, 0] = img0 * 255
        img[:, :, 1] = fd_img[0] * 255
        img[:, :, 2] = fd_img[1] * 255

        # convert output array to a rgb image
        img = Image.fromarray(img, 'RGB')
        newsize = (400, 400)
        # resize image to desired CyTex dimensions
        img = img.resize(newsize)
        cnt += 1

        # save Calm images
        if Input_filename[1] == '2':
            img.save(output_dir + 'Calm/image{}.png'.format(c))
            c += 1
        # save Happy images
        elif Input_filename[1] == '3':
            img.save(output_dir + 'Happy/image{}.png'.format(h))
            h += 1
        # save Sad images
        elif Input_filename[1] == '4':
            img.save(output_dir + 'Sad/image{}.png'.format(u))
            u += 1
        # save Angry images
        elif Input_filename[1] == '5':
            img.save(output_dir + 'Angry/image{}.png'.format(a))
            a += 1
        # save Fearful images
        elif Input_filename[1] == '6':
            img.save(output_dir + 'Fearful/image{}.png'.format(f))
            f += 1
        # save Disgust images
        elif Input_filename[1] == '7':
            img.save(output_dir + 'Disgust/image{}.png'.format(d))
            d += 1
        # save Surprised images
        elif Input_filename[1] == '8':
            img.save(output_dir + 'Surprised/image{}.png'.format(s))
            s += 1
        # save Neutral images
        else:
            img.save(output_dir + 'Neutral/image{}.png'.format(n))
            n += 1
        # *** close plot here is redundant??? What is its purpose?
        # plt.close()
