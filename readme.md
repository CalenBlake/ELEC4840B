# ELEC4840B -  CyTex Deep Learning
This repository contains the experiments 
that pertain to the Final Year Project of
Calen Blake. This project is supervised by
Stephan Chalup and Ali Bahkshi of UON.

The key focus of this repository is to explore
the CyTex transform as a means of classifying 
the emotional state of a human, given a speech
sampled input.

**Script Structure:**
Preprocessing Scripts/CyTex_Image_Gen.py:
This script is used to generate CyTex images from the EMODB dataset. The input is EMODB Database/wav/ and the output is one of the RGB_IMG folders, depending on the configuration of the script. Output is in the format of a folder housing subfolders for each emotional class of the data. Multiple CyTex images will be generated for each sample from the original EMODB dataset.

Preprocessing Scripts/RAVDESS_Image_Gen.py:
Repeats the process above for the RAVDESS sataset. I/O procedure is essentially identical to that specified above.

Pilot_Testing_EMODB.py:
Used for initial parameters tuning after k-fold cross validation was implemented in CyTex_Model_T1.py. This script trains and tests the model a single time and is used for rapid tests to determine optimal DCNN parameters like learning rate and weight decay, along with data augmentations and transfer learning base model. 

CyTex_Model_T1.py:
DCNN model based on ResNet architecture for the classification of emotional classes, using EMODB CyTex images as input. Includes k-fold cross validation to supplement the creation of a validation set. Also includes sections whihc can be uncommented to see additional information about tyhe input data and its manipulations. This includes characteristics like tensor sizes, plots of training data, ... etc. 

CyTex_Model_R1.py:
DCNN model based on ResNet architecture for the classification of emotional classes, using RAVDESS CyTex images as input. Script construction based on CyTex_Model_T1.py. Simplified and condensed, discounting additional plotted information like batch samples of the training data and error testing. 
