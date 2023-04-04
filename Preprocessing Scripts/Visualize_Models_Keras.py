"""
# -----------------------------------
# Plot box view of models... Poor visualization, however.
#
#
# Author: Calen Blake
# Date: 04-04-23
# NOTE: No equivalent method exists for completing this process in PyTorch
# Thus, this script operates independently of all others in the repository.
# -----------------------------------
"""
from keras import applications
from keras.utils import plot_model
import visualkeras
import matplotlib
from matplotlib import pyplot as plt
import image
import tensorflow as tf
from tensorflow import keras
# from keras.models import Sequential
# from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
# from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras import regularizers, optimizers

rn50 = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

matplotlib.use("TkAgg")
im = visualkeras.layered_view(rn50, legend=True)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.imshow(im)
plt.show()

# NOTE: Models are too big for clean visualization, try Draw.io approach

# cytex_rn50 = keras.applications.ResNet50(
#     # don't include the final pooling layer and fc layer
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000
# )

