import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydicom as pydicom
# Importing Image and ImageOps module from PIL package
from PIL import Image as im, ImageOps
# import cv2
from pathlib import Path


# --------------------------------------------------- save the image ---------------------------------------------------
def save_images(file_name_without_extension, type=['.tiff', '.png', '.jpg', '.bmp']):
    plt.savefig(os.path.join('Results', file_name_without_extension + type[0]))
    plt.savefig(os.path.join('Results', file_name_without_extension + type[1]))
    plt.savefig(os.path.join('Results', file_name_without_extension + type[2]))

    im.open("Results/" + file_name_without_extension + ".PNG").save(
        "Results/" + file_name_without_extension + type[3])


# ----------------------------------------------------- load the image -------------------------------------------------
file_name = 'pics/lena_gray.gif'
file_name_without_extension = Path('pics/lena_gray.gif').stem
print(file_name_without_extension)

# Open the image and convert to numpy array
file_name_without_extension_numpy = mpimg.imread(file_name)

# summarize some details about the image
print('numpy array:', file_name_without_extension_numpy.dtype)
print(file_name_without_extension_numpy.shape)

# --------------------------------------------------- display the image ------------------------------------------------
fig = plt.figure(1)
plt.title('Gray-Scale Image file')
plt.imshow(file_name_without_extension_numpy, cmap='gray')

# --------------------------------------------------- save & show image(s) ---------------------------------------------------
save_images(file_name_without_extension)
plt.show()

# --------------------------------------------------- 3.1 Color image ----------------------------------------------

file_name = 'pics/lena_color.gif'
file_name_without_extension = Path('pics/lena_color.gif').stem
print(file_name_without_extension)

# Open the image and convert to numpy array
file_name_without_extension_numpy = mpimg.imread(file_name)

# summarize some details about the image
print('numpy array:', file_name_without_extension_numpy.dtype)
print(file_name_without_extension_numpy.shape)

# --------------------------------------------------- display the image ------------------------------------------------
fig = plt.figure(2)
plt.title('Color Image file')
plt.imshow(file_name_without_extension_numpy)

# --------------------------------------------------- save & show image(s) ---------------------------------------------------
save_images(file_name_without_extension)
plt.show()

# --------------------------------------------------- 3.2 Color image, color channels ------------------------------------------------
plt.subplot(2, 2, 1)
plt.title('Color Image')
plt.imshow(file_name_without_extension_numpy)

# ---- extract channels -------
redChannel = file_name_without_extension_numpy.copy()
redChannel[:, :, 1] = 0  # set blue channel to 0
redChannel[:, :, 2] = 0  # set green channel to 0

blueChannel = file_name_without_extension_numpy.copy()
blueChannel[:, :, 0] = 0  # set red channel to 0
blueChannel[:, :, 1] = 0  # set blue channel to 0

greenChannel = file_name_without_extension_numpy.copy()
greenChannel[:, :, 0] = 0  # set red channel to 0
greenChannel[:, :, 2] = 0  # set green channel to 0

# Warum geht das nicht?
image = file_name_without_extension_numpy
r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2] # for RGB images

# ------ display rgb -----
plt.subplot(2, 2, 2)
plt.title('Red Image')
plt.imshow(greenChannel)

plt.subplot(2, 2, 3)
plt.title('Green Image')
plt.imshow(redChannel)

plt.subplot(2, 2, 4)
plt.title('Blue Image')
plt.imshow(blueChannel)

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images(file_name_without_extension + ', Color channels')
plt.show()

# --------------------------------------------------- convert to greyscale ---------------------------------------------
# Using Pillow
# Creating an original_image object
original_image = im.open("./pics/lena_color.gif")

# applying grayscale method
gray_image = ImageOps.grayscale(original_image)
gray_image.save("./Results/lena_color_converted_to_gray_scale.gif")

# Using matplotlib, considering channel-dependent luminance perception
# Source: https://e2eml.school/convert_rgb_to_grayscale.html
def greyScaleConvert(original_image):
    r, g, b = original_image[:, :, 0], original_image[:, :, 1], original_image[:, :, 2]
    imgGray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return plt.imshow(imgGray, cmap='gray')

file_name = 'pics/lena_gray.gif'
file_name_without_extension = Path('pics/lena_gray.gif').stem
print(file_name_without_extension)

# Open the image and convert to numpy array
file_name_without_extension_numpy = mpimg.imread(file_name)

plt.subplot(2, 2, 1)
plt.title('Gray image')
plt.imshow(file_name_without_extension_numpy, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Red Channel')
greyScaleConvert(redChannel)

plt.subplot(2, 2, 3)
plt.title('Blue Channel')
greyScaleConvert(blueChannel)

plt.subplot(2, 2, 4)
plt.title('Green Channel')
greyScaleConvert(greenChannel)

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.show()

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images(file_name_without_extension + ', Gray scale channels')
plt.show()

# --------------------------------------------------- Ex. 3.3 Medical Images --------------------
# Install pydicom
# Read the document
dcmBrain = pydicom.dcmread('pics/brain/brain_001.dcm')
# Print meta data
print('\nDCM Metadata :', dcmBrain)

# Show corresponding image
plt.subplot(1, 3, 1)
plt.title('Brain1')
plt.imshow(dcmBrain.pixel_array, cmap=plt.cm.gray)

# Show horizontally flipped image
# https://numpy.org/doc/stable/reference/generated/numpy.flip.html
dcmBrainFlip = dcmBrain.copy()
dcmBrainFlipHori = np.flipud(dcmBrainFlip.pixel_array)

plt.subplot(1, 3, 2)
plt.title('Horizontally Flipped')
plt.imshow(dcmBrainFlipHori, cmap=plt.cm.gray)

# Show vertically flipped image
# https://numpy.org/doc/stable/reference/generated/numpy.flip.html
dcmBrainFlip = dcmBrain.copy()
dcmBrainFlipVerti = np.fliplr(dcmBrainFlip.pixel_array)

plt.subplot(1, 3, 3)
plt.title('Vertically Flipped')
plt.imshow(dcmBrainFlipVerti, cmap=plt.cm.gray)

plt.subplots_adjust(wspace=0.6)
plt.show()


# ---convert to uint8 without normalizing----
# Image information is lost, problem over/under flow of values
pixelArray = dcmBrain.pixel_array
print(f'Pixel array shape {pixelArray.shape}')
uint8Image = pixelArray.astype(np.uint8)
plt.subplot(1,3,1)
plt.title('Converted to uint8')
plt.imshow(uint8Image)

# ---convert to uint8 with normalizing----
# prevent over/under flow of values
# Another problem occurred:
# Because of the uint8 format information is lost due to the mathematical calculations
info = np.iinfo(pixelArray.dtype)
print(f'Info {info}')
rawData = pixelArray.astype(np.float64) / info.max
normalizedData = rawData * 255  # stretch to 255
uint8ImageNormalized = normalizedData.astype(np.uint8)
plt.subplot(1,3,2)
plt.title('uint8 w/ normalizing')
plt.imshow(uint8ImageNormalized)

# ---convert uint 8 to double----
# Precention of information loss by converting the uint8 to doubles
doubleUint8Image = uint8Image.astype(np.double)
plt.subplot(1,3,3)
plt.title('Converted to double')
plt.imshow(doubleUint8Image)
plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.show()


# --------------------------------------------------- Ex. 3.3.2 Image Sequence -----------------------------------------
# Variables
path = 'pics/brain'
x = 0
i = 1
brain3DArrayTraversal = []

# Show all images
for filename in os.scandir(path):
    imageBrain = pydicom.dcmread(filename)
    plt.subplot(4, 5, i)
    i += 1
    plt.imshow(imageBrain.pixel_array)

    # Stack the images
    brain3DArrayTraversal.append(np.stack(imageBrain.pixel_array))

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.show()

# Take the traversally stacked pictures and arrange them also sagittally and frontally
brain3DArrayFrontal = np.stack(brain3DArrayTraversal,-1)
brain3DArraySagital = np.stack(brain3DArrayFrontal,1)

# Combine all images
combinedImagesTraversal = [[0]*256]*256
combinedImagesFrontal =   [[0]*20]*256
combinedImagesSagital =   [[0]*20]*256

for brainImage in brain3DArrayTraversal:  # brainimage = 2D Array -> [x][y] pixelwert-array
    combinedImagesTraversal += brainImage * (1 / len(brain3DArrayTraversal)) # 1/20 of pixel value and add -> this results in average pixelvalue

for brainImage in brain3DArrayFrontal:
    combinedImagesFrontal += brainImage * (1 / len(brain3DArrayFrontal))

for brainImage in brain3DArraySagital:
    combinedImagesSagital += brainImage * (1 / len(brain3DArraySagital))

plt.subplot(3, 1, 1)
plt.imshow(combinedImagesTraversal)
plt.title('Combination of all 20 brain slices traversal')

plt.subplot(3, 1, 2)
plt.imshow(combinedImagesFrontal, aspect='auto')
plt.title('Combination of all 20 brain slices frontal')

plt.subplot(3, 1, 3)
plt.imshow(combinedImagesSagital, aspect='auto')
plt.title('Combination of all 20 brain slices sagital')

plt.subplots_adjust(wspace=0.6,
                    hspace=0.6)

plt.show()


# Let it run in the terminal

# animated_line_plot.py

from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# create empty lists for the x and y data
x = []
y = []

# create the figure and axes objects
fig, ax = plt.subplots()
# function that draws each frame of the animation
def animate(i):
    pt = randint(1,9) # grab a random integer to be the next y-value in the animation
    x.append(i)
    y.append(pt)

    ax.clear()
    ax.plot(x, y)
    ax.set_xlim([0,20])
    ax.set_ylim([0,10])

# run the animation
ani = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

im = plt.imshow(f(x, y), animated=True)


def updatefig(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()

# https://plotly.com/python/visualizing-mri-volume-slices/

# https://www.kaggle.com/code/pranavkasela/interactive-and-animated-ct-scan

