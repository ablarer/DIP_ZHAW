import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# Define frames
import plotly.graph_objects as go
import pydicom as pydicom
from IPython import display
# Importing Image and ImageOps module from PIL package
from PIL import Image as im, ImageOps
from skimage import io


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
file_name = 'pics/lena_color.gif'
file_name_without_extension = Path('pics/lena_color.gif').stem
print(file_name_without_extension)

# Open the image and convert to numpy array
file_name_without_extension_numpy = mpimg.imread(file_name)

# ---- extract channels by subtraction of not needed channels -------
redChannel = file_name_without_extension_numpy.copy()
redChannel[:, :, 1] = 0  # set blue channel to 0
redChannel[:, :, 2] = 0  # set green channel to 0

blueChannel = file_name_without_extension_numpy.copy()
blueChannel[:, :, 0] = 0  # set red channel to 0
blueChannel[:, :, 1] = 0  # set blue channel to 0

greenChannel = file_name_without_extension_numpy.copy()
greenChannel[:, :, 0] = 0  # set red channel to 0
greenChannel[:, :, 2] = 0  # set green channel to 0

# ---- extract channels by selecting the needed channel -------
image = file_name_without_extension_numpy
r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]  # for RGB images

# ------ display images -----

# -----Real gray scale images-----
plt.subplot(2, 4, 1)
plt.title('Gray Image')
plt.imshow(mpimg.imread('pics/lena_gray.gif'), cmap='gray')

plt.subplot(2, 4, 2)
plt.title('Red Image')
plt.imshow(r, cmap='gray')

plt.subplot(2, 4, 3)
plt.title('Green Image')
plt.imshow(g, cmap='gray')

plt.subplot(2, 4, 4)
plt.title('Blue Image')
plt.imshow(b, cmap='gray')

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)

# -----"Fake" gray scale color images-----
plt.subplot(2, 4, 5)
plt.title('RGB-Channels')
composed_grey_channels = (redChannel + greenChannel + blueChannel)
plt.imshow(composed_grey_channels)

plt.subplot(2, 4, 6)
plt.title('R-Channel')
plt.imshow(redChannel)

plt.subplot(2, 4, 7)
plt.title('G-Channel')
plt.imshow(greenChannel)

plt.subplot(2, 4, 8)
plt.title('B-Channel')
plt.imshow(blueChannel)

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images(file_name_without_extension + ', Extracted gray scale and fake-colored channels')
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

# Gray scale images
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

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images('Converted color channels to gray scale')
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

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images('Medical non-flipped and flipped pictures')
plt.show()

# ---convert to uint8 without normalizing----
# Image information is lost, problem over/under flow of values
pixelArray = dcmBrain.pixel_array
print(f'Pixel array shape {pixelArray.shape}')
uint8Image = pixelArray.astype(np.uint8)
plt.subplot(1, 3, 1)
plt.title('Converted to uint8')
plt.imshow(uint8Image)

# ---convert to uint8 with normalizing----
# prevent over/under flow of values
# Another problem occurred:
# Because of the uint8 format, information is lost due to the mathematical calculations
info = np.iinfo(pixelArray.dtype)
print(f'Info {info}')
rawData = pixelArray.astype(np.float64) / info.max
normalizedData = rawData * 255  # stretch to 255
uint8ImageNormalized = normalizedData.astype(np.uint8)
plt.subplot(1, 3, 2)
plt.title('uint8 w/ normalizing')
plt.imshow(uint8ImageNormalized)

# ---convert uint 8 to double----
# Presentation of information loss by converting the uint8 to doubles
doubleUint8Image = uint8Image.astype(np.double)
plt.subplot(1, 3, 3)
plt.title('Converted to double')
plt.imshow(doubleUint8Image)
plt.subplots_adjust(wspace=0.6, hspace=0.6)

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images('Medical image conversions')
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
# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images('All transversal medical images')
plt.show()

# Take the transversally stacked pictures and arrange them also sagittally and frontally
brain3DArrayFrontal = np.stack(brain3DArrayTraversal, -1) # Stack column-wise
brain3DArraySagital = np.stack(brain3DArrayFrontal, 1)  # Stack row-wise

# Combine all images
combinedImagesTraversal = [[0] * 256] * 256
combinedImagesFrontal   = [[0] * 20] * 256
combinedImagesSagital   = [[0] * 20] * 256

for brainImage in brain3DArrayTraversal:  # brain image = 2D Array -> [x][y] pixelwert-array
    combinedImagesTraversal += brainImage * (
            1 / len(brain3DArrayTraversal))  # 1/20 of pixel value and add -> This results in average pixel value

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

# --------------------------------------------------- save & show image(s) ---------------------------------------------
save_images('Stacked medical images')
plt.show()

# -------------------------------------------------- Experimental Animations -------------------------------------------

# *******Let the following run in the terminal*********
# use python3 3.1 Histogram and getting_started.py
# Close one window view after the other to see the next visualization

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
    pt = randint(1, 9)  # grab a random integer to be the next y-value in the animation
    x.append(i)
    y.append(pt)

    ax.clear()
    ax.plot(x, y)
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 10])


# run the animation
ani = FuncAnimation(fig, animate, frames=20, interval=500, repeat=False)

writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("Results/Animated line plot.mp4", writer=writer)

plt.show()

# -------------------------------------------------- Experimental Animations -------------------------------------------

# *******Let the following run in the terminal*********
# use python3 3.1 Histogram and getting_started.py
# Close one window view after the other to see the next visualization

# shows a sort of volcano lamp

fig, ax = plt.subplots()


def f(x, y):
    return np.sin(x) + np.cos(y)


x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = ax.imshow(f(x, y), animated=True)
    if i == 0:
        ax.imshow(f(x, y))  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("Results/Volcano Lamp.mp4", writer=writer)

plt.show()

# -------------------------------------------------- Experimental Animations -------------------------------------------

# *******Let the following run in the terminal*********
# use python3 3.1 Histogram and getting_started.py
# Close one window view after the other to see the next visualization

# initializing a figure
fig, ax = plt.subplots()

# Create images from directory to show
path = 'pics/brain'
imageBrain = []
for filename in os.scandir(path):
    imageBrain_item = pydicom.dcmread(filename)
    imageBrain_item_pixel_array = imageBrain_item.pixel_array
    imageBrain.append(imageBrain_item_pixel_array)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(20):
    im = ax.imshow(imageBrain[i], animated=True)
    if i == 0:
        ax.imshow(imageBrain[1])  # show an initial one first
    ims.append([im])

# Make animation
ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)

writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save("Results/Brain scan series.mp4", writer=writer)

plt.show()

## The following below works probably only in Jupyter Notebook

# converting to an html5 video
video = ani.to_html5_video()

# embedding for the video
html = display.HTML(video)

# draw the animation
display.display(html)
plt.close()

## The following above works probably only in Jupyter Notebook

# -------------------------------------------------- Experimental Animations -------------------------------------------
# *******Let the following run in the terminal*********
# https://plotly.com/python/visualizing-mri-volume-slices/

# --------------------------------------------------- Compose the image ------------------------------------------------

# Todo: compose the Combined Transversal Image.tiff for the following code
# imageBrain is already an array with the 20 individual transversal pictures

# ----------------------------------------------------- load the image -------------------------------------------------
vol = io.imread('pics/attention-mri.tif')
# vol = io.imread('Results/Combined Transversal Image.tiff') # Geht nicht
volume = vol.T
r, c = volume[0].shape

nb_frames = 68  # 20 for the "Combined Transversal Image.tiff"

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=(6.7 - k * 0.1) * np.ones((r, c)),
    surfacecolor=np.flipud(volume[67 - k]),
    cmin=0, cmax=200
    ),
    name=str(k)  # you need to name the frame for the animation to behave properly
    )
    for k in range(nb_frames)])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=6.7 * np.ones((r, c)),
    surfacecolor=np.flipud(volume[67]),
    colorscale='Gray',
    cmin=0, cmax=200,
    colorbar=dict(thickness=20, ticklen=4)
))


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]

# Layout
fig.update_layout(
    title='Slices in volumetric data',
    width=600,
    height=600,
    scene=dict(
        zaxis=dict(range=[-0.1, 6.8], autorange=False),
        aspectratio=dict(x=1, y=1, z=1),
    ),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders
)

fig.show()

# https://www.kaggle.com/code/pranavkasela/interactive-and-animated-ct-scan
