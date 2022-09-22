import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import PIL

# ----------------------------------------------------- load cell image ------------------------------------------------
file_name = 'pics/xRayChest.tif'
print('Pillow Version:', PIL.__version__)

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
gray_pixels = np.array(Image.open(file_name))

# summarize some details about the image
print(image.format)
print('numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)


# --------------------------------------------------- save the image ---------------------------------------------------
def save_image(fileName):
    plt.savefig(os.path.join('Results', fileName + ' Histogram.tif'))
    return


# -------------------------------------------------Get histograms-------------------------------------------------------


# Source:
# https://
# dsp.stackexchange.com/questions/54647/python-how-to-compute-the-gray-level-histogram-features-as-mentioned-in-the-pap
#
# Load the image
# Get the histogram of the Red Channel (call it 𝐻𝑟)
# Get the histogram of the Green Channel (call it 𝐻𝑔)
# Get the histogram of the Blue Channel (call it 𝐻𝑏)
# Produce the combined histogram as 𝐻=𝐻𝑟+𝐻𝑏+𝐻𝑔
# Get the 98% of its mass:
# Sort 𝐻 in reverse order of values (the largest count comes first)
# Get the cumulative sum of the histogram values until you have covered the 98% of the total sum.
# Find the minimum and maximum values of the indices of the values that contribute to the sum
# Calculate the 𝑞𝑐𝑡 as the difference between the maximum and minimum of the indices.
#
# ----------------------------------------------------- Load the image -------------------------------------------------

def openTifImage(path):
    fileName = path
    gray_pixels = np.array(Image.open(fileName))
    return gray_pixels


# -------------------------------------------------Get file name from path ---------------------------------------------
def extractFileName(path):
    fileName = Path(path).stem
    return fileName


# ------------------------- Separate the red, green, and blue Channel (𝐻𝑟, Hg, Hb) -------------------------------------
def separateChannels(rgb_image):
    Hr, Hg, Hb = image[:, :, 0], image[:, :, 1], image[:, :, 2]  # for RGB images
    return Hr, Hg, Hb


# -----------------------------Convert pixel array to uint8 and normalise the array-------------------------------------
def convertToUint8(pixelArray):
    info = np.iinfo(pixelArray.dtype)  # Get the minimum and maximum values of the pixelArray
    rawData = pixelArray.astype(np.float64) / info.max  # Divide all values by the maximum value
    normalizedData = rawData * 255  # Scale all values back to 255
    uint8ImageNormalized = normalizedData.astype(np.uint8)
    return uint8ImageNormalized


# --------------------------------------Fill array for histogram with values--------------------------------------------

def fillHistogramArray(uint8Image):
    histogramArray = [0] * 256
    imageSize = uint8Image.shape
    if imageSize[1] > imageSize[0]:
        for i in range(imageSize[1]):
            for j in range(imageSize[0]):
                value = uint8Image[j][i]
                histogramArray[value] += 1
    else:
        for i in range(imageSize[0]):
            for j in range(imageSize[1]):
                value = uint8Image[i][j]
                histogramArray[value] += 1
    return histogramArray


# -----------------------Show the image and its respective histogram for its chosen color channel-----------------------

def showHistogram(uint8Image, histogramArray):
    plt.subplot(1, 2, 1)
    plt.imshow(uint8Image, cmap='gray')
    plt.title(extractFileName(path))
    plt.subplot(1, 2, 2)
    xList = range(len(histogramArray))
    plt.bar(xList, height=histogramArray)
    plt.title("Histogram")
    plt.tight_layout()
    save_image(extractFileName(path))
    plt.show()
    return


# --------------------------------------------Assemble the above functions----------------------------------------------

def fillAndShowHistogram(path):
    myPixels = openTifImage(path)
    myUint8 = convertToUint8(myPixels)
    myHistogram = fillHistogramArray(myUint8)
    showHistogram(myUint8, myHistogram)
    return


# --------------------------------------------Call fillAndShowHistogram-------------------------------------------------

if __name__ == '__main__':

    path = ['pics/bloodCells.tif',
            'pics/xRayChest.tif',
            'pics/ctSkull.tif']
    for path in path:
        fillAndShowHistogram(path)
