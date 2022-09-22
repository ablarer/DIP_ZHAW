import os
from pathlib import Path

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    plt.savefig(os.path.join('Results', fileName + ' Histogram equalized.tif'))
    return


# -------------------------------------------------Applay Gamma corrections---------------------------------------------

# Source:
# https://stackoverflow.com/questions/41118808/difference-between-contrast-stretching-and-histogram-equalization
# Contrast Stretching
# It is all about increasing the difference between the maximum intensity value in an image and
# the minimum one. All the rest of the intensity values are spread out between this range.

# It works like mapping. It maps minimum intensity in the image to the minimum value in the range( 84 ==> 0 in
# the example above).
# With the same way, it maps maximum intensity in the image to the maximum value in the range( 153 ==> 255
# in the example above).

# This is why Contrast Stretching is un-reliable, if there exist only two pixels have 0 and 255 intensity,
# it is totally useless.

# In contrast stretching, there exists a one-to-one relationship of the intensity values between the source image and
# the target image i.e., the original image can be restored from the contrast-stretched image.

# Histogram Equalization
# It is about modifying the intensity values of all the pixels in the image such that the
# histogram is "flattened" (in reality, the histogram can't be exactly flattened, there would be some peaks and
# some valleys, but that's a practical problem).

# However, once histogram equalization is performed, there is no way of getting back the original image.

# See also:
# https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43

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


# -----------------------------Apply a histogram equalization algorithm-------------------------------------------------
def histogramEqualization1(myUint8, myHistogram):
    size = myUint8.shape
    numberOfPixels = size[0] * size[1]
    pdf = np.array(myHistogram).astype(np.float64) / numberOfPixels

    count = 0
    distributionArray = np.zeros(256)
    for i in range(len(pdf)):
        count += pdf[i]
        distributionArray[i] = count

    transformationFunction = distributionArray * 255

    newImage = np.zeros(size)

    if size[1] > size[0]:
        for i in range(size[1]):
            for j in range(size[0]):
                newImage[j][i] = transformationFunction[myUint8[j][i]]


    else:
        for i in range(size[0]):
            for j in range(size[1]):
                newImage[i][j] = transformationFunction[myUint8[i][j]]

    return newImage


# ---------------------------------Show the images----------------------------------------

def showTheImages(myUint8, newImage):
    plt.subplot(1, 2, 1)
    plt.imshow(myUint8, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(newImage, cmap='gray')
    plt.title('Histogram equalized Image')
    plt.tight_layout()
    save_image(extractFileName(path))
    plt.show()
    return


def showHistogram(uint8Image, newImage, histogramArray):
    plt.subplot(1, 3, 1)
    plt.imshow(uint8Image, cmap='gray')
    plt.title(extractFileName(path))
    plt.subplot(1, 3, 2)
    plt.imshow(newImage, cmap='gray')
    plt.title('Histogram equalized\n' + 'Image')
    plt.subplot(1, 3, 3)
    xList = range(len(histogramArray))
    plt.bar(xList, height=histogramArray)
    plt.title("Histogram")
    plt.tight_layout()
    save_image(extractFileName(path))
    plt.show()
    return

# --------------------------------------------Assemble the above functions----------------------------------------------

def manipulateAndShowImages(path):
    myPixels = openTifImage(path)
    myUint8 = convertToUint8(myPixels)
    myHistogram = fillHistogramArray(myUint8)
    newImage = histogramEqualization1(myUint8, myHistogram)
    showTheImages(myUint8, newImage)
    showHistogram(myUint8, newImage, myHistogram)


# --------------------------------------------1 to 0.8 Code from Internet----------------------------------------------
# Source:
# https://levelup.gitconnected.com/introduction-to-histogram-equalization-for-digital-image-enhancement-420696db9e43
def histogramEqualization2():
    img_filename = 'bloodCells.tif'
    save_filename = 'output_image.jpg'

    ######################################
    # READ IMAGE FROM FILE
    ######################################
    # load file as pillow Image
    img = Image.open(img_filename)

    # convert to grayscale
    imgray = img.convert(mode='L')

    # convert to NumPy array
    img_array = np.asarray(imgray)

    ######################################
    # PERFORM HISTOGRAM EQUALIZATION
    ######################################

    """
    STEP 1: Normalized cumulative histogram
    """
    # flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)

    # normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels

    # normalized cumulative histogram
    chistogram_array = np.cumsum(histogram_array)

    """
    STEP 2: Pixel mapping lookup table
    """
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

    """
    STEP 3: Transformation
    """
    # flatten image array into 1D list
    img_list = list(img_array.flatten())

    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]

    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    ######################################
    # WRITE EQUALIZED IMAGE TO FILE
    ######################################
    # convert NumPy array to pillow Image and write to file
    eq_img = Image.fromarray(eq_img_array, mode='L')
    eq_img.save('Results' + save_filename)


# -----------------------------------Call manipulateAndShowImages & 1 to 0.8 Code from Internet-------------------------

if __name__ == '__main__':
    path = ['pics/bloodCells.tif',
            'pics/xRayChest.tif',
            'pics/ctSkull.tif']
    for path in path:
        manipulateAndShowImages(path)
    histogramEqualization2()  # Code from Internet
