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
    plt.savefig(os.path.join('Results', fileName + ' Gamma correction.tif'))
    return


# -------------------------------------------------Applay Gamma corrections---------------------------------------------

# Source:
# https://www.cambridgeincolour.com/tutorials/gamma-correction.htm

# https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# There are two (easy) ways to apply gamma correction using OpenCV and Python. The first method is to simply leverage
# the fact that Python + OpenCV represents images as NumPy arrays. All we need to do is scale the pixel intensities to
# the range [0, 1.0], apply the transform, and then scale back to the range [0, 255].
# Overall, the NumPy approach involves a division, raising to a power, followed by a multiplication — this tends to be
# very fast since all these operations are vectorized.
#
# However, there is an even faster way to perform gamma correction thanks to OpenCV.
# All we need to do is build a table (i.e. dictionary) that maps the input pixel values to the output gamma corrected
# values. OpenCV can then take this table and quickly determine the output value for a given pixel in O(1) time.
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


# -----------------------------Scale the pixel array within a range of [0, 1.0]-----------------------------------------
def scaleToZeroOne(pixelArray):
    info = np.iinfo(pixelArray.dtype)  # Get the minimum and maximum values of the pixelArray
    scaledPixels = pixelArray.astype(np.float64) / info.max
    return scaledPixels


# -----------------------------------------Calculate the gamma correction-----------------------------------------------

def correctGamma(image, gamma):
    imageSize = image.shape
    gammaCorrectedImage = np.zeros(imageSize)  # template image array for receiving the calculated values


    if imageSize[1] > imageSize[0]:
        for i in range(imageSize[1]):
            for j in range(imageSize[0]):
                r = image[j][i]
                s = pow(r, gamma)
                gammaCorrectedImage[j][i] = s

    else:
        for i in range(imageSize[0]):
            for j in range(imageSize[1]):
                r = image[i][j]
                s = pow(r, gamma)
                gammaCorrectedImage[i][j] = s

    return gammaCorrectedImage


# ---------------------------------Show the original and gamma correct4ed images----------------------------------------

def showOriginalandGammaCorrectedImages(scaledImage, correctedImage, gamma):
    plt.subplot(1, 2, 1)
    plt.imshow(scaledImage, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(correctedImage, cmap='gray')
    plt.title('Gamma-corrected Image ' + str(gamma))
    plt.tight_layout()
    save_image(extractFileName(path) + ' Gamma ' + str(gamma))
    plt.show()


# --------------------------------------------Assemble the above functions----------------------------------------------

def correctGammaAndShowImages(path, gamma):
    myImage = openTifImage(path)
    scaledImage = scaleToZeroOne(myImage)
    correctedImage = correctGamma(scaledImage, gamma)
    showOriginalandGammaCorrectedImages(scaledImage, correctedImage, gamma)


# ------------------------------------------Call correctGammaandShowImages----------------------------------------------

if __name__ == '__main__':
    gamma = [0.5, 1., 1.5, 2.]
    lenght_gamma = len(gamma)
    i = 0
    path = ['pics/bloodCells.tif',
            'pics/xRayChest.tif',
            'pics/ctSkull.tif']
    for path in path:
        for i in range(lenght_gamma):
            correctGammaAndShowImages(path, gamma[i])
