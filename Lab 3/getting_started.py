import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from PIL import Image
import PIL

import cv2

xRayChest = cv2.imread('pics/xRayChest.tif')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(xRayChest, cv2.COLOR_BGR2RGB))
plt.title('X-Ray Chest')
plt.show()

# ----------------------------------------------------- load cell image ------------------------------------------------------------------------------
file_name = 'pics/xRayChest.tif'
print('Pillow Version:', PIL.__version__)

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
gray_pixels = np.array(Image.open(file_name))

# summarize some details about the image
print(image.format)
print('numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)

# -------------------------------------------------- compute histogram -------------------------------------------------
my_filter = np.ones((9, 9)).astype(float) / 81.  # averaging filter
new_image = signal.convolve2d(gray_pixels.astype(float), my_filter, 'same')
new_image = new_image.astype(int)

# ---------------------------------------------------- display images --------------------------------------------------
fig = plt.figure(1)
plt.subplot(2, 1, 1)
plt.title('Gray-Scale Image')
plt.imshow(gray_pixels, cmap='gray')

ax = fig.add_subplot(2, 1, 2)
plt.title('Filtered Image')
plt.imshow(new_image, cmap='gray')

plt.show()

# --------------------------------------------------------- Task 3.1 --------------------------------------------------
#
# ----------------------------------------- Open the image with pillow and convert to numpy array-----------------------
def open_image(file_name):
    gray_pixels = np.array(Image.open(file_name))
    return gray_pixels

# ------------------------------------------------------------Define filters--------------------------------------------

# Average filter
def average_filter():
    average_filter = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    return average_filter


# Gaussian filter
def gaussian_filter():
    gaussian_filter = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    return gaussian_filter

# Laplace filter, sharpening image
def laplace_filter():
    laplace_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return laplace_filter

# ---------------------------------------------------------Plot pictures & histograms----------------------------------------------

def plot_pict_and_hist(gray_pixels, old_image, new_image, title_one="Orignal", title_two="New Image 1",  title_three="New Image 2"):

    intensity_values = np.array([x for x in range(256)])

    plt.subplot(3, 2, 1)
    plt.imshow(gray_pixels, cmap='gray')
    plt.title(title_one)

    plt.subplot(3, 2, 3)
    plt.imshow(old_image, cmap='gray')
    plt.title(title_two)

    plt.subplot(3, 2, 5)
    plt.imshow(new_image, cmap='gray')
    plt.title(title_three)
    plt.xlabel('Pictures')

    plt.subplot(3, 2, 2)
    old_image = old_image.astype('uint8')
    hist = cv2.calcHist([gray_pixels], [0], None, [256], [0, 256])
    plt.bar(intensity_values, hist[:, 0], width=1)  # width = bin size
    plt.title(title_one)

    plt.subplot(3, 2, 4)
    old_image = old_image.astype('uint8')
    hist = cv2.calcHist([old_image], [0], None, [256], [0, 256])
    plt.bar(intensity_values, hist[:, 0], width=1) # width = bin size
    plt.title(title_two)

    plt.subplot(3, 2, 6)
    new_image = new_image.astype('uint8')
    hist = cv2.calcHist([new_image], [0], None, [256], [0, 256])
    plt.bar(intensity_values, hist[:, 0], width=1)
    plt.title(title_three)
    plt.xlabel('Intensity')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1,
                        hspace=0.5)
    plt.show()




# -----------------------------------------------------Apply filters----------------------------------------------------


def apply_filters(gray_pixels):
    average_Image = signal.convolve2d(gray_pixels.astype(float), average_filter(), 'same')
    average_Image = average_Image.astype(int)
    print('Average image', average_Image)

    gaussian_image = signal.convolve2d(gray_pixels.astype(float), gaussian_filter(), 'same')
    gaussian_image = gaussian_image.astype(int)
    print('Gaussian image', gaussian_image)

    return average_Image, gaussian_image

# --------------------------------------------------------- Task 3.2 --------------------------------------------------

def apply_high_pass_filter(picture_original, picture_two):
    high_passed_filtered_picture = picture_original - picture_two
    print('High pass image', high_passed_filtered_picture)

    return high_passed_filtered_picture

# --------------------------------------------------------- Task 3.3 --------------------------------------------------

def apply_laplace_filter(gray_pixels):
    laplace_Image = signal.convolve2d(gray_pixels.astype(float), laplace_filter(), 'same')
    laplace_Image = laplace_Image.astype(int)
    print('Laplace image', laplace_Image)

    return laplace_Image


#------------------------------------------------------------Main-------------------------------------------------------

if __name__ == '__main__':
    # Task 3.1
    gray_pixels = open_image('pics/ctSkull.tif')
    average_Image, gaussian_Image = apply_filters(gray_pixels)
    plot_pict_and_hist(gray_pixels, average_Image, gaussian_Image, "Gray pixel", "Average filtering", "Gaussian filtering")

    # Task 3.2
    gray_pixels = open_image('pics/ctSkull.tif')
    average_Image, gaussian_Image = apply_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_Image)
    plot_pict_and_hist(gray_pixels, average_Image, high_passed_filtered_picture, "Gray pixel", "Low pass", "High pass")

    gray_pixels = open_image('pics/xRayChest.tif')
    average_Image, gaussian_Image = apply_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_Image)
    plot_pict_and_hist(gray_pixels, average_Image, high_passed_filtered_picture, "Gray pixel", "Low pass", "High pass")

    # Impact of low versus high pass filtering
    # Low pass filter
    # - Used for smoothing the image
    # - Attenuates high and preserves low frequencies
    # - Low frequencies pass through
    # High pass filter
    # - Used for sharpening the image
    # - Attenuates low and preserves high frequencies
    # - High frequencies pass through

    # Task 3.3
    gray_pixels = open_image('pics/ctSkull.tif')
    laplace_filtered_picture = apply_laplace_filter(gray_pixels)
    average_Image, gaussian_Image = apply_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_Image)
    plot_pict_and_hist(gray_pixels, laplace_filtered_picture, high_passed_filtered_picture, "Gray pixel", "Laplace", "High pass")

    gray_pixels = open_image('pics/xRayChest.tif')
    laplace_filtered_picture = apply_laplace_filter(gray_pixels)
    average_Image, gaussian_Image = apply_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_Image)
    plot_pict_and_hist(gray_pixels, laplace_filtered_picture, high_passed_filtered_picture, "Gray pixel", "Laplace", "High pass")

    # TODO Comment to 2, 3.3.3, - 3.3.5





