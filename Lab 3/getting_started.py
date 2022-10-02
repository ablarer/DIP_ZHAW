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
print('OpenCV Version:', cv2.__version__)

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

# ----------------------------------------------Convert image at any type to uint8--------------------------------------

def convert_any_type_to_uint8(picture):
    info = np.iinfo(picture.dtype)  # Get the information of the incoming image type
    picture = picture.astype(np.float64) / info.max  # normalize the data to 0 - 1
    picture = 255 * picture  # Now scale by 255
    picture = picture.astype(np.uint8)
    return picture

# ---------------------------------------------------------Show pictures-------------------------------------------------


# The function cv2.imshow() is used to display an image in a window.
def show_image(picture):
    cv2.imshow('graycsale image', picture)

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(0)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
    cv2.destroyAllWindows()

    # The function cv2.imwrite() is used to write an image.
    # cv2.imwrite('grayscale.jpg',img_grayscale)



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

def plot_pict_and_hist(gray_pixels, filtered_1, filtered_2, title_one="Orignal", title_two="New Image 1", title_three="New Image 2"):

    intensity_values = np.array([x for x in range(256)])

    plt.subplot(3, 2, 1)
    plt.imshow(gray_pixels, cmap='gray')
    plt.title(title_one)

    plt.subplot(3, 2, 3)
    plt.imshow(filtered_1, cmap='gray')
    plt.title(title_two)

    plt.subplot(3, 2, 5)
    plt.imshow(filtered_2, cmap='gray')
    plt.title(title_three)

    plt.xlabel('Pictures')

    plt.subplot(3, 2, 2)
    gray_pixels_converted = gray_pixels.astype('uint8')
    hist = cv2.calcHist([gray_pixels_converted], [0], None, [256], [0, 256])
    plt.bar(intensity_values, hist[:, 0], width=1)  # width = bin size
    plt.title(title_one)

    plt.subplot(3, 2, 4)
    filtered_1_converted = filtered_1.astype('uint8')
    hist = cv2.calcHist([filtered_1_converted], [0], None, [256], [0, 256])
    plt.bar(intensity_values, hist[:, 0], width=1)
    plt.title(title_two)

    plt.subplot(3, 2, 6)
    filtered_2_converted = filtered_2.astype('uint8')
    hist = cv2.calcHist([filtered_2_converted], [0], None, [256], [0, 256])
    plt.bar(intensity_values, hist[:, 0], width=1)
    plt.title(title_three)
    plt.xlabel('Intensity')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1,
                        hspace=0.5)
    plt.show()


# -----------------------------------------------------Apply filters----------------------------------------------------


def average_and_gaussian_filters(gray_pixels):
    average_Image = signal.convolve2d(gray_pixels.astype(float), average_filter(), 'same')
    average_Image = average_Image.astype(int)
    # print('Average image', average_Image)

    gaussian_image = signal.convolve2d(gray_pixels.astype(float), gaussian_filter(), 'same')
    gaussian_image = gaussian_image.astype(int)
    # print('Gaussian image', gaussian_image)

    return average_Image, gaussian_image

# --------------------------------------------------------- Task 3.2 --------------------------------------------------

def apply_high_pass_filter(picture_original, picture_two):
    high_passed_filtered_picture = picture_original - picture_two
    # print('High pass image', high_passed_filtered_picture)

    return high_passed_filtered_picture

# --------------------------------------------------------- Task 3.3 --------------------------------------------------

def apply_laplace_filter(gray_pixels):
    laplace_Image = signal.convolve2d(gray_pixels.astype(float), laplace_filter(), 'same')
    laplace_Image = laplace_Image.astype(int)
    # print('Laplace image', laplace_Image)

    return laplace_Image

def sharpen_image_1(gray_pixels, laplace_filtered_picture):
    sharpend_image_1 = gray_pixels + laplace_filtered_picture
    return sharpend_image_1

def sharpen_image_2(gray_pixels, high_pass_filtered_picture):
    sharpend_image_2 = gray_pixels + high_pass_filtered_picture
    return sharpend_image_2

def sharpen_image_3(gray_pixels, gaussian_Image):
    sharpend_image_3 = gray_pixels + gaussian_Image
    return sharpend_image_3




#------------------------------------------------------------Main-------------------------------------------------------

if __name__ == '__main__':
    # Task 3.1
    gray_pixels = open_image('pics/ctSkull.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    plot_pict_and_hist(gray_pixels, average_image, gaussian_image, "Gray pixel", "Average Filter", "Gaussian Filter")

    # Task 3.2
    gray_pixels = open_image('pics/ctSkull.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_image)
    plot_pict_and_hist(gray_pixels, average_image, high_passed_filtered_picture, "Gray Pixel", "Average Filter / Low pass", "High Pass")

    gray_pixels = open_image('pics/xRayChest.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_image)
    plot_pict_and_hist(gray_pixels, average_image, high_passed_filtered_picture, "Gray Pixel", "Average Filter / Low pass", "High Pass")

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
    # 3.3.1 + # 3.3.2
    gray_pixels = open_image('pics/ctSkull.tif')
    laplace_filtered_picture = apply_laplace_filter(gray_pixels)
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_image)
    plot_pict_and_hist(gray_pixels, laplace_filtered_picture, high_passed_filtered_picture, "Gray Pixel", "Laplace Filter", "High Pass")

    gray_pixels = open_image('pics/xRayChest.tif')
    laplace_filtered_picture = apply_laplace_filter(gray_pixels)
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_image)
    plot_pict_and_hist(gray_pixels, laplace_filtered_picture, high_passed_filtered_picture, "Gray Pixel", "Laplace Filter", "High Pass")

    # 3.3.3
    # Sharpen image with high Laplace filter
    gray_pixels = open_image('pics/ctSkull.tif')
    laplace_filtered_picture = apply_laplace_filter(gray_pixels)
    sharpend_image_1 = sharpen_image_1(gray_pixels, laplace_filtered_picture)
    plot_pict_and_hist(gray_pixels, laplace_filtered_picture, sharpend_image_1, "Gray Pixel", "Laplace Filter", "Sharpend Image")

    gray_pixels = open_image('pics/xRayChest.tif')
    laplace_filtered_picture = apply_laplace_filter(gray_pixels)
    sharpend_image_1 = sharpen_image_1(gray_pixels, laplace_filtered_picture)
    plot_pict_and_hist(gray_pixels, laplace_filtered_picture, sharpend_image_1, "Gray Pixel", "Laplace Filter", "Sharpend Image")

    # 3.3.4
    # Sharpen image with high pass filter
    gray_pixels = open_image('pics/ctSkull.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_image)
    sharpend_image_2 = sharpen_image_2(gray_pixels, high_passed_filtered_picture)
    plot_pict_and_hist(gray_pixels, high_passed_filtered_picture, sharpend_image_2, "Gray Pixel", "High Pass", "Sharpend Image")

    gray_pixels = open_image('pics/xRayChest.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    high_passed_filtered_picture = apply_high_pass_filter(gray_pixels, average_image)
    sharpend_image_2 = sharpen_image_2(gray_pixels, high_passed_filtered_picture)
    plot_pict_and_hist(gray_pixels, high_passed_filtered_picture, sharpend_image_2, "Gray Pixel", "High Pass", "Sharpend Image")

    # 3.3.5
    # Sharpen image by applying Gaussian unsharp masking instead of Laplace filtering
    gray_pixels = open_image('pics/ctSkull.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    sharpend_image_3 = sharpen_image_3(gray_pixels, gaussian_image)
    plot_pict_and_hist(gray_pixels, gaussian_image, sharpend_image_3, "Gray Pixel", "Gaussian Filter", "Sharpend Image")

    gray_pixels = open_image('pics/xRayChest.tif')
    average_image, gaussian_image = average_and_gaussian_filters(gray_pixels)
    sharpend_image_3 = sharpen_image_3(gray_pixels, gaussian_image)
    plot_pict_and_hist(gray_pixels, gaussian_image, sharpend_image_3, "Gray Pixel", "Gaussian Filter", "Sharpend Image")


# TODO Add comments and 3.4, check if histograms and pictures are accurate.
# TODO Problem: Some images are in int64 others in uint8
# TODO Bilder einzeln mit treffenden Namen speichern.

    # 3.4

    # Supplemental test
    # The function cv2.imread() is used to read an image.
    gray_pixels = cv2.imread('pics/ctSkull.tif', 0)
    show_image(convert_any_type_to_uint8(gray_pixels))



