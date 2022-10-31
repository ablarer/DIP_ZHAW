import cv2
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb, rgb_to_hsv



def get_image_array(file_name, print_info=False):
    image = Image.open(file_name)
    img = np.array(image)
    if print_info:
        print("filename:", image.filename)
        print("file format:", image.format)
        print("dtype:", img.dtype)
        print("shape:", img.shape)
        print()

    return img


def show_images(num_cols, per_image_size_px, *image_arrays, cmap="viridis", title=False):
    num_rows = int(len(image_arrays) / num_cols) + 1
    fig_size = (
        num_cols * (per_image_size_px / 100),
        num_rows * (per_image_size_px / 100),
    )
    fig = plt.figure(figsize=fig_size)
    for i, img in enumerate(image_arrays, 1):
        fig.add_subplot(num_rows, num_cols, i)
        if title:
            plt.imshow(img[0], cmap=plt.get_cmap(cmap))
            plt.title(img[1])
        else:
            plt.imshow(img, cmap=plt.get_cmap(cmap))
        plt.xticks([])
        plt.yticks([])


def rgb2gray(rgb):
    return np.rint(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])).astype(np.uint8)


def rgb2hsv(r, g, b):
    return rgb_to_hsv(r, g, b)


def hsv2rgb(h, s, v):
    return hsv_to_rgb(h / 255, s / 255, v / 255)


def convert_img(img, conversion):
    img_new = np.zeros((img.shape[0], img.shape[1], 3))
    for x, y in np.ndindex(img.shape[:2]):
        img_new[x, y] = np.asarray(conversion(*img[x, y]))

    return img_new


def display_color_channels(img):
    plt.subplot(3,1,1)
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.title('Channel Red')
    plt.subplot(3, 1, 2)
    plt.imshow(img[:, :, 1], cmap="gray")
    plt.title('Channel Green')
    plt.subplot(3, 1, 3)
    plt.imshow(img[:, :, 2], cmap="gray")
    plt.title('Channel Blue')
    plt.show()


sun_noise = get_image_array("pics/sun_noise.jpg")
plt.imshow(sun_noise)
plt.title('Original image')
plt.show()


# Apply Various Filters

# Kernels
average_kernel = np.ones((3, 3)) / 9
gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# non-linear filters
def median_filter(array):
    # Good against salt and pepper noise
    return np.median(array)


def maximum_filter(array):
    return np.max(array)


def minimum_filter(array):
    return np.min(array)


def average_filter(array):
    # linear
    return np.sum(array) / array.size


def geometric_mean(array):
    # non-linear
    product = 1
    for x, y in np.ndindex(array.shape):
        product *= array[x, y]
    return product ** (1 / array.size)


def harmonic_mean(array):
    # non-linear
    temp_sum = 0
    for x, y in np.ndindex(array.shape):
        if not array[x, y] == 0:
            temp_sum += 1 / array[x, y]

    if temp_sum == 0:
        result = 0
    else:
        result = array.size / temp_sum
    return result


def get_neighbours(img, y, x, size):
    result = np.zeros((size, size))
    h_size = int((size - 1) / 2)
    return img[y - h_size:y + h_size + 1, x - h_size:x + h_size + 1]


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def pad_border(img, border_size):
    return np.pad(img, int(border_size), pad_with)


def apply_filter(img, my_filter, filter_size):
    new_img = img.copy()
    border_size = int((filter_size - 1) / 2)
    padded_img = pad_border(img, border_size)
    for x, y in np.ndindex(img.shape):
        new_img[x, y] = my_filter(get_neighbours(padded_img, x + border_size, y + border_size, filter_size))
    return new_img


def apply_kernel_to_color_image(img, kernel):
    return cv2.filter2D(src=img, kernel=kernel, ddepth=-1)

# Show original image
plt.subplot(2, 2, 1)
plt.imshow(sun_noise)
plt.title('Original image')

# Filtering on RGB on all channels together
# Apply median filter
# Median filter on RGB Channels does not yield good results.
# False colors are generated
med_sun = ndimage.median_filter(sun_noise, size=5)
plt.subplot(2, 2, 2)
plt.imshow(med_sun, cmap="gray")
plt.title('Median, RGB, channels together ')

# Filtering channel-wise on RGB
# Apply median filter
RGB_med_sun = sun_noise.copy()
for i in range(3):
    RGB_med_sun[:, :, i] = ndimage.median_filter(RGB_med_sun[:, :, i], size=5)
plt.subplot(2, 2, 3)
plt.imshow(med_sun, cmap="gray")
plt.title('Median, RGB, channels separately')


# Filtering on HSV on the v channel
# Convert RGB to HSV
HSV_sun_noise = cv2.cvtColor(sun_noise, cv2.COLOR_RGB2HSV)
HSV_med_sun = ndimage.median_filter(HSV_sun_noise, size=3)

# Apply median filter to the v channel
HSV_sun_noise[:, :, 2] = ndimage.median_filter(HSV_sun_noise[:, :, 2], size=5)

# Convert HSV to RGB
med_sun = cv2.cvtColor(HSV_sun_noise, cv2.COLOR_HSV2RGB)
plt.subplot(2, 2, 4)
plt.imshow(med_sun, cmap="gray")
plt.title('Median, HSV, only on v Channel')
plt.show()

# Filtering HSV on all channels
# Convert RGB to HSV
HSV_sun_noise = cv2.cvtColor(sun_noise, cv2.COLOR_RGB2HSV)
HSV_med_sun = ndimage.median_filter(HSV_sun_noise, size=3)

# Apply median filter
RGB_med_sun = sun_noise.copy()
for i in range(3):
    HSV_sun_noise[:, :, i] = ndimage.median_filter(HSV_sun_noise[:, :, i], size=5)
    plt.subplot(2, 2, i + 1)
    plt.imshow(HSV_sun_noise[:, :, i])
    plt.title(f'Median on HSV Channel {i}')

# Convert HSV to RGB
plt.subplot(2, 2, 4)
med_sun = cv2.cvtColor(HSV_sun_noise, cv2.COLOR_HSV2RGB)
plt.title(f'HSV to RGB')
plt.imshow(med_sun)
plt.show()


# Filtering with open cv
# Median filter 3x3
RGB_med_sun = sun_noise.copy()
RGB_med_sun_gaussian_filtered = cv2.medianBlur(RGB_med_sun, 5)
plt.title(f'Median filter 5 x 5')
plt.imshow(RGB_med_sun_gaussian_filtered)
plt.show()

# Average filter 5x5
RGB_med_sun = sun_noise.copy()
RGB_med_sun_averaged_filtered = cv2.blur(RGB_med_sun, (5, 5), 0)
plt.title(f'Average filter 5 x 5')
plt.imshow(RGB_med_sun_averaged_filtered)
plt.show()

# Gaussian filter 5x5
RGB_med_sun = sun_noise.copy()
RGB_med_sun_gaussian_filtered = cv2.GaussianBlur(RGB_med_sun, (5, 5), 0)
plt.title(f'Gaussian filter 5 x 5')
plt.imshow(RGB_med_sun_gaussian_filtered)
plt.show()

# Maximum filter
RGB_med_sun = sun_noise.copy()
for i in range(3):
    RGB_med_sun[:, :, i] = maximum_filter(RGB_med_sun[:, :, i])
plt.title(f'Maximum filter')
plt.imshow(RGB_med_sun)
plt.show()

# Minimum filter
RGB_med_sun = sun_noise.copy()
for i in range(3):
    RGB_med_sun[:, :, i] = minimum_filter(RGB_med_sun[:, :, i])
plt.title(f'Minimum filter')
plt.imshow(RGB_med_sun)
plt.show()

# Average filter
RGB_med_sun = sun_noise.copy()
for i in range(3):
    RGB_med_sun[:, :, i] = average_filter(RGB_med_sun[:, :, i])
plt.title(f'Average filter')
plt.imshow(RGB_med_sun)
plt.show()

# The image contains salt and pepper noise on every channel
# Best salt and pepper reduction is achieved with median filtering
