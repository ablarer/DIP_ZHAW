import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb, rgb_to_hsv

#----------------------------------------Open and convert Images to np.array--------------------------------------------

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

#----------------------------------------Show Images---------------------------------------------------------------------

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
    plt.show()

#-----------------------------------------Show images--------------------------------------------------------------------

def rgb2gray(rgb):
    return np.rint(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])).astype(np.uint8)

def rgb2hsv(r, g, b):
    return rgb_to_hsv(r,g,b)

def hsv2rgb(h,s,v):
    return hsv_to_rgb(h/255,s/255,v/255)

def convert_img(img, conversion):
    img_new = np.zeros((img.shape[0], img.shape[1], 3))
    for x,y in np.ndindex(img.shape[:2]):
        img_new[x, y] = np.asarray(conversion(*img[x, y]))

    return img_new

def display_colorchannels(img, channel_names=("RED", "GREEN", "BLUE"), return_ch=False):
    plt.imshow(img)
    channels = [(channel_names[ch], img[:,:,ch]) for ch in range(3)]
    show_images(3, 500, *channels, cmap="gray", title=True)

    if return_ch:
        return [ch[1] for ch in channels]


#-----------------------------------------------Load Image-------------------------------------------------------------
image_path_1 = 'pics/landscape_1.png'
landscape_1 = get_image_array(image_path_1)
plt.imshow(landscape_1)
plt.title('Landscape')
plt.show()


def gamma_correct(channel, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(channel, table)

def show_gamma_range(img, low, high, steps):
    tests = []
    if low < 1 and high > 1:
        steps = steps / 2

    low_steps = (1 - low) / (steps)
    high_steps = (high-1) / (steps)

    if low < 1:
        for gamma_val in np.arange(low, 1, low_steps):
            gamma_val = round(gamma_val, 2)
            new_img = gamma_correct(img, gamma_val)
            tests.append((new_img, f"gamma = {gamma_val}"))

    if high > 1:
        for gamma_val in np.arange(1, high, high_steps):
            new_img = gamma_correct(img, gamma_val)
            tests.append((new_img, f"gamma = {gamma_val}"))

    show_images(6, 300, *tests, cmap="gray", title=True)
    """
    fig = plt.figure(figsize=(2, 6))
    for tests in range(1, 9, 1):
        fig.add_subplot(6, 2, i)
        plt.imshow(tests[tests], cmap="gray")
    plt.title('Gamma Correction Series')
    plt.show()
    """

sw_img = rgb2gray(landscape_1)
show_gamma_range(sw_img, 1, 3, 12)

plt.imshow(gamma_correct(sw_img, 2.3), cmap="gray")
plt.title('Gamma Corrected Image')
plt.show()


# Adjust gamma on all channels
adjusted_overall_gamma = gamma_correct(landscape_1, 2.3)
show_images(1, 1000, adjusted_overall_gamma)


# Adjust gamma on single channel
channel_to_correct = 0
adj_single_ch = landscape_1.copy()
adj_single_ch[:,:,channel_to_correct] = gamma_correct(adj_single_ch[:,:,channel_to_correct], 2.3)
show_images(1, 1000, adj_single_ch)


# Histogram Adjustment Color Image
def generate_histogram(img):
    min_val = np.min(img)
    max_val = np.max(img)
    hist = np.zeros(max_val - min_val + 1)

    for val in np.nditer(img):
        hist[val] += 1

    return hist

def generate_distribution(img):
    max_val_dtype = 255
    hist = generate_histogram(img)
    hist = np.cumsum(hist)
    max_count = np.max(hist)

    for i, val in enumerate(hist):
        hist[i] = val * max_val_dtype / max_count

    return hist


def equalize_histogram(img):
    img = img.astype(np.uint8)
    img_new = img.copy().astype(np.uint8)
    distr = generate_distribution(img.astype(np.uint8))

    for x, y in np.ndindex(img_new.shape):

        val = img[x, y]
        img_new[x, y] = distr[val]

    return img_new

overal_equ = landscape_1.copy()

for ch_no in range(3):
    ch = overal_equ[:,:, ch_no]
    overal_equ[:,:, ch_no] = equalize_histogram(ch)
show_images(1, 1000, overal_equ)
plt.imshow(overal_equ)
plt.title('Overall equalized histogram')
plt.show()


# Color change quiet a bit
# Convert to HSV, equalize the Value Channel and back to rgb
HSV_landscape = cv2.cvtColor(landscape_1, cv2.COLOR_RGB2HSV)

equ_val = equalize_histogram(HSV_landscape[:,:,2])
show_images(2, 800, HSV_landscape[:,:,2], equ_val, cmap="gray")

HSV_landscape[:,:,2] = equ_val

show_images(1, 1000, cv2.cvtColor(HSV_landscape, cv2.COLOR_HSV2RGB))
plt.imshow(overal_equ)
plt.title('')
plt.show()

# Also Equalize hue channel
equ_h = equalize_histogram(HSV_landscape[:,:,0])

HSV_landscape[:,:,0] = equ_h
show_images(1, 1000, cv2.cvtColor(HSV_landscape, cv2.COLOR_HSV2RGB))

