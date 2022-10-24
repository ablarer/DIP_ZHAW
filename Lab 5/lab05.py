import cv2
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt
from scipy import ndimage


# ----------------------------------------Open and convert Images to np.array--------------------------------------------

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


# ----------------------------------------Show Images---------------------------------------------------------------------

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


# -----------------------------------------Show images--------------------------------------------------------------------

blood_cells = get_image_array("pics/bloodCells.png", True)
squares = get_image_array("pics/squares.tif", True)
show_images(2, 500, (squares, "Squares with sizes 1x1 to 15x15"), (blood_cells, "Blood cells"), cmap="gray", title=True)
# plt.imshow(squares)
plt.show()

# ----------------------------------Task 3.1 Locate and Squares of Given Size--------------------------------------------
squares = cv2.imread("pics/squares.tif", 0)
print('Type of squares: ', type(squares))

plt.subplot(2, 2, 1)
plt.imshow(squares)
plt.title('Original image')

# Kernel defintion
kernel = np.ones((6, 6), np.uint8)

# https://www.tutorialspoint.com/opencv/opencv_morphological_operations.htm
# The morphologyEx() of the method of the class Imgproc is used to perform
# erosion and dilation on a given image.
# morphologyEx(src, dst, op, kernel)
# cv2.MORPH_TOPHAT -> Type of blurr filter

plt.subplot(2, 2, 2)
no_bigs = cv2.morphologyEx(squares, cv2.MORPH_TOPHAT, kernel)
plt.imshow(no_bigs)
plt.title('Applied blur filter MORPH_TOPHAT')

# Kernel definition
kernel = np.ones((5, 5), np.uint8)

plt.subplot(2, 2, 3)
# Erosion: Bright areas get smaller
only_5s = cv2.erode(no_bigs, kernel, iterations=1)
plt.imshow(only_5s)
plt.title('Eroded small edges')

plt.subplot(2, 2, 4)
# Dilation: Bright areas get broader
result = cv2.dilate(only_5s, kernel, iterations=1)
plt.imshow(result)
plt.title('Delated rest overs')

plt.tight_layout()
plt.subplots_adjust(wspace=0.1,
                    hspace=0.5)
plt.show()


# -----------------------Count squares by using the concept of connected components--------------------------------------
# Iterates of image and collects x- and y-coordinates of white pixels
def find_white_pixel(img, white):
    result = None
    for i_x, i_y in np.ndindex(img.shape):
        if img[i_x, i_y] >= white:
            result = (i_x, i_y)
            break

    return result


# Set connected pixels to 0
def fill_component(img, value, x, y):
    filled_comp = np.zeros(img.shape, np.uint8)
    filled_comp[x, y] = np.max(img)
    prev = img

    # .all(): iterable object in list, tuple, dictionary, i.e. in prev have to fulfill the statement
    while not (prev == filled_comp).all():
        prev = filled_comp
        filled_comp = cv2.dilate(filled_comp, np.ones((3, 3), np.uint8), iterations=1)
        # Add filled_comp and img and multiply it bei np.max(img)
        filled_comp = np.logical_and(filled_comp, img).astype(np.uint8) * np.max(img)

    subtracted_img = img - filled_comp

    for i_x, i_y in np.ndindex(filled_comp.shape):
        if filled_comp[i_x, i_y] == 1:
            filled_comp[i_x, i_y] = value

    return filled_comp, subtracted_img


def add_all_arrays(imgs):
    prev = imgs[0]
    for img in imgs[1:]:
        prev = prev + img

    return prev


def find_components(img, white):
    components = []

    while not find_white_pixel(img, white) is None:
        px = find_white_pixel(img, white)
        # Unpack x and y coordinates within px, which is obtained from
        # find_white_pixel
        comp, img = fill_component(img, 1, *px)
        components.append(comp)

    return components


img = result
components = []

# white = 255
#: Entspricht_while not find_white_pixel(img, 255) is None
while find_white_pixel(img, 255):
    px = find_white_pixel(img, 255)
    comp, img = fill_component(img, 1, *px)
    components.append(comp)

plt.subplot(1, 1, 1)
all_components = add_all_arrays([c * randint(70, 255) for c in components])
plt.imshow(all_components)
plt.title(f'{len(components)} squares of size 5x5 were found.')
plt.show()

# ----------------------------------------------Task 3.2 Counting Blood Cells--------------------------------------------
img = blood_cells
plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize = 11)

# Segment foreground from background
# cv2.THRESH_BINARY: "black" to black, "white" to white
threshold = 76
ret, img_b = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

plt.subplot(3, 2, 2)
plt.imshow(img_b, cmap='gray')
plt.title('Foreground and background separated')


# Remove border touching cells
def _pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def pad_border(img, border_size):
    return np.pad(img, int(border_size), _pad_with)


def get_border_pixels(img):
    white_border = np.ones((img.shape[0] - 2, img.shape[1] - 2), np.uint8)
    white_border = ~pad_border(white_border, 1).astype(bool)
    pixels_touching_border = np.logical_and(white_border, img).astype(np.uint8)
    return pixels_touching_border


complete_cells = img_b
pixels_touching_border = get_border_pixels(complete_cells)

while not find_white_pixel(pixels_touching_border, 1) is None:
    px = find_white_pixel(pixels_touching_border, 1)
    comp, complete_cells = fill_component(complete_cells, 1, *px)
    pixels_touching_border = get_border_pixels(complete_cells)

plt.subplot(3, 2, 3)
plt.imshow(complete_cells, cmap='gray')
plt.title('Incomplete cells on the border removed', fontsize = 11)

# Remove noise
plt.subplot(3, 2, 4)
kernel = np.ones((5, 5), np.uint8)
closed = cv2.morphologyEx(complete_cells, cv2.MORPH_OPEN, kernel)
plt.imshow(closed, cmap='gray')
plt.title('Noise removed', fontsize = 11)

# Close holes
plt.subplot(3, 2, 5)
kernel = np.ones((3, 3), np.uint8)
rem_noise = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
plt.imshow(rem_noise, cmap='gray')
plt.title('Holes removed', fontsize = 11)

# Label blood cells according to their size
def label_per_size(input):
    labels, label_sizes = np.unique(input, return_counts=True)
    size_color = {}
    hist_sizes = []

    for i, label in enumerate(labels):
        if label == 0:
            continue
        label_size = label_sizes[i]
        if label_size not in size_color:
            size_color[label_size] = -1 * i  # use minus value to avoid overlap

        hist_sizes.append(label_size)
        input = np.where(input == label, size_color[label_size], input)

    return np.abs(input), hist_sizes


# Count and label blood cells
labelled_cell_pixels, labelled_cell_count = ndimage.label(rem_noise)
labelled_cells_per_size_pixels, hist_sizes = label_per_size(labelled_cell_pixels)

# Plot color labeled blood cells according to their size
plt.subplot(3, 2, 6)
comp = find_components(rem_noise, 1)
length = len(comp)
plt.imshow(np.array(labelled_cells_per_size_pixels))
plt.title(f'Cells colored according to their size\nNumber of blood cells: {length}',fontsize = 11)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1,
                    hspace=0.5)
plt.show()


# Plot histogram for number of occurrence for each blood cell size
plt.subplot(1, 1, 1)
plt.hist(hist_sizes, bins=len(hist_sizes))
plt.title('Distribution of blood cell sizes in pixels')
plt.xlabel('Blood cell size in pixels')
plt.ylabel('Number of Occurrences ')
plt.grid(True)
plt.show()
