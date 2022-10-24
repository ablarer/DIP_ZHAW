import glob
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

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

images = {}
for image_path in glob.glob("pics/*"):
    images[Path(image_path).stem] = get_image_array(image_path)

image_path_1 = 'pics/arterie.tif'
image_path_2 = 'pics/brain.tif'
image_path_3 = 'pics/ctSkull.tif'
arterie = get_image_array(image_path_1)
brain = get_image_array(image_path_2)
ctSkull = get_image_array(image_path_3)
show_images(3, 500, [arterie, brain, ctSkull], title=True)

#---------------------------------------Convert hsv to rgb--------------------------------------------------------------------

def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in hsv_to_rgb(h, s, v))


def get_n_distinct_colors(n, sat, val, rotate=0, col_range=1, reverse=False, incr_value=False):
    n = min(n, 255)
    reverse = -1 if reverse else 1

    def get_val(val, step, n, incr):
        if incr:
            val = 1 / n * step
        return val

    return [hsv2rgb(col_range * step / n + rotate, sat, get_val(val, step, n, incr_value)) for step in range(1, n + 1)][
           ::reverse]


def show_colors(colors):
    cols = int(len(colors))
    rows = 5
    img = []
    for i, val in enumerate(range(rows)):
        img.append(colors[:])
    plt.imshow(np.array(img))
    plt.show()

cols = get_n_distinct_colors(30, .8, .95, col_range=.5, rotate=.6, reverse=True)
show_colors(cols)

# Plot Gray Level Histogram
def gen_histogram(img):
    min_val = 0
    max_val = 256
    hist = np.zeros(256)

    for x, y in np.ndindex(img.shape):
        hist[img[x, y]] += 1

    return hist

def plot_histogram(values):
    plt.bar(range(len(values)), values)
    plt.show()

def gen_distribution(hist):
    max_val_dtype = 255
    hist = np.cumsum(hist)
    max_count = np.max(hist)

    for i, val in enumerate(hist):
        hist[i] = val * max_val_dtype / max_count

    return hist

def cumulative_distr(img, colors):
    img_new = np.zeros([*img.shape, 3], dtype=np.uint8)
    hist = gen_histogram(img)
    distr = gen_distribution(hist)
    val_rang_per_col = np.max(distr) / len(colors) + 1
    for x, y in np.ndindex(img.shape):
        img_new[x, y, :] = colors[int(distr[img[x, y]] / val_rang_per_col)]

    return img_new

colors = get_n_distinct_colors(10, .8, .95, col_range=.5, rotate=.6, reverse=False, incr_value=True)
show_colors(colors)

img = images["arterie"]
show_images(1, 800, cumulative_distr(img, colors))

colors = get_n_distinct_colors(20, .8, .95, col_range=.6, rotate=1.2, reverse=False, incr_value=True)
show_colors(colors)

img = images["brain"]
show_images(1, 800, cumulative_distr(img, colors))

colors = get_n_distinct_colors(20, .8, .95, col_range=.2, rotate=.3, reverse=True, incr_value=True)
show_colors(colors)

img = images["ctSkull"]
show_images(1, 800, cumulative_distr(img, colors))
