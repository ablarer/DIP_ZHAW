#!/usr/bin/env python
# coding: utf-8

# # Lab09 - Color Segmentation

# In[1]:


import cv2
import glob
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb, rgb_to_hsv


# In[2]:


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
            plt.imshow(img[1], cmap=plt.get_cmap(cmap))
            plt.title(img[0])
        else:
            plt.imshow(img, cmap=plt.get_cmap(cmap))
        plt.xticks([])
        plt.yticks([])


# In[3]:


images = {}
for image_path in glob.glob("pics/*"):
    images[Path(image_path).stem] = get_image_array(image_path)

show_images(3, 500, *images.items(), title=True)


# ## Color Coding

# ### Color Models

# In[4]:


def rgb2cmy(r, g, b):
    return (255-r, 255-g, 255-b)

def rgb2hsv(r, g, b):
    return rgb_to_hsv(r,g,b)

def cmy2rgb(c, m, y):
    return (255-c, 255-m, 255-y)

def hsv2rgb(h,s,v):
    return hsv_to_rgb(h,s,v)

def convert_img(img, conversion):
    img_new = np.zeros((img.shape[0], img.shape[1], 3))
    for x,y in np.ndindex(img.shape[:2]):
        img_new[x, y] = np.asarray(conversion(*img[x, y]))
        
    return img_new


# In[5]:


def get_n_distinct_colors(n, sat, val, rotate=0, col_range=1, reverse=False, incr_value=False):
    n = min(n, 255)
    reverse = -1 if reverse else 1
    
    def get_val(val, step, n, incr):
        if incr:
            val = 1/n * step
        return val
    
    return [hsv2rgb(col_range * step/n + rotate, sat, get_val(val, step, n, incr_value)) for step in range(1, n+1)][::reverse]

def show_colors(colors):
    cols = int(len(colors))
    rows = 5
    img = []
    for i, val in enumerate(range(rows)):
        img.append(colors[:])
    plt.imshow(np.array(img))


# ### Plot Gray Level Histogram

# In[6]:


def gen_histogram(img):
    min_val = 0
    max_val = 256
    hist = np.zeros(256)
    
    for x,y in np.ndindex(img.shape):
        hist[img[x,y]] += 1
        
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
    img_new = np.zeros([*img.shape,3],dtype=np.uint8)
    hist = gen_histogram(img)
    distr = gen_distribution(hist)
    val_rang_per_col = np.max(distr) / len(colors) + 1
    for x,y in np.ndindex(img.shape):
        img_new[x, y, :] = colors[int(distr[img[x, y]] / val_rang_per_col)]
        
    return img_new


# ## Colorspace Conversion

# In[7]:


def display_colorchannels(img, channel_names=("RED", "GREEN", "BLUE"), return_ch=False):
    plt.imshow(img)
    channels = [(channel_names[ch], img[:,:,ch]) for ch in range(3)]
    show_images(3, 500, *channels, cmap="gray", title=True)
    
    if return_ch:
        return [ch[1] for ch in channels]


# In[8]:


display_colorchannels(images["brainCells"])


# In[9]:


CMY_braincells = convert_img(images["brainCells"], rgb2cmy)
display_colorchannels(CMY_braincells, ("CYAN", "MAGENTA", "YELLOW"))


# In[10]:


HSV_braincells = convert_img(images["brainCells"], rgb2hsv)
display_colorchannels(HSV_braincells, ("HUE", "SATURATION", "VALUE"))


# In[11]:


plt.imsave("pics/hue.png", HSV_braincells[:,:,0], cmap="gray")
plt.imsave("pics/sat.png", HSV_braincells[:,:,1], cmap="gray")
plt.imsave("pics/val.png", HSV_braincells[:,:,2], cmap="gray")


# Convert Image to HSV then use Hue values to segment blue color range, these are the cells

# In[9]:


center = 150
my_range = 255

frame_HSV = cv2.cvtColor(images["brainCells"], cv2.COLOR_BGR2HSV)
frame_threshold = cv2.inRange(frame_HSV, (center - (my_range/2), 0, 0), (center + (my_range/2), 255, 255))
show_images(1, 700, frame_threshold)


# In[11]:


def binarizeImg(img, value):
    binarized_img = np.zeros(img.shape[:2])
    for x,y in np.ndindex(img.shape):
        if img[x, y] == value:
            binarized_img[x, y] = 1
    return binarized_img


# In[12]:


def erode_dilate(img, count, kernel):
    
    for i in range(count):
         result = cv2.dilate(cv2.erode(img, kernel, iterations=1), kernel, iterations=1)
    return result


# In[13]:


cells_only = binarizeImg(frame_threshold, 0)
show_images(1, 700, cells_only)


# In[14]:


cells_only_cleaned = erode_dilate(cells_only, 3, np.ones((5,5)))
show_images(1, 700, cells_only_cleaned)


#     Count the cells

# In[25]:


def find_white_pixel(img, white):
    result = None
    for ix,iy in np.ndindex(img.shape):
        if img[ix, iy] >= white:
            result = (ix, iy)
            break
    
    return result

def fill_component(img, value, x, y):
    filled_comp = np.zeros(img.shape, np.uint8)
    filled_comp[x, y] = np.max(img)
    prev = img
    
    while not (prev == filled_comp).all():
        prev = filled_comp
        filled_comp = cv2.dilate(filled_comp, np.ones((3,3), np.uint8), iterations=1)
        filled_comp = np.logical_and(filled_comp, img).astype(np.uint8) * np.max(img)
        
    subtracted_img = img-filled_comp
    
    for ix,iy in np.ndindex(filled_comp.shape):
        if filled_comp[ix, iy] == 1:
            filled_comp[ix, iy] = value
        
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
        comp, img = fill_component(img, 1, *px)
        components.append(comp)
    
    return components

# DOES MUCH MORE THAN COUNT
def count_components(img, white_value):
    components = []
    while not find_white_pixel(img, white_value) is None:
        px = find_white_pixel(img, white_value)
        comp, img = fill_component(img, 1, *px)
        components.append(comp)
    return len(components)


# In[26]:


count_components(cells_only_cleaned, 1)


#     49 Celles where found

# In[ ]:




