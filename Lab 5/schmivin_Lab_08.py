#!/usr/bin/env python
# coding: utf-8

# # Lab 08 - Morpholgical Operations

# In[2]:


import cv2
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt


# In[3]:


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


# In[4]:


blood_cells = get_image_array(
    "../../../../ZHAW_Bachelorstudiengang_Informatik/7. Semester/DIP/Zusatzmaterial/Digital-Image-Processing-master/Lab_08/pics/bloodCells.png", True)
squares = cv2.imread("pics/squares.tif", 0)
show_images(2, 500, (blood_cells, "Need to be counted"), (squares, "sqaures from 1x1 to 15x15"), cmap="gray", title=True)


# ## 1. Locate and Count Squares of Given Size

# In[5]:


plt.imshow(squares)


#     remove all squares whos size is not 5x5

# ### remove larger squares

# In[6]:


kernel = np.ones((6,6), np.uint8)
no_bigs = cv2.morphologyEx(squares, cv2.MORPH_TOPHAT, kernel)
plt.imshow(no_bigs)


# ### remove smaller squares

# In[7]:


kernel = np.ones((5,5), np.uint8)
only_5s = cv2.erode(no_bigs, kernel, iterations=1)
plt.imshow(only_5s)


# ### bring squares back to size

# In[8]:


result = cv2.dilate(only_5s,kernel,iterations = 1)
plt.imshow(result)


#     Count squares using concept of connected components

# In[9]:


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


# In[10]:


img = result
components = []

while not find_white_pixel(img, 255) is None:
    px = find_white_pixel(img, 255)
    comp, img = fill_component(img, 1, *px)
    components.append(comp)
    
print(f"Found {len(components)} squares of size 5x5")


# In[11]:


all_components = add_all_arrays([c * randint(70,255) for c in components])
plt.imshow(all_components)


# ## Counting Bloodcells

# In[12]:


img = blood_cells
plt.imshow(img)


# In[25]:


threshold = 76
ret, img_b = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
#show_images(1, 500, *imgs, title=True)
plt.imshow(img_b)


#     Remove border touching cells

# In[26]:


def _pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
def pad_border(img, border_size):
    return np.pad(img, int(border_size), _pad_with)

def get_border_pixels(img):
    white_border = np.ones((img.shape[0]-2, img.shape[1]-2), np.uint8)
    white_border = ~pad_border(white_border, 1).astype(bool)
    pixels_touching_border = np.logical_and(white_border, img).astype(np.uint8)
    return pixels_touching_border


# In[27]:


complete_cells = img_b
pixels_touching_border = get_border_pixels(complete_cells)

while not find_white_pixel(pixels_touching_border, 1) is None:
    px = find_white_pixel(pixels_touching_border, 1)
    comp, complete_cells = fill_component(complete_cells, 1, *px)
    pixels_touching_border = get_border_pixels(complete_cells)
    
plt.imshow(complete_cells)


#     remove some leftover noise

# In[28]:


kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(complete_cells, cv2.MORPH_CLOSE, kernel)
plt.imshow(closed)


#     closing holes

# In[29]:


kernel = np.ones((3,3), np.uint8)
rem_noise = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
plt.imshow(rem_noise)


# In[30]:


comp = find_components(rem_noise, 1)
len(comp)


# In[31]:


all_components = add_all_arrays([c * randint(70,255) for c in comp])
plt.imshow(all_components)


# In[ ]:




