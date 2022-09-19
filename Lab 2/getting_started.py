import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import PIL

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
