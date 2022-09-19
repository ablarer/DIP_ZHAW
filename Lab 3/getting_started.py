import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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

# ---------------------------------------- compute histogram ---------------------------------------------------------------
my_filter = np.ones((9, 9)).astype(float) / 81.     # averaging filter
new_image = signal.convolve2d(gray_pixels.astype(float), my_filter, 'same')
new_image = new_image.astype(int)

# --------------------------------------------------------- display images --------------------------------------------------
fig = plt.figure(1)
plt.subplot(2, 1, 1)
plt.title('Gray-Scale Image')
plt.imshow(gray_pixels, cmap='gray')

ax = fig.add_subplot(2, 1, 2)
plt.title('Filtered Image')
plt.imshow(new_image, cmap='gray')


plt.show()
