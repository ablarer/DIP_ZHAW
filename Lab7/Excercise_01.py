# Excercises 01
import math

from PIL.Image import Image
from scipy import misc
import numpy as np
from numpy import array
from numpy.linalg import norm
import matplotlib.pyplot as plt
import cv2

# Uncomment an image to be processed
img = 'Test_Image.png'
# img = 'handdrawing.tif'

# Reading the image
image = cv2.imread(img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Displaying the original image
plt.imshow(image)
plt.title('RGB image')
plt.show()

# Converting into grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Displaying the converted image
plt.imshow(gray_image, cmap='gray')
plt.title('Gray scale image')
plt.show()

# Add dimension
# gray_image = np.expand_dims(gray_image, -1)

# Converting to binary image using thresholding
(thresh, binary_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
# Displaying binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary image')
plt.show()

# Add dimension
binary_image = np.expand_dims(binary_image, -1)
# https://www.moonbooks.org/Articles/Implementing-a-simple-python-code-to-detect-straight-lines-using-Hough-transform/

# Define some needed variables
# Concerning the image
from PIL import Image
import numpy as np
img = Image.open("Test_Image.png")
img = np.asarray(img)

img = Image.open('Test_Image.png')
# img = gray_image
img_shape = img.shape
print(img_shape)

x_max = img_shape[0]
y_max = img_shape[1]

# Theta ranges from 0 to Pi, which is 180º
theta_max = 1.0 * math.pi
theta_min = 0.0

# Lenght of orthogonal vector pointing to a certain line
# Maximal length at the image diagonal
# Multidimensional Euclidean distance from the origin to a point.
# For 2D: sqrt(x_max^2 + y_max^2)
r_min = 0.0
r_max = math.hypot(x_max, y_max)


r_dim = 200
theta_dim = 300

# Create the Hough space
hough_space = np.zeros((r_dim, theta_dim))

for x in range(x_max):
    for y in range(y_max):
        if img[x, y, 0] == 255: continue
        for itheta in range(theta_dim):
            theta = 1.0 * itheta * theta_max / theta_dim
            r = x * math.cos(theta) + y * math.sin(theta)
            ir = r_dim * ( 1.0 * r ) / r_max
            hough_space[ir,itheta] = hough_space[ir,itheta] + 1

plt.imshow(hough_space, origin='lower')
plt.xlim(0,theta_dim)
plt.ylim(0,r_dim)
plt.show()


# a) Precompute the normal vector for each discrete value of A.
# You will need them to compute votes for each nonzero gray value
def compute_normal_vector(matrix):
    norm_l2_matrix = norm(matrix)
    return norm_l2_matrix