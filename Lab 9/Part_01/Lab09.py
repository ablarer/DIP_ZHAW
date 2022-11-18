import math

import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import np as np
import numpy as np
import pandas as pd
import seaborn as sn
import skimage
from numpy.linalg import eig
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D

##### Aspect Ratio and Extent #####

# 1. Read image
img_bone = plt.imread('./Pictures/bone.tif')
img_bone_gray = cv2.cvtColor(img_bone, cv2.COLOR_RGBA2GRAY)
img_bone_gray = img_bone_gray.astype(np.float64) / np.max(img_bone_gray)
plt.figure()
plt.imshow(img_bone_gray, cmap='gray')
plt.title('Original image')
plt.show()

# 2. Create a matrix A with the coordinates of the pixels that belong to the bone and whose centroid is in the origin
points = np.argwhere(img_bone_gray > 0.5)
# Note that the matrix points have the dimension (number of image points x 2)
#    points[:,0] ... y coordinates
#    points[:,1] ... x coordinates
# Change the order so that 
#    points[:,0] ... x coordinates
#    points[:,1] ... y coordinates
points = points[:, ::-1]

# foo = 0  # <--
# covarianceMatrix =  # <--
df = pd.DataFrame(points)
covarianceMatrix = pd.DataFrame.cov(df)
cov_matrix = pd.DataFrame.cov(df)
sn.heatmap(cov_matrix, annot=True, fmt='g')
plt.show()

# 3. Compute the eigen vector for the largest eigen value
D, V = eig(covarianceMatrix)
idx = np.argsort(D)[-1]  # Index of largest eigenvalue
eigV = V[:, idx]  # Eigen vector for largest eigen value

# 4. Compute rotation that aligns the bone's dominant axis to the image's x-axis
height, width = img_bone_gray.shape
# 4.a) Method using the explicit rotation angle phi
# rotMatAffine = foo  # <--
# Angle calculated
# Checks orientation of x vector & selects appropriate x_axis_vector
# How to select appropriate points?

if (points[0, 0] - points[-1, 0]) < 0:
    x_axis_vector = np.array([0, -1])
else:
    x_axis_vector = np.array([0, 1])

angleInDegrees_c = np.arcsin(np.dot(eigV, x_axis_vector)) * 180 / math.pi
print('Angle calculated', angleInDegrees_c)
# Angle manually guessed
angleInDegrees_m = -28
print('Angle manually guessed', angleInDegrees_m)
image_rotated = imutils.rotate_bound(img_bone_gray, angleInDegrees_c)
# Check rotation angle
plt.imshow(image_rotated, cmap='gray')
plt.title('Rotated image with calculated angle information')
plt.show()

# Using cv2.getRotationMatrix2D()
# To get the rotation matrix
height, width = img_bone_gray.shape[:2]
center = (width / 2, height / 2)
rotMatAffine = cv2.getRotationMatrix2D(center=center, angle=-angleInDegrees_c, scale=1)

# 5. Rotate the image
# In affine transformation, all parallel lines in the original image will still
# be parallel in the output image.
# To find the transformation matrix, we need three points from the input image and
# their corresponding locations in the output image
img_bone_rot = cv2.warpAffine(img_bone_gray, rotMatAffine, (width, height), cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
plt.imshow(img_bone_rot)
plt.title('Rotated image with affine transformation')
plt.show()

# 6. Determine the bounding box for the aligned bone
alignedBoneCoords = np.argwhere(img_bone_rot > 0.5)
# y1 = foo  # <--
# x1 = foo  # <--
# y2 = foo  # <--
# x2 = foo  # <--
print('Boundig box coordiantes:')
y1 = alignedBoneCoords[0, 0]
print('y1', y1)
x1 = max(alignedBoneCoords[:, 1])  # zirka 700
print('x1', x1)
y2 = alignedBoneCoords[-1, 0]
print('y2', y2)
x2 = min(alignedBoneCoords[:, 1])  # zirka 20
print('x2', x2)
# 7. Draw the bonding box onto the image
img_bone_rot = cv2.rectangle(img_bone_rot, (x1, y1), (x2, y2), color=1)
height = y2 - y1 + 1
width = x2 - x1 + 1
print('Bounding Box: width={}, height={}'.format(np.abs(width), np.abs(height)))

# 8. Plot the aligned object bone including bounding box
plt.figure()
plt.imshow(img_bone_rot, cmap='gray')
plt.title('Rotated Object with Bounding Box')
plt.show()

###### Texture and Co-Ocurrence matrix #####
img_mc1 = plt.imread('./Pictures/mc1.tif')
plt.imshow(img_mc1, cmap='gray')
plt.show()
img_mc2 = plt.imread('./Pictures/mc2.tif')
plt.imshow(img_mc2, cmap='gray')
plt.show()
img_mc3 = plt.imread('./Pictures/mc3.tif')
plt.imshow(img_mc3, cmap='gray')
plt.show()
img_mc4 = plt.imread('./Pictures/mc4.tif')
plt.imshow(img_mc4, cmap='gray')
plt.show()

img_ut = img_mc1.copy()
height, width = img_ut.shape

resizeParam = 1
# img_ut = cv2.resize(img_ut, (resizeParam*width, resizeParam*height))

plt.figure()
plt.imshow(img_ut, cmap='gray')
plt.show()

# Computation of the Co-Occurrence Matrix
cooMat = np.zeros((256, 256))

M, N = img_ut.shape

count = 1
for ii in range(1, M - 1):
    for jj in range(1, N - 1):
        if ii + height in range(height) and jj + width in range(width):
            cooMat[img_mc1[ii, jj, 0], img_mc1[ii + height, jj + width, 0], 0] += 1
            cooMat[img_mc1[ii, jj, 1], img_mc1[ii + height, jj + width, 0], 1] += 1
            cooMat[img_mc1[ii, jj, 2], img_mc1[ii + height, jj + width, 0], 2] += 1
        if ii + height in range(height) and jj + width in range(width):
            cooMat[img_mc2[ii, jj, 0], img_mc2[ii + height, jj + width, 0], 0] += 1
            cooMat[img_mc2[ii, jj, 1], img_mc2[ii + height, jj + width, 0], 1] += 1
            cooMat[img_mc2[ii, jj, 2], img_mc2[ii + height, jj + width, 0], 2] += 1
        # <--
        # <--
        # <--
        # <--
        # <--
        # <--
        # <--
        # <--
        # <--
        count = count + 4
cooMat = cooMat / count

# Plotting the Co-Occurrence matrix
X = np.arange(cooMat.shape[1])
Y = np.arange(cooMat.shape[0])
X, Y = np.meshgrid(X, Y)
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, cooMat, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

energy = 0
contrast = 0
entropy = 0
homogenity = 0

for ii in range(256):
    for jj in range(256):
        energy += math.pow(cooMat[ii, jj], 2)
        contrast += math.pow(ii - jj, 2) * cooMat[ii, jj]
        entropy += -np.sum(cooMat*np.log2(cooMat + (cooMat==0)))
        homogenity += cooMat[ii, jj] / (1 + abs(ii - jj))
        # energy = foo
        # contrast = foo
        # entropy = foo
        # homogenity = foo

print("energy: %5.5f\ncontrast: %5.5f\nentropy: %5.5f\nhomogenity: %5.5f" % (energy, contrast, entropy, homogenity))

########## https://stackoverflow.com/questions/67801519/gray-level-co-occurrence-matrix-of-image-in-python
# GLCM properties
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte

img = io.imread('./Pictures/mc1.tif')

image = plt.imread('./Pictures/mc1.tif')
plt.imshow(image, cmap='gray')
plt.show()

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
inds = np.digitize(image, bins)

max_value = inds.max() + 1
matrix_coocurrence = skimage.feature.greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value, normed=False,
                                  symmetric=False)


def contrast_feature(matrix_coocurrence):
    contrast = skimage.feature.graycoprops(matrix_coocurrence, 'contrast')
    return "Contrast = ", contrast


def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = skimage.feature.graycoprops(matrix_coocurrence, 'dissimilarity')
    return "Dissimilarity = ", dissimilarity


def homogeneity_feature(matrix_coocurrence):
    homogeneity = skimage.feature.graycoprops(matrix_coocurrence, 'homogeneity')
    return "Homogeneity = ", homogeneity


def energy_feature(matrix_coocurrence):
    energy = skimage.feature.graycoprops(matrix_coocurrence, 'energy')
    return "Energy = ", energy


def correlation_feature(matrix_coocurrence):
    correlation = skimage.feature.graycoprops(matrix_coocurrence, 'correlation')
    return "Correlation = ", correlation


def entropy_feature(matrix_coocurrence):
    entropy = skimage.feature.graycoprops(matrix_coocurrence, 'entropy')
    return "Entropy = ", entropy


print(contrast_feature(matrix_coocurrence))
print(dissimilarity_feature(matrix_coocurrence))
print(homogeneity_feature(matrix_coocurrence))
print(energy_feature(matrix_coocurrence))
print(correlation_feature(matrix_coocurrence))
