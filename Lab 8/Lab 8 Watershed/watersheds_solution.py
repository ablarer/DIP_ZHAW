import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
def compute_binary_image_otsu(image, show=False):
    foo = 0
    # threshold, image_binary = (foo , foo) # <--
    threshold, image_binary = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    if show:
        plt.imshow(image_binary, 'gray')
        plt.show()
        plt.figure()
        plt.hist(image.ravel(), bins=256, range=(0, 255))
        plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
        plt.show()
    return image_binary


def distance_transform_implementation_example(binaryImage, show=False):
    g = 1000 * np.uint16(binaryImage > 0)
    for iy in range(1, g.shape[0] - 1):
        for ix in range(1, g.shape[1] - 1):
            neighborMinValue = np.min(
                [g[iy - 1, ix - 1], g[iy - 1, ix], g[iy, ix - 1], g[iy - 1, ix + 1], g[iy + 1, ix + 1], g[iy + 1, ix],
                 g[iy, ix + 1], g[iy + 1, ix - 1]])
            if g[iy, ix]:
                g[iy, ix] = neighborMinValue + 1

    for iy in range(g.shape[0] - 2, 1, -1):
        for ix in range(g.shape[1] - 2, 1, -1):
            neighborMinValue = np.min(
                [g[iy - 1, ix - 1], g[iy - 1, ix], g[iy, ix - 1], g[iy - 1, ix + 1], g[iy + 1, ix + 1], g[iy + 1, ix],
                 g[iy, ix + 1], g[iy + 1, ix - 1]])
            if g[iy, ix]:
                g[iy, ix] = neighborMinValue + 1

    if show:
        plt.figure()
        plt.imshow(g, 'gray')
        plt.show()

    return g


def main():
    image = plt.imread("yeast.tif")
    if len(image.shape) == 3:
        image = image[:, :, 0]

    imageBinary = compute_binary_image_otsu(image, True)
    foo = 0
    # imageDistance = foo  # <--
    # Compute Euclidean distance from every binary pixel
    # to the nearest zero pixel then find peaks
    imageDistance = distance_transform_implementation_example(imageBinary, True)

    # Find peaks
    # peakCoords = foo  # <--
    peakCoords = peak_local_max(imageDistance, min_distance=20, threshold_abs=9, exclude_border=1)

    # Create a mask containing the seedpoints, 
    # mask = foo  # <--
    # Make empty black image
    # mask = np.zeros((imageDistance.shape[0], imageDistance.shape[1], 1), np.uint8)
    mask = np.array((peakCoords[0], peakCoords[1], 1))
    # mask[10, 5] = [0, 0, 255]
    print(mask)
    # mask[foo] = True  # <--
    mask[:, :, 1] = True

    # Give the seedpoints different integer values they will be used to label the identified objects
    # Label all local maximums with different positive values starting from 1.
    # So, in case we have 10 objects in the image each of them will be labeled with a value from 1 to 10.
    # seedMask, _ = foo  # <--
    seedMask = ndi.label(peakCoords, mask)[0]

    # Watershed algorithm grows the regions around the seedpoints and labels them in a mask, the label is
    # determined by the seedpoint label
    # labeled_regions = foo  # <--
    labeled_regions = watershed(-imageDistance, seedMask)
    labels = labeled_regions * imageBinary

    # Visualize the intermediate and final results
    plt.figure()
    fig, axes = plt.subplots(ncols=5, figsize=(12, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(-image, cmap=plt.cm.gray, interpolation='none')
    ax[0].set_title('Original Image')
    ax[1].imshow(imageBinary, cmap=plt.cm.gray, interpolation='none')
    ax[1].set_title('Otsu: image_binary')
    ax[2].imshow(imageDistance / np.amax(imageDistance), cmap=plt.cm.jet, interpolation='none')
    ax[3].scatter(peakCoords[:, 1].reshape((-1,)), peakCoords[:, 0].reshape((-1,)), s=10, marker='o')
    ax[4].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='none')
    ax[4].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
