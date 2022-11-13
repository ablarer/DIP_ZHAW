import math
from statistics import variance

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def open_image(image):
    image = plt.imread(image)
    image = np.array(image)
    return image

def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.show()
    return image

def basic_thresholding(image, e):
    T0 = (image.max() + image.mean()) / 2
    T0 = 190
    Tdiff = 10
    G1, G2 = 0,0
    while (Tdiff > e):
        G1 = (image <= T0) * image
        G2 = (image > T0) * image
        m1 = G1[G1 > 0].mean()
        m2 = G2[G2 > 0].mean()
        T1 = 0.5 * (m1 + m2)
        Tdiff = np.abs(T0 - T1)
        T0 = T1
    return T0, G1, G2


def make_histogram(image):
    # Create 1D array
    image_1D = image.ravel()

    # Calculate the basic threshold
    # <-- your input
    n_bins = 128
    plt.hist(image_1D, bins=n_bins)
    plt.title('Original image histogram')
    plt.show()

    # Pixel intensities and bin borders / edges
    counts, edges = np.histogram(image, bins=n_bins)

    # Calculate the centers of the bin
    bin_centers = edges[:-1] + np.diff(edges) / 2.0

    return n_bins, counts, bin_centers


def global_variance(counts, image):
    p = counts / image.size
    i = np.arange(0, len(counts))
    mg = (p * i).sum()
    global_variance_ = ((i - mg) ** 2 * p).sum()
    return global_variance_, mg, p, i

def between_class_variance(global_variance_, mg, p, i, counts):
    mu_res = []
    between_class_variance_ = []
    e = 10 ** -8

    for T in range(0, len(counts)):
        p1 = p[:T].sum()
        mk = (i[:T] * p[:T]).sum()
        if p1 == 0:
            p1 += e
        if p1 == 1:
            p1 -= e
        between = ((p1 * mg - mk) ** 2) / (p1 * (1 - p1))
        mu = between / global_variance_
        mu_res.append(mu)
        between_class_variance_.append(between)
    return between_class_variance_

def ssd(counts, centers):
    # Sum of squared deviations from the left and right mean
    n = np.sum(counts)
    # Mean, µ , mu
    mu = np.sum(centers * counts) / n
    return np.sum(counts * ((centers - mu) ** 2))


def my_otsu(image, n_bins, counts, bin_centers):
    # Sum of the squared deviations from the left and right means
    total_ssds = []
    for bin_no in range(1, n_bins):
        # SSD k left
        left_ssd = ssd(counts[:bin_no], bin_centers[:bin_no])
        # SSD k right
        right_ssd = ssd(counts[bin_no:], bin_centers[bin_no:])
        # SSD k total
        total_ssds.append(left_ssd + right_ssd)
    # Threshold that minimizes the SSD k total in the bin z
    # Returns the index, i.e. the bin of the minimum value within the array
    z = np.argmin(total_ssds)
    print('The threshold value is in the middle of bin: ', z)
    # Otsu threshold within the bin z
    threshold = bin_centers[z]
    print(f'The Otsu threshold value in the middle of bin {z} is {threshold}.')
    # Returns boolean values for pixels that are larger than the threshold
    # Make all pixels >= threshold black
    binary_image = image >= threshold

    from skimage.filters import threshold_otsu
    print("Otsu's Method: Output Threshold via skimage", threshold_otsu(image, 128))

    return binary_image, threshold
    # return binary_image, between_class_variance, threshold, separability

def main():
    # Choose one of the images below
    # image = "thGonz.tif"
    image = "binary_test_image.png"

    image = open_image(image)
    display_image(image)

    if len(image.shape) == 3:
        image = image[:, :, 0]

    n_bins, counts, bin_centers = make_histogram(image)

    # Basic thresholding method
    e = 3
    T, G1, G2 = basic_thresholding(image, e)
    print(T)
    print(G1.shape, G2.shape)
    fig, ax = plt.subplots(2,2)
    ax[0,0].set_title(str(T))
    ax[0,0].imshow(G1)
    ax[0,1].hist(G1)
    ax[1,0].imshow(G2)
    ax[1,1].hist(G2)
    G2[G2 > 0] = 1
    plt.figure()
    plt.imshow(G2, cmap='gray')
    plt.show()


    # Otsu thresholding method
    binary_image, threshold = my_otsu(image, n_bins, counts, bin_centers)
    plt.title("Otsu's Method: Output Threshold: " + str(threshold))
    plt.imshow(binary_image, cmap="gray")
    plt.show()

    # Calculate global_variance
    global_variance_, mg, p, i = global_variance(counts, image)
    print('Global variance: ', global_variance_)

    # Calculate the in between class variance
    between_class_variance_ = between_class_variance(global_variance_, mg, p, i, counts)
    between_class_variance_max = np.array(between_class_variance_).argmax()
    print(f'In between class variance is maximal in bin {between_class_variance_max}')
    plt.plot(between_class_variance_)
    plt.title(f'In between class variance is maximal in bin {between_class_variance_max}')
    plt.show()

    # Calculate the separability
    separability = between_class_variance_max / global_variance_
    print('Separability', separability)



if __name__ == "__main__":
    main()
