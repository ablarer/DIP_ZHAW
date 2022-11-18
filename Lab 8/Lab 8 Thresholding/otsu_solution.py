import math
from statistics import variance

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Neu
threshold_values = {}
threshold = 0
pixel_intensity = [1]


def histogram(image):
    row, col = image.shape
    pixel_intensity = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            pixel_intensity[image[i, j]] += 1
    x = np.arange(0, 256)
    plt.bar(x, pixel_intensity, color='b', width=5, align='center', alpha=0.25)
    plt.show()
    return pixel_intensity


def countPixel(h):
    pixel_count = 0
    for i in range(0, len(h)):
        if pixel_intensity[i] > 0:
            pixel_count += pixel_intensity[i]
    return pixel_count


def weight(s, e):
    weight_ = 0
    for i in range(s, e):
        weight_ += pixel_intensity[i]
    return weight_


def mean(s, e):
    mean_ = 0
    weight_ = weight(s, e)
    for i in range(s, e):
        mean_ += pixel_intensity[i] * i

    return mean_ / float(weight_)


def variance(s, e):
    variance_ = 0
    mean_ = mean(s, e)
    weight_ = weight(s, e)
    for i in range(s, e):
        variance_ += ((i - mean_) ** 2) * pixel_intensity[i]
    variance_ /= weight_
    return variance_
#Neu

def basic_thresholding(image):
    t = np.mean(image.flatten()).astype(int)

    # Calculate the basic threshold
    # <-- your input

    bin_img = image > t
    return bin_img, t


def my_otsu(image):
    # dummy variables, replace with your own code:
    threshold = 128
    between_class_variance = 0
    separability = 0
    # until here

    binary_image = image >= threshold
    return binary_image, between_class_variance, threshold, separability


def threshold(pixel_intensity):
    pixel_count = countPixel(pixel_intensity)
    for i in range(1, len(pixel_intensity)):

        vb = variance(0, i)
        wb = weight(0, i) / float(pixel_count)
        mb = mean(0, i)

        vf = variance(i, len(pixel_intensity))
        wf = weight(i, len(pixel_intensity)) / float(pixel_count)
        mf = mean(i, len(pixel_intensity))

        # Within class variance
        variance_within_class = wb * (vb) + wf * (vf)
        # Between class variance
        variance_between_classes = wb * wf * (mb - mf) ** 2

        print('T=' + str(i) + "\n")
        print('Wb=' + str(wb) + "\n")
        print('Mb=' + str(mb) + "\n")
        print('Vb=' + str(vb) + "\n")
        print('Wf=' + str(wf) + "\n")
        print('Mf=' + str(mf) + "\n")
        print('Vf=' + str(vf) + "\n")

        print('within class variance=' + str(variance_within_class) + "\n")
        print('between class variance=' + str(variance_between_classes) + "\n")
        print("\n")

        if not math.isnan(variance_within_class):
            threshold_values[i] = variance_within_class

    return threshold_values, variance_within_class, variance_between_classes

def main():
    # image = np.asarray(Image.open("polymersome_cells_10_36.png"))
    image = np.asarray(Image.open("thGonz.tif"))

    # image = np.asarray(Image.open("binary_test_image.png"))
    if len(image.shape) == 3:
        image = image[:, :, 0]

     # Neu
    pixel_intensity = histogram(image)
    # threshold_values, variance_within_class, variance_between_classes = threshold(pixel_intensity)
    # Neu

    binary_image, t = basic_thresholding(image)
    print("Basic Thresholding. Output Threshold: " + str(t))

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()

    binary_image, between_class_variance, threshold, separability = my_otsu(image)

    print("Otsu's Method. Output Threshold: " + str(threshold))
    print("Separability: " + str(separability))
    # print(separability)

    plt.plot(between_class_variance * 100)
    plt.show()

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
