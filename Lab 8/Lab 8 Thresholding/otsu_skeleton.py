import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

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


def main():
    #image = np.asarray(Image.open("polymersome_cells_10_36.png"))
    image = np.asarray(Image.open("thGonz.tif"))

    #image = np.asarray(Image.open("binary_test_image.png"))
    if len(image.shape)==3:
        image=image[:,:,0]

    binary_image, t = basic_thresholding(image)
    print("Basic Thresholding. Output Threshold: "+str(t))

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()

    binary_image, between_class_variance, threshold, separability = my_otsu(image)

    print("Otsu's Method. Output Threshold: "+str(threshold))
    print("Separability: "+str(separability))
    #print(separability)

    plt.plot(between_class_variance*100)
    plt.show()

    plt.imshow(binary_image.astype(int), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

