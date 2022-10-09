import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL.ImageDraw import ImageDraw
from scipy import ndimage, fft
from PIL import Image


# ----------------------------------------------------- Load blurred image ---------------------------------------------
file_name = 'pics/MenInDesert.jpg'

# Open the image with pillow and convert to numpy array
image = Image.open(file_name)
color_pixels = np.asarray(Image.open(file_name))
gray_pixels = np.asarray(Image.open(file_name).convert('L'))

# Summarize some details about the image
print(image.format)
print('Numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)

# -------------------------------------------- Generate the motion blur filter -----------------------------------------
nFilter = 91
angle = 30
my_filter = np.zeros((nFilter, nFilter))
my_filter[nFilter//2, :] = 1.0 / nFilter
my_filter = scipy.ndimage.rotate(my_filter, angle, reshape=False)
print('My filter:\n', my_filter)

# here goes your code ...
nRows = gray_pixels.shape[0]
nCols = gray_pixels.shape[1]
# Axes
nFFT = 1024

# fft2: The two-dimensional standard fft
image_spectrum = scipy.fft.fft2(gray_pixels, (nFFT, nFFT))
filter_spectrum = scipy.fft.fft2(my_filter, (nFFT, nFFT))
# ------------------------------------------------------Filter the image -----------------------------------------------

modified_image_spectrum = image_spectrum * filter_spectrum

# --------------------------------------------------- Reconstruct the image --------------------------------------------

# ifft2: The inverse two-dimensional standard fft
# Original image
image_back_transformed = scipy.fft.ifft2(image_spectrum)
image_back_transformed = np.real(image_back_transformed)[nFilter:nRows + nFilter, nFilter:nCols + nFilter]
# Filtered image
modified_image_back_transformed = scipy.fft.ifft2(modified_image_spectrum)
modified_image_back_transformed = np.real(modified_image_back_transformed)[nFilter:nRows + nFilter, nFilter:nCols + nFilter]

print()
print('Image dimension before transforming')
print('Original Image:', gray_pixels.shape)
print()
print('Image dimensions after back transforming')
print('Original Image:', image_back_transformed.shape)
print()
print('Image dimensions')
print('Modified Image', modified_image_back_transformed.shape)

# --------------------------------------------------------- Display images ---------------------------------------------
fig = plt.figure(1)
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_pixels, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Motion Blur Filter')
plt.imshow(my_filter, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Modified Image')
plt.imshow(modified_image_back_transformed, cmap='gray')


plt.subplot(2, 2, 4)
plt.title('Reconstructed Image')
# here goes your reconstructed image
plt.imshow(image_back_transformed, cmap='gray')

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)

plt.show()

# ------------------------------------------Apply Fourier transformation-------------------------------------------------
imageFourier = scipy.fft.fft2(gray_pixels)
modImageFourier = abs(imageFourier)
logScaleImage = np.log1p(modImageFourier)
print('Image Dimension:', gray_pixels.shape)

plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(gray_pixels, cmap='gray')

plt.subplot(1,2,2)
plt.title('DFT')
plt.imshow(logScaleImage, cmap='gray')

plt.subplots_adjust(wspace=0.4,
                    hspace=0.4)
plt.show()

# ------------------------------------------------------Task 3.2--------------------------------------------------------
# Wiener Filter
def get_image_array(file_name, print_info=True):
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


def rgb2gray(rgb):
    return np.rint(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])).astype(np.uint8)


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# Frequency Domain

desert = get_image_array("pics/MenInDesert.jpg", True)
desert = rgb2gray(desert)
plt.imshow(desert, cmap="gray")


#     Original Image converted to grayscale

# Appling a Filter in Frequency Domain


def next_biggest_square(size):
    result = 2
    while (result < size):
        result *= 2
    return result


def create_filter(size, low_frequ_value, high_frequ_value):
    scale_factor = 1.1
    e_x, e_y = size, size

    b_box = [(int(size * scale_factor) - size, int(size * scale_factor) - size), (size, size)]
    low_pass = Image.new("L", (int(size * scale_factor), int(size * scale_factor)), color=high_frequ_value)
    draw1 = ImageDraw.Draw(low_pass)
    draw1.ellipse(b_box, fill=low_frequ_value)

    return np.transpose(np.array(low_pass))


def img_to_freq(img, size=2):
    if size == 2:
        while (size < np.max(img.shape)):
            size *= 2

    img = img.astype(float)
    return np.fft.fft2(img, (size, size))


def freq_to_img(freq):
    return np.abs(np.fft.ifft2(freq))


def apply_freq_filter(img, my_filter):
    img_spectrum = img_to_freq(img)
    filtered_img_spectrum = np.multiply(img_spectrum, img_to_freq(my_filter, np.max(img_spectrum.shape)))
    return freq_to_img(filtered_img_spectrum)


def convert_freq(freq):
    return (np.log(1 + np.abs(np.fft.fftshift(freq))))


def show_spectrum(freq):
    plt.imshow(convert_freq(freq))


def crop_image(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]



img = get_image_array('pics/blurred_image.jpg', print_info=False)

img_spectrum = img_to_freq(img)
my_filter_spectrum = img_to_freq(my_filter)
show_images(1, 1024, apply_freq_filter(img, my_filter), cmap="gray")

img_spectrum = img_to_freq(img)
my_filter_spectrum = img_to_freq(my_filter, np.max(img.shape))
distorted_img_spectrum_f = img_spectrum * my_filter_spectrum
distorted_img_f = freq_to_img(distorted_img_spectrum_f)
distorted_img = convert(distorted_img_f, 0, 255, np.uint8)
distorted_img_spectrum = img_to_freq(distorted_img)

def wiener_filter(distorted_img_spectrum, filter_spectrum, K):
    return np.conjugate(filter_spectrum) * distorted_img_spectrum / (np.power(np.abs(filter_spectrum), 2.0) + K)

reconstructed_spectrum = wiener_filter(distorted_img_spectrum, filter_spectrum, 300000)
show_spectrum(reconstructed_spectrum)



# Adapted from https://github.com/lvxiaoxin/Wiener-filter
def motion_process(len, size):
    sx, sy = size
    PSF = np.zeros((sy, sx))
    PSF[int(sy / 2):int(sy /2 + 1), int(sx / 2 - len / 2):int(sx / 2 + len / 2)] = 1
    return PSF / PSF.sum()


def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred

def wiener(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    result = np.fft.ifft2(input_fft / PSF_fft)
    result = np.abs(np.fft.fftshift(result))
    return result

image = Image.open('pics/MenInDesert.jpg').convert('L')
plt.figure(1)
plt.xlabel("Original Image")
plt.gray()
plt.imshow(image)

plt.figure(2)
plt.gray()
data = np.asarray(image.getdata()).reshape(image.size)
PSF = motion_process(30, data.shape)
blurred = np.abs(make_blurred(data, PSF, 1e-3))

plt.subplot(221)
plt.xlabel("Motion blurred")
plt.imshow(blurred)

result = wiener(blurred, PSF, 1e-3)
plt.subplot(222)
plt.xlabel("Wiener Deblurred")
plt.imshow(result)

blurred += 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)

plt.subplot(223)
plt.xlabel("Motion & Noisy blurred")
plt.imshow(blurred)

result = wiener(blurred, PSF, 0.1 + 1e-3)
plt.subplot(224)
plt.xlabel("Wiener Deblurred")
plt.imshow(result)

plt.show()
