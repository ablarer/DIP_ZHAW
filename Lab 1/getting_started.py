import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ----------------------------------------------------- load the image -------------------------------------------------
file_name = 'pics/lena_gray.gif'

# Open the image and convert to numpy array
gray_pixels = mpimg.imread(file_name)

# summarize some details about the image
print('numpy array:', gray_pixels.dtype)
print(gray_pixels.shape)

# --------------------------------------------------- display the image ------------------------------------------------
fig = plt.figure(1)
plt.title('Gray-Scale Image')
plt.imshow(gray_pixels, cmap='gray')
plt.show()
