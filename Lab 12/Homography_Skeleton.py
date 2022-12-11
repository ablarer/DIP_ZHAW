
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2


FIGURE_SIZE = (12, 9)
# Bilder einlesen
imgFolder = ''

# imgName = 'HausPerspektivisch.png'
# imgName='Klassenfoto.jpg'
imgName = 'chessboard_perspective.jpg'

img_src = plt.imread(imgFolder+imgName)

plt.figure(figsize=FIGURE_SIZE)
plt.imshow(img_src)
plt.show()

# Coordiantes are defined as [vertical, horizontal]
if imgName == "chessboard_perspective.jpg":
    src = np.array([[380, 40], [940, 170], [710, 670], [50, 420]])    # TODO: Additional Points go here
    points_u = np.array([[20, 20], [200, 20],[200, 200],[20, 200]])# TODO: Coordinates of undistorted Points go here

# draw markings on the source image
for i, pts in enumerate(src):
    center_coordinates = (pts[0], pts[1])
    cv2.putText(img_src, str(i + 1), (pts[0] + 15, pts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 215, 255), 5)
    cv2.circle(img_src, center_coordinates, 10, (0, 215, 255), 10)

# TODO: Please add your code here
t_source_image = img_src.copy()

def get_homography_matrix(source, destination):
    """ Calculates the entries of the Homography matrix between two sets of matching points.

    Args
    ----
        - `source`: Source points where each point is int (x, y) format.
        - `destination`: Destination points where each point is int (x, y) format.

    Returns
    ----
        - A numpy array of shape (3, 3) representing the Homography matrix.

    Raises
    ----
        - `source` and `destination` is lew than four points.
        - `source` and `destination` is of different size.
    """
    assert len(source) >= 4, "must provide more than 4 source points"
    assert len(destination) >= 4, "must provide more than 4 destination points"
    assert len(source) == len(destination), "source and destination must be of equal length"
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))

h = get_homography_matrix(src, points_u)
destination_image = cv2.warpPerspective(t_source_image, h, (300, 300))

figure = plt.figure(figsize=(12, 6))

subplot1 = figure.add_subplot(1, 2, 1)
subplot1.title.set_text("Source Image")
subplot1.imshow(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB))

subplot2 = figure.add_subplot(1, 2, 2)
subplot2.title.set_text("Destination Image")
subplot2.imshow(cv2.cvtColor(destination_image, cv2.COLOR_BGR2RGB))


plt.savefig("output.png")

plt.show()