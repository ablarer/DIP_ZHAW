import cv2 as cv2
from matplotlib import pyplot as plt

import sift_visualizer as sv

image_1 = 'clutteredDesk.jpg'
image_2 = 'stapleRemover.jpg'

image_1 = plt.imread(image_1)
plt.imshow(image_1, cmap='gray')
plt.show()
image_2 = plt.imread(image_2)
plt.imshow(image_2, cmap='gray')
plt.show()

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)

image_1_keypoint = cv2.drawKeypoints(image_1, keypoints_1, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
image_2_keypoint = cv2.drawKeypoints(image_2, keypoints_2, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Displaying the image with keypoints as the
# Output on the screen
plt.imshow(image_1_keypoint)
plt.show()
plt.imshow(image_2_keypoint)
plt.show()


image_1_sift = sv.SiftVisualizer(image_1, keypoints_1, descriptors_1)
image_2_sift = sv.SiftVisualizer(image_2, keypoints_2, descriptors_2)

image_1_sift.investigator(image_1)
image_2_sift.investigator(image_2)

# create BFMatcher object
bf_1 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# Match descriptors.
matches_1 = bf_1.match(descriptors_1, descriptors_2)
# Sort them in the order of their distance.
matches = sorted(matches_1, key = lambda x:x.distance)
# Draw first 10 matches.
image_matched_1 = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches[:15], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(image_matched_1),plt.show()


# create BFMatcher object
bf_2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# Match descriptors.
matches_2 = bf_2.match(descriptors_1, descriptors_2)
# Sort them in the order of their distance.
matches = sorted(matches_2, key = lambda x:x.distance)
# Draw first 10 matches.
image_matched_2 = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches[:15], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(image_matched_2),plt.show()