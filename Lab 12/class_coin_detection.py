import matplotlib.pyplot as plt
import numpy as np
import cv2

class Coin_detection:
    def run_main(self, image):

        def __init__(self, max_image_width=1024, max_image_height=1024):
            self._max_image_width = max_image_width
            self._max_image_height = max_image_height

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # fontScale
        fontScale = 1

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(roi, (15, 15), 0)

        rows = roi.shape[0]

        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, rows / 2, param1=50, param2=30, minRadius=0,
                                   maxRadius=0)
        circles = np.uint16(np.around(circles))
        largestRadius = 0
        for i in circles[0, :]:
            if largestRadius < i[2]:
                largestRadius = i[2]

        change = 0
        for i in circles[0, :]:
            cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)
            center = (i[0] - 15, i[1])
            radius = i[2]
            ratio = ((radius * radius) / (largestRadius * largestRadius))
            # Zwei Franken: 187.69/248.0625 = 0.566238347
            # Ein Franken: 0.5424439405
            # 50 Rappen: 0.3338271605
            # 20 Rappen: 0.4658604182
            # 10 Rappen: 0.3832199546
            # 5 Rappen: 0.3086419753
            if (ratio > 0.7):
                value = 5.0
                change = change + value  # 5.00 funktioniert
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)

            elif ((ratio >= 0.5424439405) and (ratio <= 0.7)):
                value = 2.0
                change = change + value  # 2.00 funktioniert
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)

            elif ((ratio >= 0.4658604182) and (ratio < 0.5424439405)):
                value = 1.0
                change = change + value  # 1.00 funktioniert
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)

            elif ((ratio >= 0.3832199546) and (ratio < 0.4658604182)):
                value = 0.2
                change = change + value
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)

            elif ((ratio >= 0.3338271605) and (ratio < 0.3832199546)):
                value = 0.1
                change = change + value
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)

            elif ((ratio >= 0.30) and (ratio < 0.3338271605)):
                value = 0.05
                change = change + value
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)

            elif (ratio < 0.30):
                value = 0.5
                change = change  + value
                roi = cv2.putText(roi, str(value), center, font, fontScale, color, thickness, cv2.LINE_AA)


        text = "Total: " + str(round(change, 2)) + " CHF"
        cv2.putText(roi, text, (0, 200), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Result', roi)
        cv2.waitKey()

        return change, roi

