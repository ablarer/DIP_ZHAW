import matplotlib.pyplot as plt
import cv2
import numpy as np

LEFT_BUTTON = 1
RIGHT_BUTTON = 3


class SiftVisualizer():
    def __init__(self, img, keypoints_sift, descriptors_sift):
        self.img = img
        self.keypoints = keypoints_sift
        self.descriptors = descriptors_sift


    def on_click(self, event):
        print("button={}, x={}, y={}, xdata={}, ydata={}\n".format(
            event.button, event.x, event.y, event.xdata, event.ydata))
        if (event.button == RIGHT_BUTTON):
            keypoint_id = self.query_keypoint_id((event.xdata, event.ydata))
            
            octave, layer, scale, radius, _ = self.unpackSIFTOctave(self.keypoints[keypoint_id])
            print("img.size = {}, kpt.size = {}, radius = {}".format(self.img.shape, self.keypoints[keypoint_id].size, radius))
            
            self.display_sift_feature(self.keypoints[keypoint_id], self.descriptors[keypoint_id], keypoint_id)


    def euclidean_distance(self, point_a, point_b):
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        d = np.sqrt(dx**2 + dy**2)
        return d


    def query_keypoint_id(self, point):
        lowest_distance = np.infty
        closest_keypoint_id = None
        for id, keypoint in enumerate(self.keypoints):
            distance = self.euclidean_distance(keypoint.pt, point)
            if (distance < lowest_distance):
                lowest_distance = distance
                closest_keypoint_id = id
        return closest_keypoint_id


    def display_sift_feature(self, keypoint, descriptor, id, color=(255, 255, 0)):
        octave, layer, scale, dhc, _ = self.unpackSIFTOctave(keypoint)
        height = int(np.round(3*dhc))
        width = int(np.round(3*dhc))

        img_keypoint = np.zeros((2*height+1, 2*width+1, 3))
        # Consider only integer part of keypoint coordinate
        kp_x = int(keypoint.pt[0])
        kp_y = int(keypoint.pt[1])

        # Find coordinates on source
        y0 = kp_y - height
        y1 = kp_y + height
        x0 = kp_x - width
        x1 = kp_x + width

        # Limit source coordinates to image boundaries
        M = self.img.shape[0]
        N = self.img.shape[1]
        y0 = np.max((0, y0))
        y1 = np.min((y1, M-1))
        x0 = np.max((0, x0))
        x1 = np.min((x1, N-1))
        plt.figure()

        # Compute coordinates on target Image
        t_center = (width, height)
        t_x0 = t_center[0] - (kp_x - x0)
        t_y0 = t_center[1] - (kp_y - y0)
        t_x1 = t_center[0] + (x1 - kp_x)
        t_y1 = t_center[1] + (y1 - kp_y)

        # Zoom in to roi
        ZOOM_TARGET_RADIUS = 150
        zoom_factor = int(round(ZOOM_TARGET_RADIUS/dhc))
        zoomed_size = (img_keypoint.shape[0] * zoom_factor, img_keypoint.shape[0] * zoom_factor)
        img_keypoint[t_y0:t_y1, t_x0:t_x1, :] = np.repeat(np.expand_dims(self.img[y0:y1, x0:x1], 2), 3, axis=2)
        img_keypoint_zoomed = cv2.resize(img_keypoint, zoomed_size, cv2.INTER_NEAREST)
        img_descriptor_zoomed = img_keypoint_zoomed.copy()

        # Draw keypoint properties
        img_overview = self.img.copy()

        img_overview = self.img_sift_overview(img_overview, x0, x1, y0, y1)
        plt.subplot(1, 3, 1)
        max = np.amax(img_overview)

        plt.imshow(img_overview, cmap='gray', vmin=0, vmax=max )
        plt.title("square showing zoomed area")
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        img_keypoint = self.img_sift_keypoint(img_keypoint_zoomed, keypoint, descriptor, id, color, zoom_factor)
        plt.subplot(1, 3, 2)
        plt.imshow(img_keypoint)
        ax = plt.gca()
        ax.set_xticks(dhc * zoom_factor * np.arange(1, 6, 1))
        ax.set_xticklabels(dhc*np.arange(-2, 3, 1))
        ax.set_yticks(dhc * zoom_factor * np.arange(1, 6, 1))
        ax.set_yticklabels(dhc*np.arange(-2, 3, 1))
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        plt.title("keypoint properties")
        # Draw descriptor properties
        img_descriptor = self.img_sift_descriptor(img_descriptor_zoomed, keypoint, descriptor, id, color, zoom_factor)
        plt.subplot(1, 3, 3)
        plt.imshow(img_descriptor)
        ax = plt.gca()
        ax.set_xticks(dhc * zoom_factor * np.arange(1, 6, 1))
        ax.set_xticklabels(dhc*np.arange(-2, 3, 1))
        ax.set_yticks(dhc * zoom_factor * np.arange(1, 6, 1))
        ax.set_yticklabels(dhc*np.arange(-2, 3, 1))
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        plt.title("descriptor properties")
        plt.show()


    def unpackSIFTOctave(self, kpt):
        # unpackSIFTOctave(kpt)->(octave,layer,scale)
        # @created by Silencer at 2018.01.23 11:12:30 CST
        # @brief Unpack Sift Keypoint by Silencer
        # @param kpt: cv2.KeyPoint (of SIFT)
        # computes dhc in units of pixels (weie 2.12.2020)
        _octave = kpt.octave
        octave = _octave & 0xFF
        layer = (_octave >> 8) & 0xFF
        if octave >= 128:
            octave |= -128
        if octave >= 0:
            scale = float(1/(1 << octave))
        else:
            scale = float(1 << -octave)
        radius_in_scaled_img = int(3*np.sqrt(2)*2.5*kpt.size*scale)
        pixelsBetweenHistogramCenters = int(kpt.size*1.5)
        return octave, layer, scale, pixelsBetweenHistogramCenters, radius_in_scaled_img


    def img_sift_descriptor(self, img, key, descr, id, color=(255, 255, 255), zoom_factor=1):
        angrad = np.deg2rad(key.angle)
        _, _, _, unzoomed_dhc , _ = self.unpackSIFTOctave(key)
        dhc = zoom_factor*unzoomed_dhc
        M = img.shape[0]
        N = img.shape[1]
        center = np.array((int(N/2), int(M/2))).reshape(2, 1)
        # Create 8 directions
        angle_pih = np.deg2rad(-45)
        rotmat_pih = np.array([[np.cos(angle_pih), -np.sin(angle_pih)], [np.sin(angle_pih), np.cos(angle_pih)]])
        directions = np.zeros((2, 8))
        directions[:, 0] = np.array([1, 0])
        for n in range(1, 8):
            directions[:, n] = rotmat_pih @ directions[:, n-1]

        # loop over all histograms and create directions and origins for all 16 HoGs (fields)
        DIRS = 8  # Directions of Histogram
        HOGS = 16  # Number of Histograms
        RHOGS = 4  # sqrt(HOGS)
        delta = (RHOGS-1)/2  # Distance between origin and outher most histogram center along an axis
        weighted_directions = np.zeros((2, HOGS, DIRS))
        origins = np.zeros((2, HOGS))
        rotmat_angle = np.array([[np.cos(angrad), -np.sin(angrad)], [np.sin(angrad), np.cos(angrad)]])
        for col in range(0, RHOGS):
            for raw in range(0, RHOGS):
                field = (raw * RHOGS) + col
                hist = descr[DIRS * field: DIRS * (field + 1)]
                origins[:, field] = rotmat_angle @ (dhc*np.array((col-delta, raw-delta)))
                for dir in range(0, DIRS):
                    weighted_directions[:, field, dir] = rotmat_angle @ (directions[:, dir] * 0.7* dhc * hist[dir] / 255)

        # draw histogram bins as lines with direction
        for field in range(0, HOGS):
            for dir in range(0, DIRS):
                origin = origins[:, field].reshape(2,) + center.reshape(2,)
                target = weighted_directions[:, field, dir].reshape((2, )) + origin
                cv2.line(img, tuple(origin.astype(np.int)), tuple(target.astype(np.int)),
                        color, thickness=2, lineType=cv2.LINE_AA, shift=0)
        # compute grid lines
        begin_pt_h = np.zeros((2, RHOGS+1))
        end_pt_h = np.zeros((2, RHOGS+1))
        pos = -(RHOGS/2)
        # horizontal lines
        for n in range(0, RHOGS+1):
            begin_pt_h[:, n] = np.array([pos, -RHOGS/2]) * dhc
            end_pt_h[:, n] = np.array([pos, RHOGS/2]) * dhc
            pos += 1
        # vertical lines
        rot_mat_90 = np.array([[0, -1], [1, 0]])
        begin_pt = np.column_stack((begin_pt_h, rot_mat_90 @ begin_pt_h))
        end_pt = np.column_stack((end_pt_h, rot_mat_90 @ end_pt_h))
        # rotate grid lines
        bpt = (rotmat_angle @ begin_pt) + np.repeat(center, begin_pt.shape[1], axis=1)
        ept = (rotmat_angle @ end_pt) + np.repeat(center, begin_pt.shape[1], axis=1)
        # draw grid lines
        self.plot_lines_on_image(img, bpt, ept, color)
        return img/255


    def plot_lines_on_image(self, img, begin_pt, end_pt, color=(255, 255, 255)):
        N = begin_pt.shape[1]
        assert(N == end_pt.shape[1])

        for n in range(0, N):
            pt1 = tuple(begin_pt[:, n].astype(np.int))
            pt2 = tuple(end_pt[:, n].astype(np.int))
            cv2.line(img, pt1, pt2, color, thickness=1)


    def img_sift_keypoint(self, img, key, descr, id, color=(255, 255, 255), zoom_factor=1):
        # Interpretation of key.octave according to
        # https://stackoverflow.com/questions/8561352/opencv-keypoint-information-about-angle-and-octave
        octave, layer, scale, radius_unzoomed_pixel, _ = self.unpackSIFTOctave(key)
        radius = int(zoom_factor*key.size)
        angle = int(np.round(key.angle))
        response = np.round(key.response*1000)/1000
        key_prop_str = []
        key_prop_str += ["key_id {}, octave {}".format(id, octave)]
        key_prop_str += ["layer {}, scale {}".format(layer, scale)]
        key_prop_str += ["angle {}, response {}".format(angle, response)]
        key_prop_str += ["size {}".format(key.size)]

        # Write text on the image
        M = img.shape[0]
        N = img.shape[1]
        text_x_pos = int(np.round(N / 20))
        text_y_pos = int(np.round(M / 20))
        text_scale = 0.8 * M / 300
        dy = int(text_scale*16)
        for str in key_prop_str:
            cv2.putText(img, str,
                        (text_x_pos, text_y_pos), cv2.FONT_HERSHEY_PLAIN,
                        text_scale, color, thickness=2,
                        lineType=cv2.LINE_AA,
                        bottomLeftOrigin=False)
            text_y_pos += dy
        # rotation by angle
        angrad = np.deg2rad(key.angle)
        rotmat_angle = np.array([[np.cos(angrad), -np.sin(angrad)], [np.sin(angrad), np.cos(angrad)]])

        # Draw descriptor size as circle
        center = (int(N/2), int(M/2))
        cv2.circle(img, center, radius, color, thickness=1, lineType=cv2.LINE_AA, shift=0)
        # draw cross wires on circle
        rot_points = np.array([[int(np.round(radius*2/3)), int(np.round(radius*5/4))], [0, 0]])
        rot_points = rotmat_angle @ rot_points
        rotmat = np.array([[0, -1], [1, 0]])
        for rot in range(0, 4):
            rot_points = (rotmat @ rot_points)
            line_points = rot_points.astype(np.int) + np.repeat(np.array(center).reshape(2, 1), 2, axis=1)
            pt1 = tuple(line_points[:, 0].reshape(2,))
            pt2 = tuple(line_points[:, 1].reshape(2,))
            cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA, shift=0)

        # draw orientation bullet
        bullet_center = np.array([int(radius*5/4), 0])
        bullet_radius = int(round(radius/12))
        bullet_center = tuple((rotmat_angle @ bullet_center).astype(np.int) + np.array(center))
        cv2.circle(img, bullet_center, bullet_radius, color, thickness=1, lineType=cv2.LINE_AA, shift=0)            
        return img/255
    
    def img_sift_overview(self, img_overview, x0, x1, y0, y1, color=(255, 255, 255)):
        cv2.line(img_overview, (x0,y0), (x1, y0), color, thickness=1, lineType=cv2.LINE_AA, shift=0)
        cv2.line(img_overview, (x1,y0), (x1, y1), color, thickness=1, lineType=cv2.LINE_AA, shift=0)
        cv2.line(img_overview, (x1,y1), (x0, y1), color, thickness=1, lineType=cv2.LINE_AA, shift=0)
        cv2.line(img_overview, (x0,y1), (x0, y0), color, thickness=1, lineType=cv2.LINE_AA, shift=0)
        return img_overview/255

    def investigator(self, imageName='Overview Image'):

        img_keypoints = cv2.drawKeypoints(self.img, self.keypoints, None)
        plt.figure()
        ax = plt.imshow(img_keypoints)
        plt.title(imageName)
        fig = ax.get_figure()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()


