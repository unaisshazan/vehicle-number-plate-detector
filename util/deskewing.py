import numpy as np
import cv2


class PlateDeskewingTransformer:

    def four_point_transform(self, image, points):
        rect = self._order_corner_points(points)
        (top_left, top_right, bottom_right, bottom_left) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
        height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        warp_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, warp_matrix, (max_width, max_height))

        return warped

    def _order_corner_points(self, points):
        points = [item for sublist in points for item in sublist]
        points = np.array([(arr[0], arr[1]) for arr in points])
        # initialize a list of coordinates that will be ordered top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        ooints_sum = points.sum(axis=1)
        rect[0] = points[np.argmin(ooints_sum)]
        rect[2] = points[np.argmax(ooints_sum)]
        # top-right point will have the smallest difference, bottom-left will have the largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect
