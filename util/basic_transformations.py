import cv2
import numpy as np


class BasicTransformations:

    def __init__(self, display_helper=None):
        self.display_helper = display_helper

    def gray_scale(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.display_helper.add_to_plot(image, title='Grayscale', fix_colors=True)
        return image

    def blur(self, image):
        image = cv2.blur(image, ksize=(3, 5))
        self.display_helper.add_to_plot(image, title='Blur', fix_colors=True)
        return image

    def bilateral_filter(self, image, d=16, sigma_color=32, sigma_space=32):
        image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        self.display_helper.add_to_plot(image, title='Bilateral', fix_colors=True)
        return image

    def histogram_equalization(self, image):
        image = cv2.equalizeHist(image)
        self.display_helper.add_to_plot(image, title='Hist equalization', fix_colors=True)
        return image

    def contrast_brightness(self, image, alpha=2, beta=50):
        image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)
        self.display_helper.add_to_plot(image, title='Contrast-brightness', fix_colors=True)
        return image

    def canny_edge_detection(self, image, low_thresh=170, high_thresh=200):
        image = cv2.Canny(image, low_thresh, high_thresh)
        self.display_helper.add_to_plot(image, title='Canny', fix_colors=True)
        return image

    def sobel_vertical_edge_detection(self, image):
        vertical_image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        image = self._normalize_sobel_to_cv8u(vertical_image)
        self.display_helper.add_to_plot(image, title='Vertical sobel', fix_colors=True)
        return image

    def sobel_horizontal_edge_detection(self, image):
        horizontal_image = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        image = self._normalize_sobel_to_cv8u(horizontal_image)
        self.display_helper.add_to_plot(image, title='Horizontal sobel', fix_colors=True)
        return image

    def _normalize_sobel_to_cv8u(self, sobel_image):
        sobelx_64f = sobel_image - np.min(sobel_image)  # to have only positive values
        div = np.max(sobelx_64f) / 255  # calculate the normalize divisor
        sobel_8u = np.uint8(sobelx_64f / div)
        return sobel_8u

    def binary_threshold(self, image, thresh):
        _, threshed = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        self.display_helper.add_to_plot(threshed, title='Binary threshold', fix_colors=True)
        return threshed

    def otsu_threshold(self, image, threshold=0, maxval=255):
        _, threshed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.display_helper.add_to_plot(threshed, title='Otsu threshold', fix_colors=True)
        return threshed

    def negative(self, image):
        negative = image
        non_zeros = np.count_nonzero(image)
        zeros = image.size - non_zeros
        if non_zeros > zeros:
            negative = cv2.bitwise_not(image)
            self.display_helper.add_to_plot(negative, title='Negative', fix_colors=True)
        return negative

    def skeletonize(self, image):
        size = np.size(image)
        skeletonized = np.zeros(image.shape, np.uint8)

        image = self.otsu_threshold(image)
        image = self.negative(image)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(image, temp)
            skeletonized = cv2.bitwise_or(skeletonized, temp)
            image = eroded.copy()

            zeros = size - cv2.countNonZero(image)
            if zeros == size:
                done = True

        self.display_helper.add_to_plot(skeletonized, title='Skeletonization', fix_colors=True)
        return skeletonized

    def morphological_opening(self, image, kernel_size=(7, 3), iterations=15):
        opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=opening_mask, iterations=iterations)
        image = cv2.subtract(image, opening_image)
        self.display_helper.add_to_plot(image, title='Morph opening', fix_colors=True)
        return image

    def morphological_closing(self, image, kernel_size=(3, 3), iterations=6):
        kernel = np.ones(kernel_size, np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        self.display_helper.add_to_plot(image, title='Morph closing', fix_colors=True)
        return image

    def erosion(self, image, kernel_size=(3, 3), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        image = cv2.erode(image, kernel, iterations=iterations)
        self.display_helper.add_to_plot(image, title='Erosion', fix_colors=True)
        return image

    def dilation(self, image, kernel_size=(3, 3), iterations=1):
        kernel = np.ones(kernel_size, np.uint8)
        image = cv2.dilate(image, kernel, iterations=iterations)
        self.display_helper.add_to_plot(image, title='Dilation', fix_colors=True)
        return image

    def color_mask(self, image, color):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color == 'yellow':
            lower_mask = np.array([10, 100, 100])  # Yellow
            upper_mask = np.array([60, 255, 255])  # Yellow
        elif color == 'green':
            lower_mask = np.array([73, 100, 100])  # Green
            upper_mask = np.array([93, 255, 255])  # Green
        elif color == 'red':
            lower_mask = np.array([0, 30, 60])  # Red
            upper_mask = np.array([10, 120, 100])  # Red
        elif color == 'blue':
            lower_mask = np.array([20, 100, 100])  # Blue
            upper_mask = np.array([30, 255, 255])  # Blue
        else:
            raise Exception('Specified color not supported')

        mask = cv2.inRange(image_hsv, lower_mask, upper_mask)
        self.display_helper.add_to_plot(mask, title='{} mask'.format(color), fix_colors=True)
        return mask
