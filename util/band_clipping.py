import math

import numpy as np
from scipy import signal
import util.utils as utils
import copy

mask_0 = [1, 3, 5, 7, 5, 9, 3, 1]
mask_1 = [1, 5, 9, 12, 15, 12, 9, 5, 1]
mask_2 = [4, 7, 16, 26, 41, 26, 16, 7, 4]
mask_3 = [.006, .061, .242, .383, .242, .061, .006]
mask_4 = [1, 4, 7, 16, 26, 41, 26, 16, 7, 4, 1]
mask_5 = [.000229, .005977,	.060598, .241732, .382928, .241732, .060598, .005977, .000229]
mask_6 = [1, 4, 16, 24, 36, 24, 16, 4, 1]
mask_7 = [2, 5, 8, 16, 20, 24, 30, 37, 30, 24, 20, 16, 8, 5, 2]
mask_8 = [2, 5, 8, 16, 20, 24, 30, 37, 42, 37, 30, 24, 20, 16, 8, 5, 2]


class BandsFinder:

    def __init__(self, image):
        self.image = np.array(image / np.max(image))
        self.mask = mask_8
        self.y_c = 0.30
        self.x_c = 0.42
        self.trim_c = 0.2
        self.derivation_step = 4

    def _find_band(self, projection, c):
        pick = np.argmax(projection)
        pick_value = projection[pick]
        threshold = c * pick_value

        # Find left band
        left_pick_side = projection[0:pick]

        b0 = 0
        for index, intensity in reversed(list(enumerate(left_pick_side))):
            if intensity <= threshold:
                b0 = index
                break

        # Find right band
        right_pick_side = projection[pick + 1:projection.size + 1]

        b1 = pick
        for index, intensity in enumerate(right_pick_side):
            if intensity <= threshold:
                b1 = index
                break

        return b0, pick + b1 + 1

    def _find_y_bands(self, bands_count_limit=5):
        before = y_projection = np.sum(self.image, axis=1).tolist()
        before = y_projection = y_projection / np.max(y_projection)
        y_projection = signal.convolve(y_projection, self.mask, mode='same')

        # utils.plot_histograms(before, y_projection, str(self.mask[0:5]))

        bands = []
        projection = np.copy(y_projection)
        for i in range(bands_count_limit):
            (y0, y1) = self._find_band(projection, c=self.y_c)

            if y1-y0 >= 10:
                bands.append((y0, y1))
                # utils.show_one_image(self.image[y0:y1,:])

            projection[y0:y1+1] = 0

            # utils.plot_histograms(before, projection, str(self.mask[0:5]))

        return bands

    def _find_x_bands_phase_one(self, image, bands_count_limit=3, plate_min_width=25):
        before = x_projection = np.sum(image, axis=0).tolist()
        before = x_projection = x_projection / np.max(x_projection)
        x_projection = signal.convolve(x_projection, self.mask, mode='same')

        # utils.plot_histograms(before, x_projection, str(self.mask[0:5]))
        # utils.show_one_image(image)

        bands = []
        projection = np.copy(x_projection)
        for i in range(bands_count_limit):
            before = copy.copy(projection)
            (x0, x1) = self._find_band(projection, c=self.x_c)

            if x1-x0 >= plate_min_width:
                bands.append((x0, x1))
                # utils.show_one_image(self.image[:, x0:x1])
            projection[x0:x1+1] = 0
            # utils.plot_histograms(before, projection, str(self.mask[0:5]))


        return bands

    def _find_x_bands_phase_two(self, bands):
        trimmed_bands = []

        for y0, y1, x0, x1 in bands:
            plate_area = self.image[y0:y1, x0:x1]
            x_projection = np.sum(plate_area, axis=0).tolist()
            if len(x_projection) > 8:
                (nx0, nx1) = self._trim_plate_area(x_projection, c=self.trim_c)
                trimmed_bands.append((y0, y1, x0+nx0, x0+nx1+1))

        return trimmed_bands

    def _derivate(self, projection, derivation_step):
        derivative = []
        for i in range(derivation_step, len(projection)):
            numerator = (projection[i] - projection[i - derivation_step])
            denominator = derivation_step
            derivative.append(numerator / denominator)

        return derivative

    def _trim_plate_area(self, projection, c):
        derivative = self._derivate(projection, self.derivation_step)
        center_index = math.ceil(len(derivative) / 2)

        # Find where the plate begins
        max_threshold = c * max(derivative)
        left_derivative_side = derivative[0:center_index]

        b0 = center_index
        for index, f_x in enumerate(left_derivative_side):
            if f_x >= max_threshold:
                b0 = index
                break

        # Find where the plate ends
        min_threshold = c * min(derivative)
        right_derivative_side = derivative[center_index + 1:]

        b1 = center_index
        for index, f_x in reversed(list(enumerate(right_derivative_side))):
            if f_x <= min_threshold:
                b1 = index
                break

        # utils.plot_histograms(projection, derivative, str(self.mask[0:5]))

        return b0, center_index + b1 + 1

    def find_bands(self, phase_two_image=None):
        bands = []

        for y0, y1 in self._find_y_bands():
            if y1-y0 <= 10:
                continue

            if phase_two_image is not None:
                band = phase_two_image[y0:y1, ...]
            else:
                band = self.image[y0:y1, ...]

            x_bands = self._find_x_bands_phase_one(band)

            for x0, x1 in x_bands:
                bands.append((y0, y1, x0, x1))

        # bands = self._find_x_bands_phase_two(bands)

        return bands


sobel_method = 'sobel'
opening_method = 'opening'
color_method = 'color'


def find_candidates(method, *image):
    bf = BandsFinder(image[0])

    if method == sobel_method:
        bands = bf.find_bands(image[1])
    else:
        bands = bf.find_bands(image[0])

    return bands
