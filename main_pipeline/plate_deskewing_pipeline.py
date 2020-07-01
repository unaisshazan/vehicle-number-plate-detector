import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from util.basic_transformations import BasicTransformations
from util.deskewing import PlateDeskewingTransformer
from util.image_display_helper import ImageDisplayHelper
from util.plate_connected_component import PlateConnectedComponentExtractor
from util.plate_contours import PlateContoursFinder

display_helper = ImageDisplayHelper(True, 2, 10)
bt = BasicTransformations(display_helper)
cf = PlateContoursFinder()
ex = PlateConnectedComponentExtractor(bt)
ds = PlateDeskewingTransformer()


def process_path(image_path):
    process_image(cv2.imread(image_path), image_path)


def process_image(image, image_path=''):
    print('\nProcessing {}...'.format(image_path))
    image = Image.fromarray(image)
    contrast_image = ImageEnhance.Contrast(image)
    image = contrast_image.enhance(9)
    image = np.asarray(image.copy())
    channels_list = cv2.split(image)
    contrast_image = cv2.merge([channels_list[2], channels_list[1], channels_list[0]])  # b, g, r

    display_helper.add_to_plot(contrast_image, title="Contrast bump")
    gray_image = bt.gray_scale(contrast_image)
    binarized_image = bt.otsu_threshold(gray_image)

    plate_component_image = ex.extract_plate_connected_component(binarized_image)
    display_helper.add_to_plot(plate_component_image, title="Plate connected component")

    plate_polygon = cf.find_plate_contours(plate_component_image)

    if plate_polygon is not None:
        polygon_image = cf.draw_plate_polygon(image.copy(), plate_polygon)
        display_helper.add_to_plot(polygon_image, title="Approx polygon")

        # contours = np.asarray(cv2.findContours(plate_component_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
        # hough_lines(plate_component_image, image.copy())

        deskewed_image = None
        if plate_polygon.shape[0] == 4:
            deskewed_image = ds.four_point_transform(gray_image, plate_polygon)
            display_helper.add_to_plot(deskewed_image, title="Deskewed image")

        display_helper.plot_results()
        display_helper.reset_subplot()

        # cv2.imwrite('../output/ocr_ready/{}'.format(image_path.split('/')[-1]), deskewed_image)

        return deskewed_image


def hough_lines(gray_image, img):
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    min_line_length = 50
    max_line_gap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        display_helper.add_to_plot(img, title="Hough Lines Prob")


if __name__ == '__main__':
    dir_path = '../dataset/skewed_trimmed_samples/'
    for filename in os.listdir(dir_path):
        if filename.startswith("IMG"):
            process_path('{}{}'.format(dir_path, filename))
