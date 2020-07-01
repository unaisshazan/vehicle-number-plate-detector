import numpy as np
from PIL import Image, ImageEnhance

from util.basic_transformations import BasicTransformations
from util.image_display_helper import ImageDisplayHelper
from util.input_output import *

display = ImageDisplayHelper(False, 2, 5)
bt = BasicTransformations(display)


def process(image_path):
    display = ImageDisplayHelper(False, 2, 5)
    bt = BasicTransformations(display)

    image = Image.open(image_path)
    contrast_bumped_image = np.asarray(ImageEnhance.Contrast(image).enhance(5))
    image = np.asarray(image)

    gray_image = bt.gray_scale(contrast_bumped_image)
    binarized_image = bt.otsu_threshold(gray_image)
    eroded_image = bt.erosion(binarized_image)
    closed_image = bt.morphological_closing(binarized_image, iterations=5)
    eroded_closed_image = bt.morphological_closing(eroded_image, iterations=5)

    _, result_polygon = _find_plate_contour(eroded_closed_image, image)
    polygon_flat_list = [item for sublist in result_polygon for item in sublist]
    plate_corners_list = [(arr[0], arr[1]) for arr in polygon_flat_list]

    deskewed_image = four_point_transform(gray_image, np.array(plate_corners_list))
    _draw_plate_polygons(image, result_polygon)

    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray_image)
    cv2.imshow("Contrast bumped", contrast_bumped_image)
    cv2.imshow("Bin", binarized_image)
    cv2.imshow("Bin -> Erosion", eroded_image)
    cv2.imshow("Bin -> Erosion -> Closing", eroded_closed_image)
    cv2.imshow("Bin -> Closing", closed_image)
    cv2.imshow("Result Polygon", image)
    cv2.imshow("Deskewed image", deskewed_image)
    cv2.imwrite("ocr-ready.jpg", deskewed_image)

    cv2.waitKey()


def deskew(image):
    gray_image = bt.gray_scale(image)
    contrast_bumped_image = bt.contrast_brightness(gray_image, alpha=2, beta=50)

    binarized_image = bt.otsu_threshold(gray_image)
    eroded_image = bt.erosion(binarized_image)
    closed_image = bt.morphological_closing(binarized_image, iterations=5)
    eroded_closed_image = bt.morphological_closing(eroded_image, iterations=5)
    # ut.show_one_image(eroded_closed_image)

    _, result_polygon = _find_plate_contour(eroded_closed_image, image)
    polygon_flat_list = [item for sublist in result_polygon for item in sublist]
    plate_corners_list = [(arr[0], arr[1]) for arr in polygon_flat_list]

    deskewed_image = four_point_transform(gray_image, np.array(plate_corners_list))
    _draw_plate_polygons(image, result_polygon)
    return deskewed_image


if __name__ == '__main__':
    process("dataset/skewed_trimmed_samples/skewed_009.jpg")
