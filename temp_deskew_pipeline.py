from random import randint

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def _find_plate_contour(preprocessed_image, original_image):
    _, contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # take into consideration only 10 contours covering greatest area

    polygons_with_areas = []  # [(contour, area), (...)]

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        # print("Contour area: ", contour_area, " contour is ", contour)
        contour_perimeter = cv2.arcLength(contour, True)
        approximated_contour_polygon = cv2.approxPolyDP(contour, 0.03 * contour_perimeter, closed=True)
        # print("approximated contour polygon", approximated_contour_polygon)
        polygons_with_areas.append((approximated_contour_polygon, contour_area))

    result_polygon = max(polygons_with_areas, key=lambda item: item[1])[0]
    return original_image, result_polygon


def _draw_plate_polygons(image, approximated_polygon):
    return cv2.drawContours(image, [approximated_polygon], -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 3)


def erosion(image, kernel_size=(3, 3), iterations=1):
    kernel_size = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel_size, iterations=iterations)


def morphological_closing(image, kernel_size=(3, 3), iterations=6):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphological_opening(image, kernel_size=(7, 3), iterations=15):
    opening_mask = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=opening_mask, iterations=iterations)
    image = cv2.subtract(image, opening_image)
    return image


# Loads the image then enhances it
image = Image.open('dataset/skewed_trimmed_samples/I00000.png')
contrast = ImageEnhance.Contrast(image)
img = contrast.enhance(10)
img = np.asarray(img)
r, g, b, a = cv2.split(img)
contrast = cv2.merge([b, g, r])
# Reads the enhanced image and converts it to grayscale, creates new file
gray_image = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)  # there is a problem here

ret2, th2 = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

_, result_polygon = _find_plate_contour(th2, img)
polygon_image = _draw_plate_polygons(img, result_polygon)

# writes enhanced and thresholded img
cv2.imwrite('temp3.png', th2)

cv2.imshow("contrast bumped", img)
cv2.imshow("grayscale after contrast", gray_image)
cv2.imshow("otsu", th2)
cv2.imshow("approx polygon", polygon_image)
cv2.waitKey()
