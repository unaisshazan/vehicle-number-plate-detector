import cv2

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PINK = (255, 192, 203)


def apply_bounding_boxes(image, bands, color=GREEN):
    for band in bands:
        __draw_box(image, band, color)

    return image


def __draw_box(image, band, color):
    x0 = band[2]
    y0 = band[0]
    x1 = band[3]
    y1 = band[1]

    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
