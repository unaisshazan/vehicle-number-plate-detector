import cv2
import numpy as np

from datasets import DatasetsProvider, sample
from util.band_clipping import BandsFinder

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def show_bounds(img, band, color):
    x = band[2]
    y = band[0]
    x1 = band[3]
    y1 = band[1]

    cv2.rectangle(img, (x, y), (x1, y1), color, 2)


def canny_method(image):
    canny_image = canny_edge_detection(image)

    bf = BandsFinder(canny_image)
    bands = bf.find_bands()

    return bands


def run_pipelines_real_dataset():
    dp = DatasetsProvider(
        source_path='/home/lukasz/Studia/Analiza obrazow i wideo/Datasets/UFPR-ALPR dataset/'
    )
    for image, position, number in dp.images():
        grayscale_image = gray_scale(image)
        noise_removed_image = bilateral_filter(grayscale_image)

        sobel_bands = skeletonized_sobel_method(noise_removed_image)
        canny_bands = canny_method(noise_removed_image)
        thresh_bands = opening_method(noise_removed_image)

        for band in canny_bands:
            show_bounds(image, band, GREEN)

        for band in thresh_bands:
            show_bounds(image, band, RED)

        for band in sobel_bands:
            show_bounds(image, band, BLUE)

        save_image(image, number, position)

def process():

    image, name = sample('006')
    grayscale_image = gray_scale(image)
    noise_removed_image = bilateral_filter(grayscale_image)

    horizontal_sobel = sobel_horizontal_edge_detection(noise_removed_image)
    cv2.threshold(horizontal_sobel, 135, 255, cv2.THRESH_TOZERO, horizontal_sobel)
    horizontal_sobel_skeleton, horizontal_sobel_thresh = skeletonization(horizontal_sobel)

    vertical_sobel = sobel_vertical_edge_detection(noise_removed_image)
    vertical_sobel_skeleton, vertical_sobel_thresh = skeletonization(vertical_sobel)

    bf = BandsFinder(horizontal_sobel_skeleton)
    bands = bf.find_bands()

    for band in bands:
        show_bounds(image, band, RED)

    display.add_to_plot(grayscale_image, 1, 'grayscale')
    display.add_to_plot(noise_removed_image, 2, 'bilateral')
    display.add_to_plot(image, 3, 'image with bounds')
    display.add_to_plot(horizontal_sobel, 3, 'horizontal sobel')
    display.add_to_plot(horizontal_sobel_thresh, 4, 'horizontal sobel threshold')
    display.add_to_plot(horizontal_sobel_skeleton, 5, 'horizontal sobel skeleton')
    display.add_to_plot(vertical_sobel, 6, 'vertical sobel')
    display.add_to_plot(vertical_sobel_thresh, 7, 'vertical sobel threshold')
    display.add_to_plot(vertical_sobel_skeleton, 8, 'vertical sobel skeleton')

    plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    fig = plt.gcf()
    fig.set_size_inches(10, 15)
    # print('calculated')
    plt.show()


if __name__ == '__main__':

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower_red = np.array([20, 100, 100]) # Yellow
    upper_red = np.array([30, 255, 255]) # Yellow

    # lower_red = np.array([0, 100, 50])  # Green
    # upper_red = np.array([100, 255, 255])  # Green

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(im, im, mask=mask)


    plt.imshow(res)
    plt.show()
    # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)

    lower_red = np.array([0, 100, 50])  # Green
    upper_red = np.array([100, 255, 255])  # Green

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(im, im, mask=mask)

    plt.imshow(res)
    plt.show()
    # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)
