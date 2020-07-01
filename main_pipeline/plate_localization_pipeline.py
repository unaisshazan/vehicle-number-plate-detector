import argparse
import os
import sys
from copy import copy

import cv2

import util.band_clipping as bc
import util.bounding_boxes as bb
import util.heuristics as heuristics
import util.input_output as io
from main_pipeline.candidates import Candidates
from util.basic_transformations import BasicTransformations
from util.image_display_helper import ImageDisplayHelper
from util.pipeline_transformations import PipelineTransformations
from util.vehicles_detection import VehiclesDetector
import main_pipeline.plate_deskewing_pipeline as pdp

image_helper = ImageDisplayHelper(False, subplot_width=2, subplot_height=10)
transformations = PipelineTransformations(BasicTransformations(image_helper))
vehicle_detector = VehiclesDetector()


def main(argv):
    args = parse()

    img_loader = io.BatchImageLoader()
    img_saver = io.ImageSaver(args.output_dir)

    for image in img_loader.load_images(args.input_dir):
        counter = 0
        counter_ocr = 0
        for sub_image in vehicle_detector.detect_vehicles(image.image):
            image.image = sub_image
            candidates = process(image.image)

            image_boxes = apply_bounding_boxex(image.image, candidates)
            image.image = image_boxes
            img_saver.save_image(image, counter)
            counter = counter + 1
            image.image = sub_image

            numrows = len(image.image)
            numcols = len(image.image[0])

            candidates_filtered = filter_heuristically(candidates.all, (numrows, numcols))
            image_boxes = bounding_box_filtered(image.image, candidates_filtered)

            image.image = image_boxes
            img_saver.save_image(image, counter)
            counter = counter + 1

            for idx, bond in enumerate(candidates_filtered):
                y0, y1, x0, x1 = bond
                print(idx, y0, y1, x0, x1)
                image.image = sub_image[y0:y1, x0:x1]
                write_deskewed(image, counter_ocr)
                counter_ocr = counter_ocr + 1

                # ut.show_one_image(sub_image[y0:y1, x0:x1])
                # deskewed = pdp.process_image(sub_image[y0:y1, x0:x1])

            #     if deskewed is not None:
            #         # ut.show_one_image(deskewed)
            #         image.image = deskewed
            #         write_deskewed(image, counter_ocr)
            # #         counter_ocr = counter_ocr + 1
            #         # ocr.read_text(ocr_file)

            image_helper.plot_results()
            image_helper.reset_subplot()


def write_deskewed(image, counter):
    root = '../final_solution/results/toocr/'
    source_name = image.path.split('/')[-1]
    source_name_raw = source_name.split('.')[-2]

    name = '{}_{}.jpg'.format(source_name_raw , counter)
    path = os.path.join(root, name)
    cv2.imwrite(path, image.image)
    print('Saved in: ', path)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_dir', required=True, type=str)
    parser.add_argument('-o', action='store', dest='output_dir', required=True, type=str)

    return parser.parse_args()


def process(image):
    working_image = copy(image)
    working_image = transformations.preprocess(working_image)

    vert_sobel_image, hor_sobel_image = transformations.apply_skeletonized_sobel(copy(working_image))
    opening_method_image = transformations.apply_morph_opening(copy(working_image))
    color_method_image = transformations.apply_color_masks(copy(image))

    try:
        sobel_candidates = bc.find_candidates(bc.sobel_method, vert_sobel_image, hor_sobel_image)
    except ValueError:
        sobel_candidates = []

    try:
        opening_candidates = bc.find_candidates(bc.opening_method, opening_method_image)
    except ValueError:
        opening_candidates = []

    color_candidates = []
    for image_color in color_method_image:
        try:
            candidates = bc.find_candidates(bc.color_method, image_color)
            color_candidates.extend(candidates)
        except ValueError:
            continue

    candidates = Candidates(
        sobel_candidates=sobel_candidates,
        opening_candidates=opening_candidates,
        color_candidates=color_candidates
    )
    return candidates


def apply_bounding_boxex(image, candidates):
    image_boxes = copy(image)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.sobel_candidates, bb.GREEN)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.opening_candidates, bb.RED)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates.color_candidates, bb.BLUE)
    image_helper.add_to_plot(image_boxes, title='Final candidates')
    return image_boxes


def bounding_box_filtered(image, candidates_filtered):
    image_boxes = copy(image)
    image_boxes = bb.apply_bounding_boxes(image_boxes, candidates_filtered, bb.PINK)
    return image_boxes


def filter_heuristically(candidates, image_size):
    candidates = heuristics.remove_big_areas(candidates, image_size)
    candidates = heuristics.remove_vertical(candidates)
    candidates = heuristics.remove_horizontal(candidates, image_size[1])
    candidates = heuristics.join_separated_2(candidates)
    candidates = heuristics.enhance_area(candidates)

    return candidates


if __name__ == '__main__':
    main(sys.argv)
