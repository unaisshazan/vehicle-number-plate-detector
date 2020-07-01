from copy import copy


class PipelineTransformations:
    bt = None
    pipeline_debug_enabled = True

    def __init__(self, basic_transformations):
        self.bt = basic_transformations

    def preprocess(self, image):
        image = self.bt.gray_scale(image)
        image = self.bt.bilateral_filter(image)
        # image = self.bt.contrast_brightness(image)
        return image

    def apply_skeletonized_sobel(self, image):
        image_vertical = self.bt.sobel_vertical_edge_detection(copy(image))
        # image_horizontal = self.bt.sobel_horizontal_edge_detection(copy(image))
        image_horizontal = self.bt.canny_edge_detection(copy(image))

        image_vertical = self.bt.skeletonize(image_vertical)
        image_horizontal = self.bt.skeletonize(image_horizontal)

        return image_vertical, image_horizontal

    def apply_morph_opening(self, image):
        image = self.bt.histogram_equalization(image)
        image = self.bt.morphological_opening(image)
        image = self.bt.binary_threshold(image, 80)
        return image

    def apply_color_masks(self, image):
        image_yellow = self.bt.color_mask(copy(image), 'yellow')
        image_red = self.bt.color_mask(copy(image), 'red')
        image_green = self.bt.color_mask(copy(image), 'green')
        # image = self.transforms.color_mask(copy(image), 'blue')

        return [image_yellow, image_green, image_red]
