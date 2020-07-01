import numpy as np
import cv2


class PlateConnectedComponentExtractor:
    def __init__(self, basic_transformations):
        self.bt = basic_transformations

    def extract_plate_connected_component(self, binarized_image):
        eroded_image = self.bt.erosion(binarized_image)
        components_count, output, stats, centroids = cv2.connectedComponentsWithStats(eroded_image, connectivity=4)

        sizes = stats[:, -1]
        centroids_areas = np.column_stack((
            np.arange(components_count, dtype=int),
            centroids,
            np.zeros(components_count),
            sizes
        ))

        # delete background component
        components_areas_centroids = np.delete(centroids_areas, 0, axis=0)
        # sort components descending by size column
        sorted_components_info = components_areas_centroids[components_areas_centroids[:, -1].argsort()[::-1]]

        largest_component_info = self._choose_plate_component(binarized_image, sorted_components_info)
        largest_components_image = np.zeros(output.shape)
        # mask the image to only contain the chosen connected component
        if largest_component_info is not None and largest_component_info[0] is not None:
            largest_components_image[output == largest_component_info[0]] = 255

        return cv2.normalize(largest_components_image, dst=None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    def _choose_plate_component(self, image, sorted_components_info):
        # take two largest components (plate might not be the largest one)
        largest_components_info = sorted_components_info[:2, :]
        image_center = image.shape[::-1]

        def calculate_centroid_distance(row):
            import math
            x_dist = image_center[0] / 2 - row[1]
            y_dist = image_center[1] / 2 - row[2]
            row[3] = math.sqrt(pow(x_dist, 2) + pow(y_dist, 2))
            return row

        # take largest component if it's much bigger than second largest
        # otherwise, choose component, for which distance from image center to its centroid is the smallest
        largest_component_area_trust_threshold = 4
        largest_components_sizes = np.sort(largest_components_info[:, -1])
        # print(largest_components_info)
        if len(largest_components_sizes) == 2:
            if largest_components_sizes[1] / largest_components_sizes[0] <= largest_component_area_trust_threshold:
                np.apply_along_axis(calculate_centroid_distance, 1, largest_components_info)
                # sort ascending by centroid-image center distance
                return largest_components_info[largest_components_info[:, -2].argsort()][0, :]
            else:
                return largest_components_info[0, :]
        elif len(largest_components_sizes) == 1:
            return largest_components_info[0]
