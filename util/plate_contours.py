from random import randint

import cv2


class PlateContoursFinder:

    def find_plate_contours(self, preprocessed_image):
        _, contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # take into consideration only 10 contours covering greatest area

        polygons_with_areas = []  # [(polygon, area), (...)]

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # print("Contour area: ", contour_area, " contour is ", contour)
            approximated_contour_polygon = self.approx(contour)
            # print("approximated contour polygon", approximated_contour_polygon)
            polygons_with_areas.append((approximated_contour_polygon, contour_area))

        if polygons_with_areas:
            result_polygon = cv2.convexHull(max(polygons_with_areas, key=lambda item: item[1])[0])
            print(result_polygon)
            epsilon = 0.03
            while result_polygon.shape[0] is not 4:
                result_polygon = self.approx(result_polygon, epsilon)
                epsilon = epsilon + 0.01
                print(result_polygon)
            return result_polygon

    def approx(self, contour, epsilon=0.02):
        contour_perimeter = cv2.arcLength(contour, closed=True)
        # print(epsilon)
        approximated_contour_polygon = cv2.approxPolyDP(contour, epsilon * contour_perimeter, closed=True)
        return approximated_contour_polygon

    def draw_plate_polygon(self, image, approx_polygon):
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        return cv2.drawContours(image, [approx_polygon], -1, color, 3)
