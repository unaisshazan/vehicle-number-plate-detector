import os

import cv2
import numpy as np

classes = [line.strip() for line in open("yolo_suite/classes.txt", 'r').readlines()]
classes_of_interest = ['car', 'motorbike', 'bus', 'truck']
classes_of_interest_ids = [classes.index(class_name) for class_name in classes_of_interest]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def _get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def detect_vehicles(image_path):
    image = cv2.imread(image_path)
    image_width = image.shape[1]
    image_height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet("yolo_suite/yolov3.weights", "yolo_suite/yolov3.cfg")
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(_get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    confidence_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    _show_and_save_detected_vehicles_predictions(boxes, class_ids, confidences, image, image_path, indices)
    # _create_temp_dataset(image, boxes, image_path, indices)


def _show_and_save_detected_vehicles_predictions(boxes, class_ids, confidences, image, image_path, indices):
    for i in indices:
        i = i[0]
        if class_ids[i] < len(classes) and class_ids[i] in classes_of_interest_ids:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            _draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    # print(image_path.split('/')[-1])
    cv2.imwrite("output/yolov3/{}".format(image_path.split('/')[-1]), image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def _draw_prediction(image, class_id, confidence, x, y, x_plus_w, y_plus_h):
    if class_id < len(classes):
        class_name = str(classes[class_id])
        label = '{} {}'.format(class_name, confidence)
        color = COLORS[class_id]
        cv2.rectangle(image, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _create_temp_dataset(image, boxes, image_path, indices):
    boxes_areas = [calculate_area(box) for box in boxes]
    largest_box_index = np.argmax(boxes_areas)
    box = boxes[largest_box_index]

    x = int(max(box[0], 0))
    y = int(max(box[1], 0))
    w = int(abs(box[2]))
    h = int(abs(box[3]))

    cropped_image = image[y: y + h, x: x + w]
    # cv2.imshow("cropped", cropped_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite("output/{}".format(image_path), cropped_image)


def calculate_area(box):
    return box[2] * box[3]


if __name__ == '__main__':
    for index, file_name in enumerate(os.listdir("dataset/UFPR-ALPR-snapshots"), 1):
        if any(file_name.endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
            # print("Processing image {}: {}".format(index, file_name))
            detect_vehicles("dataset/UFPR-ALPR-snapshots/{}".format(file_name))
