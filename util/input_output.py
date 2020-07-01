import os

import cv2


class Image:

    def __init__(self, image, path):
        self.image = image
        self.path = path


class BatchImageLoader:

    def load_images(self, source_path):
        """
        Generate images from source.

        :param source_path:
        :return: image: Image
        """

        for filename in os.listdir(source_path):
            if any(extension in filename for extension in [".jpg", ".jpeg", ".png"]):
                relative_path = "{}/{}".format(source_path, filename)
                image = cv2.imread(relative_path)
                yield Image(
                    image=image,
                    path=filename
                )


class ImageSaver:

    def __init__(self, path):
        self.path = path

    def save_image(self, image, counter):
        path = self.__make_save_path(image.path, counter)
        cv2.imwrite(path, image.image)
        print('Image saved at:' + path)

    def __make_save_path(self, source_path, counter):
        source_name = source_path.split('/')[-1]

        name = source_name + str(counter) + source_name
        path = os.path.join(self.path, name)

        return path
