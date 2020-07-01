import logging

from matplotlib import pyplot as plt

logger = logging.getLogger()


def plot(figure, subplot, image, title):
    figure.subplot(subplot)

    figure.imshow(image)
    figure.xlabel(title)
    figure.xticks([])
    figure.yticks([])
    return True


def plot_(figure, subplot, image, title):
    figure.subplot(subplot)

    figure.plot(image)
    figure.xlabel(title)

    return True


def show_one_image(image):
    plt.figure("Show image", figsize=(30, 30))
    plot(plt, 111, image, "Original image")
    plt.tight_layout()
    plt.show()

    return True


def plot_histograms(hist_1, hist_2, title):
    plt.figure("Histograms", figsize=(10, 5))
    plot_(plt, 121, hist_1, "Before")
    plot_(plt, 122, hist_2, "After")

    plt.title(title)
    plt.show()
