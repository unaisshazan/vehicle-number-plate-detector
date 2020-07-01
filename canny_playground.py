from util.utils import *


def cannyEdge():
    global img, minT, maxT
    edge = cv2.Canny(img, minT, maxT)
    cv2.imshow("edges", edge)


def adjustMinT(v):
    global minT
    minT = v
    cannyEdge()


def adjustMaxT(v):
    global maxT
    maxT = v
    cannyEdge()


if __name__ == '__main__':
    img = cv2.imread("license_plate_snapshots/test_001.jpg", cv2.IMREAD_GRAYSCALE)
    img = bilateral_filter(img)

    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    minT = 30
    maxT = 150
    cv2.createTrackbar("minT", "edges", minT, 255, adjustMinT)
    cv2.createTrackbar("maxT", "edges", maxT, 255, adjustMaxT)
    cannyEdge()
    cv2.waitKey(0)
