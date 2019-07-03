import cv2
import numpy as np

def histCalculation_BGR (image):
    bgr_planes = cv2.split(image)
    hist_size = 256
    hist_range = (0, 256)  # the upper boundary is exclusive
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [hist_size], hist_range, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [hist_size], hist_range, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [hist_size], hist_range, accumulate=accumulate)
    return [b_hist, g_hist, r_hist]


def histCalulation_HSV(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist


def histDraw (histograms, hist_size):
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / hist_size))
    hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    for histogram in histograms:
        cv2.normalize(histogram, histogram, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i in range(1, hist_size):
            cv2.line(hist_image, (bin_w * (i - 1), hist_h - int(np.round(histogram[i - 1]))),
                     (bin_w * i, hist_h - int(np.round(histogram[i], 0))),
                     colors[(i-1) % len(colors)],
                     thickness=2)

    return hist_image


def his_equal_bgr(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    eqimage = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return eqimage

def hist_H_calculation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    hist_range = (0, 180)  # the upper boundary is exclusive
    h_hist = cv2.calcHist(hsv_planes, [0], None, [180], hist_range, accumulate=False)
    return h_hist

def count_colors(image):
    h_hist = histograms.hist_H_calculation(image)
    res = sum(map(lambda i: i > 100, h_hist))
    return res

