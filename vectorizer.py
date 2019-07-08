import cv2
import quantization
import os
import vectorfile
from skimage import io
from urllib.parse import urlparse


class Vectorizer:
    def processimage(self, image):
        # Quantized Image to reduce color map using kmeans
        clusters = quantization.quatization_colors(image)
        image_q = quantization.generate_quantized_image(image, clusters)
        first_pixel = image_q[0, 0]

        # Generate one image per region and get contours, to avoid overlaped areas
        general_contours = []
        for i in range(len(clusters.cluster_centers_)):
            centroid = clusters.cluster_centers_[i]
            if i != clusters.labels_[0]:
                image_region_BN = quantization.generate_quantized_imageBW(image, clusters, i)
                # Create a binary image
                (thres, image_bw) = cv2.threshold(image_region_BN, 200, 255, cv2.THRESH_BINARY)

                # Transform to gray scale
                image_gray = cv2.cvtColor(image_bw, cv2.COLOR_BGR2GRAY)
                # Find Contours
                contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = list(filter(lambda x: cv2.contourArea(x) > 1.0, contours))
                general_contours += contours

        return image_q, general_contours, first_pixel

    def processURL(self, url):
        print("URL: " + url)
        image_raw = io.imread(url)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image_q, general_contours, first_pixel = self.processimage(image)
        vectorfile.savetosvg(image_q, general_contours, first_pixel, 'testOutput')
        directory = os.getcwd()
        return str('file://' + directory + 'testOutput.svg')
