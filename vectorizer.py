import cv2
import numpy as np
import quantization
import vectorfile

#Load Image
image = cv2.imread("./Input/artwork01.jpg")
cv2.imshow('image', image)

#Quantized Image to reduce color map using kmeans
clusters = quantization.quatization_colors(image)
image_q = quantization.generate_quantized_image(image, clusters)
cv2.imshow('imageQuantized', image_q)
first_pixel = image_q[0, 0]

#Generate one image per region and get contours, to avoid overlaped areas
general_contours = []
for i in range(len(clusters.cluster_centers_)):
    centroid = clusters.cluster_centers_[i]
    if i != clusters.labels_[0]:
        image_region_BN = quantization.generate_quantized_imageBW(image, clusters, i)
        # Create a binary image
        (thres, image_bw) = cv2.threshold(image_region_BN, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow('imageBW' + str(i), image_bw)

        # Transform to gray scale
        image_gray = cv2.cvtColor(image_bw, cv2.COLOR_BGR2GRAY)
        # Find Contours
        contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 1.0, contours))
        general_contours += contours


#Draw contours
image_contours = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
cv2.drawContours(image_contours, general_contours, -1, (255, 255, 255), 1)
cv2.imshow('imageContours', image_contours)

vectorfile.savetosvg(image_q, general_contours, first_pixel)

cv2.waitKey(0)
cv2.destroyAllWindows()
