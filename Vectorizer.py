import cv2
import numpy as np
import quantization

#Load Image
image = cv2.imread("./Input/artwork01.jpg")
print('Image Shape' + str(image.shape))
cv2.imshow('image', image)

#Quantized Image to reduce color map using kmeans
cluster = quantization.quatization_colors(image)
image_q = quantization.generate_quantized_image(image, cluster)
cv2.imshow('imageQuantized', image_q)

first_pixel = image_q[0, 0]
background = [0,0,0]
image_qbw = image_q.copy()
print(first_pixel)
image_qbw[np.where((image_q == first_pixel).all(axis=2))] = background

cv2.imshow('imageQuantizedBN', image_qbw)

#Create a binary image
(thres, image_bw) = cv2.threshold(image_qbw, 1, 255, cv2.THRESH_BINARY)
cv2.imshow('imageBW', image_bw)

#Transform to gray scale
image_gray = cv2.cvtColor(image_bw, cv2.COLOR_BGR2GRAY)
cv2.imshow('imageGray', image_gray)

#Find Edges
image_edges = cv2.Canny(image_gray, 1, 200)
cv2.imshow('imageEdges', image_edges)

#Find Contours
contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda x: cv2.contourArea(x) > 1.0, contours))

#Draw contours
image_contours = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 1)
cv2.imshow('imageContours', image_contours)

savetosvg(image_q,contours,first_pixel)

cv2.waitKey(0)
cv2.destroyAllWindows()
