import cv2
import numpy as np
import quantization

#Load Image
image = cv2.iread("./Input/artwork01.jpg")
cv2.imshow('image', image)

#Quantized Image to reduce color map using kmeans
cluster = quantization.quatization_colors(image)
image_q = quantization.generate_quantized_image(image, cluster)
cv2.imshow('imageQuantized', image_q)

#Blurring image
image_blur = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('imageBlur', image_blur)

#Transform to gray scale
image_gray = cv2.cvtColor(image_q, cv2.COLOR_BGR2GRAY)
cv2.imshow('imageGray', image_gray)

#Find Edges
image_edges = cv2.Canny(image_gray, 50, 100)
cv2.imshow('imageEdges', image_edges)

#Find Contours
contours = cv2.findContours(image_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
image_contours = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 1)
cv2.imshow('imageContours', image_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()
