
from sklearn.cluster import KMeans
import cv2
import numpy as np
from matplotlib import pyplot as plot


def plotOptimarColors(image):
    results = quatization_colors(image)
    print(results)
    plot.plot(range(1, 20), results, 'bx-')
    plot.xlabel('k')
    plot.ylabel('Sum_of_squared_distances')
    plot.title('Elbow Method For Optimal k')
    plot.show()



def quatization_colors(image):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vector = imageRGB.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=1)
    clt.fit(vector)
    previous_res = clt
    results = [previous_res.inertia_]
    for i in range(2, 20):
        clt = KMeans(n_clusters=i)
        clt.fit(vector)
        ratio = (previous_res.inertia_ - clt.inertia_)/previous_res.inertia_
        results.append(ratio)
        if ratio < 0.2:
            break
        else:
            previous_res = clt

    print(results)
    return previous_res


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def plot_quantization(image):
    kmeans = quatization_colors(image)
    hist = centroid_histogram(kmeans)
    bar = plot_colors(hist, kmeans.cluster_centers_)
    # show our color bart
    plot.figure()
    plot.axis("off")
    plot.imshow(bar)
    plot.show()

def generate_quantized_image(image, clusters):
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vector = imageRGB.reshape((image.shape[0] * image.shape[1], 3))
    for i in range(vector.shape[0]):
        vector[i] = clusters.cluster_centers_[clusters.labels_[i]]

    image_quant = vector.reshape(image.shape[0], image.shape[1], 3)
    return cv2.cvtColor(image_quant, cv2.COLOR_RGB2BGR)
