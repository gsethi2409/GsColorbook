import cv2 as cv
import numpy as np
import sys

from sklearn.cluster import KMeans

class ColorBook(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = cv.imread(img_path)
        self.x, self.y, self.channels = self.img.shape
        # call pre-blur function here
        self.preblur()

        self.raw_img = np.asarray(self.img)
        self.color_centroids = 10
        # call normalize image function here
        self.normalize()

    def preblur(self):
        self.im = cv.GaussianBlur(self.img,(5,5),0)
    
    def normalize(self):
        self.max_depth = np.max(np.max(np.max(self.raw_img)))
        self.scaled_img = self.raw_img/self.max_depth
        self.flattened = np.reshape(self.scaled_img, (self.x*self.y,3))

    def perform_kmeans(self):
        self.init_centroids = np.array([[k/self.color_centroids,
                                         k/self.color_centroids,
                                         k/self.color_centroids]
                                        for k in range(self.color_centroids)], 
                                         dtype = 'float32')
        # print(self.init_centroids)
        self.kmeans = KMeans(n_clusters=self.color_centroids, init='k-means++', 
                             max_iter=10, n_init=3)
        self.cluster_indices=self.kmeans.fit_predict(self.flattened)
        self.cluster_centers=self.kmeans.cluster_centers_

        # print(self.cluster_centers)
        # print(self.cluster_indices)

    def quantize_image(self):
        self.colorized = np.zeros(np.shape(self.flattened),dtype='uint8')
        for i, pixel in enumerate(self.flattened):
            cluster_index = self.cluster_indices[i]
            center = np.array(self.cluster_centers[cluster_index]*255, 
                              dtype='uint8')
            self.colorized[i] = center
        cv.imshow("Display window", self.img)
        k = cv.waitKey(0)

if __name__ == "__main__":
    colorbook = ColorBook("/home/gsethi2409/Personal-Projects/image-processing/ColorBooked/examples/barrhorn/barrhorn_colorized12.png")
    colorbook.perform_kmeans()
    colorbook.quantize_image()