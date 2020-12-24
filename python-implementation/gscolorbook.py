import cv2 as cv
import numpy as np
import sys

from PIL import Image
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageDraw
from sklearn.cluster import KMeans
from skimage import feature, color

import matplotlib                                                                  
import matplotlib.pyplot as plt                                      
import matplotlib.gridspec as gridspec                                             

class ColorBook(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.im = Image.open(img_path)

        if(self.im.mode != 'RGB'):
            background = Image.new("RGB", self.im.size, (255, 255, 255))
            background.paste(self.im, mask=self.im.split()[3]) # 3 is the alpha channel
            self.im = background

        self.ypix, self.xpix = self.im.size
        # call pre-blur function here
        self.preblur()

        self.raw_img = np.asarray(self.im)
        self.color_centroids = 5
        # call normalize image function here
        

        print("Raw image <%s> read in:"%(self.img_path))
        print("Pixels (width=%d x height=%d):"%(self.xpix, self.ypix))
        print("Image Mode = ", self.im.mode)

        self.normalize()

    def preblur(self):
        blur=max(self.xpix,self.ypix)/500
        self.im = self.im.filter(ImageFilter.GaussianBlur(blur))

    def normalize(self):
        self.max_depth = np.max(np.max(np.max(self.raw_img)))
        self.scaled_img = self.raw_img/self.max_depth
        self.flattened = np.reshape(self.scaled_img, (self.xpix*self.ypix,3))
        # cv.imwrite("normalized.png", self.img)

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
        # print(self.cluster_indices)
        self.cluster_centers=self.kmeans.cluster_centers_
        # print(self.cluster_centers)

        self.quantize_image()
        self.img_final = np.reshape(self.colorized,(self.xpix, self.ypix, 3))
        im = Image.fromarray(self.img_final)

        # draw = ImageDraw.Draw(im)
        # font = ImageFont.truetype('/home/gsethi2409/Personal-Projects/image-processing/GsColorbook/Aaargh.ttf', 16)
        # draw.text((0,0), "lalalalalala", (255,0,0), font = font)
        
        im.save("colorized.png") 
        
        # print(self.cluster_centers)
        # print(self.cluster_indices)

    def quantize_image(self):
        self.colorized = np.zeros(np.shape(self.flattened),dtype='uint8')
        # draw = ImageDraw.Draw(Image.fromarray(self.colorized))
        # font = ImageFont.truetype('/home/gsethi2409/Personal-Projects/image-processing/GsColorbook/Aaargh.ttf', 16)

        print(self.im.size)
        print(self.flattened.shape)
        
        for i, pixel in enumerate(self.flattened):
            # print(i, ' : ', pixel)
            cluster_index = self.cluster_indices[i]
            # print(cluster_index)
            center = np.array(self.cluster_centers[cluster_index]*255, 
                              dtype='uint8')
            # print(center)
            self.colorized[i] = center
            # draw.text((pixel[0],pixel[1]), "lalalalalala", fill='blue', font = font)

    def outline_img(self):

        sigma=0.1

        edges_r = feature.canny(self.img_final[:,:,0],sigma=sigma)
        edges_g = feature.canny(self.img_final[:,:,1],sigma=sigma)
        edges_b = feature.canny(self.img_final[:,:,2],sigma=sigma)

        final_edges = np.logical_or(edges_r, edges_g)
        final_edges = np.logical_or(final_edges, edges_b) 
        final_edges = np.array(np.logical_not(final_edges)*self.max_depth,
                               dtype = 'uint8')

        final_edges[:,0] = 0
        final_edges[:,np.shape(final_edges)[1]-1] = 0
        final_edges[0,:] = 0
        final_edges[np.shape(final_edges)[0]-1,:] = 0

        towrite = Image.fromarray(final_edges)
        towrite.save("outlined.png")

        # generates palatte
        curr_fig=0                                                              
        plt.figure(curr_fig,figsize=(7,0.6))                                    
        gs = gridspec.GridSpec(1,self.color_centroids)
        gs.update(left=0.09, right=0.95, top=0.95, bottom=0.15,wspace=0.3)      
        for i in range(self.color_centroids):
            ax = plt.subplot(gs[i])
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
            ax.fill_between(xlim,[ylim[0],ylim[0]],[ylim[1],ylim[1]],color=self.cluster_centers[i])

            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        plt.savefig("palette.png")
        plt.close()


if __name__ == "__main__":
    colorbook = ColorBook("../images/abs.png")
    colorbook.perform_kmeans()
    colorbook.quantize_image()
    colorbook.outline_img()