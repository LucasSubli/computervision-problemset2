# USAGE
# python problem1.py --image image/image1.jpg

# import the necessary packages
import numpy as np
import argparse
from matplotlib import pyplot as plt
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())



#-------------------------------------------------------------
#Loading a color (RGB) image
img = cv2.imread(args["image"])

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#-------------------------------------------------------------
#Perform some neat edge detection
cannyEdges = cv2.Canny(img,100,200)

#Perform some neat edge detection
laplacianEdges = cv2.Laplacian(grayImg,cv2.CV_8U)

#Edge detect on the edge detector (inception)
cannyThenLaplacian = cv2.Laplacian(cannyEdges,cv2.CV_8U)

#The other way around
laplacianThanCanny = cv2.Canny(laplacianEdges,100,200)


#-------------------------------------------------------------
#Plot everything


#Convert opencv BGR to RGB
#Y u do this to me OpenCv...
plt.subplot(321),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#Plot the gray level image
plt.subplot(322),plt.imshow(grayImg, cmap = 'gray')
plt.title('Gray level Image'), plt.xticks([]), plt.yticks([])

#Plot the canny edge detection
plt.subplot(323),plt.imshow(cannyEdges,cmap = 'gray')
plt.title('Canny Image'), plt.xticks([]), plt.yticks([])

#Plot the laplacian edge detection
plt.subplot(324),plt.imshow(laplacianEdges,cmap = 'gray')
plt.title('Laplacian Image'), plt.xticks([]), plt.yticks([])

#Plot the Canny into laplacian edge detection
plt.subplot(325),plt.imshow(cannyThenLaplacian,cmap = 'gray')
plt.title('Canny -> Laplacian'), plt.xticks([]), plt.yticks([])

#Plot the laplacian into canny edge detection
plt.subplot(326),plt.imshow(laplacianThanCanny,cmap = 'gray')
plt.title('Laplacian -> Canny'), plt.xticks([]), plt.yticks([])

#Show everything
plt.show()


