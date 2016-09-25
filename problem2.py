# USAGE
# python problem2.py --image image/image1.jpg

# import the necessary packages
import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())


#-------------------------------------------------------------
#Helper function
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



#-------------------------------------------------------------
# load the image and define the window width and height
imgColor = cv2.imread(args["image"])
img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
result = img.copy()
(winW, winH) = (5, 5)

#get the standard deviation
_, stdDev = cv2.meanStdDev(img)

# MANUAL IMPLEMENTATION JUST FOR CURIOSITY -- UNUSED
# WILL KEET IT HERE FOR DOCUMENTATION
#-------------------------------------------------------------
# # loop over the sliding window
# for (x, y, window) in sliding_window(img, stepSize=1, windowSize=(winW, winH)):
# 	# if the window does not meet our desired window size, ignore it
# 	# AKA border window
# 	if window.shape[0] != winH or window.shape[1] != winW:
# 		continue


# 	filter = window[(window > window[3,3] - stdDev) & (window < window[3,3] + stdDev)] 

# 	#print(window[3,3] - stdDev)
# 	#print(window[3,3] + stdDev)
	
# 	#print(window)
# 	#print(filter)
# 	#print(np.mean(filter))
# 	result[y,x] = np.mean(filter)

# 	# since we do not have a classifier, we'll just draw the window
# 	clone = img.copy()
# 	cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
# 	cv2.imshow("Window", clone)
# 	cv2.waitKey(1)
# 	#time.sleep(0.002)
	

	
plt.subplot(531),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#Plot the gray level image
plt.subplot(532),plt.imshow(cv2.equalizeHist(img), cmap = 'gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])

#Plot the canny edge detection
plt.subplot(533),plt.hist(cv2.equalizeHist(img).ravel(),256,[0,256]) 
plt.title('Equalized Hist'), plt.xticks([]), plt.yticks([])

#this function does no like 0 so we use something close to 0
gaussianImg = cv2.GaussianBlur(img,(5,5), 0.001)
equalizedImg = cv2.equalizeHist(gaussianImg)

plt.subplot(534),plt.imshow(gaussianImg, cmap = 'gray')
plt.title('r=0 Image'), plt.xticks([]), plt.yticks([])

#Plot the gray level image
plt.subplot(535),plt.imshow(cv2.equalizeHist(gaussianImg), cmap = 'gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])

#Plot the canny edge detection
plt.subplot(536),plt.hist(cv2.equalizeHist(gaussianImg).ravel(),256,[0,256]) 
plt.title('Equalized Hist'), plt.xticks([]), plt.yticks([])

gaussianImg = cv2.GaussianBlur(img,(5,5), 0.5)
equalizedImg = cv2.equalizeHist(gaussianImg)

plt.subplot(537),plt.imshow(gaussianImg, cmap = 'gray')
plt.title('r=0.5 Image'), plt.xticks([]), plt.yticks([])

#Plot the gray level image
plt.subplot(538),plt.imshow(cv2.equalizeHist(gaussianImg), cmap = 'gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])

#Plot the canny edge detection
plt.subplot(539),plt.hist(cv2.equalizeHist(gaussianImg).ravel(),256,[0,256]) 
plt.title('Equalized Hist'), plt.xticks([]), plt.yticks([])

gaussianImg = cv2.GaussianBlur(img,(5,5), 1)
equalizedImg = cv2.equalizeHist(gaussianImg)

plt.subplot(5,3,10),plt.imshow(gaussianImg, cmap = 'gray')
plt.title('r=1 Image'), plt.xticks([]), plt.yticks([])

#Plot the gray level image
plt.subplot(5,3,11),plt.imshow(cv2.equalizeHist(gaussianImg), cmap = 'gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])

#Plot the canny edge detection
plt.subplot(5,3,12),plt.hist(cv2.equalizeHist(gaussianImg).ravel(),256,[0,256]) 
plt.title('Equalized Hist'), plt.xticks([]), plt.yticks([])

gaussianImg = cv2.GaussianBlur(img,(5,5), 10)
equalizedImg = cv2.equalizeHist(gaussianImg)

plt.subplot(5,3,13),plt.imshow(gaussianImg, cmap = 'gray')
plt.title('r=10 Image'), plt.xticks([]), plt.yticks([])

#Plot the gray level image
plt.subplot(5,3,14),plt.imshow(cv2.equalizeHist(gaussianImg), cmap = 'gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])

#Plot the canny edge detection
plt.subplot(5,3,15),plt.hist(cv2.equalizeHist(gaussianImg).ravel(),256,[0,256]) 
plt.title('Equalized Hist'), plt.xticks([]), plt.yticks([])

#Show everything
plt.show()
