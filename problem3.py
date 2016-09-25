# USAGE
# python problem3.py --image image/image1.jpg

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


#Loading the image and making it a grey level one
img = cv2.imread(args["image"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# code from
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Fourier_Transform_FFT_DFT.php
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft[dft == 0] = 0.0000001

#Magnitude and phase
magnitude, phase = 20*np.log(cv2.cartToPolar(dft[:,:,0],dft[:,:,1]))

#Pre allocate some variables
width, height = img.shape[:2]
amp2Phase1 = np.zeros((width,height,2))

#convert it back
amp2Phase1[:,:,0], amp2Phase1[:,:,1] = cv2.polarToCart(magnitude, phase)

#plot the images
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(phase, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# mouse events for the image
def on_mouse_click(event,x,y,flags,param):
	#define the k for the window size (will be 2k +1, 41 in this case)
	size = 20
	if event == cv2.EVENT_LBUTTONDOWN:
		img = cv2.imread(args["image"])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#we have to use copy to avoid pointers interfering with edge detection after
		#we draw the rectangle in the next few lines
		img2 = img.copy()[y-size:y+size, x-size:x+size]

		#Draw the window
		cv2.rectangle(img,(x-size,y+size),(x+size,y-size), (0, 255, 0), 2)
		cv2.imshow("Image2", img)

		#take the DFT
		dft = cv2.dft(np.float32(img2),flags = cv2.DFT_COMPLEX_OUTPUT)
		#Avoid the warning of log 0
		dft[dft == 0] = 0.0000001

		#Take the magnitude and phases
		magnitudeWindow, phaseWindow = 20*np.log(cv2.cartToPolar(dft[:,:,0],dft[:,:,1]))

		#Apply some edge detection on the image
		img2 = cv2.Canny(img2,100,200)

		#Apply some edge detection on the magnitude
		slicemagnitude = np.uint8(magnitudeWindow)
		magnitude2 = cv2.Canny(slicemagnitude,100,300)

		#Apply some edge detection on the phase
		slicephase = np.uint8(phaseWindow)
		phase2 = cv2.Canny(slicephase,100,300)

		#Print all the edge detections
		cv2.imshow("Spacial", img2)
		cv2.imshow("Magnitude", magnitude2)
		cv2.imshow("Phase", phase2)

		#Threshold everything for the nfinal result
		img2 = img2[img2>150]
		magnitude2 = magnitude2[magnitude2>150]
		phase2 = phase2[phase2>150]

		#Print the info we are looking for
		print('Window edges = ',img2.size)
		print('Magnitude edges = ',magnitude2.size)
		print('Phase edges = ',phase2.size)
		
		#helper text
		print('Click on the image to get some informations or press ENTER to exit\n')

#draw the initial window
cv2.namedWindow('Image2')

#prepare the mouse callback
cv2.setMouseCallback('Image2',on_mouse_click)

#helper info
print('Click on the image or press ENTER to exit\n')

#keep the program running
while(1):
	cv2.imshow('Image2',img)
	k = cv2.waitKey(0)
	if k:
		break

#destroy all windows
cv2.destroyAllWindows()
		