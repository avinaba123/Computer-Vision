# USAGE
# python detect.py --images images

# import the necessary packages
# from __future__ import print_function
from datetime import datetime
from imutils.object_detection import non_max_suppression
# from imutils import paths
import numpy as np
# import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
# imagePaths = list(paths.list_images(args["images"]))

# for imagePath in imagePaths:
# 	# load the image and resize it to (1) reduce detection time
# 	# and (2) improve detection accuracy
# 	image = cv2.imread(imagePath)
# 	image = imutils.resize(image, width=min(400, image.shape[1]))
# 	orig = image.copy()

# Input Video stream
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('vtest.avi')

frame_counter = 0

now = datetime.now()
start_time = now.strftime("%H : %M : %S")
print("Start Time : " + start_time)


while 1:

	ret, image = cap.read()

	if ret == False:
		break


	image = imutils.resize(image, width=min(600, image.shape[1]))

	# image = imutils.resize(image, width=min(600, image.shape[1]))
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	# (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.2)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

	# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	pick = non_max_suppression(rects, probs=None, overlapThresh=0.50)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	# filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format("VideoInput", len(rects), len(pick)))

	# show the output images
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)

	frame_counter += 1

	# cv2.waitKey(0)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()

now = datetime.now()
end_time = now.strftime("%H : %M : %S")
print("End Time : " + end_time)