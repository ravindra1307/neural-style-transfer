import numpy as np
import argparse
import random
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-v", "--visualize", type=int, default=0,
	help="whether or not we are going to visualize each instance")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())


labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")


colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
COLORS = open(colorsPath).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")

weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])


print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)


image = cv2.imread(args["image"])
(H, W) = image.shape[:2]


blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
end = time.time()


print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
print("[INFO] boxes shape: {}".format(boxes.shape))
print("[INFO] masks shaspe: {}".format(masks.shape))



for i in range(0, boxes.shape[2]):
	
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]

	
	if confidence > args["confidence"]:
		
		clone = image.copy()
		clone[:] =(255, 255, 255)


		
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY

		
		mask = masks[i, classID]
		mask = cv2.resize(mask, (boxW, boxH),interpolation=cv2.INTER_NEAREST)
		mask = (mask > args["threshold"])

		
		roi = clone[startY:endY, startX:endX]

		visMask = (mask * 255).astype("uint8")
		cv2.imshow("Mask", visMask)
		cv2.imwrite("/home/ravi/Desktop/mask-rcnn/abc.jpg", visMask)


		if args["visualize"] > 0:
			
			
			instance = cv2.bitwise_and(roi, roi, mask=visMask)

			#cv2.imshow("ROI", roi)
			
			

			#cv2.imshow("Segmented", instance)

		
		roi = roi[mask]

		#print(COLORS[1])
		color = (COLORS[0])
		blended = ((1 * color) + (0 * roi)).astype("uint8")
		




		
		clone[startY:endY, startX:endX][mask] = blended

		


		
		#color = [int(c) for c in color]
		#cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

		#text = "{}: {:.4f}".format(LABELS[classID], confidence)
		#cv2.putText(clone, text, (startX, startY - 5),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		cv2.imshow("Output", clone)
		cv2.imwrite("/home/ravi/Desktop/mask-rcnn/abc.jpg", clone)
		cv2.waitKey(0)
