from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# OCR tool for extracting text from images
from PIL import Image
import pytesseract

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", type=str, help="path to the receipt")
ap.add_argument("-east", "--east", type=str, help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320, help="resized image width (should be multiple of 32)")
ad.add_argument("-e", "--height", type=int, default=320, help="resized image height (should be multiple of 32)")

args = vars(ap.parse_args())

# load the receipt and grab the image dimensions
image = cv2.imgread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine 
# the ratio in change for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

#resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
    # This is the first layer of the network.
    # This layer is responsible for giving the probability of a region containing text or not.
    "feature_fusion/Conv_7/Sigmoid",
    # This layer is responsible for giving the bounding box coordinates of the text in the image.
    # Later I'd like to see if we can find a layer for determining the angle of the text.
    "feature_fusion/concat_3"
]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then preform 
# a forward pass of the model to obtain the two output layer sets
# `blobFromImage` essentially converts the image into a blob of data.
# The way it does this is by getting the mean of the RGB
# values of the image and subtraction the mean from each RGB value
# relative to the image. This is done to normalize the image.
# Read more here: https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
# start a time to determine how long it takes to run the model
start = time.time()
# Set the blob as the input to the network
net.setInput(blob)
# Deconstruct `scores` and `geometry` from the output of the network via `forward`
#look more into the forward function
(scores, geometry) = net.forward(layerNames)
# End the timer
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume
# then initialize our set of bounding box rectangles and corresponding confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    # extract the scores (probabilities), 
    # followed by the geometrical data used to derive 
    # potential bounding box coordinates that surround text
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]
    
    # loop over the number of columns
    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue
        
        # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
        (offsetX, offsetY) = (x * 4.0, y * 4.0)
        
        # extract the rotation angle for the prediction and then compute the sin and cosine
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)
        
        # use the geometry volume to derive the width and height of the bounding box
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]
        
        # compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        
        # add the bounding box coordinates and probability score to our respective lists
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])
        
# Finally we need to apply a `non-maxima suppression` to the bounding boxes
# This will help us to remove overlapping bounding boxes
# Read more here: https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
boxes = non_max_suppression(np.array(rects), probs=confidences)


# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    
    # HERE IS WHERE WE NEED TO DETERMINE HOW TO EXTRACT THE ACTUAL CONTENTS OF THE TEXT
    # I think we can use the bounding box coordinates to crop the image and then use OCR to extract the text
    
    # draw the bounding box on the image
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # crop the image based on startX, startY, endX, endY and then use OCR to extract the text
    cv2.imshow("cropped", orig[startY:endY, startX:endX])
    
    
'''
    THINGS TO DO:
    HELPFUL GUIDES:
        OpenCV: https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
        PyTesseract: https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
    
    1. Use the coordinates of the bounding box to crop the image and then use OCR to extract the text
    2. Store the coordinates of the bounding box and the text in a tuple
    3. Store the tuple in a list
    4. Eventually we will need to store the data in a json file.
    5. Far down the line we'll use the coordinates of each text and the contents of the text to determine what part of the receipt is what.
    6. Then, we'll have a JSON file that contains each part of the receipt.
'''