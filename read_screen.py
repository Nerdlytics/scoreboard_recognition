import numpy as np
import pandas as pd
import argparse
import time
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ocr_functions import detect_text, display_ocr_text, ocr_text
from PIL import ImageGrab
import time

def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab())
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def screen_ocr_text(east, bbox=None):
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=bbox))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        img = detect_all_text(printscreen, east)

        cv2.imshow('window',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def detect_all_text(img, east, min_confidence=0.5, width=320, height=320, padding=0.5):
    # load the input image and grab the image dimensions
    image = img
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = east

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # A more accurate bounding box for rotated text
            offsetX = offsetX + cos * xData1[x] + sin * xData2[x]
            offsetY = offsetY - sin * xData1[x] + cos * xData2[x]

            # calculate the UL and LR corners of the bounding rectangle
            p1x = -cos * w + offsetX
            p1y = -cos * h + offsetY
            p3x = -sin * h + offsetX
            p3y = sin * w + offsetY

            # add the bounding box coordinates
            rects.append((p1x, p1y, p3x, p3y))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]
        text = ocr_text(img=roi)
        results.append(((startX, startY, endX, endY), text))

    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r: r[0][1])
    output = orig.copy()
    for ((startX, startY, endX, endY), text) in results:
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        print(text)

        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return output

def simple_screen_ocr(bbox=None):
    last_time = time.time()
    while(True):
        printscreen =  np.array(ImageGrab.grab(bbox=bbox))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        text = ocr_text(printscreen)
        print(text)

        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def premleague_ocr():
    while(True):
        printscreen =  np.array(ImageGrab.grab(bbox=(68,75,720,123)))

        boxes = [(70, 0, 162, 48),  # Team 1
                 (166, 0, 234, 48),  # Score 1
                 (261, 0, 330, 48),  # Score 2
                 (334, 0, 432, 48),  # Team 2
                 (500, 0, 720, 48)]  # Game Clock

        # (Width Start, Height Start, Width End, Height End)

        results = []
        pimg = process_img(printscreen)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the actual padded ROI
            roi = pimg[startY:endY, startX:endX]
            #roi = printscreen[startY:endY, startX:endX]
            #roi = process_img(roi)
            # roi = cv2.resize(roi, (int(endX*1.5), int(endY*1.5)))
            text = ocr_text(img=roi, psm="7")
            results.append(((startX, startY, endX, endY), text))
            #cv2.imshow('window', cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        print(results[0][1] + ':' + results[1][1] + ' ' + results[3][1] + ':' + results[2][1] + ' ' + results[4][1])

        cv2.imshow('window', cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.fastNlMeansDenoising(processed_img)
    ret,processed_img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV)
    #processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_img = cv2.fastNlMeansDenoising(processed_img)

    return processed_img
