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

east = cv2.dnn.readNet('frozen_east_text_detection.pb')
image = cv2.imread('images/espn2_uc_tenn.JPG')

#detect_text(image, east)

scoreboard = image[890:960, 360:1600]
plt.imshow(scoreboard)
plt.show()