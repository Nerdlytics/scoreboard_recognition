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

def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.fastNlMeansDenoising(processed_img)
    #ret,processed_img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)
    processed_img = cv2.fastNlMeansDenoising(processed_img)

    return processed_img