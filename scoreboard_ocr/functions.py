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
    """
    Process a given image for optimal text recognition
    """
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.fastNlMeansDenoising(processed_img)
    #ret,processed_img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)
    processed_img = cv2.fastNlMeansDenoising(processed_img)

    return processed_img

def ocr_text(img, language="eng", oem="1", psm="7"):
    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    config = ("-l "+language+" --oem "+oem+" --psm "+psm+" -c tessedit_char_whitelist=abc123")
    text = pytesseract.image_to_string(img, config=config)

    # add the bounding box coordinates and OCR'd text to the list
    # of results
    return(text)

def map_leds(position, character):
    """
    Take the position and character and return a dict with the LEDs and an on/off
    """
    led_dict = dict(zip(position, character))
    return led_dict
