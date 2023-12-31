import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime as dt

import cv2
import easyocr
import imutils
import numpy as np
import pandas as pd
from cv2 import IMREAD_GRAYSCALE, Canny, imread, imwrite, line
from numpy import argwhere as arg
from PIL import Image as im
from PIL import ImageTk   
import base64
from flask import Flask, request, jsonify  
import io


app = Flask(__name__)

@app.route("/image", methods=["POST"])
def convertImageToPlate():
    data = request.get_json()   
    imgdata = base64.b64decode(str(data["base64"])) 
    img = im.open(io.BytesIO(imgdata))
    # opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return process(img)  


@app.route("/get", methods=["GET"])
def hello():
    return "Hello"




def process(img):
    # img=imread(file)
    
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    image = im.fromarray(gray)
    width, height = image.size
    bytes_per_line = width


    adjusted = cv2.convertScaleAbs(gray, alpha=-1, beta=40)
        
    iterations = 2
    bfilter = adjusted.copy()
    for _ in range(iterations):
        bfilter = cv2.bilateralFilter(bfilter, 11, 17, 17)
    edged = cv2.Canny(bfilter, 50, 150)
    image2 = im.fromarray(edged)
    width, height = image2.size
    bytes_per_line = width

    
    threshold = cv2.adaptiveThreshold(
        bfilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    image3 = im.fromarray(threshold)
    width, height = image3.size
    bytes_per_line = width

    
    keypoints = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(np.array(img), np.array(img), mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
    image4 = im.fromarray(cropped_image)
    width, height = image4.size
    bytes_per_line = width

    
    text_box=[]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    for i in range(len(result)):
        text_box.append(result[i][-2])
    result_string = ''.join(map(str, text_box))
    last_string=result_string
    
    return last_string
    
def extract_number_plate(file): 
    last_string = process(file)
    license_plate_text = last_string.upper()

    # Apply regex and formatting to the license plate text
    license_plate_text1 = re.sub(r"\s+", "", license_plate_text)

    license_plate_text1 = re.sub(r"[^A-Z0-9]+", "", license_plate_text) 

    pattern = r"^[A-Za-z]{0,2}\d{0,2}[A-Za-z]{0,2}\d{0,4}$"
    matches = re.findall(pattern, license_plate_text1)
    result = "".join(matches)
    # if len(self.result) == 10:
    # self.ui.Extract_det.setPlainText(f"Extracted number plate \n {self.result}") 
    return result  

if __name__ == "__main__":
    app.run(host='localhost')