import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_document(image, kernel_size=21):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(gray, kernel_size)

    #Subtract background
    subtracted = cv2.subtract(background, gray)
    result = cv2.bitwise_not(subtracted)

    # Power Law (Gamma Correction)
    norm = result/255
    gamma_corrected = np.power(norm, 3)

    return gamma_corrected
