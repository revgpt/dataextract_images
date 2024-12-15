import os
import pandas as pd
import numpy as np
import shutil
import requests
import multiprocessing
import time
from time import time as timer
from tqdm import tqdm
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image
import csv

# installing the libraries

!pip install pytesseract
import re
import pytesseract
import cv2


#Function for preprocessing ad saving the images


def process_and_save_images(directory_path, csv_output_path):
    # Initialize lists to hold processed images
    gray_images = []
    bw_images = []
    processed_images = []
    image_files = []

    # Define paths for saving processed images
    gray_save_dir = '/kaggle/working/gray_images/'
    bw_save_dir = '/kaggle/working/bw_images/'
    processed_save_dir = '/kaggle/working/processed_images/'
    
    # Ensure the save directories exist
    os.makedirs(gray_save_dir, exist_ok=True)
    os.makedirs(bw_save_dir, exist_ok=True)
    os.makedirs(processed_save_dir, exist_ok=True)

    # Get all jpg image files from the directory
    for image_file in Path(directory_path).glob('*.jpg'):
        image_files.append(image_file)
    
    # Prepare the CSV file for writing
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Filename', 'Extracted Text'])  # Header row
        
        # Process each image file
        for i, image_file in enumerate(image_files, start=1):
            # Read the image
            img = cv2.imread(str(image_file))
            
            # Check if the image was loaded properly
            if img is None:
                print(f"Error loading image: {image_file}")
                continue
            
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_images.append(gray_image)
            gray_output_path = os.path.join(gray_save_dir, f'gray_image_{i}.jpg')
            cv2.imwrite(gray_output_path, gray_image)

            # Apply binary thresholding
            _, bw_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)
            bw_images.append(bw_image)
            bw_output_path = os.path.join(bw_save_dir, f'bw_image_{i}.jpg')
            cv2.imwrite(bw_output_path, bw_image)
            
            # Noise removal on black and white image
            kernel = np.ones((1, 1), np.uint8)
            noise_removed = cv2.dilate(bw_image, kernel, iterations=1)
            kernel = np.ones((1, 1), np.uint8)
            noise_removed = cv2.erode(noise_removed, kernel, iterations=1)
            noise_removed = cv2.morphologyEx(noise_removed, cv2.MORPH_CLOSE, kernel)
            noise_removed = cv2.medianBlur(noise_removed, 1)
            processed_images.append(noise_removed)
           
            # Save the noise-removed image
            processed_output_path = os.path.join(processed_save_dir, f'processed_image_{i}.jpg')
            cv2.imwrite(processed_output_path, noise_removed)
            
            # Extract text using PyTesseract
            extracted_text = pytesseract.image_to_string(noise_removed)
            extracted_text = extracted_text.strip()  # Remove leading/trailing whitespace
            extracted_text = ' '.join(extracted_text.split())  # Replace multiple spaces/newlines with a single space
            extracted_text = extracted_text[:1000]  # Limit text length to 1000 characters (adjust as needed)
            
            # Write the filename and compacted extracted text to the CSV file
            csv_writer.writerow([image_file.name, extracted_text])

# input which contains the download set of images
dataset='/kaggle/input/dataset/dataset'

#file to save the extracted text
output ='/kaggle/working/output.csv'

#function call
process_and_save_images(dataset,output)