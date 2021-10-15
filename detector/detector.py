import random
import os
from wand.image import Image
import matplotlib.pyplot as plt 
import PIL
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pandas as pd
import requests # to get image from the web
import shutil # to save it locally
import time
import numpy as np

def getFileNameFromUrl(url):
  firstpos=url.rindex("/")
  lastpos=len(url)
  filename=url[firstpos+1:lastpos]
  print(f"url={url} firstpos={firstpos} lastpos={lastpos} filename={filename}")
  return filename


def downloadImage(imageUrl, destinationFolder):
  filename = getFileNameFromUrl(imageUrl)
  # Open the url image, set stream to True, this will return the stream content.
  r = requests.get(imageUrl, stream = True)

  # Check if the image was retrieved successfully
  if r.status_code == 200:
      # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
      r.raw.decode_content = True
      
      # Open a local file with wb ( write binary ) permission.
      filePath = os.path.join(destinationFolder, filename)
      if not os.path.exists(filePath):
        with open(filePath,'wb') as f:
            shutil.copyfileobj(r.raw, f)
        print('Image sucessfully Downloaded: ',filename)
        print("Sleeping for 1 seconds before attempting next download")
        time.sleep(1)
      else:
        print(f'Skipping image {filename} as it is already Downloaded: ')

  else:
      print(f'Image url={imageUrl} and filename={filename} Couldn\'t be retreived. HTTP Status={r.status_code}')


def get_images_from_dataset(dataset_path, destinationFolder):
    df = pd.read_csv(dataset_path)
    os.makedirs(destinationFolder, exist_ok=True)

    for i, row in df.iterrows():
        print(f"Index: {i}")
        print(f"{row['Image URL']}\n")

        downloadImage(row["Image URL"], destinationFolder)


def resize_images():
    os.makedirs('resized', exist_ok=True)
    count = 0
    for file in os.listdir('science'):
        img = Image(filename='science/' + file)
        if (img.width * img.height > 1048576):
            img.transform(resize='200x')
            img.save(filename = 'resized/' + file)
            count += 1
    print('Resized ' + str(count) + ' images...')

def detect():
    return random.choice(['diagram', 'graph', 'bar_chart', 'system', 'scheme', 'formula'])
    

#resize_images()

get_images_from_dataset('datasets/article_dataset.csv', "content/dataset/article")
get_images_from_dataset('datasets/other_dataset.csv', "content/dataset/others")
get_images_from_dataset('datasets/test_dataset.csv', "content/test_dataset")