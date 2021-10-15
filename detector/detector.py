import random
import os
from wand.image import Image
import pathlib
import os
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

def map_to_numeric_values():
    batch_size = 32
    test_batch_size=37
    img_height = 200
    img_width = 200
    data_dir = pathlib.Path("content/dataset/")
    test_data_dir = pathlib.Path("content/test_dataset")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size, label_mode='binary')
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,label_mode='binary')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=200,
        image_size=(img_height, img_width),
        batch_size=test_batch_size,label_mode='binary')
    class_names = train_ds.class_names
    print(class_names)

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

#пересохраняет изображение и, если размер изображения слишком большой, уменьшает его.
def resize_images():
    count = 0;
    #обработка научных изображений
    os.makedirs('resized', exist_ok=True)
    count = 0
    for file in os.listdir('content/dataset/article'):
        img = Image(filename='content/dataset/article/' + file)
        if (img.width * img.height > 1048576):
            img.transform(resize='200x')
            count += 1
        img.save(filename = 'content/resized/dataset/1/' + file)
    #обработка ненаучных изображений
    for file in os.listdir('notscience'):
        img = Image(filename='content/dataset/article/' + file)
        if (img.width * img.height > 1048576):
            img.transform(resize='200x')
            count += 1
        img.save(filename = 'content/resized/dataset/0/' + file)
    # ресайз тестовой выборки данных???
    print('Resized ' + str(count) + ' images...')
    #map_to_numeric_values()
    
def dir_making():
    os.makedirs('content/resized', exist_ok=True)
    os.makedirs('content/resized/dataset', exist_ok=True)
    os.makedirs('content/resized/dataset/1', exist_ok=True)
    os.makedirs('content/resized/dataset/0', exist_ok=True)

def dataset_proccessing():
    dir_making()
    resize_images()

def detect():
    return random.choice(['diagram', 'graph', 'bar_chart', 'system', 'scheme', 'formula'])
    


get_images_from_dataset('datasets/article_dataset.csv', "content/dataset/article")
get_images_from_dataset('datasets/other_dataset.csv', "content/dataset/others")
get_images_from_dataset('datasets/test_dataset.csv', "content/test_dataset")
dataset_proccessing()