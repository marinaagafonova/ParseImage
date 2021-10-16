import random
import os
from wand.image import Image
import pathlib
import os
import matplotlib.pyplot as plt 
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pandas as pd
import requests
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

def get_file_name_from_url(url):
  firstpos=url.rindex("/")
  lastpos=len(url)
  filename=url[firstpos+1:lastpos]
  print(f"url={url} firstpos={firstpos} lastpos={lastpos} filename={filename}")
  return filename

def download_image(imageUrl, destinationFolder):
  filename = get_file_name_from_url(imageUrl)
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


def get_images_from_dataset(dataset_path, destinationFolder, is_testdataset=False):
    df = pd.read_csv(dataset_path)
    os.makedirs(destinationFolder, exist_ok=True)
    if not is_testdataset:
        for i, row in df.iterrows():
            print(f"Index: {i}")
            print(f"{row['Image URL']}\n")

            download_image(row["Image URL"], destinationFolder)
    else:
        for i, row in df.iterrows():
            labelName = row["Label"]
            print(f"Index: {i}")
            print(f"{row['Image URL']}\n")
            destinationFolderLabel= os.path.join(destinationFolder, labelName)
            os.makedirs(destinationFolderLabel, exist_ok=True)
            download_image(row["Image URL"], destinationFolderLabel)

def prefetch():
    global train_ds
    global val_ds
    memory = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=memory)
    val_ds = val_ds.cache().prefetch(buffer_size=memory)

def data_augmentation():
    global data_augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

def standardize_data():
    global train_ds
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    
# def training_view(history, epochs):
    # print(history.history)
    # acc = history.history['binary_accuracy']
    # val_acc = history.history['val_binary_accuracy']
    # # acc = history.history['accuracy']
    # # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs_range = range(epochs)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

# def create_model():
    # global model
    # img_height = 300
    # img_width = 300
    # model = Sequential([
        # layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        # layers.Conv2D(16, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(32, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(1, activation='sigmoid')
# ])
    # model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

# def model_train():
    # epochs=10
    # history = model.fit(
        # train_ds,
        # validation_data=val_ds,
        # epochs=epochs
# )
    # training_view(history, epochs)

def predict_with_test_dataset(model):
    labelMappings={"0":"Others", "1":"Article", 0.0:"Others", 1.0:"Article"}
    image_batch, label_batch = test_ds.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)
    correctPredictions=0
    plt.figure(figsize=(20, 20))
    print(f"number predictions={len(predictions)}")
    for i in range(len(predictions)):
        ax = plt.subplot(8, 5, i +1)
        plt.imshow(image_batch[i].astype("uint8"))
        prediction = class_names[predictions[i]]
        predictionLabel = labelMappings[prediction]
        gtLabel = labelMappings[label_batch[i][0]]
    if gtLabel == predictionLabel:
        correctPredictions += 1
    plt.title(f"P={predictionLabel} GT={gtLabel}")
    plt.axis("off")
    accuracy = correctPredictions/len(predictions)
    print(f"Accuracy:{accuracy}")


def map_to_numeric_values():
    global train_ds, val_ds, test_ds
    batch_size = 32
    test_batch_size=37
    img_height = 300
    img_width = 300
    data_dir = pathlib.Path("content/resized/dataset")
    test_data_dir = pathlib.Path("content/resized/test_dataset")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size, label_mode='binary')
    class_names_train = train_ds.class_names
    print("train classes " + str(class_names_train))
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,label_mode='binary')
    class_names_val = val_ds.class_names
    print("val classes " + str(class_names_val))
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=200,
        image_size=(img_height, img_width),
        batch_size=test_batch_size,label_mode='binary')
    class_names_test = test_ds.class_names
    print("test classes " + str(class_names_test))


def resize_and_move(wo, wohin, maxsize = 90000):
    count_resized = 0
    count_moved = 0
    for file in os.listdir(wo):
        img = Image(filename = wo + "/" + file)
        if (img.width * img.height > maxsize):
            img.sample(300, 300)
            count_resized += 1
        img.save(filename = wohin + "/" + file)
        count_moved += 1
    print(str(count_moved) + " images was moved from " + wo + " to " + wohin + ". Resized " + str(count_resized) + " images.")
    

#пересохраняет изображение и, если размер изображения слишком большой, уменьшает его.
def resize_images():
    #обработка научных изображений из dataset, сохранение в директорию 1
    resize_and_move('content/dataset/article', 'content/resized/dataset/1')
    #обработка ненаучных изображений из dataset, сохранение в директорию 0
    resize_and_move('content/dataset/others', 'content/resized/dataset/0')
    #аналогичная обработка изображений из test_dataset
    resize_and_move('content/test_dataset/article', 'content/resized/test_dataset/1')
    resize_and_move('content/test_dataset/others', 'content/resized/test_dataset/0')
    map_to_numeric_values()
    
def dir_making():
    os.makedirs('content/resized/dataset/1', exist_ok=True)
    os.makedirs('content/resized/dataset/0', exist_ok=True)
    os.makedirs('content/resized/test_dataset/1', exist_ok=True)
    os.makedirs('content/resized/test_dataset/0', exist_ok=True)

def dataset_proccessing():
    dir_making()
    resize_images()

def detect():
    return random.choice(['diagram', 'graph', 'bar_chart', 'system', 'scheme', 'formula'])
    

def visualize_data_from_training():
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            if labels[i] == 1.0:
                title = "Article"
            else:
                title = "Others"

            plt.title(title)
            plt.axis("off")
    plt.show()

def parse_datasets():
    get_images_from_dataset('datasets/article_dataset.csv', "content/dataset/article")
    get_images_from_dataset('datasets/other_dataset.csv', "content/dataset/others")
    get_images_from_dataset('datasets/test_dataset.csv', "content/test_dataset", is_testdataset=True)

train_ds = []
val_ds = []
test_ds = []

parse_datasets()
dataset_proccessing()
#visualize_data_from_training()
prefetch()
standardize_data()
data_augmentation()
# create_model()
# model_train()
# keras.backend.clear_session()