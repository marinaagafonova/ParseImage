import random
import os
from wand.image import Image
import pathlib
#import tensorflow as tf
import os
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential

def map_to_numeric_values():
    batch_size = 32
    test_batch_size=37
    img_height = 200
    img_width = 200
    data_dir = pathlib.Path("/dataset")
    test_data_dir = pathlib.Path("/test_dataset")
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
    #То, что в директории res_notscience, перемещается в директорию 0
    #То что в директории res_science, перемещается в директорию 1

#пересохраняет изображение и, если размер изображения слишком большой, уменьшает его.
def resize_images():
    count = 0;
    #обработка научных изображений
    for file in os.listdir('science'):
        img = Image(filename='science/' + file)
        if (img.width * img.height > 1048576):
            img.transform(resize='200x')
            count += 1
        img.save(filename = 'dataset/1/' + file)
    #обработка ненаучных изображений
    for file in os.listdir('notscience'):
        img = Image(filename='notscience/' + file)
        if (img.width * img.height > 1048576):
            img.transform(resize='200x')
            count += 1
        img.save(filename = 'dataset/0/' + file)
    # ресайз тестовой выборки данных???
    print('Resized ' + str(count) + ' images...')
    #map_to_numeric_values()
    
def dir_making():
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('test_dataset', exist_ok=True)
    os.makedirs('dataset/0', exist_ok=True)
    os.makedirs('dataset/1', exist_ok=True)
    os.makedirs('test_dataset/0', exist_ok=True)
    os.makedirs('test_dataset/1', exist_ok=True)

def dataset_proccessing():
    dir_making()
    resize_images()

def detect():
    return random.choice(['diagram', 'graph', 'bar_chart', 'system', 'scheme', 'formula'])
    

dataset_proccessing()