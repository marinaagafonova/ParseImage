import random
import os
from wand.image import Image

def resize_images():
    os.makedirs('resized', exist_ok=True)
    count = 0;
    for file in os.listdir('science'):
        img = Image(filename='science/' + file)
        if (img.width * img.height > 1048576):
            img.transform(resize='200x')
            img.save(filename = 'resized/' + file)
            count += 1
    print('Resized ' + str(count) + ' images...')

def detect():
    return random.choice(['diagram', 'graph', 'bar_chart', 'system', 'scheme', 'formula'])
    

resize_images()