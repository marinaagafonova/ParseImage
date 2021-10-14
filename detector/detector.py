import random
import os
from wand.image import Image

def resize_images():
    os.makedirs('resized', exist_ok=True)
    for file in os.listdir('science'):
        img = Image(filename='science/' + file)
        img.transform(resize='200x')
        img.save(filename = 'resized/' + file)

def detect():
    return random.choice(['diagram', 'graph', 'bar_chart', 'system', 'scheme', 'formula'])
    

resize_images()