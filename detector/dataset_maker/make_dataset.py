from PIL import Image, ImageOps
import numpy as np
import sys
import os
import csv

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def create_file_list(directory, format='.jpg'):
    file_list = []
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            if name.endswith(format):
                fullname = os.path.join(root, name)
                file_list.append(fullname)
    return file_list

def save_to_csv(file_list):
    with open(dataset_name + '.csv', 'a',  newline='') as file:
        file.truncate(0)

    for file in file_list:
        
        img_file = Image.open(file)
        
        #Greyscale
        img_grey = img_file.convert('L')

        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1]), img_grey.size[0])
        value = value.flatten()
        with open(dataset_name + '.csv', 'a',  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(value)

def check():
    rows = []
    with open('test.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        c = 1
        for row in spamreader:
            print('row: ' + str(c))   
            c += 1

path = 'path/to/images_folder'
dataset_name = 'name_of_dataset'

dataset_filelist = create_file_list(path)
save_to_csv(dataset_filelist)

