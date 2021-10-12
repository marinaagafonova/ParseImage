from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import os
import sys
sys.path.append(".")
sys.path.append("./detector")
from detector import detect


# 1. получение массива ссылок на изображения на странице.
def get_html_images(url):
    try:
        html = urlopen(url)
        bs = BeautifulSoup(html, "html.parser")
    except HTTPError as e:
        print("The server returned an HTTP error")
    except URLError as e:
        print("The server could not be found!")
    
    imagesList = bs.findAll('img')
    print("Getting images list from " + url)
    return imagesList

#2. получение изображений по ссылкам из массива.
def get_url_images(url):
    images = get_html_images(url)
    src = []
    for img in images:
        src.append(urljoin(url,img['src']))
    return src

#3. сохранение изображений в директории saved.
def save_images(url):
    links = get_url_images(url);
    path = './saved'
    os.makedirs(path, exist_ok=True)
    for link in links:
        url = urlparse(link)
        name = os.path.basename(url.path)
        savein = path + '/' + detect() + name;
        try:
            urlretrieve(link, savein)
        except:
            print("Saving arror")
    print('Images were saved successfully. You can find them in the folder ../saved.')
        
def parse_class_from_web(url = "https://habr.com/ru/news/t/477922/"):
    save_images(url);