import argparse
import sys
sys.path.append(".")
sys.path.append("/image_parser")
from image_parser.parse_pdf import extract_images_from_pdf
from image_parser.parse_url import parse_class_from_web

parser = argparse.ArgumentParser(description='Parse your source for images and detect the article')

parser.add_argument("-u", "--url", help="URL of the page, that should be parsed")
parser.add_argument("-p", "--pdf", help="Full path to pdf-file, that should be parsed")
parser.add_argument("-j", "--jpeg", help="Full path to jpeg-image, that should be parsed")

args = parser.parse_args()

if args.url is not None:
    print(args.url + " processing... ")
    parse_class_from_web(args.url)
elif args.pdf is not None:
    extract_images_from_pdf(args.pdf)
elif args.jpeg is not None:
    #go to parse_jpeg
    #parse_jpeg(args.jpeg)
    print(args.jpeg)
else:
    print("""None of sources is setted. Please, write your source with corresponding flag.
For more information execute command with flag -h """)