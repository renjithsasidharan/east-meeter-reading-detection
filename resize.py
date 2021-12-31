#!/usr/bin/python
from PIL import Image
import os, sys

path = "tmp/images/"
output = "tmp/images_resized/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            imResize = im.resize((320, 320), Image.ANTIALIAS)
            imResize.save(output+item, 'JPEG', quality=90)

resize()