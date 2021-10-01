#!/bin/env python

import os
import cv2
import sys

file_list = sys.argv[1:]

for fpath in file_list: # iterate through the files
    try:
        img=cv2.imread(fpath)
        size=img.shape
    except:
        print(f'file {fpath} is not a valid image file ')
        os.remove(fpath)