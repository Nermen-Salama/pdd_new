#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:27:49 2019

@author: asem
"""

import os
from os import listdir
from PIL import Image
import numpy as np
from imageio import imread

dir_paths = ["Abietinella_abietina","pleurozium_schreberi"]

for dir_path in dir_paths:
    for filename in listdir(dir_path):
        if filename.endswith('.jpg'):
            try:
                img = Image.open(dir_path+"/"+filename) # open the image file
                img.verify() # verify that it is, in fact an image
                img = imread( dir_path+"/"+filename )
                if img.shape != (256,256,3):
                    print("shape mismatch: {},{}".format( img.shape, dir_path+"/"+filename))
                
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename)