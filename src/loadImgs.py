#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:55:11 2018

@author: ali
"""
import glob
import PIL
from PIL import Image
        
import util



def save_resized_pictures(height, width):
    for cloud_kind, subkinds in util.cloud_kinds.items():
        if util.ensure_directory_exists("../temp/classic/" + cloud_kind):
            # directory exists already, we assume that the resized pictures
            # are in there
            continue

        counter = 0

        for subkind in subkinds:
            for element in glob.glob("../data/" + subkind + "/*"):
                img = Image.open(element)
                resized = img.resize((height, width), PIL.Image.ANTIALIAS)
                path = "../temp/classic/%s/%s%s.jpg" % (cloud_kind, cloud_kind, counter)
                resized.save(path)
                counter += 1


save_resized_pictures(500, 500)
