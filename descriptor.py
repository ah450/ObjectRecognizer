"""
Process a single image.
"""
import cv2
import os
import glob
import itertools
import random
import numpy

def sift_file_name(filename):
    filename = os.path.split(filename)[-1]
    filename = ".".join(filename.split('.')[0:-1]) + '.sift'
    return os.path.join(filename)

def process(filename, out_path):
    """Generates a descriptor image."""
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (300, 250))
    locations = list(itertools.product(range(0, 300, 5), range(0, 250, 5)))
    locations = [cv2.KeyPoint(x, y, 5) for (x, y) in locations]
    sift = cv2.DescriptorExtractor_create('SIFT')
    locations, desc = sift.compute(img, locations)
    with open(out_path, "wb") as f:
        numpy.save(f, desc)

def select_sample(path_glob):
    """returns a list of 75 random .sift file names"""
    files = glob.glob(path_glob)
    # Use Reservoir sampling to choose 75 items at random from files
    result = [None] * 75
    for i in range(0, 75):
        result[i] = files[i]
    for i in range(75, len(files)):
        j = random.randint(0, i)
        if j < 75:
            result[j] = files[i]
    return result
