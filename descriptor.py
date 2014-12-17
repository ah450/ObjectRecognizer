"""
Process a single image.
"""
import cv2
import os
import itertools
import random
import numpy

def process(filename):
    """Generates a descriptor image."""
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (300, 250))

    desc_file_name = os.path.split(filename)[-1]
    desc_file_name = desc_file_name.split('.')[0] + '.sift'
    desc_file_name = os.path.join(os.path.dirname(filename), desc_file_name)
    locations = list(itertools.product(range(0, 300, 5), range(0, 250, 5)))
    locations = [cv2.KeyPoint(x, y, 1) for (x, y) in locations]
    sift = cv2.DescriptorExtractor_create('SIFT')
    locations, desc = sift.compute(img, locations)
    with open(desc_file_name, "wb") as f:
        numpy.save(f, desc)


def select_sample(path1, path2, path3):
    """returns a list of 75 random .sift file names"""
    files1 = [os.path.join(path1, x) for x in os.listdir(path1) if ".sift" in x]
    files2 = [os.path.join(path2, x) for x in os.listdir(path2) if ".sift" in x]
    files3 = [os.path.join(path3, x) for x in os.listdir(path3) if ".sift" in x]
    files = files1 + files2 + files3
    # Use Reservoir sampling to choose 75 items at random from files
    result = [None] * 75
    for i in range(0, 75):
        result[i] = files[i]
    for i in range(75, len(files)):
        j = random.randint(0, i)
        if j < 75:
            result[j] = files[i]
    return result

    