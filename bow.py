"""
kmeans and bow.
"""
import cv2
import os
import itertools
import random
import numpy
import itertools
import scipy.spatial as spatial

def kmeans(images):
    """ takes a list of 75 randomly chosen descriptor file names, return  100 centroids matrix."""

    descriptors = numpy.zeros((0,128), numpy.float32)
    for image in images:
        descriptors = numpy.concatenate((descriptors, numpy.load(image)))

    num_images = len(images)
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    ret, labels, centers = cv2.kmeans(descriptors, 100, term_crit, 10, cv2.KMEANS_RANDOM_CENTERS)

    return centers

def bow(path, centers):
    """ takes path of sift files, centroids calculated by kmeans, saves a bow file containing histogram for each sift file """
    files = [os.path.join(path, x) for x in os.listdir(path) if ".sift" in x]

    for file in files:
        image = numpy.load(file)
        bow = numpy.zeros(len(centers)+1)
        for descriptor in image:
            minVal = 1e17
            minCenter = 0
            for index, center in enumerate(centers):
                sumOfSquaresOfDiff = spatial.distance.euclidean(center, descriptor)
                if sumOfSquaresOfDiff < minVal:
                    minVal, minCenter = sumOfSquaresOfDiff, index
            bow[minCenter] += 1
        bow_file_name = os.path.split(file)[-1]
        bow_file_name = bow_file_name.split('.')[0] + '.bow'
        bow_file_name = os.path.join(os.path.dirname(file), bow_file_name)
        with open(bow_file_name, "wb") as f:
            numpy.save(f, bow)
