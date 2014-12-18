"""
kmeans and bow.
"""
import cv2
import os
import itertools
import random
import numpy
import scipy.spatial as spatial
import itertools


def kmeans(images):
    """ takes a list of 75 randomly chosen descriptor file names, return  100 centroids matrix."""

    descriptors = numpy.zeros((0,128), numpy.float32)
    for image in images:
        descriptors = numpy.concatenate((descriptors, numpy.load(image)))

    num_images = len(images)
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    ret, labels, centers = cv2.kmeans(descriptors, 100, term_crit, 10, cv2.KMEANS_RANDOM_CENTERS)

    return centers

def bow_kdtree(centers, sift_path, bow_path):
    """ Generates and saves a BoW from a .sift file and centroids calculated by kmeans"""

    import kdtree

    class CentroidWrapper(list):
        def __init__(self, centroid, cluster):
            super(CentroidWrapper, self).__init__(centroid)
            self.cluster = cluster


    wrapped_centers = [CentroidWrapper(c.tolist(), i) for i, c in enumerate(centers)]
    # Construct kd-tree
    tree = kdtree.create(point_list=wrapped_centers, dimensions=128)
    if not tree.is_balanced:
        tree = tree.rebalance()

    sift = numpy.load(sift_path)
    bow = numpy.zeros(len(centers)+1)
    for descriptor in sift:
        cluster = tree.search_nn(descriptor)[0].data.cluster
        bow[cluster] += 1
    with open(bow_path, "wb") as f:
        numpy.save(f, bow)


def bow(centers, sift_path, bow_path):
    """ Generates and saves a BoW from a .sift file and centroids calculated by kmeans"""

    sift = numpy.load(sift_path)
    bow = numpy.zeros(len(centers)+1)
    for descriptor in sift:
        minVal = 1e17
        minCenter = 0
        for index, center in enumerate(centers):
            sumOfSquaresOfDiff = spatial.distance.euclidean(center, descriptor)
            if sumOfSquaresOfDiff < minVal:
                minVal, minCenter = sumOfSquaresOfDiff, index
        bow[minCenter] += 1

    with open(bow_path, "wb") as f:
        numpy.save(f, bow)

