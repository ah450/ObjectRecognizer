import numpy as np
import cv2

# pos and neg are positive and negative instances
# each is a list of files of nparray dumps,
# nparray of BoW histograms; shape = (n, 101)
# of the class to be trained for
def build_trained_classifier(pos_files, neg_files):
    total = len(pos_files) + len(neg_files)
    samples = np.empty((total, 101), np.float32)

    i = 0
    for pos_file in pos_files:
        samples[i] = np.load(pos_file)
        i = i + 1
    for neg_file in neg_files:
        samples[i] = np.load(neg_file)
        i = i + 1

    labels = np.empty((total, 1), np.float32)
    labels[0:len(pos_files), 0] = 1.0
    labels[len(pos_files):, 0] = 0.0

    return cv2.NormalBayesClassifier(samples, labels)

