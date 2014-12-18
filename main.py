#!/usr/bin/env python2
import os
import glob
import sys
import numpy as np

import descriptor
import bow
import bayes

from multiprocessing import Pool

if len(sys.argv) < 3:
    print "usage: ./main.py SamplesDir DataDir"
    exit(1)

def section(msg):
    print
    print "="*80
    print msg
    print "="*80
    print


def file_name_ext(filename, ext):
    filename = os.path.split(filename)[-1]
    filename = ".".join(filename.split('.')[0:-1]) + '.' + ext
    return os.path.join(filename)

samples_dir = sys.argv[1]
data_dir = sys.argv[2]

section("Generating dense SIFT descriptors for all classes inside %s" % samples_dir)

sift_dir = os.path.join(data_dir, "sift")
if not os.path.isdir(sift_dir):
    os.makedirs(sift_dir)

for klass in os.listdir(samples_dir):
    klass_samples_dir = os.path.join(samples_dir, klass)
    klass_sift_dir = os.path.join(sift_dir, klass)
    if not os.path.isdir(klass_sift_dir):
        os.makedirs(klass_sift_dir)

    print "Generating descriptors for %s/*" % klass
    for img in os.listdir(klass_samples_dir):
        sample_file = os.path.join(klass_samples_dir, img)
        sift_file = file_name_ext(sample_file, 'sift')
        sift_file = os.path.join(klass_sift_dir, sift_file)
        if os.path.isfile(sift_file):
            print "Found " + sift_file
        else:
            print "Generating " + sift_file
            try:
                descriptor.process(sample_file, sift_file)
            except:
                print "Failed for " + sift_file


section("100 word vocabulary out of 75 random files")
vocab_file = os.path.join(data_dir, "vocab_centroids.data")
if os.path.isfile(vocab_file):
    print "Found ", vocab_file
    vocab = np.load(vocab_file)
else:
    print "Generating ", vocab_file
    vocab_sample_files = descriptor.select_sample(os.path.join(sift_dir, "**/**"))
    vocab = bow.kmeans(vocab_sample_files)
    with open(vocab_file, "wb") as f:
        np.save(f, vocab)


section("BoWs for all images")
bow_dir = os.path.join(data_dir, "bow")
if not os.path.isdir(bow_dir):
    os.makedirs(bow_dir)

for klass in os.listdir(sift_dir):
    klass_sift_dir = os.path.join(sift_dir, klass)
    klass_bow_dir = os.path.join(bow_dir, klass)
    if not os.path.isdir(klass_bow_dir):
        os.makedirs(klass_bow_dir)

    print "Generating BoWs for %s/*" % klass
    batch = []
    for sift_file in os.listdir(klass_sift_dir):
        bow_file = file_name_ext(sift_file, 'bow')
        bow_file = os.path.join(klass_bow_dir, bow_file)
        sift_file = os.path.join(klass_sift_dir, sift_file)

        if os.path.isfile(bow_file):
            print "Found " + bow_file
        else:
            batch.append((vocab, sift_file, bow_file))

    def doit(b):
        vocab, sift_file, bow_file = b
        print "Generating " + bow_file
        try:
            bow.bow(vocab, sift_file, bow_file)
        except:
            "Failed for " + bow_file

    p = Pool(4)
    res = p.map_async(doit, batch, 10)
    res.get()
    p.close()
    p.join()

section("Training classifiers")
classifiers = {}
class_train_files = {}
class_test_files = {}

# globbing is lexically ordered
# will use first 80% for training, remaining 20% for testing

all_class_bow_dirs  = [(klass, os.path.join(bow_dir, klass)) for klass in os.listdir(bow_dir)]
for klass, klass_bow_dir in all_class_bow_dirs:
    pos_files = glob.glob(os.path.join(klass_bow_dir, "*"))
    num_train = int(len(pos_files) * 0.8)
    class_train_files[klass] = pos_files[0:num_train]
    class_test_files[klass] = pos_files[num_train:]
    print "Splitting data for %s into %i training images and %i test images"\
        % (klass, num_train, len(pos_files) - num_train)

for klass in class_train_files:
    neg_files = []
    for other_klass in class_train_files:
        if other_klass == klass: continue
        neg_files += class_train_files[other_klass]

    print "Training classifier for %s" % klass
    classifiers[klass] = bayes.build_trained_classifier(class_train_files[klass], neg_files)

    #classifier_path = os.path.join(data_dir, klass + ".classifier")
    # print "Saving classifier for %s to %s" % (klass, classifier_path)
    # FIXME dump to file?


section("Testing Time!")
for klass in class_test_files:
    miss_classification = 0
    tests = 0

    for other_klass in class_test_files:
        if other_klass == klass:
            actual = 1.0
        else:
            actual = 0.0

        for test_file in class_test_files[other_klass]:
            tests += 1
            bow = np.empty((1, 101), np.float32)
            bow[0] = np.load(test_file)

            prediction = classifiers[klass].predict(bow)
            if np.abs(prediction[1] - actual) > 0.00001:
                miss_classification += 1

    result = float(miss_classification) / tests
    print "Misclassification rate for %s classifier: %i/%i = %f" % (klass, miss_classification, tests, result)
