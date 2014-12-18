#!/usr/bin/env python2
import os
import sys

import descriptor
import bow

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


section("Generating 100 word vocabulary out of 75 random files")
vocab_sample_files = descriptor.select_sample(os.path.join(sift_dir, "**/**/"))
vocab = bow.kmeans(vocab_sample_files)

section("Generating BoWs for all images")

bow_dir = os.path.join(data_dir, "bow")
if not os.path.isdir(bow_dir):
    os.makedirs(bow_dir)

for klass in os.listdir(sift_dir):
    klass_sift_dir = os.path.join(sift_dir, klass)
    klass_bow_dir = os.path.join(bow_dir, klass)
    if not os.path.isdir(klass_bow_dir):
        os.makedirs(klass_bow_dir)

    print "Generating BoWs for %s/*" % klass
    for sift_file in os.listdir(klass_sift_dir):
        sift_file = os.path.join(klass_sift_dir, sift_file)
        bow_file = file_name_ext(sift_file, 'bow')
        bow_file = os.path.join(klass_bow_dir, bow_file)

        if os.path.isfile(bow_file):
            print "Found " + bow_file
        else:
            print "Generating " + bow_file
            try:
                bow.bow(sift_file, bow_file)
            except:
                "Failed for " + bow_file

section("Training classifiers")

for klass in os.listdir(sift_dir):
    klass_sift_dir = os.path.join(sift_dir, klass)

    print "Training clas for %s/*" % klass
    for img in os.listdir(klass_samples_dir):
