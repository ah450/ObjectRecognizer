#!/usr/bin/env python2
import os
import sys

import descriptor

if len(sys.argv) < 2:
    print "Specify the samples directory as the first argument"
    exit(1)

samples_dir = sys.argv[1]
print "Generating dense SIFT descriptors for all classes inside %s" % samples_dir

sift_dir = os.path.join(samples_dir, "sift")
if not os.path.isdir(sift_dir):
    os.makedirs(sift_dir)

for klass in os.listdir(samples_dir):
    if klass == 'sift': continue
    klass_samples_dir = os.path.join(samples_dir, klass)
    klass_sift_dir = os.path.join(sift_dir, klass)
    if not os.path.isdir(klass_sift_dir):
        os.makedirs(klass_sift_dir)

    print "Generating descriptors for %s/*" % klass
    for img in os.listdir(klass_samples_dir):
        sample_file = os.path.join(klass_samples_dir, img)
        sift_file = descriptor.sift_file_name(sample_file)
        sift_file = os.path.join(klass_sift_dir, sift_file)
        if os.path.isfile(sift_file):
            print "Found " + sift_file
        else:
            print "Generating " + sift_file
            try:
                descriptor.process(sample_file, sift_file)
            except:
                "Failed for " + sift_file


