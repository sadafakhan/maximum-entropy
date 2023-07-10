"""Sadaf Khan, LING572, HW5, 02/06/2022. Calculates empirical expectation of a MaxEnt classifier."""

import os
import sys
from collections import defaultdict

training_data = sys.argv[1]
output_file = sys.argv[2]

training_formatted = open(os.path.join(os.path.dirname(__file__), training_data), 'r').read().split("\n")[:-1]
raw = {"talk.politics.guns": defaultdict(int), "talk.politics.misc": defaultdict(int),
       "talk.politics.mideast": defaultdict(int)}
vocab = defaultdict(int)

# collect raw word counts per class
for vec in training_formatted:
    split = vec.split()
    class_label = split[0]
    word_counts = split[1:-1]
    for pair in word_counts:
        feat = pair.split(":")[0]
        count = int(pair.split(":")[1])
        vocab[feat] += count
        raw[class_label][feat] += count

alphabetic = sorted(vocab.keys(), key=lambda x: x.lower())

with open(output_file, 'w') as o:
    for label in raw:
        for feat in alphabetic:
            o.write(label + " " + feat + " " + "%.5f" % (raw[label][feat] / 2700) + " " + str(raw[label][feat]))
            o.write("\n")
