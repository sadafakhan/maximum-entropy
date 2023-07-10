"""Sadaf Khan, LING572, HW5, 02/06/2022. Calculates model expectation of a MaxEnt classifier."""

import sys
import os
import numpy as np
from math import exp
from collections import defaultdict

# take in data
training_data = sys.argv[1]
training_formatted = open(os.path.join(os.path.dirname(__file__), training_data), 'r').read().split("\n")[:-1]
output_file = sys.argv[2]

if len(sys.argv) == 4:
    model_file = sys.argv[3]
    model_formatted = open(os.path.join(os.path.dirname(__file__), model_file), 'r').read().split("\n")[:-1]
else:
    model_file = None

# initialize counters, data structures
feat2col = 0
vocab = {}
columns = {}

# extract the vocabulary (columns) and number of instances (rows)
for vec in training_formatted:
    word_counts = vec.split()[1:]
    for label in word_counts:
        feat = label.split(":")[0]

        # if an encountered word-feature isn't accounted for in the training vocabulary
        # add it into the reference dicts
        if feat not in vocab:
            vocab[feat] = feat2col
            columns[feat2col] = feat
            feat2col += 1

# create array of training instance x word count
training_array = np.zeros((len(training_formatted), feat2col))
for i in range(len(training_formatted)):
    split = training_formatted[i].split()
    word_counts = split[1:]

    # change cell word counts
    for pair in word_counts:
        feat = pair.split(":")[0]
        count = int(pair.split(":")[1])
        training_array[i, vocab[feat]] = count


def model_tracker(range1, range2):
    default = 0
    class_label = {}
    for i in range(range1, range2):
        pair = model_formatted[i].split()
        feat = pair[0]
        weight = float(pair[1])
        if feat.startswith("<"):
            default = weight
        else:
            class_label[feat] = weight
    return default, class_label


if model_file != None:
    guns_def, guns_weights = model_tracker(1, 34508)
    mideast_def, mideast_weights = model_tracker(34509, 69016)
    misc_def, misc_weights = model_tracker(69017, 103524)
    defaults = {"talk.politics.guns": guns_def, "talk.politics.misc": misc_def, "talk.politics.mideast": mideast_def}
    weights = {"talk.politics.guns": guns_weights, "talk.politics.misc": misc_weights,
               "talk.politics.mideast": mideast_weights}


def classify(vector):
    if model_file != None:
        distribution = {"talk.politics.guns": 0.0, "talk.politics.misc": 0.0, "talk.politics.mideast": 0.0}
        Z = 0

        # calculate probability numerator per class
        for label in distribution:
            lbd = defaults[label]
            sum = 0

            # iterate over the words in the vectors
            for j in range(len(vector)):
                if vector[j] != 0.0:
                    sum += weights[label][columns[j]]

            # cast numerator to label dictionary
            numerator = exp(lbd + sum)
            distribution[label] = numerator
            Z += numerator

        # divide by Z to get probabilities
        for label in distribution:
            distribution[label] = (distribution[label] / Z)
    else:
        distribution = {"talk.politics.guns": (1 / 3), "talk.politics.misc": (1 / 3), "talk.politics.mideast": (1 / 3)}
    return distribution


model_expect = {"talk.politics.guns": defaultdict(int), "talk.politics.misc": defaultdict(int), "talk.politics.mideast": defaultdict(int)}

# for each instance x in the training data
for i in range(len(training_array)):
    # get P(y|x) for every y in Y
    dist = classify(training_array[i])

    # for each feature t in x
    for t in range(len(training_array[i])):
        if training_array[i][t] != 0:
            feat = columns[t]
            # for each y in Y
            for label in dist:
                model_expect[label][feat] += dist[label]

alphabetic = sorted(vocab.keys(), key=lambda x: x.lower())

with open(output_file, 'w') as o:
    for label in model_expect:
        for feat in alphabetic:
            o.write(label + " " + feat + " " + "%.5f" % (model_expect[label][feat]/2700) + " " +
                    "%.5f" % (model_expect[label][feat]))
            o.write("\n")
