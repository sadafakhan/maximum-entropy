"""Sadaf Khan, LING572, HW5, 02/06/2022. Classifies test data given a MaxEnt model learned from training data."""

import os
import sys
import numpy as np
import pandas as pd
from numpy import exp
from sklearn.metrics import confusion_matrix, accuracy_score

test_data = sys.argv[1]
model_file = sys.argv[2]
sys_output = sys.argv[3]

testing_formatted = open(os.path.join(os.path.dirname(__file__), test_data), 'r').read().split("\n")[:-1]
model_formatted = open(os.path.join(os.path.dirname(__file__), model_file), 'r').read().split("\n")[:-1]

# initialize counters, data structures
feat2col = 0
vocab = {}
columns = {}
te_y_real = []
te_y_pred = []

# create numpy arrays representing test vectors
# extract the vocabulary (columns) and number of instances (rows)
for vec in testing_formatted:
    word_counts = vec.split()[1:]
    for label in word_counts:
        feat = label.split(":")[0]

        # if an encountered word-feature isn't accounted for in the training vocabulary
        # add it into the reference dicts
        if feat not in vocab:
            vocab[feat] = feat2col
            columns[feat2col] = feat
            feat2col += 1

# create array of testing instance x word count
testing_array = np.zeros((len(testing_formatted), feat2col))
for i in range(len(testing_formatted)):
    split = testing_formatted[i].split()
    class_label = split[0]
    word_counts = split[1:]

    # keep track of actual testing labels. i-th item in actual_labels is the label for i-th row/vector in array
    te_y_real.append(class_label)

    # change cell word counts
    for pair in word_counts:
        feat = pair.split(":")[0]
        count = int(pair.split(":")[1])

        # ignore OOV
        if feat in vocab:
            testing_array[i, vocab[feat]] = count


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


guns_def, guns_weights = model_tracker(1, 34508)
mideast_def, mideast_weights = model_tracker(34509, 69016)
misc_def, misc_weights = model_tracker(69017, 103524)

defaults = {"talk.politics.guns": guns_def, "talk.politics.misc": misc_def, "talk.politics.mideast": mideast_def}
weights = {"talk.politics.guns": guns_weights, "talk.politics.misc": misc_weights, "talk.politics.mideast": mideast_weights}

# classify and write to file
with open(sys_output, 'w') as m:
    m.write("%%%%% test data:\n")
    # get probability distributions
    # iterate over the test vectors
    for i in range(len(testing_array)):
        m.write("array:" + str(i) + " " + te_y_real[i] + " ")
        distribution = {"talk.politics.guns": 0.0, "talk.politics.misc": 0.0, "talk.politics.mideast": 0.0}
        Z = 0

        # calculate probability numerator per class
        for label in distribution:
            lbd = defaults[label]
            sum = 0

            # iterate over the words in the test vectors
            for j in range(len(testing_array[i])):
                if testing_array[i][j] != 0.0:
                    sum += weights[label][columns[j]]

            # cast numerator to label dictionary
            numerator = exp(lbd + sum)
            distribution[label] = numerator
            Z += numerator

        # divide by Z to get probabilities
        for label in distribution:
            distribution[label] = (distribution[label] / Z)

        sorted_dist = sorted(distribution, key=distribution.get, reverse=True)
        te_y_pred.append(sorted_dist[0])
        for label in sorted_dist:
            m.write(label + " " + "%.5f" % distribution[label] + " ")
        m.write("\n")


# header order for confusion matrix
label_set = ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc"]

# create confusion matrices and accuracy scores
test_cm = confusion_matrix(te_y_real, te_y_pred, labels=label_set)
test_accuracy = accuracy_score(te_y_real, te_y_pred)
test_formatted = pd.DataFrame(test_cm, index=label_set, columns=label_set)

pd.set_option('display.expand_frame_repr', False)

print("Confusion matrix for the testing data:")
print("row is the truth, column is the system output \n")
print(test_formatted)
print("\n")
print("Testing accuracy=" + str(test_accuracy))
