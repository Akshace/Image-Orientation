import math
import numpy as np
import pandas as pd
from collections import Counter


def train_knn(train_data):
    train_data = pd.read_csv("train-data (1).txt", sep=" ", header=None)
    train_new = train_data.drop(columns=[0], axis=1)
    """
    Removing Image Name from training Data
    """
    train_new.to_csv('nearest_model.txt', sep=' ', index=False, header=None)
    print("Model file for KNN Generated")

def test_knn(model):
    print("Running time is around 20-25 mins for kNN")
    test_list = []
    train = []
    for line in model:
        orientation_ = line.split(" ")[0]
        img_array = np.asarray(line.split(" ")[1:], dtype=int)
        train.append([orientation_, img_array])

    test = open('test-data.txt', "r")

    k = [30, 40, 50]

    for line in test:
        img_info_ = line.split(" ")[0:2]
        img_array_ = np.asarray(line.split(" ")[2:], dtype=int)
        test_list.append([img_info_, img_array_])

    file = open("output_results-kNN.txt", "w")

    """
    Running the loop for 3 values of k
    to get optimized k - value.
    """

    for k_value in k:
        file.write("Analysis for k-value = " + str(k_value)+ "\n")
        correct_predict = 0
        for img in test_list:
            dist = []
            for img_train in train:
                """
                Calculating euclidean distance between two
                image arrays
                """
                euc_dist = math.sqrt(np.sum((img_train[1] - img[1])**2))
                dist.append([img_train[0], euc_dist])
            cls_dict = Counter([img_data[0] for img_data in sorted(dist, key=lambda z: z[1])[0:k_value]])
            # cls_dict = Counter(img_data[0])
            """
            Selecting the label with highest count
            """
            predicted_label = max(cls_dict, key=lambda x: cls_dict[x])
            file.write(" ".join([img[0][0], predicted_label, "\n"]))
            if img[0][1] == predicted_label:
                correct_predict += 1
            else:
                file.write("Incorrectly predicted = " + img[0][0] + " should be " + str(img[0][1]) + "\n")
        print("Accuracy = ", str(float(correct_predict) * 100 / len(test_list)) + " for k value = " + str(k_value))