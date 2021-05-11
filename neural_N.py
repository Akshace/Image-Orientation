import numpy as np
import random

no_of_features = 192
hidden_nodes = 20
output_nodes = 4
epochs = 100

def create_w_matrix(row, column):
    w_mat = []
    for i in range(row):
        inn_mat = []
        for j in range(column):
            value = random.uniform(1,-1)
            inn_mat.append(value)
        w_mat.append(inn_mat)
    # print(w_mat)
    return w_mat


def sigmoid_function(z):
    z = 1.0 / (1.0 + np.exp(np.negative(z)))
    return z



"""
https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
"""
def diff_sigmoid_function(z):
    a = (z * (1 - z))
    return a


def init_weights():
    w1 = create_w_matrix(no_of_features, hidden_nodes)
    w2 = create_w_matrix(hidden_nodes, output_nodes)
    w1 = np.array(w1)
    w2 = np.array(w2)
    return w1,w2

def actual_output(img_):
    output = []
    for i in range(len(img_)):
        orient = img_[i][0][1]
        if orient == "0":
            output.append((1, 0, 0, 0))
        elif orient == "90":
            output.append((0, 1, 0, 0))
        elif orient == "180":
            output.append((0, 0, 1, 0))
        elif orient == "270":
            output.append((0, 0, 0, 1))
    return np.asarray(output)


def feedforward(w1, w2, img_array):
    z2 = np.dot(img_array, w1)
    a1 = sigmoid_function(z2)
    z3 = np.dot(a1, w2)
    a2 = sigmoid_function(z3)
    return a1, a2, z3, z2


def neural_network(img_):
    w1, w2 = init_weights()
    y = actual_output(img_)
    for j in range(0, epochs):

        for i in range(0, len(img_)):
            x = img_[i][1]

            """
            Dividing by 1000 to avoid overflow error
            """

            over_ = np.true_divide(x,1000)
            # print(x)
            x = np.array(over_)[np.newaxis]
            a1, a2, z3, z2 = feedforward(w1, w2, x)
            delta3 = np.multiply((y[i] - a2), diff_sigmoid_function(a2))
            adj_w2 = np.dot(a1.T, delta3)
            delta2 = np.dot(delta3, w2.T) * diff_sigmoid_function(a1)
            adj_w1 = np.dot(x.T, delta2)  # error
            learning_rate = 0.1
            w1 = w1 + learning_rate * adj_w1
            w2 = w2 + learning_rate * adj_w2

    return w1, w2



def train_nn(train_file, model_file):
    tr_file = open(train_file,"r")
    img_array_ = []
    # img_info_ = []
    img_ = []
    for line in tr_file:
        img_info_ = line.split(" ")[0:2]
        img_array_ = np.asarray(line.split(" ")[2:], dtype = int)
        img_.append([img_info_, img_array_])

    w1, w2 = neural_network(img_)
    f = open(model_file, "w")
    for r in w1:
        for c in r:
            # print(c, end =" ")
            f.write(str(c) + " ")
        f.write("\n")
    f.write("\n")
    # print()
    for r1 in w2:
        for c1 in r1:
            # print(c1, end = " ")
            f.write(str(c1) + " ")
        f.write("\n")
    f.write("\n")
    # print()
    f.close()
    print("Model file generated.")

def test_nn(test_file,model_file):
    t_file = open(test_file, "r")
    img_ = []
    for line in t_file:
        img_orient_ = line.split(" ")[0:2]
        img_array_ = np.asarray(line.split(" ")[2:], dtype=int)
        img_.append([img_orient_, img_array_])

    model_file = open(model_file, "r")
    data = model_file.read()
    weights = data.split('\n\n')
    print("Model file loaded.")

    w2 = weights[1].split('\n')
    w2 = [x.split(' ') for x in w2 if x != '']
    for x in w2:
        for y in x:
            if y == '':
                x.remove(y)

    """
    Removing '' from weights array due to
    error when converting them into
    numpy array below 
    """
    w1 = weights[0].split('\n')
    w1 = [x.split(' ') for x in w1 if x != '']
    for x in w1:
        for y in x:
            if y == '':
                x.remove(y)

    predicted_array = []
    label = ['0', '90', '180', '270']
    w1 = np.array(w1, dtype=float)
    w2 = np.array(w2, dtype=float)
    for i in range(len(img_)):
        x = np.array(img_[i][1])
        x = np.true_divide(x,1000)
        z2 = np.dot(x, w1)
        a2 = sigmoid_function(z2)
        z3 = np.dot(a2, w2)
        yhat = sigmoid_function(z3)
        max = 0
        max_label = 0
        for j in range(0, len(yhat)):
            if yhat[j] > max:
                max = yhat[j]
                max_label = j

        if max_label == 0:
            predicted_array.append(label[0])
        elif max_label == 1:
            predicted_array.append(label[1])
        elif max_label == 2:
            predicted_array.append(label[2])
        elif max_label == 3:
            predicted_array.append(label[3])
    correct = 0.0
    f = open("output_results-NN.txt", "w")
    for i in range(0, len(img_)):
        # print "Predicted",predicted_array[i]
        # print "Actual",newtest[i][1]
        f.write(" ".join([img_[i][0][0], predicted_array[i], "\n"]))
        # f.write(img_[i][0][0] + " ")
        # f.write(predicted_array[i])
        # f.write("\n")
        if predicted_array[i] == img_[i][0][1]:
            correct += 1
        else:
            f.write("Incorrectly predicted = " + img_[i][0][0] + " should be " + str(img_[i][0][1]) + "\n")
    f.close()
    print("-----" * 18)
    print("Test images with their predicted orientations have been written into output_results-NN.txt.")
    print("Predicted orientation of", int(correct), "images correctly out of", len(img_), "test images.")
    print("-----" * 18)
    accuracy = float(float(correct) / len(img_))
    print("accuracy", accuracy * 100, "%")
