import kNN
import sys
import neural_N


method, model_type = sys.argv[1], sys.argv[4]
if method == "train":
    if model_type == "knn":
        train_data = sys.argv[2]
        kNN.train_knn(train_data)
    elif model_type == "neural" or model_type == "best":
        input = sys.argv[2]
        model_file = sys.argv[3]
        neural_N.train_nn(input, model_file)
if method == "test":
    if model_type == "knn":
        input = open('nearest_model.txt', "r")
        # model_file = open(sys.argv[3],"w")
        kNN.test_knn(input)
    elif model_type == "neural" or model_type == "best":
        testfile = sys.argv[2]
        model_file = sys.argv[3]
        neural_N.test_nn(testfile,model_file)

