## Machine Learning

## Image Orientation

### Problem Statement

In this assignment, we need to identify the orientation of images in a FlickR image dataset. Our goal is to implement k-nearest neighbors as a baseline, and then another approach of our choice, I chose to implement Neural Network.

### Preprovided Code and Data :

1. train-data (1).txt - The training file contains image_name image_orientation and the image array in the form of 1x3mn.
2. test_data - The test file contains around 1000 entries in similar format to training data
3. Images - Actual Images have also been provided for further analysis.

### Approach to the Problem

#### 1. kNN

The approach to kNN was to calculate the euclidean distance of each test image with the training image data, and sort them in ascending order. After that, we would select first k values from the distances, and select the orienatation which appears max times.
The selected orientation is the predicted orientation for the test_image.

Now, the code checks whether the predicted matches the actual or not, if it matches the correct prediction counter increases, and the accuracy is calculated.

#### Training

The training involves creating a copy of traning data in the model file, with the name of images column removed because we do not need it during the testing phase.

### Results and Accuarcy

Train -> python3 Orientation.py "train" "train-data (1).txt" "best_model.txt" "neural"
Test ->  python3 Orientation.py "test" "test-data.txt" "best_model.txt" "neural"

For the ouput results, I have created a file "output_results-kNN.txt", in which the code adds the test file name, and its predicted orientation. If the prediction is not correct, another line is added displaying that the orienatation predicted is not correct and the actual orientation is printed along with it.

Since, the appropriate value of **k should be sqrt(N)**. I ran the code for three values of k : [30,40,50]. The results are below:

Accuracy =  70.20148462354189 for k value = 30
Accuracy =  71.89819724284199 for k value = 40
Accuracy =  70.83775185577943 for k value = 50

For k = 40, I got the highest accuarcy.


### 2. Neural Network

Neural Networks are system of interconnected neurons, which transfer information to each other. The main components of neural networks are :
1. Input Layer
2. Hidden Layer
3. Output Layer

A set of weights are implemented in neural networks which are updated after each epoch. 

#### Forward Propogation

We use the set of weights in the layers of neural networks and obtain the output.

#### Backward Propogation

Backward propogation involves going from output layer to input layer after calculation of error between actual output and predicted output. This error is backpropogated and the weights are updated accordingly.

#### Architecture Details

[![IMG-20210504-174327.jpg](https://i.postimg.cc/sxP5gzWL/IMG-20210504-174327.jpg)](https://postimg.cc/H8xr6fB0)

#### NN Algorithm

##### Training
1. Assign weights randomly to w1 and w2 matrices.

				Forward propogation:
					1.  z1 = x.w1 ,where x is image
					2.  a1 = sigmoid(z1)
					3.  z2 = a1.w2
					4. yhat = sigmoid(z2)

				Backward Propogation:
					5. error = actual output - yhat
					6. derivative_yhat = derivate of sigmoid func of yhat
					7. delta = derivative_yhat * error
					8. adj_w2 = dot product of a1.T and delta
					9. derivate_a1 = derivative of sigmoid function of a1
					10. delta2 = dot product of multiplication of delta and w2.T with derivative_a1 
					11. adj_w1 = dot product of x and delta2
					12. Adjust weights using learning rate and adj_w1 and adj_w2
	
	
	The obtained w1 and w2 after n epochs are stored in model file.

##### Testing

For each image:
1.  z2 =x.w1
2.  a2 = sigmoid(z2)
3.  z3 = dot product of a2 and w2
4.  yhat = sigmoid(z3) 

We check the maximum value in yhat and select the label which is present at the index of the maximum value.

We do it for all images, predicting labels and checking whether the prediction was correct or not, and thus calculate the accuracy. Also, we create an output file 
"output-NN.txt" which contains test_image names and predicted orientation.

#### Results 

Train -> python3 Orientation.py "train" "train-data (1).txt" "best_model.txt" "neural"
Test ->  python3 Orientation.py "test" "test-data.txt" "best_model.txt" "neural"


Test images with their predicted orientations have been written into 
output_results-NN.txt.
Predicted orientation of 707 images correctly out of 943 test images.
accuracy 74.97348886532343 %


**Note** - I have referenced the Neural Network code from Git repository - https://github.com/Hasika11/Image-Orientation-Detection/blob/master/orient.py
Initially, I had only planned to submit kNN code due to the lack of time, but I really wanted to implement Neural Network from scratch and understand it's working. That's why, I implemented the code myself from scratch after referencing from above. 

After submission, I will try and increase the hidden layers and the number of nodes in them, to check if I can increase the accuracy.


