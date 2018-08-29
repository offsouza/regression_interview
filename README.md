# Regression NN

## Overview
The purpose of this task is to test your knowledge and capabilities using neural networks to solve problems.

## The Task
The task is to create a neural network which takes a set of 10 points as inputs, and outputs slope and the y-intercept of the best-fitting line for the given points. The points are noisy, i.e. they won't fit perfectly on a line, so the net must figure out the best-fit line.

Note that this is a toy task, simple enough to be solved with other means. However, we're curious to see how you use a neural network to solve the task.

Important: This task is simple enough to solve by other means. However, your result should be one neural network that any set of 10 points can be fed into to get the answer. That is, you should not create any sort of algorithm that figures out the slope and intercept on a row-by-row basis. The end result is one network with one set of trained weights.

# Methodology

First I start with a Neural Network with a 4 layers and 100 neurons each, during all process was used Relu activation function for all layers except for output layers where was used linear function and also Adam optimizer for all process. The results wasn't good for the fist test with 500 epochs.

I tried many type of the configuration, increasing and reduce the number of neurons, layers, epochs and learning rate, but always I had a problem. So I plot the loss chart and I saw that a network was probably overfitting. 

I split the database into training (70%) and test (30%) to check how the model behave with a new data. I again a tried several setting as Dropout for the regulation, update learning rate, dacay but the best configuration was when I put L2 regularization  in the hidden layers with 1000 neurons each. 

After this, the network was able to the low loss of training and test data without increasing the loss during the epochs.
I used some methods to the data normalization, but made no many difference in the final result.

Was trained the last model using 1000 epochs.

Because of the training time I could not test more parameters, but maybe it's possible improve the result of this network, reducing the learning rate, trying other optimizers and analyzing better the input data.

## For Run

In folder `req` there is the archive `requeriments.txt`, you will find the librarys necessary for run.
> pip install -r req.txt

However if you used Anaconda, cuda 8.0 and have a GPU, you can create a env base in basetgpu, follwed next code to create.
In directory `req` you will find `basetfgpu.yml` .In terminal put the codes below:

> conda env create -n teste -f basetfgpu.yml

To activative in linux and Mac:

> source activate basetfgpu

or for Windows:

> activate basetfgpu

So run the code: 

> python interview.py


Then, to see the loss score:

> python evaluate.py results/submission_train.csv data/train_100k.truth.csv 
