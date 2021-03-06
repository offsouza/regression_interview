# Neural Network Regression

## Overview
The purpose of this task is to test your knowledge and capabilities using neural networks to solve problems.

## The Task
The task is to create a neural network which takes a set of 10 points as inputs, and outputs slope and the y-intercept of the best-fitting line for the given points. The points are noisy, i.e. they won't fit perfectly on a line, so the net must figure out the best-fit line.

Note that this is a toy task, simple enough to be solved with other means. However, we're curious to see how you use a neural network to solve the task.

Important: This task is simple enough to solve by other means. However, your result should be one neural network that any set of 10 points can be fed into to get the answer. That is, you should not create any sort of algorithm that figures out the slope and intercept on a row-by-row basis. The end result is one network with one set of trained weights.

## Methodology

First of all, I start with a Neural Network with a 4 layers and 100 neurons each, during all the process was used Relu activation function for all layers except for output layers where was used linear function and also Adam optimizer for all process. The result wasn't good for the fist test with 500 epochs.

I test many types of the configuration, increasing and reducing the number of neurons, layers, epochs and learning rate, but always I had a problem. So I ploted the loss chart and I saw that a network was probably overfitting.

I split the database into training (70%) and test (30%) to check how the model behaves with a new data. I tried several settings as Dropout for the regulation, update learning rate, decay but the best configuration was when I put L2 regularization in the hidden layers with 1000 neurons each.

After this, the network was able to have the low loss of training and test data without increasing the loss during the epochs. I used some methods to the data normalization, but did not make difference in the final result.

The last model was trained using 1000 epochs.

Because of the training time I could not test more parameters, but maybe it's possible improve the result of this network reducing the learning rate, trying other optimizers and analyzing better the input data.

## For Run
First, unzip train.zip and test.zip in directory `data/`. Then, you will have a 3 files in `data/`: `test_100k.csv, train_100k.csv and train_100k.truth.csv`.

In folder `req` there is the file `requeriments.txt`, where you will find the librarys necessary for run.
To install librarys:

> pip install -r req.txt

However if you used Anaconda, cuda 8.0 and have a GPU, you can create an env based on the basetgpu file, follow the next codes to create.
In directory `req` you will find `basetfgpu.yml` .In terminal put the codes below:

> conda env create -n tfgpu -f basetfgpu.yml

To activative in linux and Mac:

> source activate tfgpu

or for Windows:

> activate tfgpu

Run the script `interview.py`: 

> python interview.py


Then, to see the loss score:

> python evaluate.py results/submission_train.csv data/train_100k.truth.csv 

## Results

I got this loss score :

`Slope mse: 0.03596314032215982
Intercept mae: 4.598533536702228`
