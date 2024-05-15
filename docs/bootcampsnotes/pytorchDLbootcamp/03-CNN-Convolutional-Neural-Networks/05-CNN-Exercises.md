================
by Jawad Haider

- <a href="#cnn-exercises" id="toc-cnn-exercises">CNN Exercises</a>
  - <a href="#perform-standard-imports-load-the-fashion-mnist-dataset"
    id="toc-perform-standard-imports-load-the-fashion-mnist-dataset">Perform
    standard imports, load the Fashion-MNIST dataset</a>
  - <a href="#create-data-loaders" id="toc-create-data-loaders">1. Create
    data loaders</a>
  - <a href="#examine-a-batch-of-images"
    id="toc-examine-a-batch-of-images">2. Examine a batch of images</a>
  - <a href="#downsampling" id="toc-downsampling">Downsampling</a>
    - <a
      href="#if-the-sample-from-question-3-is-then-passed-through-a-2x2-maxpooling-layer-what-is-the-resulting-matrix-size"
      id="toc-if-the-sample-from-question-3-is-then-passed-through-a-2x2-maxpooling-layer-what-is-the-resulting-matrix-size">4.
      If the sample from question 3 is then passed through a 2x2 MaxPooling
      layer, what is the resulting matrix size?</a>
  - <a href="#cnn-definition" id="toc-cnn-definition">CNN definition</a>
    - <a href="#define-a-convolutional-neural-network"
      id="toc-define-a-convolutional-neural-network">5. Define a convolutional
      neural network</a>
  - <a href="#trainable-parameters" id="toc-trainable-parameters">Trainable
    parameters</a>
    - <a
      href="#what-is-the-total-number-of-trainable-parameters-weights-biases-in-the-model-above"
      id="toc-what-is-the-total-number-of-trainable-parameters-weights-biases-in-the-model-above">6.
      What is the total number of trainable parameters (weights &amp; biases)
      in the model above?</a>
    - <a href="#define-loss-function-optimizer"
      id="toc-define-loss-function-optimizer">7. Define loss function &amp;
      optimizer</a>
    - <a href="#train-the-model" id="toc-train-the-model">8. Train the
      model</a>
    - <a href="#evaluate-the-model" id="toc-evaluate-the-model">9. Evaluate
      the model</a>
  - <a href="#great-job" id="toc-great-job">Great job!</a>

# CNN Exercises

For these exercises we’ll work with the
<a href='https://www.kaggle.com/zalando-research/fashionmnist'>Fashion-MNIST</a>
dataset, also available through
<a href='https://pytorch.org/docs/stable/torchvision/index.html'><tt><strong>torchvision</strong></tt></a>.
Like MNIST, this dataset consists of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale
image, associated with a label from 10 classes: 0. T-shirt/top 1.
Trouser 2. Pullover 3. Dress 4. Coat 5. Sandal 6. Shirt 7. Sneaker 8.
Bag 9. Ankle boot

<div class="alert alert-danger" style="margin: 10px">

<strong>IMPORTANT NOTE!</strong> Make sure you don’t run the cells
directly above the example output shown, <br>otherwise you will end up
writing over the example output!

</div>

## Perform standard imports, load the Fashion-MNIST dataset

Run the cell below to load the libraries needed for this exercise and
the Fashion-MNIST dataset.<br> PyTorch makes the Fashion-MNIST dataset
available through
<a href='https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist'><tt><strong>torchvision</strong></tt></a>.
The first time it’s called, the dataset will be downloaded onto your
computer to the path specified. From that point, torchvision will always
look for a local copy before attempting another download.

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='../Data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='../Data', train=False, download=True, transform=transform)

class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']
```

## 1. Create data loaders

Use DataLoader to create a <tt>train_loader</tt> and a
<tt>test_loader</tt>. Batch sizes should be 10 for both.

``` python
# CODE HERE


```

``` python
# DON'T WRITE HERE
```

## 2. Examine a batch of images

Use DataLoader, <tt>make_grid</tt> and matplotlib to display the first
batch of 10 images.<br> OPTIONAL: display the labels as well

``` python
# CODE HERE





```

``` python
# DON'T WRITE HERE
# IMAGES ONLY
```

![](05-CNN-Exercises_files/figure-gfm/cell-6-output-1.png)

``` python
# DON'T WRITE HERE
# IMAGES AND LABELS
```

    Label:  [9 2 5 9 4 2 1 2 7 3]
    Class:  Boot Sweater Sandal Boot Coat Sweater Trouser Sweater Sneaker Dress

![](05-CNN-Exercises_files/figure-gfm/cell-7-output-2.png)

## Downsampling

<h3>

3.  If a 28x28 image is passed through a Convolutional layer using a 5x5
    filter, a step size of 1, and no padding, what is the resulting
    matrix size?
    </h3>

<div style="border:1px black solid; padding:5px">

<br><br>

</div>

``` python
##################################################
###### ONLY RUN THIS TO CHECK YOUR ANSWER! ######
################################################

# Run the code below to check your answer:
conv = nn.Conv2d(1, 1, 5, 1)
for x,labels in train_loader:
    print('Orig size:',x.shape)
    break
x = conv(x)
print('Down size:',x.shape)
```

### 4. If the sample from question 3 is then passed through a 2x2 MaxPooling layer, what is the resulting matrix size?

<div style="border:1px black solid; padding:5px">

<br><br>

</div>

``` python
##################################################
###### ONLY RUN THIS TO CHECK YOUR ANSWER! ######
################################################

# Run the code below to check your answer:
x = F.max_pool2d(x, 2, 2)
print('Down size:',x.shape)
```

## CNN definition

### 5. Define a convolutional neural network

Define a CNN model that can be trained on the Fashion-MNIST dataset. The
model should contain two convolutional layers, two pooling layers, and
two fully connected layers. You can use any number of neurons per layer
so long as the model takes in a 28x28 image and returns an output of 10.
Portions of the definition have been filled in for convenience.

``` python
# CODE HERE
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X):
        pass 
        return 
    
torch.manual_seed(101)
model = ConvolutionalNetwork()
```

## Trainable parameters

### 6. What is the total number of trainable parameters (weights & biases) in the model above?

Answers will vary depending on your model definition.

<div style="border:1px black solid; padding:5px">

<br><br>

</div>

``` python
# CODE HERE
```

### 7. Define loss function & optimizer

Define a loss function called “criterion” and an optimizer called
“optimizer”.<br> You can use any functions you want, although we used
Cross Entropy Loss and Adam (learning rate of 0.001) respectively.

``` python
# CODE HERE


```

``` python
# DON'T WRITE HERE
```

### 8. Train the model

Don’t worry about tracking loss values, displaying results, or
validating the test set. Just train the model through 5 epochs. We’ll
evaluate the trained model in the next step.<br> OPTIONAL: print
something after each epoch to indicate training progress.

``` python
# CODE HERE




```

``` python
```

    1 of 5 epochs completed
    2 of 5 epochs completed
    3 of 5 epochs completed
    4 of 5 epochs completed
    5 of 5 epochs completed

### 9. Evaluate the model

Set <tt>model.eval()</tt> and determine the percentage correct out of
10,000 total test images.

``` python
# CODE HERE




```

``` python
```

    Test accuracy: 8733/10000 =  87.330%

## Great job!

<center>

<a href=''> ![Logo](../logo1.png) </a>

</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
