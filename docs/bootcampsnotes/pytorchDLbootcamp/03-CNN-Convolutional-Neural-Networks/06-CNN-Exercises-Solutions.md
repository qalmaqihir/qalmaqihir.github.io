PyTorch BootCamp - Convolutional Neural Networks
================
by Jawad Haider

- <a href="#cnn-exercises---solutions"
  id="toc-cnn-exercises---solutions">CNN Exercises - Solutions</a>
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

# CNN Exercises - Solutions

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
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
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
for images,labels in train_loader: 
    break

im = make_grid(images, nrow=10)
plt.figure(figsize=(12,4))

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
```

![](06-CNN-Exercises-Solutions_files/figure-gfm/cell-6-output-1.png)

``` python
# DON'T WRITE HERE
# IMAGES AND LABELS
for images,labels in train_loader: 
    break

print('Label: ', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow=10)
plt.figure(figsize=(12,4))

plt.imshow(np.transpose(im.numpy(), (1, 2, 0)));
```

    Label:  [3 0 0 3 1 3 5 4 8 8]
    Class:  Dress T-shirt T-shirt Dress Trouser Dress Sandal Coat Bag Bag

![](06-CNN-Exercises-Solutions_files/figure-gfm/cell-7-output-2.png)

## Downsampling

<h3>

3.  If a 28x28 image is passed through a Convolutional layer using a 5x5
    filter, a step size of 1, and no padding, what is the resulting
    matrix size?
    </h3>

<div style="border:1px black solid; padding:5px">

A 5x5 filter leaves a two-pixel border on each side, so the overall
dimension is reduced by 4.<br> The result is a 24x24 matrix.

</div>

``` python
# Run the code below to check your answer:
conv = nn.Conv2d(1, 1, 5, 1)
for x,labels in train_loader:
    print('Orig size:',x.shape)
    break
x = conv(x)
print('Down size:',x.shape)
```

    Orig size: torch.Size([10, 1, 28, 28])
    Down size: torch.Size([10, 1, 24, 24])

### 4. If the sample from question 3 is then passed through a 2x2 MaxPooling layer, what is the resulting matrix size?

<div style="border:1px black solid; padding:5px">

If a 2x2 pooling layer is applied to a 24x24 matrix, each side is
divided by two, and rounded down if necessary.<br> The result is a 12x12
matrix.

</div>

``` python
# Run the code below to check your answer:
x = F.max_pool2d(x, 2, 2)
print('Down size:',x.shape)
```

    Down size: torch.Size([10, 1, 12, 12])

## CNN definition

### 5. Define a convolutional neural network

Define a CNN model that can be trained on the Fashion-MNIST dataset. The
model should contain two convolutional layers, two pooling layers, and
two fully connected layers. You can use any number of neurons per layer
so long as the model takes in a 28x28 image and returns an output of 10.
Portions of the definition have been filled in for convenience.

``` python
# DON'T WRITE HERE
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)
    
torch.manual_seed(101)
model = ConvolutionalNetwork()
```

## Trainable parameters

### 6. What is the total number of trainable parameters (weights & biases) in the model above?

Answers will vary depending on your model definition.

<div style="border:1px black solid; padding:5px">

$\quad\begin{split}(1\times6\times3\times3)+6+(6\times16\times3\times3)+16+(400\times100)+100+(100\times10)+10 &=\\ 54+6+864+16+40000+100+1000+10 &= 42,050\end{split}$<br>

</div>

``` python
# Run the code below to check your answer:
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')
    
count_parameters(model)
```

        54
         6
       864
        16
     40000
       100
      1000
        10
    ______
     42050

### 7. Define loss function & optimizer

Define a loss function called “criterion” and an optimizer called
“optimizer”.<br> You can use any functions you want, although we used
Cross Entropy Loss and Adam (learning rate of 0.001) respectively.

``` python
# DON'T WRITE HERE
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### 8. Train the model

Don’t worry about tracking loss values, displaying results, or
validating the test set. Just train the model through 5 epochs. We’ll
evaluate the trained model in the next step.<br> OPTIONAL: print
something after each epoch to indicate training progress.

``` python
# DON'T WRITE HERE
epochs = 5

for i in range(epochs):
    for X_train, y_train in train_loader:

        # Apply the model
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # OPTIONAL print statement
    print(f'{i+1} of {epochs} epochs completed')
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
# DON'T WRITE HERE
model.eval()

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        y_val = model(X_test)
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
        
print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')
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
