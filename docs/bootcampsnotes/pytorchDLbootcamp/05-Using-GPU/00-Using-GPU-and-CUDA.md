PyTorch BootCamp - Using GPU
================
by Jawad Haider

- <a href="#what-is-cuda" id="toc-what-is-cuda">What is CUDA?</a>
- <a href="#how-do-i-install-pytorch-for-gpu"
  id="toc-how-do-i-install-pytorch-for-gpu">How do I install PyTorch for
  GPU?</a>
- <a href="#how-do-i-know-if-i-have-cuda-available"
  id="toc-how-do-i-know-if-i-have-cuda-available">How do I know if I have
  CUDA available?</a>
- <a href="#using-gpu-and-cuda" id="toc-using-gpu-and-cuda">Using GPU and
  CUDA</a>
- <a href="#using-cuda-instead-of-cpu"
  id="toc-using-cuda-instead-of-cpu">Using CUDA instead of CPU</a>
  - <a href="#sending-models-to-gpu" id="toc-sending-models-to-gpu">Sending
    Models to GPU</a>
  - <a href="#convert-tensors-to-.cuda-tensors"
    id="toc-convert-tensors-to-.cuda-tensors">Convert Tensors to .cuda()
    tensors</a>

# What is CUDA?

Most people confuse CUDA for a language or maybe an API. It is not.

It’s more than that. CUDA is a parallel computing platform and
programming model that makes using a GPU for general purpose computing
simple and elegant. The developer still programs in the familiar C, C++,
Fortran, or an ever expanding list of supported languages, and
incorporates extensions of these languages in the form of a few basic
keywords.

These keywords let the developer express massive amounts of parallelism
and direct the compiler to the portion of the application that maps to
the GPU.

# How do I install PyTorch for GPU?

Refer to video, its dependent on whether you have an NVIDIA GPU card or
not.

# How do I know if I have CUDA available?

``` python
import torch
torch.cuda.is_available()
# True
```

    True

# Using GPU and CUDA

We’ve provided 2 versions of our yml file, a GPU version and a CPU
version. To use GPU, you need to either manually create a virtual
environment, please watch the video related to this lecture, as not
every computer can run GPU, you need CUDA and an NVIDIA GPU.

``` python
## Get Id of default device
torch.cuda.current_device()
```

    0

``` python
# 0
torch.cuda.get_device_name(0) # Get name device with ID '0'
```

    'GeForce GTX 1080 Ti'

``` python
# Returns the current GPU memory usage by 
# tensors in bytes for a given device
torch.cuda.memory_allocated()
```

    0

``` python
# Returns the current GPU memory managed by the
# caching allocator in bytes for a given device
torch.cuda.memory_cached()
```

    0

# Using CUDA instead of CPU

``` python
# CPU
a = torch.FloatTensor([1.,2.])
```

``` python
a
```

    tensor([1., 2.])

``` python
a.device
```

    device(type='cpu')

``` python
# GPU
a = torch.FloatTensor([1., 2.]).cuda()
```

``` python
a.device
```

    device(type='cuda', index=0)

``` python
torch.cuda.memory_allocated()
```

    512

## Sending Models to GPU

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

``` python
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
```

``` python
torch.manual_seed(32)
model = Model()
```

``` python
# From the discussions here: discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda
next(model.parameters()).is_cuda
```

    False

``` python
gpumodel = model.cuda()
```

``` python
next(gpumodel.parameters()).is_cuda
```

    True

``` python
df = pd.read_csv('../Data/iris.csv')
X = df.drop('target',axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)
```

## Convert Tensors to .cuda() tensors

``` python
X_train = torch.FloatTensor(X_train).cuda()
X_test = torch.FloatTensor(X_test).cuda()
y_train = torch.LongTensor(y_train).cuda()
y_test = torch.LongTensor(y_test).cuda()
```

``` python
trainloader = DataLoader(X_train, batch_size=60, shuffle=True)
testloader = DataLoader(X_test, batch_size=60, shuffle=False)
```

``` python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

``` python
import time
epochs = 100
losses = []
start = time.time()
for i in range(epochs):
    i+=1
    y_pred = gpumodel.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    # a neat trick to save screen space:
    if i%10 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print(f'TOTAL TRAINING TIME: {time.time()-start}')
```

    epoch:  1  loss: 1.15071142
    epoch: 11  loss: 0.93773186
    epoch: 21  loss: 0.77982736
    epoch: 31  loss: 0.60996711
    epoch: 41  loss: 0.40083539
    epoch: 51  loss: 0.25436994
    epoch: 61  loss: 0.15052448
    epoch: 71  loss: 0.10086147
    epoch: 81  loss: 0.08127660
    epoch: 91  loss: 0.07230931
    TOTAL TRAINING TIME: 0.4668765068054199

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
