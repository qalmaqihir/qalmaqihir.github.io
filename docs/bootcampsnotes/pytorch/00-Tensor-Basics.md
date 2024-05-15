================

by Jawad Haider
# **00 - Tensor Basics**
------------------------------------------------------------------------
<center>
<a href=''>![Image](../../assets/img/logo1.png)</a>
</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
------------------------------------------------------------------------

- <a href="#tensor-basics" id="toc-tensor-basics"><span
  class="toc-section-number">1</span> Tensor Basics</a>
  - <a href="#perform-standard-imports"
    id="toc-perform-standard-imports"><span
    class="toc-section-number">1.1</span> Perform standard imports</a>
  - <a href="#converting-numpy-arrays-to-pytorch-tensors"
    id="toc-converting-numpy-arrays-to-pytorch-tensors"><span
    class="toc-section-number">1.2</span> Converting NumPy arrays to PyTorch
    tensors</a>
  - <a href="#copying-vs.-sharing" id="toc-copying-vs.-sharing"><span
    class="toc-section-number">1.3</span> Copying vs. sharing</a>
  - <a href="#class-constructors" id="toc-class-constructors"><span
    class="toc-section-number">1.4</span> Class constructors</a>
  - <a href="#creating-tensors-from-scratch"
    id="toc-creating-tensors-from-scratch"><span
    class="toc-section-number">1.5</span> Creating tensors from scratch</a>
    - <a href="#uninitialized-tensors-with-.empty"
      id="toc-uninitialized-tensors-with-.empty"><span
      class="toc-section-number">1.5.1</span> Uninitialized tensors with
      <tt>.empty()</tt></a>
    - <a href="#initialized-tensors-with-.zeros-and-.ones"
      id="toc-initialized-tensors-with-.zeros-and-.ones"><span
      class="toc-section-number">1.5.2</span> Initialized tensors with
      <tt>.zeros()</tt> and <tt>.ones()</tt></a>
    - <a href="#tensors-from-ranges" id="toc-tensors-from-ranges"><span
      class="toc-section-number">1.5.3</span> Tensors from ranges</a>
    - <a href="#tensors-from-data" id="toc-tensors-from-data"><span
      class="toc-section-number">1.5.4</span> Tensors from data</a>
    - <a href="#changing-the-dtype-of-existing-tensors"
      id="toc-changing-the-dtype-of-existing-tensors"><span
      class="toc-section-number">1.5.5</span> Changing the dtype of existing
      tensors</a>
    - <a href="#random-number-tensors" id="toc-random-number-tensors"><span
      class="toc-section-number">1.5.6</span> Random number tensors</a>
    - <a href="#random-number-tensors-that-follow-the-input-size"
      id="toc-random-number-tensors-that-follow-the-input-size"><span
      class="toc-section-number">1.5.7</span> Random number tensors that
      follow the input size</a>
    - <a href="#setting-the-random-seed"
      id="toc-setting-the-random-seed"><span
      class="toc-section-number">1.5.8</span> Setting the random seed</a>
  - <a href="#tensor-attributes" id="toc-tensor-attributes"><span
    class="toc-section-number">1.6</span> Tensor attributes</a>
    - <a href="#great-job" id="toc-great-job"><span
      class="toc-section-number">1.6.1</span> Great job!</a>

------------------------------------------------------------------------

# Tensor Basics

This section covers: \* Converting NumPy arrays to PyTorch tensors \*
Creating tensors from scratch

## Perform standard imports

``` python
import torch
import numpy as np
```

Confirm you’re using PyTorch version 1.1.0

``` python
torch.__version__
```

    '1.1.0'

## Converting NumPy arrays to PyTorch tensors

A
<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.Tensor</tt></strong></a>
is a multi-dimensional matrix containing elements of a single data
type.<br> Calculations between tensors can only happen if the tensors
share the same dtype.<br> In some cases tensors are used as a
replacement for NumPy to use the power of GPUs (more on this later).

``` python
arr = np.array([1,2,3,4,5])
print(arr)
print(arr.dtype)
print(type(arr))
```

    [1 2 3 4 5]
    int32
    <class 'numpy.ndarray'>

``` python
x = torch.from_numpy(arr)
# Equivalent to x = torch.as_tensor(arr)

print(x)
```

    tensor([1, 2, 3, 4, 5], dtype=torch.int32)

``` python
# Print the type of data held by the tensor
print(x.dtype)
```

    torch.int32

``` python
# Print the tensor object type
print(type(x))
print(x.type()) # this is more specific!
```

    <class 'torch.Tensor'>
    torch.IntTensor

``` python
arr2 = np.arange(0.,12.).reshape(4,3)
print(arr2)
```

    [[ 0.  1.  2.]
     [ 3.  4.  5.]
     [ 6.  7.  8.]
     [ 9. 10. 11.]]

``` python
x2 = torch.from_numpy(arr2)
print(x2)
print(x2.type())
```

    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]], dtype=torch.float64)
    torch.DoubleTensor

Here <tt>torch.DoubleTensor</tt> refers to 64-bit floating point data.

<h2>
<a href='https://pytorch.org/docs/stable/tensors.html'>Tensor
Datatypes</a>
</h2>
<table style="display: inline-block">
<tr>
<th>
TYPE
</th>
<th>
NAME
</th>
<th>
EQUIVALENT
</th>
<th>
TENSOR TYPE
</th>
</tr>
<tr>
<td>
32-bit integer (signed)
</td>
<td>
torch.int32
</td>
<td>
torch.int
</td>
<td>
IntTensor
</td>
</tr>
<tr>
<td>
64-bit integer (signed)
</td>
<td>
torch.int64
</td>
<td>
torch.long
</td>
<td>
LongTensor
</td>
</tr>
<tr>
<td>
16-bit integer (signed)
</td>
<td>
torch.int16
</td>
<td>
torch.short
</td>
<td>
ShortTensor
</td>
</tr>
<tr>
<td>
32-bit floating point
</td>
<td>
torch.float32
</td>
<td>
torch.float
</td>
<td>
FloatTensor
</td>
</tr>
<tr>
<td>
64-bit floating point
</td>
<td>
torch.float64
</td>
<td>
torch.double
</td>
<td>
DoubleTensor
</td>
</tr>
<tr>
<td>
16-bit floating point
</td>
<td>
torch.float16
</td>
<td>
torch.half
</td>
<td>
HalfTensor
</td>
</tr>
<tr>
<td>
8-bit integer (signed)
</td>
<td>
torch.int8
</td>
<td>
</td>
<td>
CharTensor
</td>
</tr>
<tr>
<td>
8-bit integer (unsigned)
</td>
<td>
torch.uint8
</td>
<td>
</td>
<td>
ByteTensor
</td>
</tr>
</table>

## Copying vs. sharing

<a href='https://pytorch.org/docs/stable/torch.html#torch.from_numpy'><strong><tt>torch.from_numpy()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.as_tensor'><strong><tt>torch.as_tensor()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.tensor'><strong><tt>torch.tensor()</tt></strong></a><br>

There are a number of different functions available for
<a href='https://pytorch.org/docs/stable/torch.html#creation-ops'>creating
tensors</a>. When using
<a href='https://pytorch.org/docs/stable/torch.html#torch.from_numpy'><strong><tt>torch.from_numpy()</tt></strong></a>
and
<a href='https://pytorch.org/docs/stable/torch.html#torch.as_tensor'><strong><tt>torch.as_tensor()</tt></strong></a>,
the PyTorch tensor and the source NumPy array share the same memory.
This means that changes to one affect the other. However, the
<a href='https://pytorch.org/docs/stable/torch.html#torch.tensor'><strong><tt>torch.tensor()</tt></strong></a>
function always makes a copy.

``` python
# Using torch.from_numpy()
arr = np.arange(0,5)
t = torch.from_numpy(arr)
print(t)
```

    tensor([0, 1, 2, 3, 4], dtype=torch.int32)

``` python
arr[2]=77
print(t)
```

    tensor([ 0,  1, 77,  3,  4], dtype=torch.int32)

``` python
# Using torch.tensor()
arr = np.arange(0,5)
t = torch.tensor(arr)
print(t)
```

    tensor([0, 1, 2, 3, 4], dtype=torch.int32)

``` python
arr[2]=77
print(t)
```

    tensor([0, 1, 2, 3, 4], dtype=torch.int32)

## Class constructors

<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.Tensor()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.FloatTensor()</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/tensors.html'><strong><tt>torch.LongTensor()</tt></strong></a>,
etc.<br>

There’s a subtle difference between using the factory function
<font color=black><tt>torch.tensor(data)</tt></font> and the class
constructor <font color=black><tt>torch.Tensor(data)</tt></font>.<br>
The factory function determines the dtype from the incoming data, or
from a passed-in dtype argument.<br> The class constructor
<tt>torch.Tensor()</tt>is simply an alias for
<tt>torch.FloatTensor(data)</tt>. Consider the following:

``` python
data = np.array([1,2,3])
```

``` python
a = torch.Tensor(data)  # Equivalent to cc = torch.FloatTensor(data)
print(a, a.type())
```

    tensor([1., 2., 3.]) torch.FloatTensor

``` python
b = torch.tensor(data)
print(b, b.type())
```

    tensor([1, 2, 3], dtype=torch.int32) torch.IntTensor

``` python
c = torch.tensor(data, dtype=torch.long)
print(c, c.type())
```

    tensor([1, 2, 3]) torch.LongTensor

## Creating tensors from scratch

### Uninitialized tensors with <tt>.empty()</tt>

<a href='https://pytorch.org/docs/stable/torch.html#torch.empty'><strong><tt>torch.empty()</tt></strong></a>
returns an <em>uninitialized</em> tensor. Essentially a block of memory
is allocated according to the size of the tensor, and any values already
sitting in the block are returned. This is similar to the behavior of
<tt>numpy.empty()</tt>.

``` python
x = torch.empty(4, 3)
print(x)
```

    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])

### Initialized tensors with <tt>.zeros()</tt> and <tt>.ones()</tt>

<a href='https://pytorch.org/docs/stable/torch.html#torch.zeros'><strong><tt>torch.zeros(size)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.ones'><strong><tt>torch.ones(size)</tt></strong></a><br>
It’s a good idea to pass in the intended dtype.

``` python
x = torch.zeros(4, 3, dtype=torch.int64)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])

### Tensors from ranges

<a href='https://pytorch.org/docs/stable/torch.html#torch.arange'><strong><tt>torch.arange(start,end,step)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.linspace'><strong><tt>torch.linspace(start,end,steps)</tt></strong></a><br>
Note that with <tt>.arange()</tt>, <tt>end</tt> is exclusive, while with
<tt>linspace()</tt>, <tt>end</tt> is inclusive.

``` python
x = torch.arange(0,18,2).reshape(3,3)
print(x)
```

    tensor([[ 0,  2,  4],
            [ 6,  8, 10],
            [12, 14, 16]])

``` python
x = torch.linspace(0,18,12).reshape(3,4)
print(x)
```

    tensor([[ 0.0000,  1.6364,  3.2727,  4.9091],
            [ 6.5455,  8.1818,  9.8182, 11.4545],
            [13.0909, 14.7273, 16.3636, 18.0000]])

### Tensors from data

<tt>torch.tensor()</tt> will choose the dtype based on incoming data:

``` python
x = torch.tensor([1, 2, 3, 4])
print(x)
print(x.dtype)
print(x.type())
```

    tensor([1, 2, 3, 4])
    torch.int64
    torch.LongTensor

Alternatively you can set the type by the tensor method used. For a list
of tensor types visit https://pytorch.org/docs/stable/tensors.html

``` python
x = torch.FloatTensor([5,6,7])
print(x)
print(x.dtype)
print(x.type())
```

    tensor([5., 6., 7.])
    torch.float32
    torch.FloatTensor

You can also pass the dtype in as an argument. For a list of dtypes
visit
https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.dtype<br>

``` python
x = torch.tensor([8,9,-3], dtype=torch.int)
print(x)
print(x.dtype)
print(x.type())
```

    tensor([ 8,  9, -3], dtype=torch.int32)
    torch.int32
    torch.IntTensor

### Changing the dtype of existing tensors

Don’t be tempted to use <tt>x = torch.tensor(x, dtype=torch.type)</tt>
as it will raise an error about improper use of tensor cloning.<br>
Instead, use the tensor <tt>.type()</tt> method.

``` python
print('Old:', x.type())

x = x.type(torch.int64)

print('New:', x.type())
```

    Old: torch.IntTensor
    New: torch.LongTensor

### Random number tensors

<a href='https://pytorch.org/docs/stable/torch.html#torch.rand'><strong><tt>torch.rand(size)</tt></strong></a>
returns random samples from a uniform distribution over \[0, 1)<br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randn'><strong><tt>torch.randn(size)</tt></strong></a>
returns samples from the “standard normal” distribution \[σ = 1\]<br>
    Unlike <tt>rand</tt> which is uniform, values closer to zero are
more likely to appear.<br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randint'><strong><tt>torch.randint(low,high,size)</tt></strong></a>
returns random integers from low (inclusive) to high (exclusive)

``` python
x = torch.rand(4, 3)
print(x)
```

    tensor([[0.0211, 0.2336, 0.6775],
            [0.4790, 0.5132, 0.9878],
            [0.7552, 0.0789, 0.1860],
            [0.6712, 0.1564, 0.3753]])

``` python
x = torch.randn(4, 3)
print(x)
```

    tensor([[ 0.7164, -0.1538, -0.9980],
            [-1.8252,  1.1863, -0.1523],
            [ 1.4093, -0.0212, -1.5598],
            [ 0.1831, -0.6961,  1.3497]])

``` python
x = torch.randint(0, 5, (4, 3))
print(x)
```

    tensor([[0, 3, 0],
            [1, 3, 4],
            [1, 2, 3],
            [4, 4, 3]])

### Random number tensors that follow the input size

<a href='https://pytorch.org/docs/stable/torch.html#torch.rand_like'><strong><tt>torch.rand_like(input)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randn_like'><strong><tt>torch.randn_like(input)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.randint_like'><strong><tt>torch.randint_like(input,low,high)</tt></strong></a><br>
these return random number tensors with the same size as <tt>input</tt>

``` python
x = torch.zeros(2,5)
print(x)
```

    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])

``` python
x2 = torch.randn_like(x)
print(x2)
```

    tensor([[-0.5442, -0.3149,  0.0922,  1.1829, -0.7873],
            [ 0.3143,  0.9465,  0.4534,  0.4623,  2.2044]])

The same syntax can be used with<br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.zeros_like'><strong><tt>torch.zeros_like(input)</tt></strong></a><br>
<a href='https://pytorch.org/docs/stable/torch.html#torch.ones_like'><strong><tt>torch.ones_like(input)</tt></strong></a>

``` python
x3 = torch.ones_like(x2)
print(x3)
```

    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])

### Setting the random seed

<a href='https://pytorch.org/docs/stable/torch.html#torch.manual_seed'><strong><tt>torch.manual_seed(int)</tt></strong></a>
is used to obtain reproducible results

``` python
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)
```

    tensor([[0.8823, 0.9150, 0.3829],
            [0.9593, 0.3904, 0.6009]])

``` python
torch.manual_seed(42)
x = torch.rand(2, 3)
print(x)
```

    tensor([[0.8823, 0.9150, 0.3829],
            [0.9593, 0.3904, 0.6009]])

## Tensor attributes

Besides <tt>dtype</tt>, we can look at other
<a href='https://pytorch.org/docs/stable/tensor_attributes.html'>tensor
attributes</a> like <tt>shape</tt>, <tt>device</tt> and <tt>layout</tt>

``` python
x.shape
```

    torch.Size([2, 3])

``` python
x.size()  # equivalent to x.shape
```

    torch.Size([2, 3])

``` python
x.device
```

    device(type='cpu')

PyTorch supports use of multiple
<a href='https://pytorch.org/docs/stable/tensor_attributes.html#torch-device'>devices</a>,
harnessing the power of one or more GPUs in addition to the CPU. We
won’t explore that here, but you should know that operations between
tensors can only happen for tensors installed on the same device.

``` python
x.layout
```

    torch.strided

PyTorch has a class to hold the
<a href='https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.layout'>memory
layout</a> option. The default setting of
<a href='https://en.wikipedia.org/wiki/Stride_of_an_array'>strided</a>
will suit our purposes throughout the course.

### Great job!
