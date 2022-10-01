Crash Course: Numpy 
================
by Jawad Haider
# **00 - Numpy Arrays**
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

- <a href="#numpy" id="toc-numpy"><span
  class="toc-section-number">1</span> NumPy</a>
  - <a href="#installation-instructions"
    id="toc-installation-instructions"><span
    class="toc-section-number">1.1</span> Installation Instructions</a>
  - <a href="#using-numpy" id="toc-using-numpy"><span
    class="toc-section-number">1.2</span> Using NumPy</a>
- <a href="#numpy-arrays" id="toc-numpy-arrays"><span
  class="toc-section-number">2</span> NumPy Arrays</a>
  - <a href="#creating-numpy-arrays" id="toc-creating-numpy-arrays"><span
    class="toc-section-number">2.1</span> Creating NumPy Arrays</a>
    - <a href="#from-a-python-list" id="toc-from-a-python-list"><span
      class="toc-section-number">2.1.1</span> 1. From a Python List</a>
    - <a href="#built-in-methods" id="toc-built-in-methods"><span
      class="toc-section-number">2.1.2</span> 2. Built-in Methods</a>
    - <a href="#arange" id="toc-arange"><span
      class="toc-section-number">2.1.3</span> arange</a>
    - <a href="#zeros-and-ones" id="toc-zeros-and-ones"><span
      class="toc-section-number">2.1.4</span> zeros and ones</a>
    - <a href="#linspace" id="toc-linspace"><span
      class="toc-section-number">2.1.5</span> linspace</a>
    - <a href="#eye" id="toc-eye"><span
      class="toc-section-number">2.1.6</span> eye</a>
  - <a href="#random" id="toc-random"><span
    class="toc-section-number">2.2</span> Random</a>
    - <a href="#rand" id="toc-rand"><span
      class="toc-section-number">2.2.1</span> 1. rand</a>
    - <a href="#randn" id="toc-randn"><span
      class="toc-section-number">2.2.2</span> 2. randn</a>
    - <a href="#randint" id="toc-randint"><span
      class="toc-section-number">2.2.3</span> 3. randint</a>
    - <a href="#seed" id="toc-seed"><span
      class="toc-section-number">2.2.4</span> 4. seed</a>
  - <a href="#array-attributes-and-methods"
    id="toc-array-attributes-and-methods"><span
    class="toc-section-number">2.3</span> Array Attributes and Methods</a>
    - <a href="#reshape" id="toc-reshape"><span
      class="toc-section-number">2.3.1</span> 1. Reshape</a>
    - <a href="#max-min-argmax-argmin" id="toc-max-min-argmax-argmin"><span
      class="toc-section-number">2.3.2</span> 2. max, min, argmax, argmin</a>
    - <a href="#shape" id="toc-shape"><span
      class="toc-section-number">2.3.3</span> 3. Shape</a>
    - <a href="#dtype" id="toc-dtype"><span
      class="toc-section-number">2.3.4</span> 4. dtype</a>
- <a href="#great-job-thats-the-end-of-this-part."
  id="toc-great-job-thats-the-end-of-this-part."><span
  class="toc-section-number">3</span> Great Job! Thats the end of this
  part.</a>

------------------------------------------------------------------------

# NumPy

NumPy is a powerful linear algebra library for Python. What makes it so
important is that almost all of the libraries in the
<a href='https://pydata.org/'>PyData</a> ecosystem (pandas, scipy,
scikit-learn, etc.) rely on NumPy as one of their main building blocks.
Plus we will use it to generate data for our analysis examples later on!

NumPy is also incredibly fast, as it has bindings to C libraries. For
more info on why you would want to use arrays instead of lists, check
out this great [StackOverflow
post](http://stackoverflow.com/questions/993984/why-numpy-instead-of-python-lists).

We will only learn the basics of NumPy. To get started we need to
install it!

## Installation Instructions

**NumPy is already included in your environment! You are good to go if
you are using the course environment!**

------------------------------------------------------------------------

**For those not using the provided environment:**

**It is highly recommended you install Python using the Anaconda
distribution to make sure all underlying dependencies (such as Linear
Algebra libraries) all sync up with the use of a conda install. If you
have Anaconda, install NumPy by going to your terminal or command prompt
and typing:**

    conda install numpy

**If you do not have Anaconda and can not install it, please refer to
[Numpy’s official documentation on various installation
instructions.](https://www.scipy.org/install.html)**

------------------------------------------------------------------------

## Using NumPy

Once you’ve installed NumPy you can import it as a library:

``` python
import numpy as np
```

NumPy has many built-in functions and capabilities. We won’t cover them
all but instead we will focus on some of the most important aspects of
NumPy: vectors, arrays, matrices and number generation. Let’s start by
discussing arrays.

# NumPy Arrays

NumPy arrays are the main way we will use NumPy throughout the course.
NumPy arrays essentially come in two flavors: vectors and matrices.
Vectors are strictly 1-dimensional (1D) arrays and matrices are 2D (but
you should note a matrix can still have only one row or one column).

Let’s begin our introduction by exploring how to create NumPy arrays.

## Creating NumPy Arrays

### 1. From a Python List

We can create an array by directly converting a list or list of lists:

``` python
my_list = [1,2,3]
my_list
```

    [1, 2, 3]

``` python
np.array(my_list)
```

    array([1, 2, 3])

``` python
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix
```

    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

``` python
np.array(my_matrix)
```

    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

### 2. Built-in Methods

There are lots of built-in ways to generate arrays.

### arange

Return evenly spaced values within a given interval.\[[Numpy ndarray
arange](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.arange.html)\]

``` python
np.arange(0,10)
```

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

``` python
np.arange(0,11,2)
```

    array([ 0,  2,  4,  6,  8, 10])

### zeros and ones

Generate arrays of zeros or ones.\[[Numpy ndarray
zeros](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.zeros.html)\]

np.zeros(3)

``` python
np.zeros((5,5))
```

    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

``` python
np.ones(3)
```

    array([1., 1., 1.])

``` python
np.ones((3,3))
```

    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

### linspace

Return evenly spaced numbers over a specified interval.\[[Numpy ndarray
linspace](https://www.numpy.org/devdocs/reference/generated/numpy.linspace.html)\]

``` python
np.linspace(0,10,3)
```

    array([ 0.,  5., 10.])

``` python
np.linspace(0,5,20)
```

    array([0.        , 0.26315789, 0.52631579, 0.78947368, 1.05263158,
           1.31578947, 1.57894737, 1.84210526, 2.10526316, 2.36842105,
           2.63157895, 2.89473684, 3.15789474, 3.42105263, 3.68421053,
           3.94736842, 4.21052632, 4.47368421, 4.73684211, 5.        ])

<font color=green>Note that `.linspace()` *includes* the stop value. To
obtain an array of common fractions, increase the number of
items:</font>

``` python
np.linspace(0,5,21)
```

    array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ,
           2.75, 3.  , 3.25, 3.5 , 3.75, 4.  , 4.25, 4.5 , 4.75, 5.  ])

### eye

Creates an identity matrix \[[Numpy
eye](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.eye.html)\]

``` python
np.eye(4)
```

    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])

## Random

Numpy also has lots of ways to create random number arrays. Here we will
go through some of the most used methods from random module

### 1. rand

Creates an array of the given shape and populates it with random samples
from a uniform distribution over `[0, 1)`. \[[Random
rand](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.rand.html)\]

``` python
np.random.rand(2)
```

    array([0.69647666, 0.14395438])

``` python
np.random.rand(5,5)
```

    array([[0.9112553 , 0.75849901, 0.43392287, 0.4134459 , 0.10902179],
           [0.66881652, 0.21265267, 0.21783956, 0.08716564, 0.46147918],
           [0.16064897, 0.38241433, 0.50076915, 0.58926492, 0.69837196],
           [0.88502465, 0.2996012 , 0.49291933, 0.75316852, 0.29998398],
           [0.42345042, 0.57034504, 0.94797283, 0.70571464, 0.35788149]])

### 2. randn

Returns a sample (or samples) from the “standard normal” distribution
\[σ = 1\]. Unlike **rand** which is uniform, values closer to zero are
more likely to appear. \[[Radnom
randn](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randn.html)\]

``` python
np.random.randn(2)
```

    array([-0.55673554, -0.08515858])

``` python
np.random.randn(5,5)
```

    array([[ 0.83041645, -1.22369138,  0.258011  ,  0.90984287, -0.48702078],
           [-0.88539528, -0.54034218, -0.39928196,  0.85910869, -0.36305332],
           [ 0.132046  , -1.28709664,  0.49352402,  0.80293611,  0.2601146 ],
           [ 0.74912365,  0.16013944,  0.39345536, -0.52355146,  1.0536796 ],
           [ 0.00293273, -0.14715505, -1.22460234, -0.65347358, -0.31514422]])

### 3. randint

Returns random integers from `low` (inclusive) to `high` (exclusive).
\[[Random
randint](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randint.html)\]

``` python
np.random.randint(1,100)
```

    42

``` python
np.random.randint(1,100,10)
```

    array([33, 26, 51, 78, 89, 15, 42, 68, 14, 62])

### 4. seed

Can be used to set the random state, so that the same “random” results
can be reproduced. \[[Numpy ndarray
random](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.seed.html)\]

``` python
np.random.seed(42)
np.random.rand(4)
```

    array([0.37454012, 0.95071431, 0.73199394, 0.59865848])

``` python
np.random.seed(42)
np.random.rand(4)
```

    array([0.37454012, 0.95071431, 0.73199394, 0.59865848])

## Array Attributes and Methods

Let’s discuss some useful attributes and methods for an array:  
**In particular, the reshape attribute and max,min,argmax, argmin, shape
& the dytpe methods**  
Let’s first create two numpy arrays to experiment with :)

``` python
arr = np.arange(25)
ranarr = np.random.randint(0,50,10)
```

``` python
arr
```

    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24])

``` python
ranarr
```

    array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

### 1. Reshape

Returns an array containing the same data with a new shape.\[[Numpy
ndarray
reshape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.reshape.html)\]

``` python
arr.reshape(5,5)
```

    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

### 2. max, min, argmax, argmin

These are useful methods for finding max or min values. Or to find their
index locations using argmin or argmax

``` python
ranarr
```

    array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

``` python
ranarr.max()
```

    39

``` python
ranarr.argmax()
```

    7

``` python
ranarr.min()
```

    2

``` python
ranarr.argmin()
```

    9

### 3. Shape

Shape is an attribute that arrays have (not a method):\[[Numpy ndarray
Shape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html)\]

``` python
# Vector
arr.shape
```

    (25,)

``` python
# Notice the two sets of brackets
arr.reshape(1,25)
```

    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24]])

``` python
arr.reshape(1,25).shape
```

    (1, 25)

``` python
arr.reshape(25,1)
```

    array([[ 0],
           [ 1],
           [ 2],
           [ 3],
           [ 4],
           [ 5],
           [ 6],
           [ 7],
           [ 8],
           [ 9],
           [10],
           [11],
           [12],
           [13],
           [14],
           [15],
           [16],
           [17],
           [18],
           [19],
           [20],
           [21],
           [22],
           [23],
           [24]])

``` python
arr.reshape(25,1).shape
```

    (25, 1)

### 4. dtype

You can also grab the data type of the object in the array:\[[Numpy
ndarray
dtype](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.dtype.html)\]

``` python
arr.dtype
```

    dtype('int64')

``` python
arr2 = np.array([1.2, 3.4, 5.6])
arr2.dtype
```

    dtype('float64')

# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`
