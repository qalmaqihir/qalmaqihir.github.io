Tensorflow BootCamp - Low Level Tensorflow
================
by Jawad Haider

- <a href="#basic-computation" id="toc-basic-computation">Basic
  Computation</a>

## Basic Computation

``` python
# Install TensorFlow
# !pip install -q tensorflow-gpu==2.0.0-beta1

try:
  %tensorflow_version 2.x  # Colab only.
except Exception:
  pass

import tensorflow as tf
print(tf.__version__)
```

    `%tensorflow_version` only switches the major version: 1.x or 2.x.
    You set: `2.x  # Colab only.`. This will be interpreted as: `2.x`.


    TensorFlow 2.x selected.
    2.2.0-rc2

``` python
a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.sqrt(a**2 + b**2)
print("c:", c)

# if you use Python 3 f-strings it will print
# the tensor as a float
print(f"c: {c}") 
```

    c: tf.Tensor(5.0, shape=(), dtype=float32)
    c: 5.0

``` python
# Get the Numpy version of a Tensor
c.numpy()
```

    5.0

``` python
type(c.numpy())
```

    numpy.float32

``` python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print(f"b: {b}")
c = tf.tensordot(a, b, axes=[0,0])
print(f"c: {c}")
```

    b: [4 5 6]
    c: 32

``` python
a.numpy().dot(b.numpy())
```

``` python
import numpy as np
A0 = np.random.randn(3, 3)
b0 = np.random.randn(3, 1)
c0 = A0.dot(b0)
print(f"c0: {c0}")

A = tf.constant(A0)
b = tf.constant(b0)
c = tf.matmul(A, b)
print(f"c: {c}")
```

    c0: [[ 1.13966116]
     [-0.31443995]
     [-0.78649886]]
    c: [[ 1.13966116]
     [-0.31443995]
     [-0.78649886]]

``` python
# Broadcasting
A = tf.constant([[1,2],[3,4]])
b = tf.constant(1)
C = A + b
print(f"C: {C}")
```

    C: [[2 3]
     [4 5]]

``` python
# Element-wise multiplication
A = tf.constant([[1,2],[3,4]])
B = tf.constant([[2,3],[4,5]])
C = A * B
print(f"C: {C}")
```

    C: [[ 2  6]
     [12 20]]

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
