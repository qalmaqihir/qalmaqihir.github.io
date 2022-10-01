Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 1 - Introduction to Numpy**

# 04 - Computation on Arrays: Broadcasting

------------------------------------------------------------------------
<center>
<a href=''>![Image](../../../assets/img/logo1.png)</a>
</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
----------------------------------------------------------------------------

- <a href="#computation-on-arrays-broadcasting"
  id="toc-computation-on-arrays-broadcasting">Computation on Arrays:
  Broadcasting</a>
  - <a href="#introducing-broadcasting"
    id="toc-introducing-broadcasting">Introducing Broadcasting</a>
    - <a href="#rules-of-broadcasting" id="toc-rules-of-broadcasting">Rules of
      Broadcasting</a>
  - <a href="#broadcasting-in-practice"
    id="toc-broadcasting-in-practice">Broadcasting in Practice</a>
    - <a href="#centering-an-array" id="toc-centering-an-array">Centering an
      array</a>
    - <a href="#plotting-a-two-dimensional-function"
      id="toc-plotting-a-two-dimensional-function">Plotting a two-dimensional
      function</a>

------------------------------------------------------------------------

# Computation on Arrays: Broadcasting

We saw in the previous section how NumPy’s universal functions can be
used to vec‐ torize operations and thereby remove slow Python loops.
Another means of vectoriz‐ ing operations is to use NumPy’s broadcasting
functionality. Broadcasting is simply a set of rules for applying binary
ufuncs (addition, subtraction, multiplication, etc.) on arrays of
different sizes.

## Introducing Broadcasting

Recall that for arrays of the same size, binary operations are performed
on an element-by-element basis:

``` python
import numpy as np
a=np.array([0,2,3,4,5])
b=np.array([7,8,9,9,6])
a+b
```

    array([ 7, 10, 12, 13, 11])

Broadcasting allows these types of binary operations to be performed on
arrays of dif‐ ferent sizes—for example, we can just as easily add a
scalar (think of it as a zero- dimensional array) to an array

``` python
a+5
```

    array([ 5,  7,  8,  9, 10])

**We can think of this as an operation that stretches or duplicates the
value 5 into the array `[5, 5, 5]`, and adds the results. The advantage
of NumPy’s broadcasting is that this duplication of values does not
actually take place, but it is a useful mental model as we think about
broadcasting.**\_

``` python
b-9
```

    array([-2, -1,  0,  0, -3])

``` python
b*10
```

    array([70, 80, 90, 90, 60])

``` python
a/3
```

    array([0.        , 0.66666667, 1.        , 1.33333333, 1.66666667])

``` python
m=np.ones((3,3))
m
```

    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

``` python
m+a
```

    ValueError: operands could not be broadcast together with shapes (3,3) (5,) 

``` python
m=np.ones((5,5))
m
```

    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])

``` python
m+a
```

    array([[1., 3., 4., 5., 6.],
           [1., 3., 4., 5., 6.],
           [1., 3., 4., 5., 6.],
           [1., 3., 4., 5., 6.],
           [1., 3., 4., 5., 6.]])

Here the one-dimensional array a is stretched, or broadcast, across the
second dimension in order to match the shape of M.

``` python
# Broadcasting of both arrays
a=np.arange(3)
b=np.arange(3)[:,np.newaxis]
```

``` python
a
```

    array([0, 1, 2])

``` python
b
```

    array([[0],
           [1],
           [2]])

``` python
a+b
```

    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

``` python
b+a
```

    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

***Here the one-dimensional array a is stretched, or broadcast, across
the second dimension in order to match the shape of M.***

### Rules of Broadcasting

Broadcasting in NumPy follows a strict set of rules to determine the
interaction  
between the two arrays:  
\* Rule 1: If the two arrays differ in their number of dimensions, the
shape of the one with fewer dimensions is padded with ones on its
leading (left) side.  
\* Rule 2: If the shape of the two arrays does not match in any
dimension, the array with shape equal to 1 in that dimension is
stretched to match the other shape.  
\* Rule 3: If in any dimension the sizes disagree and neither is equal
to 1, an error is raised.

``` python
m=np.ones((2,3))
a=np.arange(3)
m,a
```

    (array([[1., 1., 1.],
            [1., 1., 1.]]),
     array([0, 1, 2]))

``` python
m+a
```

    array([[1., 2., 3.],
           [1., 2., 3.]])

``` python
a=np.arange(3).reshape((3,1))
b=np.arange(3)
```

``` python
a
```

    array([[0],
           [1],
           [2]])

``` python
b
```

    array([0, 1, 2])

``` python
a+b
```

    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

## Broadcasting in Practice

### Centering an array

``` python
x=np.random.random((10,3))
x
```

    array([[0.83823659, 0.53520822, 0.88885158],
           [0.89147897, 0.41948707, 0.42342134],
           [0.96876136, 0.46312611, 0.31711275],
           [0.47652192, 0.25392243, 0.25454173],
           [0.6239708 , 0.97644488, 0.97034953],
           [0.33707785, 0.24628063, 0.25686595],
           [0.12018414, 0.77360894, 0.19437752],
           [0.90201827, 0.18844719, 0.99512986],
           [0.84278929, 0.69803136, 0.53420956],
           [0.13620165, 0.90381885, 0.78541803]])

``` python
x_mean=x.mean(0)
```

``` python
x_mean
```

    array([0.61372408, 0.54583757, 0.56202779])

``` python
# center the x arry by subtracting the mean (a boradcasting operation)
x_centered=x-x_mean
```

``` python
x_centered
```

    array([[ 0.22451251, -0.01062935,  0.3268238 ],
           [ 0.27775488, -0.1263505 , -0.13860644],
           [ 0.35503727, -0.08271145, -0.24491504],
           [-0.13720216, -0.29191514, -0.30748605],
           [ 0.01024672,  0.43060732,  0.40832174],
           [-0.27664623, -0.29955694, -0.30516184],
           [-0.49353995,  0.22777137, -0.36765026],
           [ 0.28829418, -0.35739038,  0.43310208],
           [ 0.22906521,  0.15219379, -0.02781823],
           [-0.47752244,  0.35798128,  0.22339024]])

### Plotting a two-dimensional function

One place that broadcasting is very useful is in displaying images based
on two- dimensional functions. If we want to define a function z = f(x,
y), broadcasting can be used to compute the function across the gri

``` python
x= np.linspace(0,5,50)
y=np.linspace(0,5,50)[:,np.newaxis]
```

``` python
x
```

    array([0.        , 0.10204082, 0.20408163, 0.30612245, 0.40816327,
           0.51020408, 0.6122449 , 0.71428571, 0.81632653, 0.91836735,
           1.02040816, 1.12244898, 1.2244898 , 1.32653061, 1.42857143,
           1.53061224, 1.63265306, 1.73469388, 1.83673469, 1.93877551,
           2.04081633, 2.14285714, 2.24489796, 2.34693878, 2.44897959,
           2.55102041, 2.65306122, 2.75510204, 2.85714286, 2.95918367,
           3.06122449, 3.16326531, 3.26530612, 3.36734694, 3.46938776,
           3.57142857, 3.67346939, 3.7755102 , 3.87755102, 3.97959184,
           4.08163265, 4.18367347, 4.28571429, 4.3877551 , 4.48979592,
           4.59183673, 4.69387755, 4.79591837, 4.89795918, 5.        ])

``` python
y
```

    array([[0.        ],
           [0.10204082],
           [0.20408163],
           [0.30612245],
           [0.40816327],
           [0.51020408],
           [0.6122449 ],
           [0.71428571],
           [0.81632653],
           [0.91836735],
           [1.02040816],
           [1.12244898],
           [1.2244898 ],
           [1.32653061],
           [1.42857143],
           [1.53061224],
           [1.63265306],
           [1.73469388],
           [1.83673469],
           [1.93877551],
           [2.04081633],
           [2.14285714],
           [2.24489796],
           [2.34693878],
           [2.44897959],
           [2.55102041],
           [2.65306122],
           [2.75510204],
           [2.85714286],
           [2.95918367],
           [3.06122449],
           [3.16326531],
           [3.26530612],
           [3.36734694],
           [3.46938776],
           [3.57142857],
           [3.67346939],
           [3.7755102 ],
           [3.87755102],
           [3.97959184],
           [4.08163265],
           [4.18367347],
           [4.28571429],
           [4.3877551 ],
           [4.48979592],
           [4.59183673],
           [4.69387755],
           [4.79591837],
           [4.89795918],
           [5.        ]])

``` python
z=np.sin(x)**10 + np.cos(10+x*y)*np.cos(x)
```

``` python
%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(z,origin='lower',extent=[0,5,0,5],cmap='viridis')
plt.colorbar();
```

![](04_Computation%20on%20Arrays%20Broadcasting_files/figure-gfm/cell-33-output-1.png)
