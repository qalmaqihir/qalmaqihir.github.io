Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 1 - Introduction to Numpy**

# 02 - Computation on NumPy Arrays: Universal Functions

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
--------------------------------------------------------------------------


- <a href="#computation-on-numpy-arrays-universal-functions"
  id="toc-computation-on-numpy-arrays-universal-functions">Computation on
  NumPy Arrays: Universal Functions</a>
  - <a href="#the-slowness-of-loops" id="toc-the-slowness-of-loops">The
    Slowness of Loops</a>
  - <a href="#introducing-ufuncs" id="toc-introducing-ufuncs">Introducing
    UFuncs</a>
  - <a href="#exploring-numpys-ufuncs"
    id="toc-exploring-numpys-ufuncs">Exploring NumPy’s UFuncs</a>
    - <a href="#array-arithmetic" id="toc-array-arithmetic">Array
      arithmetic</a>
  - <a href="#specialized-ufuncs" id="toc-specialized-ufuncs">Specialized
    ufuncs</a>
  - <a href="#advanced-ufunc-features"
    id="toc-advanced-ufunc-features">Advanced Ufunc Features</a>
    - <a href="#specifying-output" id="toc-specifying-output">Specifying
      output</a>
    - <a href="#aggregates" id="toc-aggregates">Aggregates</a>
    - <a href="#outer-products" id="toc-outer-products">Outer products</a>

------------------------------------------------------------------------


# Computation on NumPy Arrays: Universal Functions

Computation on NumPy arrays can be very fast, or it can be very slow.
The key to making it fast is to use vectorized operations, generally
implemented through Num‐ Py’s universal functions (ufuncs).

## The Slowness of Loops

Python’s default implementation (known as CPython) does some operations
very slowly. This is in part due to the dynamic, interpreted nature of
the language: the fact that types are flexible, so that sequences of
operations cannot be compiled down to efficient machine code as in
languages like C and Fortran.

``` python
# Example of reciprocal of each item in the list
import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output=np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output

values = np.random.randint(1,10,size=5)
compute_reciprocals(values)
```

    array([0.16666667, 1.        , 0.25      , 0.25      , 0.125     ])

``` python
big_array=np.random.randint(1,100,size=100000)
%timeit compute_reciprocals(big_array)
```

    167 ms ± 5.95 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

## Introducing UFuncs

For many types of operations, NumPy provides a convenient interface into
just this kind of statically typed, compiled routine. This is known as a
vectorized operation. You can accomplish this by simply performing an
operation on the array, which will then be applied to each element. This
vectorized approach is designed to push the loop into the compiled layer
that underlies NumPy, leading to much faster execution.

``` python
print(compute_reciprocals(values))
print(1.0/values)
```

    [0.16666667 1.         0.25       0.25       0.125     ]
    [0.16666667 1.         0.25       0.25       0.125     ]

``` python
%timeit (1/big_array)
```

    129 µs ± 12 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

***Vectorized operations in NumPy are implemented via ufuncs, whose main
purpose is to quickly execute repeated operations on values in NumPy
arrays. Ufuncs are extremely flexible—before we saw an operation between
a scalar and an array, but we can also operate between two arrays:***

``` python
np.arange(5)/np.arange(1,6)
```

    array([0.        , 0.5       , 0.66666667, 0.75      , 0.8       ])

``` python
# Works for multi-d arraays
x=np.arange(9).reshape((3,3))
2*x
```

    array([[ 0,  2,  4],
           [ 6,  8, 10],
           [12, 14, 16]])

## Exploring NumPy’s UFuncs

### Array arithmetic

``` python
print('x = ', x)
```

    x =  [[0 1 2]
     [3 4 5]
     [6 7 8]]

``` python
print('x-3 \n' ,np.subtract(x,3))
print('x+13 \n' ,np.add(x,13))
print('x*9 \n' ,np.multiply(x,9))
print('x/6 \n' ,np.subtract(x,6))
print('x//6 \n' ,np.subtract(x,6))
print('x**4 \n' ,np.power(x,4))
print('x%4 \n' ,np.mod(x,4))
```

    x-3 
     [[-3 -2 -1]
     [ 0  1  2]
     [ 3  4  5]]
    x+13 
     [[13 14 15]
     [16 17 18]
     [19 20 21]]
    x*9 
     [[ 0  9 18]
     [27 36 45]
     [54 63 72]]
    x/6 
     [[-6 -5 -4]
     [-3 -2 -1]
     [ 0  1  2]]
    x//6 
     [[-6 -5 -4]
     [-3 -2 -1]
     [ 0  1  2]]
    x**4 
     [[   0    1   16]
     [  81  256  625]
     [1296 2401 4096]]
    x%4 
     [[0 1 2]
     [3 0 1]
     [2 3 0]]

``` python
# Absolute value
x=np.array([-2,3,-4,-9,0])
abs(x)
```

    array([2, 3, 4, 9, 0])

``` python
np.absolute(x)
```

    array([2, 3, 4, 9, 0])

``` python
# Trig
thet=np.linspace(0,np.pi,3)
print("Theta       =",thet)
print("Sin(theta)  =",np.sin(thet))
print("Cos(theta)  =",np.cos(thet))

print("\nInverse tri\n")
print("Theta       =",thet)
print("arcSin(theta)  =",np.arcsin(thet))
print("arcCos(theta)  =",np.arccos(thet))
```

    Theta       = [0.         1.57079633 3.14159265]
    Sin(theta)  = [0.0000000e+00 1.0000000e+00 1.2246468e-16]
    Cos(theta)  = [ 1.000000e+00  6.123234e-17 -1.000000e+00]
    Inverse tri

    Theta       = [0.         1.57079633 3.14159265]
    arcSin(theta)  = [ 0. nan nan]
    arcCos(theta)  = [1.57079633        nan        nan]

    /tmp/ipykernel_22625/2951825817.py:9: RuntimeWarning: invalid value encountered in arcsin
      print("arcSin(theta)  =",np.arcsin(thet))
    /tmp/ipykernel_22625/2951825817.py:10: RuntimeWarning: invalid value encountered in arccos
      print("arcCos(theta)  =",np.arccos(thet))

``` python
# Logarthm and Exponents
x = [1, 2, 3]
print("x=", x)
print("e^x=", np.exp(x))
print("2^x=", np.exp2(x))
print("3^x=", np.power(3, x))
```

    x= [1, 2, 3]
    e^x= [ 2.71828183  7.3890561  20.08553692]
    2^x= [2. 4. 8.]
    3^x= [ 3  9 27]

## Specialized ufuncs

NumPy has many more ufuncs available, including hyperbolic trig
functions, bitwise arithmetic, comparison operators, conversions from
radians to degrees, rounding and remainders, and much more. A look
through the NumPy documentation reveals a lot of interesting
functionality.

Another excellent source for more specialized and obscure ufuncs is the
submodule scipy.special. If you want to compute some obscure
mathematical function on your data, chances are it is implemented in
scipy.special. There are far too many functions to list them all, but
the following snippet shows a couple that might come up in a statistics
context

``` python
from scipy import special
#Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x)=", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)=", special.beta(x, 2))
```

    gamma(x)= [1.0000e+00 2.4000e+01 3.6288e+05]
    ln|gamma(x)| = [ 0.          3.17805383 12.80182748]
    beta(x, 2)= [0.5        0.03333333 0.00909091]

``` python
# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))
```

    erf(x) = [0.         0.32862676 0.67780119 0.84270079]
    erfc(x) = [1.         0.67137324 0.32219881 0.15729921]
    erfinv(x) = [0.         0.27246271 0.73286908        inf]

## Advanced Ufunc Features

### Specifying output

For large calculations, it is sometimes useful to be able to specify the
array where the result of the calculation will be stored. Rather than
creating a temporary array, you can use this to write computation
results directly to the memory location where you’d like them to be.

``` python
# THis can be done using the out argument of the function
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)
```

    [ 0. 10. 20. 30. 40.]

``` python
# This can be done with array views
y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)
```

    [ 1.  0.  2.  0.  4.  0.  8.  0. 16.  0.]

``` python
np.power(2,x,out=2**x)
```

    array([ 1,  2,  4,  8, 16])

### Aggregates

For binary ufuncs, there are some interesting aggregates that can be
computed directly from the object. For example, if we’d like to reduce
an array with a particular operation, we can use the reduce method of
any ufunc. A reduce repeatedly applies a given operation to the elements
of an array until only a single result remains.

``` python
# Calling reduce on the add ufunc
x = np.arange(1, 6)
np.add.reduce(x)
```

    15

``` python
x=np.arange(1,6)
np.sum(x)
```

    15

``` python
np.multiply(x)
```

    TypeError: multiply() takes from 2 to 3 positional arguments but 1 were given

``` python
np.add(x)
```

    TypeError: add() takes from 2 to 3 positional arguments but 1 were given

``` python
np.multiply.reduce(x)
```

    120

``` python
# To store all the imtermmediate results of the computation we can use accumulate
np.add.accumulate(x)
```

    array([ 1,  3,  6, 10, 15])

``` python
x
```

    array([1, 2, 3, 4, 5])

***Note that for these particular cases, there are dedicated NumPy
functions to compute the results (np.sum, np.prod, np.cumsum,
np.cumprod), The ufunc.at and ufunc.reduceat methods***

### Outer products

Finally, any ufunc can compute the output of all pairs of two different
inputs using the outer method.

``` python
np.multiply.outer(x,x)
```

    array([[ 1,  2,  3,  4,  5],
           [ 2,  4,  6,  8, 10],
           [ 3,  6,  9, 12, 15],
           [ 4,  8, 12, 16, 20],
           [ 5, 10, 15, 20, 25]])

***Another extremely useful feature of ufuncs is the ability to operate
between arrays of different sizes and shapes, a set of operations known
as broadcasting.***


# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`

