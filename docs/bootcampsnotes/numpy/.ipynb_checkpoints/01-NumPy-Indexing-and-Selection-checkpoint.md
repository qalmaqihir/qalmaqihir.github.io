Crash Course: Numpy 
================
by Jawad Haider
# **01 - Numpy Indexing and Selection**
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

 - <a href="#numpy-indexing-and-selection"
  id="toc-numpy-indexing-and-selection"><span
  class="toc-section-number">1</span> NumPy Indexing and Selection</a>
  - <a href="#bracket-indexing-and-selection"
    id="toc-bracket-indexing-and-selection"><span
    class="toc-section-number">1.1</span> Bracket Indexing and Selection</a>
  - <a href="#broadcasting" id="toc-broadcasting"><span
    class="toc-section-number">1.2</span> Broadcasting</a>
  - <a href="#indexing-a-2d-array-matrices"
    id="toc-indexing-a-2d-array-matrices"><span
    class="toc-section-number">1.3</span> Indexing a 2D array (matrices)</a>
  - <a href="#more-indexing-help" id="toc-more-indexing-help"><span
    class="toc-section-number">1.4</span> More Indexing Help</a>
  - <a href="#conditional-selection" id="toc-conditional-selection"><span
    class="toc-section-number">1.5</span> Conditional Selection</a>
- <a href="#great-job-thats-the-end-of-this-part."
  id="toc-great-job-thats-the-end-of-this-part."><span
  class="toc-section-number">2</span> Great Job! Thats the end of this
  part.</a>

------------------------------------------------------------------------

# NumPy Indexing and Selection

In this lecture we will discuss how to select elements or groups of
elements from an array.

``` python
import numpy as np
```

``` python
#Creating sample array
arr = np.arange(0,11)
```

``` python
#Show
arr
```

    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

## Bracket Indexing and Selection

The simplest way to pick one or some elements of an array looks very
similar to python lists:

``` python
#Get a value at an index
arr[8]
```

    8

``` python
#Get values in a range
arr[1:5]
```

    array([1, 2, 3, 4])

``` python
#Get values in a range
arr[0:5]
```

    array([0, 1, 2, 3, 4])

## Broadcasting

NumPy arrays differ from normal Python lists because of their ability to
broadcast. With lists, you can only reassign parts of a list with new
parts of the same size and shape. That is, if you wanted to replace the
first 5 elements in a list with a new value, you would have to pass in a
new 5 element list. With NumPy arrays, you can broadcast a single value
across a larger set of values:

``` python
#Setting a value with index range (Broadcasting)
arr[0:5]=100

#Show
arr
```

    array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])

``` python
# Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)

#Show
arr
```

    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

``` python
#Important notes on Slices
slice_of_arr = arr[0:6]

#Show slice
slice_of_arr
```

    array([0, 1, 2, 3, 4, 5])

``` python
#Change Slice
slice_of_arr[:]=99

#Show Slice again
slice_of_arr
```

    array([99, 99, 99, 99, 99, 99])

Now note the changes also occur in our original array!

``` python
arr
```

    array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

Data is not copied, it’s a view of the original array! This avoids
memory problems!

``` python
#To get a copy, need to be explicit
arr_copy = arr.copy()

arr_copy
```

    array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

## Indexing a 2D array (matrices)

The general format is `arr_2d[row][col] or arr_2d[row,col]`. I recommend
using the comma notation for clarity.

``` python
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

#Show
arr_2d
```

    array([[ 5, 10, 15],
           [20, 25, 30],
           [35, 40, 45]])

``` python
#Indexing row
arr_2d[1]
```

    array([20, 25, 30])

``` python
# Format is arr_2d[row][col] or arr_2d[row,col]

# Getting individual element value
arr_2d[1][0]
```

    20

``` python
# Getting individual element value
arr_2d[1,0]
```

    20

``` python
# 2D array slicing

#Shape (2,2) from top right corner
arr_2d[:2,1:]
```

    array([[10, 15],
           [25, 30]])

``` python
#Shape bottom row
arr_2d[2]
```

    array([35, 40, 45])

``` python
#Shape bottom row
arr_2d[2,:]
```

    array([35, 40, 45])

## More Indexing Help

Indexing a 2D matrix can be a bit confusing at first, especially when
you start to add in step size. Try google image searching *NumPy
indexing* to find useful images, like this one:  
------------
![Image](../../assets/img/numpy_indexing.png)


*[Image source](http://www.scipy-lectures.org/intro/numpy/numpy.html)*

## Conditional Selection

This is a very fundamental concept that will directly translate to
pandas later on, make sure you understand this part!

Let’s briefly go over how to use brackets for selection based off of
comparison operators.

``` python
arr = np.arange(1,11)
arr
```

    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

``` python
arr > 4
```

    array([False, False, False, False,  True,  True,  True,  True,  True,
            True])

``` python
bool_arr = arr>4
```

``` python
bool_arr
```

    array([False, False, False, False,  True,  True,  True,  True,  True,
            True])

``` python
arr[bool_arr]
```

    array([ 5,  6,  7,  8,  9, 10])

``` python
arr[arr>2]
```

    array([ 3,  4,  5,  6,  7,  8,  9, 10])

``` python
x = 2
arr[arr>x]
```

    array([ 3,  4,  5,  6,  7,  8,  9, 10])

# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`
