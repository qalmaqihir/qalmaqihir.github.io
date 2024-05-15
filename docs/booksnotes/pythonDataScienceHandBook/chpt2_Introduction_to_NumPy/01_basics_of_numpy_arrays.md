================
by Jawad Haider
# **Chpt 1 - Introduction to Numpy**

# 01 - The Basics of NumPy Arrays

------------------------------------------------------------------------

- <a href="#the-basics-of-numpy-arrays"
  id="toc-the-basics-of-numpy-arrays">The Basics of NumPy Arrays</a>
  - <a href="#array-attributes" id="toc-array-attributes">Array
    Attributes</a>
  - <a href="#array-indexing-accessing-single-elements"
    id="toc-array-indexing-accessing-single-elements">Array Indexing:
    Accessing Single Elements</a>
  - <a href="#array-slicing-accessing-subarrays"
    id="toc-array-slicing-accessing-subarrays">Array Slicing: Accessing
    Subarrays</a>
    - <a href="#one-dimensional-subarrays"
      id="toc-one-dimensional-subarrays">One-dimensional subarrays</a>
    - <a href="#multidimensional-subarrays"
      id="toc-multidimensional-subarrays">Multidimensional subarrays</a>
    - <a href="#accessing-array-rows-and-columns."
      id="toc-accessing-array-rows-and-columns.">Accessing array rows and
      columns.</a>
    - <a href="#subarrays-as-no-copy-views"
      id="toc-subarrays-as-no-copy-views">Subarrays as no-copy views</a>
    - <a href="#creating-copies-of-arrays"
      id="toc-creating-copies-of-arrays">Creating copies of arrays</a>
  - <a href="#reshaping-of-arrays" id="toc-reshaping-of-arrays">Reshaping of
    Arrays</a>
  - <a href="#array-concatenation-and-splitting"
    id="toc-array-concatenation-and-splitting">Array Concatenation and
    Splitting</a>
    - <a href="#concatenation-of-arrays"
      id="toc-concatenation-of-arrays">Concatenation of arrays</a>
    - <a href="#splitting-of-arrays" id="toc-splitting-of-arrays">Splitting of
      arrays</a>

------------------------------------------------------------------------

# The Basics of NumPy Arrays

This section will present several examples using NumPy array
manipulation to access data and subarrays, and to split, reshape, and
join the arrays.  
<hr>

We’ll cover a few categories of basic array manipulations here:

*Attributes of arrays*

    Determining the size, shape, memory consumption, and data types of arrays

*Indexing of arrays*

    Getting and setting the value of individual array elements

*Slicing of arrays*

    Getting and setting smaller subarrays within a larger array  

*Reshaping of arrays*

    Changing the shape of a given array  

*Joining and splitting of arrays*

    Combining multiple arrays into one, and splitting one array into many

## Array Attributes

``` python
import numpy as np
np.random.seed(0)
x1 = np.random.randint(10,size=6) # 1-d array
x2 = np.random.randint(10,size=(3,4)) # 2-d array (matrix)
x3 = np.random.randint(10,size=(3,4,5)) # 3-d array
```

``` python
print(x3)
print("x3 ndim: ",x3.ndim)
print("x3.shape: ", x3.shape)
print("x3.size: ", x3.size)
```

    [[[8 1 5 9 8]
      [9 4 3 0 3]
      [5 0 2 3 8]
      [1 3 3 3 7]]

     [[0 1 9 9 0]
      [4 7 3 2 7]
      [2 0 0 4 5]
      [5 6 8 4 1]]

     [[4 9 8 1 1]
      [7 9 9 3 6]
      [7 2 0 3 5]
      [9 4 4 6 4]]]
    x3 ndim:  3
    x3.shape:  (3, 4, 5)
    x3.size:  60

``` python
print(x2)
print("x2 ndim: ",x2.ndim)
print("x2.shape: ", x2.shape)
print("x2.size: ", x2.size)
```

    [[3 5 2 4]
     [7 6 8 8]
     [1 6 7 7]]
    x2 ndim:  2
    x2.shape:  (3, 4)
    x2.size:  12

``` python
print(x1)
print("x1 ndim: ",x1.ndim)
print("x1.shape: ", x1.shape)
print("x1.size: ", x1.size)
```

    [5 0 3 3 7 9]
    x1 ndim:  1
    x1.shape:  (6,)
    x1.size:  6

``` python
print("dtype: ", x3.dtype)
```

    dtype:  int64

``` python
print("itemsize: ", x3.itemsize, "bytes") # size of each array element
```

    itemsize:  8 bytes

``` python

print("nbytes: ",x3.nbytes," bytes") # total size of the array -> sum of bytes of each array element
```

    nbytes:  480  bytes

## Array Indexing: Accessing Single Elements

If you are familiar with Python’s standard list indexing, indexing in
NumPy will feel quite familiar. In a one-dimensional array, you can
access the ith value (counting from zero) by specifying the desired
index in square brackets, just as with Python lists:

``` python
x1
```

    array([5, 0, 3, 3, 7, 9])

``` python
x1[-1]
```

    9

``` python
x1[0]
```

    5

``` python
#In a multidimensional array, you access items using a comma-separated tuple of indices:
x2
```

    array([[3, 5, 2, 4],
           [7, 6, 8, 8],
           [1, 6, 7, 7]])

``` python
x2[0,0]
```

    3

``` python
# same for modifying any element of the array
x2[0,3]
```

    4

``` python
x2[0,3]=-4
```

``` python
x2[0,3]
```

    -4

``` python
x2
```

    array([[ 3,  5,  2, -4],
           [ 7,  6,  8,  8],
           [ 1,  6,  7,  7]])

Keep in mind that, unlike Python lists, NumPy arrays have a fixed type.
This means, for example, that if you attempt to insert a floating-point
value to an integer array, the value will be silently truncated. Don’t
be caught unaware by this behavior!

``` python
x1[0]
```

    5

``` python
x1[0]=-5.78 # this will the decimal part
```

``` python
x1[0] 
```

    -5

## Array Slicing: Accessing Subarrays

Just as we can use square brackets to access individual array elements,
we can also use them to access subarrays with the slice notation, marked
by the colon (:) character. The NumPy slicing syntax follows that of the
standard Python list; to access a slice of an array x, use this:  
`x[start:stop:step]`

### One-dimensional subarrays

``` python

x= np.arange(10)
x
```

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

``` python
x[:5]
```

    array([0, 1, 2, 3, 4])

``` python
x[1:len(x)-3]
```

    array([1, 2, 3, 4, 5, 6])

``` python
x[len(x)//2:] # after mid
```

    array([5, 6, 7, 8, 9])

``` python
x[::2] # Every other element
```

    array([0, 2, 4, 6, 8])

``` python
x[1::2]
```

    array([1, 3, 5, 7, 9])

A potentially confusing case is when the step value is negative. In this
case, the defaults for start and stop are swapped. This becomes a
convenient way to reverse an array:

``` python
x[::-1]  # All elements reverse
```

    array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

``` python
x[5::-2]
```

    array([5, 3, 1])

### Multidimensional subarrays

Multidimensional slices work in the same way, with multiple slices
separated by com‐ mas.

``` python
x2
```

    array([[ 3,  5,  2, -4],
           [ 7,  6,  8,  8],
           [ 1,  6,  7,  7]])

``` python
x2[:2,:3] #two rows and three columns
```

    array([[3, 5, 2],
           [7, 6, 8]])

``` python
x2[:3,::2] # three rows and every other column
```

    array([[3, 2],
           [7, 8],
           [1, 7]])

### Accessing array rows and columns.

One commonly needed routine is accessing single rows or columns of an
array. You can do this by combining indexing and slicing, using an empty
slice marked by a single colon (:):

``` python
x2[:,0] # first column
```

    array([3, 7, 1])

``` python
x2[0,:] # First row 
```

    array([ 3,  5,  2, -4])

``` python
#In the case of row access, the empty slice can be omitted for a more compact syntax:s
x2[0]
```

    array([ 3,  5,  2, -4])

### Subarrays as no-copy views

One important—and extremely useful—thing to know about array slices is
that they return views rather than copies of the array data. This is one
area in which NumPy array slicing differs from Python list slicing: in
lists, slices will be copies

``` python
x2
```

    array([[ 3,  5,  2, -4],
           [ 7,  6,  8,  8],
           [ 1,  6,  7,  7]])

``` python
# Extracting a 2x2 subarray from this
x2_sub=x2[:2,:2]
x2_sub
```

    array([[3, 5],
           [7, 6]])

``` python
# Modifying this subarray will also modify the original array
x2_sub[0,0]=-9
```

``` python
x2
```

    array([[-9,  5,  2, -4],
           [ 7,  6,  8,  8],
           [ 1,  6,  7,  7]])

``` python
x2_sub
```

    array([[-9,  5],
           [ 7,  6]])

This default behavior is actually quite useful: it means that when we
work with large datasets, we can access and process pieces of these
datasets without the need to copy the underlying data buffer.

### Creating copies of arrays

Despite the nice features of array views, it is sometimes useful to
instead explicitly copy the data within an array or a subarray. This can
be most easily done with the copy() method:

``` python
x2_sub_copy=x2[:2,:2].copy()
x2_sub_copy
```

    array([[-9,  5],
           [ 7,  6]])

#### If we now modify this subarray, the original array is not touched:

``` python
x2_sub_copy[0,0]=-999
```

``` python
x2
```

    array([[-9,  5,  2, -4],
           [ 7,  6,  8,  8],
           [ 1,  6,  7,  7]])

``` python
x2_sub_copy
```

    array([[-999,    5],
           [   7,    6]])

## Reshaping of Arrays

Another useful type of operation is reshaping of arrays. The most
flexible way of doing this is with the reshape() method.

``` python
grid = np.arange(1,10).reshape((3,3))
```

``` python
grid
```

    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

Note that for this to work, the size of the initial array must match the
size of the reshaped array. Where possible, the reshape method will use
a no-copy view of the initial array, but with noncontiguous memory
buffers this is not always the case.  
Another common reshaping pattern is the conversion of a one-dimensional
array into a two-dimensional row or column matrix. You can do this with
the reshape method, or more easily by making use of the newaxis keyword
within a slice opera‐ tion:

``` python
x= np.array([1,2,3,4])
x
```

    array([1, 2, 3, 4])

``` python
x.reshape((1,4)) # row vector reshape via reshape
```

    array([[1, 2, 3, 4]])

``` python
x.reshape((len(x),1)) #Column vector reshape via 
```

    array([[1],
           [2],
           [3],
           [4]])

``` python
x[np.newaxis,:]
```

    array([[1, 2, 3, 4]])

``` python
x[:,np.newaxis]
```

    array([[1],
           [2],
           [3],
           [4]])

## Array Concatenation and Splitting

### Concatenation of arrays

Concatenation, or joining of two arrays in NumPy, is primarily
accomplished through the routines np.concatenate, np.vstack, and
`np.hstack`. `np.concatenate` takes a tuple or list of arrays as its
first argument.  
\* You can also concatenate more than two arrays at once \*
np.concatenate can also be used for two-dimensional arrays

``` python
x = np.array([2,4,6,8])
y=np.array([1,3,5,7])
np.concatenate([x,y])
```

    array([2, 4, 6, 8, 1, 3, 5, 7])

``` python
z=np.zeros(4)
z
```

    array([0., 0., 0., 0.])

``` python
np.concatenate([x,z,y]) # Three arrays concatenate in given order
```

    array([2., 4., 6., 8., 0., 0., 0., 0., 1., 3., 5., 7.])

``` python
grid1=np.random.randint(10,50,size=(2,4))
grid2=np.random.randint(0,1,size=(2,3))
grid1
```

    array([[39, 13, 44, 23],
           [49, 31, 19, 10]])

``` python
grid2
```

    array([[0, 0, 0],
           [0, 0, 0]])

``` python
np.concatenate([grid1,grid2])
```

    ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 4 and the array at index 1 has size 3

``` python
np.concatenate([grid1,grid2],axis=1)
```

    array([[39, 13, 44, 23,  0,  0,  0],
           [49, 31, 19, 10,  0,  0,  0]])

``` python
grid2=np.random.randint(0,10,size=(2,4))
grid2
```

    array([[3, 0, 5, 0],
           [1, 2, 4, 2]])

``` python
np.concatenate([grid1,grid2],axis=0)
```

    array([[13, 22, 46, 24],
           [25, 30, 45, 33],
           [ 3,  0,  5,  0],
           [ 1,  2,  4,  2]])

``` python
np.concatenate([grid1,grid2],axis=1)
```

    array([[13, 22, 46, 24,  3,  0,  5,  0],
           [25, 30, 45, 33,  1,  2,  4,  2]])

***For working with arrays of mixed dimensions, it can be clearer to use
the np.vstack (vertical stack) and np.hstack (horizontal stack)
functions***

``` python
x=np.array([1,2,3])
grid=np.random.randint(1,10,size=(2,3))
```

``` python
x
```

    array([1, 2, 3])

``` python
grid
```

    array([[5, 7, 9],
           [3, 4, 1]])

``` python
np.vstack([x,grid])
```

    array([[1, 2, 3],
           [5, 7, 9],
           [3, 4, 1]])

### Splitting of arrays

The opposite of concatenation is splitting, which is implemented by the
functions np.split, np.hsplit, and np.vsplit. For each of these, we can
pass a list of indices giving the split points:

``` python
x=[1,2,3,99,100,101,3,2,1]
x1,x2,x3=np.split(x,[3,5]) # split will be at index 3 & 5
```

``` python
x1
```

    array([1, 2, 3])

``` python
x2
```

    array([ 99, 100])

``` python
x3
#Notice that N split points lead to N + 1 subarrays. 
#The related functions np.hsplit and np.vsplit are similar:
```

    array([101,   3,   2,   1])

``` python
grid=np.arange(16).reshape((4,4))
upper,lower = np.vsplit(grid,[2])
```

``` python
upper
```

    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])

``` python
lower
```

    array([[ 8,  9, 10, 11],
           [12, 13, 14, 15]])

``` python
grid
```

    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

``` python
left,right=np.hsplit(grid,[2])
```

``` python
left
```

    array([[ 0,  1],
           [ 4,  5],
           [ 8,  9],
           [12, 13]])

``` python
right
```

    array([[ 2,  3],
           [ 6,  7],
           [10, 11],
           [14, 15]])



# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`
