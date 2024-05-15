PyTorch Basics
================

by Jawad Haider
# **02 - PyTorch Basics Exercises**
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

- <a href="#pytorch-basics-exercises"
  id="toc-pytorch-basics-exercises"><span
  class="toc-section-number">1</span> PyTorch Basics Exercises</a>
  - <a href="#perform-standard-imports"
    id="toc-perform-standard-imports"><span
    class="toc-section-number">1.0.1</span> 1. Perform standard imports</a>
  - <a href="#set-the-random-seed-for-numpy-and-pytorch-both-to-42"
    id="toc-set-the-random-seed-for-numpy-and-pytorch-both-to-42"><span
    class="toc-section-number">1.0.2</span> 2. Set the random seed for NumPy
    and PyTorch both to “42”</a>
  - <a
    href="#create-a-numpy-array-called-arr-that-contains-6-random-integers-between-0-inclusive-and-5-exclusive"
    id="toc-create-a-numpy-array-called-arr-that-contains-6-random-integers-between-0-inclusive-and-5-exclusive"><span
    class="toc-section-number">1.0.3</span> 3. Create a NumPy array called
    “arr” that contains 6 random integers between 0 (inclusive) and 5
    (exclusive)</a>
  - <a href="#create-a-tensor-x-from-the-array-above"
    id="toc-create-a-tensor-x-from-the-array-above"><span
    class="toc-section-number">1.0.4</span> 4. Create a tensor “x” from the
    array above</a>
  - <a href="#change-the-dtype-of-x-from-int32-to-int64"
    id="toc-change-the-dtype-of-x-from-int32-to-int64"><span
    class="toc-section-number">1.0.5</span> 5. Change the dtype of x from
    ‘int32’ to ‘int64’</a>
  - <a href="#reshape-x-into-a-3x2-tensor"
    id="toc-reshape-x-into-a-3x2-tensor"><span
    class="toc-section-number">1.0.6</span> 6. Reshape x into a 3x2
    tensor</a>
  - <a href="#return-the-right-hand-column-of-tensor-x"
    id="toc-return-the-right-hand-column-of-tensor-x"><span
    class="toc-section-number">1.0.7</span> 7. Return the right-hand column
    of tensor x</a>
  - <a href="#without-changing-x-return-a-tensor-of-square-values-of-x"
    id="toc-without-changing-x-return-a-tensor-of-square-values-of-x"><span
    class="toc-section-number">1.0.8</span> 8. Without changing x, return a
    tensor of square values of x</a>
  - <a
    href="#create-a-tensor-y-with-the-same-number-of-elements-as-x-that-can-be-matrix-multiplied-with-x"
    id="toc-create-a-tensor-y-with-the-same-number-of-elements-as-x-that-can-be-matrix-multiplied-with-x"><span
    class="toc-section-number">1.0.9</span> 9. Create a tensor “y” with the
    same number of elements as x, that can be matrix-multiplied with x</a>
  - <a href="#find-the-matrix-product-of-x-and-y"
    id="toc-find-the-matrix-product-of-x-and-y"><span
    class="toc-section-number">1.0.10</span> 10. Find the matrix product of
    x and y</a>
  - <a href="#great-job" id="toc-great-job"><span
    class="toc-section-number">1.1</span> Great job!</a>

------------------------------------------------------------------------

# PyTorch Basics Exercises

For these exercises we’ll create a tensor and perform several operations
on it.

<div class="alert alert-danger" style="margin: 10px">

<strong>IMPORTANT NOTE!</strong> Make sure you don’t run the cells
directly above the example output shown, <br>otherwise you will end up
writing over the example output!

</div>

### 1. Perform standard imports

Import torch and NumPy

``` python
# CODE HERE

```

### 2. Set the random seed for NumPy and PyTorch both to “42”

This allows us to share the same “random” results.

``` python
# CODE HERE

```

### 3. Create a NumPy array called “arr” that contains 6 random integers between 0 (inclusive) and 5 (exclusive)

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    [3 4 2 4 4 1]

### 4. Create a tensor “x” from the array above

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    tensor([3, 4, 2, 4, 4, 1], dtype=torch.int32)

### 5. Change the dtype of x from ‘int32’ to ‘int64’

Note: ‘int64’ is also called ‘LongTensor’

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    torch.LongTensor

### 6. Reshape x into a 3x2 tensor

There are several ways to do this.

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    tensor([[3, 4],
            [2, 4],
            [4, 1]])

### 7. Return the right-hand column of tensor x

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    tensor([[4],
            [4],
            [1]])

### 8. Without changing x, return a tensor of square values of x

There are several ways to do this.

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    tensor([[ 9, 16],
            [ 4, 16],
            [16,  1]])

### 9. Create a tensor “y” with the same number of elements as x, that can be matrix-multiplied with x

Use PyTorch directly (not NumPy) to create a tensor of random integers
between 0 (inclusive) and 5 (exclusive).<br> Think about what shape it
should have to permit matrix multiplication.

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    tensor([[2, 2, 1],
            [4, 1, 0]])

### 10. Find the matrix product of x and y

``` python
# CODE HERE

```

``` python
# DON'T WRITE HERE
```

    tensor([[22, 10,  3],
            [20,  8,  2],
            [12,  9,  4]])

## Great job!
