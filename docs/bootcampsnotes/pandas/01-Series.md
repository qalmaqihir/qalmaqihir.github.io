================
by Jawad Haider
# **01 - Series**
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

- <a href="#series" id="toc-series"><span
  class="toc-section-number">1</span> Series</a>
  - <a href="#creating-a-series" id="toc-creating-a-series"><span
    class="toc-section-number">1.1</span> Creating a Series</a>
    - <a href="#using-lists" id="toc-using-lists"><span
      class="toc-section-number">1.1.1</span> Using Lists</a>
    - <a href="#using-numpy-arrays" id="toc-using-numpy-arrays"><span
      class="toc-section-number">1.1.2</span> Using NumPy Arrays</a>
    - <a href="#using-dictionaries" id="toc-using-dictionaries"><span
      class="toc-section-number">1.1.3</span> Using Dictionaries</a>
    - <a href="#data-in-a-series" id="toc-data-in-a-series"><span
      class="toc-section-number">1.1.4</span> Data in a Series</a>
  - <a href="#using-an-index" id="toc-using-an-index"><span
    class="toc-section-number">1.2</span> Using an Index</a>
- <a href="#great-job-thats-the-end-of-this-part."
  id="toc-great-job-thats-the-end-of-this-part."><span
  class="toc-section-number">2</span> Great Job! Thats the end of this
  part.</a>

------------------------------------------------------------------------

# Series

The first main data type we will learn about for pandas is the Series
data type. Let’s import Pandas and explore the Series object.

A Series is very similar to a NumPy array (in fact it is built on top of
the NumPy array object). What differentiates the NumPy array from a
Series, is that a Series can have axis labels, meaning it can be indexed
by a label, instead of just a number location. It also doesn’t need to
hold numeric data, it can hold any arbitrary Python Object.

Let’s explore this concept through some examples:

``` python
import numpy as np
import pandas as pd
```

## Creating a Series

You can convert a list,numpy array, or dictionary to a Series:

``` python
labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}
```

### Using Lists

``` python
pd.Series(data=my_list)
```

    0    10
    1    20
    2    30
    dtype: int64

``` python
pd.Series(data=my_list,index=labels)
```

    a    10
    b    20
    c    30
    dtype: int64

``` python
pd.Series(my_list,labels)
```

    a    10
    b    20
    c    30
    dtype: int64

### Using NumPy Arrays

``` python
pd.Series(arr)
```

    0    10
    1    20
    2    30
    dtype: int64

``` python
pd.Series(arr,labels)
```

    a    10
    b    20
    c    30
    dtype: int64

### Using Dictionaries

``` python
pd.Series(d)
```

    a    10
    b    20
    c    30
    dtype: int64

### Data in a Series

A pandas Series can hold a variety of object types:

``` python
pd.Series(data=labels)
```

    0    a
    1    b
    2    c
    dtype: object

``` python
# Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])
```

    0      <built-in function sum>
    1    <built-in function print>
    2      <built-in function len>
    dtype: object

## Using an Index

The key to using a Series is understanding its index. Pandas makes use
of these index names or numbers by allowing for fast look ups of
information (works like a hash table or dictionary).

Let’s see some examples of how to grab information from a Series. Let us
create two sereis, ser1 and ser2:

``` python
ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan'])                                   
```

``` python
ser1
```

    USA        1
    Germany    2
    USSR       3
    Japan      4
    dtype: int64

``` python
ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])                                   
```

``` python
ser2
```

    USA        1
    Germany    2
    Italy      5
    Japan      4
    dtype: int64

``` python
ser1['USA']
```

    1

Operations are then also done based off of index:

``` python
ser1 + ser2
```

    Germany    4.0
    Italy      NaN
    Japan      8.0
    USA        2.0
    USSR       NaN
    dtype: float64

# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`
