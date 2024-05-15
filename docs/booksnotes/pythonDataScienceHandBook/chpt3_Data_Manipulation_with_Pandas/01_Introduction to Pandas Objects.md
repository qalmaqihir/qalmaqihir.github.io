================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 01 - Introducing Pandas Objects
------------------------------------------------------------------------

- <a href="#introducing-pandas-objects"
  id="toc-introducing-pandas-objects">Introducing Pandas Objects</a>
  - <a href="#the-pandas-series-object"
    id="toc-the-pandas-series-object">The Pandas Series Object</a>
    - <a href="#series-as-generalized-numpy-array"
      id="toc-series-as-generalized-numpy-array">Series as generalized NumPy
      array</a>
    - <a href="#series-as-specialized-dictionary"
      id="toc-series-as-specialized-dictionary">Series as specialized
      dictionary</a>
    - <a href="#constructing-series-objects"
      id="toc-constructing-series-objects">Constructing Series objects</a>
  - <a href="#the-pandas-dataframe-object"
    id="toc-the-pandas-dataframe-object">The Pandas DataFrame Object</a>
    - <a href="#dataframe-as-a-generalized-numpy-array"
      id="toc-dataframe-as-a-generalized-numpy-array">DataFrame as a
      generalized NumPy array</a>
    - <a href="#dataframe-as-specialized-dictionary"
      id="toc-dataframe-as-specialized-dictionary">DataFrame as specialized
      dictionary</a>
  - <a href="#the-pandas-index-object" id="toc-the-pandas-index-object">The
    Pandas Index Object</a>
    - <a href="#index-as-immutable-array"
      id="toc-index-as-immutable-array">Index as immutable array</a>
    - <a href="#index-as-ordered-set" id="toc-index-as-ordered-set">Index as
      ordered set</a>

------------------------------------------------------------------------


# Introducing Pandas Objects

At the very basic level, Pandas objects can be thought of as enhanced
versions of NumPy structured arrays in which the rows and columns are
identified with labels rather than simple integer indices.  
**let’s introduce the three fundamental Pandas data structures: the
Series, DataFrame, and Index.**

``` python
import numpy as np
import pandas as pd
```

## The Pandas Series Object

A Pandas Series is a one-dimensional array of indexed data. It can be
created from a list or array

``` python
data= pd.Series([0.25,0.5,3.1415,2.729,1.0,])
data
```

    0    0.2500
    1    0.5000
    2    3.1415
    3    2.7290
    4    1.0000
    dtype: float64

``` python
data.values
```

    array([0.25  , 0.5   , 3.1415, 2.729 , 1.    ])

``` python
data.index
```

    RangeIndex(start=0, stop=5, step=1)

``` python
# Like Numpy array, data can be accessed by the associated index vi the [] notation
data[1]
```

    0.5

``` python
data[1:3]
```

    1    0.5000
    2    3.1415
    dtype: float64

### Series as generalized NumPy array

From what we’ve seen so far, it may look like the Series object is
basically interchangeable with a one-dimensional NumPy array. The
essential difference is the presence of the index: while the NumPy array
has an implicitly defined integer index usedvo access the values, the
Pandas Series has an explicitly defined index associated with the
values.

``` python
# This explicit index definition gives the Series object additional capabilities...
# Like index can be not only integer
data=pd.Series([0.25,0.5,0.75,1.0], index=['a','b','c','d'])
data
```

    a    0.25
    b    0.50
    c    0.75
    d    1.00
    dtype: float64

``` python
data['b']
```

    0.5

``` python
# We can use non-contigious values for index like
data=pd.Series([0.25,0.5,0.75,1.0], index=[25,1,0,75])
data
```

    25    0.25
    1     0.50
    0     0.75
    75    1.00
    dtype: float64

``` python
data[1]
```

    0.5

### Series as specialized dictionary

In this way, you can think of a Pandas Series a bit like a
specialization of a Python dictionary. A dictionary is a structure that
maps arbitrary keys to a set of arbitrary values, and a Series is a
structure that maps typed keys to a set of typed values. This typing is
important: just as the type-specific compiled code behind a NumPy array
makes it more efficient than a Python list for certain operations, the
type information of a Pandas Series makes it much more efficient than
Python dictionaries for certain operations. We can make the
Series-as-dictionary analogy even more clear by constructing a Series
object directly from a Python dictionary

``` python
population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}
population = pd.Series(population_dict)
population
```

    California    38332521
    Texas         26448193
    New York      19651127
    Florida       19552860
    Illinois      12882135
    dtype: int64

***By default, a Series will be created where the index is drawn from
the sorted keys. From here, typical dictionary-style item access can be
performed:***

``` python
population['California']
```

    38332521

``` python
# Unlike a dictionary, though, the Series also supports array-style operations such as slicing
population['California':'Florida']
```

    California    38332521
    Texas         26448193
    New York      19651127
    Florida       19552860
    dtype: int64

### Constructing Series objects

We’ve already seen a few ways of constructing a Pandas Series from
scratch; all of them are some version of the following:  
`>>> pd.Series(data, index=index)`  
where index is an optional argument, and data can be one of many
entities.

***data can be a list or NumPy array, in which case index defaults to an
integer sequence***

``` python
pd.Series([2,4,6])
```

    0    2
    1    4
    2    6
    dtype: int64

***data can be a scalar, which is repeated to fill the specified
index:***

``` python
data=pd.Series(5, index=[100,200,300,400])
```

``` python
data
```

    100    5
    200    5
    300    5
    400    5
    dtype: int64

***data can be a dictionary, in which index defaults to the sorted
dictionary keys***

``` python
pd.Series({2:'a',3:'c',5:'e',0:'i'})
```

    2    a
    3    c
    5    e
    0    i
    dtype: object

## The Pandas DataFrame Object

The next fundamental structure in Pandas is the DataFrame. Like the
Series object discussed in the previous section, the DataFrame can be
thought of either as a gener‐ alization of a NumPy array, or as a
specialization of a Python dictionary. We’ll now take a look at each of
these perspectives.

### DataFrame as a generalized NumPy array

If a Series is an analog of a one-dimensional array with flexible
indices, a DataFrame is an analog of a two-dimensional array with both
flexible row indices and flexible column names. Just as you might think
of a two-dimensional array as an ordered sequence of aligned
one-dimensional columns, you can think of a DataFrame as a sequence of
aligned Series objects. Here, by “aligned” we mean that they share the
same index.

``` python
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}
```

``` python
area_dict
```

    {'California': 423967,
     'Texas': 695662,
     'New York': 141297,
     'Florida': 170312,
     'Illinois': 149995}

``` python
area=pd.Series(area_dict)
area
```

    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    dtype: int64

``` python
population
```

    California    38332521
    Texas         26448193
    New York      19651127
    Florida       19552860
    Illinois      12882135
    dtype: int64

``` python
type(area)
```

    pandas.core.series.Series

``` python
type(population)
```

    pandas.core.series.Series

``` python
states=pd.DataFrame({'population':population,'area':area})
states
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38332521</td>
      <td>423967</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>26448193</td>
      <td>695662</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
      <td>141297</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
      <td>170312</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12882135</td>
      <td>149995</td>
    </tr>
  </tbody>
</table>
</div>

***Like the Series object, the DataFrame has an index attribute that
gives access to the index labels***

``` python
states.columns
```

    Index(['population', 'area'], dtype='object')

``` python
states.index
```

    Index(['California', 'Texas', 'New York', 'Florida', 'Illinois'], dtype='object')

### DataFrame as specialized dictionary

Similarly, we can also think of a DataFrame as a specialization of a
dictionary. Where a dictionary maps a key to a value, a DataFrame maps a
column name to a Series of column data. For example, asking for the
‘area’ attribute returns the Series object containing the areas we saw
earlier

``` python
states['area']
```

    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    Name: area, dtype: int64

``` python
states['area'][0]
```

    423967

``` python
states['area'][:3]
```

    California    423967
    Texas         695662
    New York      141297
    Name: area, dtype: int64

``` python
states['area'][0,0]
```

    KeyError: 'key of type tuple not found and not a MultiIndex'

***Notice the potential point of confusion here: in a two-dimensional
NumPy array, `data[0]` will return the first row. For a DataFrame,
`data['col0']` will return the first column. Because of this, it is
probably better to think about DataFrames as generalized dictionaries
rather than generalized arrays, though both ways of looking at the
situa‐ tion can be useful.***

``` python
# Constructing dataframes from signle series object
population
```

    California    38332521
    Texas         26448193
    New York      19651127
    Florida       19552860
    Illinois      12882135
    dtype: int64

``` python
pd.DataFrame(data=population)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.DataFrame(data=population, columns=['population'])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# Constructing dataframes from list of dicts
data = [{'a':i, 'b':2*i} for i in range(4)]
data
```

    [{'a': 0, 'b': 0}, {'a': 1, 'b': 2}, {'a': 2, 'b': 4}, {'a': 3, 'b': 6}]

``` python
pd.DataFrame(data)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>

***Even if some keys in the dictionary are missing, Pandas will fill
them in with NaN (i.e., “not a number”) values:***

``` python
pd.DataFrame([{'a':1,'b':2},{'b':3,'c':4}])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# Constructing dataframes from a dictionary of Series Objects
pd.DataFrame({'population':population,'area':area})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>population</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>38332521</td>
      <td>423967</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>26448193</td>
      <td>695662</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>19651127</td>
      <td>141297</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
      <td>170312</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>12882135</td>
      <td>149995</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# Constructing dataframes from two-d Numpy array
pd.DataFrame(np.random.rand(3,2),columns=['foo','bar'],index=['a','b','c'])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>foo</th>
      <th>bar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.952378</td>
      <td>0.229453</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.305751</td>
      <td>0.208598</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.569426</td>
      <td>0.843111</td>
    </tr>
  </tbody>
</table>
</div>

## The Pandas Index Object

We have seen here that both the Series and DataFrame objects contain an
explicit index that lets you reference and modify data. This Index
object is an interesting structure in itself, and it can be thought of
either as an immutable array or as an ordered set (technically a
multiset, as Index objects may contain repeated values). Those views
have some interesting consequences in the operations available on Index
objects. As a simple example, let’s construct an Index from a list of
integers

``` python
ind=pd.Index([2,3,5,7,11])
ind
```

    Int64Index([2, 3, 5, 7, 11], dtype='int64')

``` python
type(ind)
```

    pandas.core.indexes.numeric.Int64Index

### Index as immutable array

The Index object in many ways operates like an array. For example, we
can use stan‐ dard Python indexing notation to retrieve values or slices

``` python
ind[1]
```

    3

``` python
ind[::2]
```

    Int64Index([2, 5, 11], dtype='int64')

``` python
# Index objects also have many of the attributes familiar from NumPy arrays:
print(ind.size, ind.shape, ind.ndim, ind.dtype)
```

    5 (5,) 1 int64

``` python
#One difference between Index objects and NumPy arrays is that indices are immuta‐
#ble—that is, they cannot be modified via the normal means
ind[1]=0
```

    TypeError: Index does not support mutable operations

### Index as ordered set

Pandas objects are designed to facilitate operations such as joins
across datasets, which depend on many aspects of set arithmetic.

the conventions used by Python’s built-in set data structure, so that
unions, intersec‐ tions, differences, and other combinations can be
computed in a familiar way the conventions used by Python’s built-in set
data structure, so that unions, intersec‐ tions, differences, and other
combinations can be computed in a familiar way.

``` python
ind_a=pd.Index([1,3,5,7,9])
ind_b=pd.Index([2,3,5,7,11])
```

``` python
ind_a & ind_b # Intersection
```

    /tmp/ipykernel_229290/4215377278.py:1: FutureWarning: Index.__and__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__and__.  Use index.intersection(other) instead.
      ind_a & ind_b # Intersection

    Int64Index([3, 5, 7], dtype='int64')

``` python
ind_a | ind_b # Union operation
```

    /tmp/ipykernel_229290/3034377863.py:1: FutureWarning: Index.__or__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__or__.  Use index.union(other) instead.
      ind_a | ind_b # Union operation

    Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')

``` python
ind_a^ind_b # Symmetric difference
```

    /tmp/ipykernel_229290/3946211992.py:1: FutureWarning: Index.__xor__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__xor__.  Use index.symmetric_difference(other) instead.
      ind_a^ind_b # Symmetric difference

    Int64Index([1, 2, 9, 11], dtype='int64')
