Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 02 - Data Indexing and Selection
------------------------------------------------------------------------

- <a href="#data-indexing-and-selection"
  id="toc-data-indexing-and-selection">Data Indexing and Selection</a>
  - <a href="#data-selection-in-series"
    id="toc-data-selection-in-series">Data Selection in Series</a>
    - <a href="#series-as-one-dimensional-array"
      id="toc-series-as-one-dimensional-array">Series as one-dimensional
      array</a>
    - <a href="#indexers-loc-iloc-and-ix"
      id="toc-indexers-loc-iloc-and-ix">Indexers: loc, iloc, and ix</a>
  - <a href="#data-selection-in-dataframe"
    id="toc-data-selection-in-dataframe">Data Selection in DataFrame</a>
    - <a href="#dataframe-as-a-dictionary"
      id="toc-dataframe-as-a-dictionary">DataFrame as a Dictionary</a>
    - <a href="#dataframe-as-2d-array"
      id="toc-dataframe-as-2d-array">DataFrame as 2D array</a>

------------------------------------------------------------------------

# Data Indexing and Selection

In Chapter 2, we looked in detail at methods and tools to access, set,
and modify values in NumPy arrays. These included indexing (e.g.,
`arr[2, 1]` ), slicing (e.g., `arr[:,1:5])`, masking (e.g.,
`arr[arr > 0]`), fancy indexing (e.g., `arr[0, [1, 5]]`), and
combinations thereof (e.g., `arr[:, [1, 5]]`). Here we’ll look at
similar means of accessing and modifying values in Pandas Series and
DataFrame objects. If you have used the NumPy patterns, the
corresponding patterns in Pandas will feel very famil‐ iar, though there
are a few quirks to be aware of.

## Data Selection in Series

As we saw in the previous section, a Series object acts in many ways
like a one- dimensional NumPy array, and in many ways like a standard
Python dictionary. If we keep these two overlapping analogies in mind,
it will help us to understand the patterns of data indexing and
selection in these arrays. \### Series as dictionary Like a dictionary,
the Series object provides a mapping from a collection of keys to a
collection of values:

``` python
import pandas as pd
data = pd.Series([0.25,0.5,0.75,1.0], index=['a','b','c','d'])
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
# also use dictionary-like python experssions and methods to examine the keys/indices and values
'a' in data
```

    True

``` python
data.keys()
```

    Index(['a', 'b', 'c', 'd'], dtype='object')

``` python
list(data.items())
```

    [('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)]

Series objects can even be modified with a dictionary-like syntax. Just
as you can extend a dictionary by assigning to a new key, you can extend
a Series by assigning to a new index value

``` python
data['e']=1.25
```

``` python
data
```

    a    0.25
    b    0.50
    c    0.75
    d    1.00
    e    1.25
    dtype: float64

### Series as one-dimensional array

A Series builds on this dictionary-like interface and provides
array-style item selec‐ tion via the same basic mechanisms as NumPy
arrays—that is, slices, masking, and fancy indexing. Examples of these
are as follows:

``` python
# slicing by explicit index
data['a':'c']
```

    a    0.25
    b    0.50
    c    0.75
    dtype: float64

``` python
#slicing by implicit integer index
data[0:2]
```

    a    0.25
    b    0.50
    dtype: float64

``` python
#masking
data[(data>0.3) & (data<0.8)]
```

    b    0.50
    c    0.75
    dtype: float64

``` python
#fancy indexing
data[['a','e']]
```

    a    0.25
    e    1.25
    dtype: float64

***Among these, slicing may be the source of the most confusion. Notice
that when you are slicing with an explicit index (i.e.,
`data['a':'c']`), the final index is included in the slice, while when
you’re slicing with an implicit index (i.e., `data[0:2]`), the final
index is excluded from the slice.***

### Indexers: loc, iloc, and ix

These slicing and indexing conventions can be a source of confusion. For
example, if your Series has an explicit integer index, an indexing
operation such as `data[1]` will use the explicit indices, while a
slicing operation like `data[1:3]` will use the implicit Python-style
index.

``` python
data =pd.Series(['a','b','c'], index=[1,3,5])
data
```

    1    a
    3    b
    5    c
    dtype: object

``` python
# explicit index when indexing
data[1]
```

    'a'

``` python
data[3]
```

    'b'

``` python
# implicit index when slicing
data[1:3]
```

    3    b
    5    c
    dtype: object

***Because of this potential confusion in the case of integer indexes,
Pandas provides some special indexer attributes that explicitly expose
certain indexing schemes. These are not functional methods, but
attributes that expose a particular slicing interface to the data in the
Series.***

``` python
# First, the loc attribute allows indexing and slicing that always references the explicit index:
data.loc[1]
```

    'a'

``` python
data.loc[1:3]
```

    1    a
    3    b
    dtype: object

``` python
# The iloc attribute allows indexing and slicing that always references the implicit Python-style index
data.iloc[1]
```

    'b'

``` python
data.iloc[0]
```

    'a'

``` python
data.loc[0]
```

    KeyError: 0

``` python
data.iloc[1:3]
```

    3    b
    5    c
    dtype: object

``` python
data.loc[1:3]
```

    1    a
    3    b
    dtype: object

## Data Selection in DataFrame

### DataFrame as a Dictionary

``` python
area = pd.Series({'California': 423967, 'Texas': 695662,
'New York': 141297, 'Florida': 170312,
'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127, 'Florida': 19552860,
'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data
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
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data['area']
```

    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    Name: area, dtype: int64

``` python
data.area
```

    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    Name: area, dtype: int64

``` python
data.area is data['area']
```

    True

``` python
data.pop is data['pop'] # pop method is refered instead of pop in our datafram
```

    False

``` python
data['density']=data['pop']/data['area']
```

``` python
data
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
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
      <td>90.413926</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
      <td>38.018740</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
      <td>85.883763</td>
    </tr>
  </tbody>
</table>
</div>

### DataFrame as 2D array

``` python
data.values
```

    array([[4.23967000e+05, 3.83325210e+07, 9.04139261e+01],
           [6.95662000e+05, 2.64481930e+07, 3.80187404e+01],
           [1.41297000e+05, 1.96511270e+07, 1.39076746e+02],
           [1.70312000e+05, 1.95528600e+07, 1.14806121e+02],
           [1.49995000e+05, 1.28821350e+07, 8.58837628e+01]])

``` python
# Transposing the values
data.T
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
      <th>California</th>
      <th>Texas</th>
      <th>New York</th>
      <th>Florida</th>
      <th>Illinois</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>area</th>
      <td>4.239670e+05</td>
      <td>6.956620e+05</td>
      <td>1.412970e+05</td>
      <td>1.703120e+05</td>
      <td>1.499950e+05</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>3.833252e+07</td>
      <td>2.644819e+07</td>
      <td>1.965113e+07</td>
      <td>1.955286e+07</td>
      <td>1.288214e+07</td>
    </tr>
    <tr>
      <th>density</th>
      <td>9.041393e+01</td>
      <td>3.801874e+01</td>
      <td>1.390767e+02</td>
      <td>1.148061e+02</td>
      <td>8.588376e+01</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data.values[0]
```

    array([4.23967000e+05, 3.83325210e+07, 9.04139261e+01])

``` python
data['area']
```

    California    423967
    Texas         695662
    New York      141297
    Florida       170312
    Illinois      149995
    Name: area, dtype: int64

``` python
data.iloc[:3,:2]
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
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data.loc[:'Illinois',:'pop']
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
      <th>area</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data.loc[data.density>100,['pop','density']]
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
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New York</th>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data.iloc[0,2]=90
data
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
      <th>area</th>
      <th>pop</th>
      <th>density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>423967</td>
      <td>38332521</td>
      <td>90.000000</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>695662</td>
      <td>26448193</td>
      <td>38.018740</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>141297</td>
      <td>19651127</td>
      <td>139.076746</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>170312</td>
      <td>19552860</td>
      <td>114.806121</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>149995</td>
      <td>12882135</td>
      <td>85.883763</td>
    </tr>
  </tbody>
</table>
</div>
