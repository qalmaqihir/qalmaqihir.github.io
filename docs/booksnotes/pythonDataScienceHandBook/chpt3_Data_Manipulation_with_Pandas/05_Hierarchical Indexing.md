Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 05 - Hierarchical Indexing
------------------------------------------------------------------------

- <a href="#hierarchical-indexing-multi-indexing"
  id="toc-hierarchical-indexing-multi-indexing">Hierarchical Indexing
  (Multi-indexing)</a>
  - <a href="#a-multiply-indexed-series"
    id="toc-a-multiply-indexed-series">A Multiply Indexed Series</a>
  - <a href="#methods-of-multiindex-creation"
    id="toc-methods-of-multiindex-creation">Methods of MultiIndex
    Creation</a>
    - <a href="#explicit-multiindex-constructors"
      id="toc-explicit-multiindex-constructors">Explicit MultiIndex
      constructors</a>
    - <a href="#multiindex-level-names"
      id="toc-multiindex-level-names">MultiIndex level names</a>
    - <a href="#multiindex-for-columns"
      id="toc-multiindex-for-columns">MultiIndex for columns</a>
  - <a href="#indexing-and-slicing-a-multiindex"
    id="toc-indexing-and-slicing-a-multiindex">Indexing and Slicing a
    MultiIndex</a>
  - <a href="#rearranging-multi-indices"
    id="toc-rearranging-multi-indices">Rearranging Multi-Indices</a>
    - <a href="#sorted-and-unsorted-indices"
      id="toc-sorted-and-unsorted-indices">Sorted and unsorted indices</a>
  - <a href="#index-setting-and-resetting"
    id="toc-index-setting-and-resetting">Index setting and resetting</a>
  - <a href="#data-aggregations-on-multi-indices"
    id="toc-data-aggregations-on-multi-indices">Data Aggregations on
    Multi-Indices</a>

------------------------------------------------------------------------


# Hierarchical Indexing (Multi-indexing)

a far more common pattern in practice is to make use of hierarchical
indexing (also known as multi-indexing) to incorporate multiple index
levels within a single index. In this way, higher-dimensional data can
be compactly represented within the familiar one-dimensional Series and
two-dimensional DataFrame objects.

## A Multiply Indexed Series

Let’s start by considering how we might represent two-dimensional data
within a one-dimensional Series. For concreteness, we will consider a
series of data where each point has a character and numerical key.

``` python
import numpy as np
import pandas as pd
```

``` python
# the bad way
index = [('California', 2000), ('California', 2010),
('New York', 2000), ('New York', 2010),
('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
18976457, 19378102,
20851820, 25145561]
```

``` python
pop=pd.Series(populations,index=index)
pop
```

    (California, 2000)    33871648
    (California, 2010)    37253956
    (New York, 2000)      18976457
    (New York, 2010)      19378102
    (Texas, 2000)         20851820
    (Texas, 2010)         25145561
    dtype: int64

``` python
# Indexing 
pop[('New York',2010):('Texas',2010)]
```

    (New York, 2010)    19378102
    (Texas, 2000)       20851820
    (Texas, 2010)       25145561
    dtype: int64

``` python
#A better way
index=pd.MultiIndex.from_tuples(index)
index
```

    MultiIndex([('California', 2000),
                ('California', 2010),
                (  'New York', 2000),
                (  'New York', 2010),
                (     'Texas', 2000),
                (     'Texas', 2010)],
               )

``` python
pop=pop.reindex(index)
```

``` python
pop
```

    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64

``` python
pop[:,2010]
```

    California    37253956
    New York      19378102
    Texas         25145561
    dtype: int64

``` python
# Multi-index as extr dimension
pop_df=pop.unstack()
pop_df
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
      <th>2000</th>
      <th>2010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>33871648</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>20851820</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pop_df.stack()
```

    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64

``` python
pop_df = pd.DataFrame({'total': pop,
'under18': [9267089, 9284094,
4687374, 4318033,
5906301, 6879014]})
pop_df
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
      <th></th>
      <th>total</th>
      <th>under18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>33871648</td>
      <td>9267089</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
      <td>9284094</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
      <td>4687374</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>19378102</td>
      <td>4318033</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Texas</th>
      <th>2000</th>
      <td>20851820</td>
      <td>5906301</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>25145561</td>
      <td>6879014</td>
    </tr>
  </tbody>
</table>
</div>

## Methods of MultiIndex Creation

The most straightforward way to construct a multiply indexed Series or
DataFrame is to simply pass a list of two or more index arrays to the
constructor.

``` python
df= pd.DataFrame(np.random.rand(4,2),
                 index=[['a','a','b','b'],[1,2,1,2]],
                 columns=['data1','data2'])
```

``` python
df
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
      <th></th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0.812128</td>
      <td>0.312338</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.769851</td>
      <td>0.255045</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>0.904529</td>
      <td>0.364216</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.139294</td>
      <td>0.501778</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data = {('California', 2000): 33871648,
('California', 2010): 37253956,
('Texas', 2000): 20851820,
('Texas', 2010): 25145561,
('New York', 2000): 18976457,
('New York', 2010): 19378102}
pd.Series(data)
```

    California  2000    33871648
                2010    37253956
    Texas       2000    20851820
                2010    25145561
    New York    2000    18976457
                2010    19378102
    dtype: int64

### Explicit MultiIndex constructors

For more flexibility in how the index is constructed, you can instead
use the class method constructors available in the pd.MultiIndex. For
example, as we did before, you can construct the MultiIndex from a
simple list of arrays, giving the index values within each level:

``` python
pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])
```

    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )

``` python
pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
```

    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )

``` python
pd.MultiIndex.from_product([['a','b'],[1,2]])
```

    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )

### MultiIndex level names

Sometimes it is convenient to name the levels of the MultiIndex. You can
accomplish this by passing the names argument to any of the above
MultiIndex constructors, or by setting the names attribute of the index
after the fact:

``` python
pop.index
```

    MultiIndex([('California', 2000),
                ('California', 2010),
                (  'New York', 2000),
                (  'New York', 2010),
                (     'Texas', 2000),
                (     'Texas', 2010)],
               )

``` python
pop.index.names=['state','year']
```

``` python
pop
```

    state       year
    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64

``` python
pop.unstack()
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
      <th>year</th>
      <th>2000</th>
      <th>2010</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>33871648</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>20851820</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>

### MultiIndex for columns

In a DataFrame, the rows and columns are completely symmetric, and just
as the rows can have multiple levels of indices, the columns can have
multiple levels as well

``` python
# hierarchical indices and columns
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
names=['subject', 'type'])
# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>43.0</td>
      <td>36.1</td>
      <td>37.0</td>
      <td>35.9</td>
      <td>39.0</td>
      <td>38.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.0</td>
      <td>35.6</td>
      <td>36.0</td>
      <td>37.6</td>
      <td>30.0</td>
      <td>35.8</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>35.0</td>
      <td>38.2</td>
      <td>47.0</td>
      <td>35.6</td>
      <td>43.0</td>
      <td>37.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43.0</td>
      <td>37.8</td>
      <td>25.0</td>
      <td>36.4</td>
      <td>34.0</td>
      <td>36.7</td>
    </tr>
  </tbody>
</table>
</div>

``` python
health_data['Guido']
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
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>37.0</td>
      <td>35.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36.0</td>
      <td>37.6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>47.0</td>
      <td>35.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25.0</td>
      <td>36.4</td>
    </tr>
  </tbody>
</table>
</div>

## Indexing and Slicing a MultiIndex

Indexing and slicing on a MultiIndex is designed to be intuitive, and it
helps if you think about the indices as added dimensions. We’ll first
look at indexing multiply indexed Series, and then multiply indexed
DataFrames.

``` python
# Mutiply indexed Series
pop
```

    state       year
    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64

``` python
pop['California']
```

    year
    2000    33871648
    2010    37253956
    dtype: int64

``` python
pop['California']['2000']
```

    KeyError: '2000'

``` python
pop['California','2000']
```

``` python
pop.loc['California':'New York']
```

``` python
pop[:,2000]
```

``` python
pop[pop>2200000]
```

``` python
pop[['California','Texas']]
```

## Rearranging Multi-Indices

One of the keys to working with multiply indexed data is knowing how to
effectively transform the data. There are a number of operations that
will preserve all the infor‐ mation in the dataset, but rearrange it for
the purposes of various computations. We saw a brief example of this in
the stack() and unstack() methods, but there are many more ways to
finely control the rearrangement of data between hierarchical indices
and columns,

### Sorted and unsorted indices

Earlier, we briefly mentioned a caveat, but we should emphasize it more
here. Many of the MultiIndex slicing operations will fail if the index
is not sorted. Let’s take a look at this here. We’ll start by creating
some simple multiply indexed data where the indices are not
lexographically sorted:

``` python
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
```

``` python
try:
    data['a':'b']
except KeyError as e:
    print(type(e))
    print(e)
```

``` python
data=data.sort_index()
data
```

``` python
try:
    print(data['a':'b'])
except KeyError as e:
    print(type(e))
    print(e)
```

``` python
pop
```

``` python
pop.unstack(level=0)
```

``` python
pop.unstack(level=1)
```

``` python
pop.unstack().stack() # Get the original dataset
```

## Index setting and resetting

Another way to rearrange hierarchical data is to turn the index labels
into columns; this can be accomplished with the reset_index method.
Calling this on the popula‐ tion dictionary will result in a DataFrame
with a state and year column holding the information that was formerly
in the index.

``` python
pop_flat=pop.reset_index(name='population')
```

``` python
pop_flat
```

``` python
pop_flat.set_index(['state','year'])
```

## Data Aggregations on Multi-Indices

We’ve previously seen that Pandas has built-in data aggregation methods,
such as mean(), sum(), and max(). For hierarchically indexed data, these
can be passed a level parameter that controls which subset of the data
the aggregate is computed on

``` python
health_data
```

``` python
data_mean=health_data.mean(level='year')
```

``` python
data_mean
```

``` python
data_mean=health_data.mean(axis=1,level='type')
```

``` python
data_mean
```

``` python
#pip install -U jupyter notebook
```

    Requirement already satisfied: jupyter in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (1.0.0)
    Requirement already satisfied: notebook in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (6.4.8)
    Collecting notebook
      Downloading notebook-6.4.12-py3-none-any.whl (9.9 MB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.9/9.9 MB 3.2 MB/s eta 0:00:00m eta 0:00:010:01:010m
    Requirement already satisfied: ipywidgets in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter) (7.6.5)
    Requirement already satisfied: ipykernel in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter) (6.9.1)
    Requirement already satisfied: qtconsole in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter) (5.3.0)
    Requirement already satisfied: nbconvert in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter) (6.4.4)
    Requirement already satisfied: jupyter-console in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter) (6.4.0)
    Requirement already satisfied: terminado>=0.8.3 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (0.13.1)
    Requirement already satisfied: argon2-cffi in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (21.3.0)
    Requirement already satisfied: ipython-genutils in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (0.2.0)
    Requirement already satisfied: nest-asyncio>=1.5 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (1.5.5)
    Requirement already satisfied: prometheus-client in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (0.14.1)
    Requirement already satisfied: jupyter-core>=4.6.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (4.10.0)
    Requirement already satisfied: traitlets>=4.2.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (5.1.1)
    Requirement already satisfied: jinja2 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (3.1.2)
    Requirement already satisfied: jupyter-client>=5.3.4 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (6.1.12)
    Requirement already satisfied: nbformat in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (5.3.0)
    Requirement already satisfied: Send2Trash>=1.8.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (1.8.0)
    Requirement already satisfied: pyzmq>=17 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (23.2.0)
    Requirement already satisfied: tornado>=6.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from notebook) (6.1)
    Requirement already satisfied: python-dateutil>=2.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter-client>=5.3.4->notebook) (2.8.2)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (0.5.13)
    Requirement already satisfied: bleach in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (4.1.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (1.5.0)
    Requirement already satisfied: jupyterlab-pygments in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (0.1.2)
    Requirement already satisfied: pygments>=2.4.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (2.11.2)
    Requirement already satisfied: testpath in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (0.6.0)
    Requirement already satisfied: defusedxml in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (0.7.1)
    Requirement already satisfied: beautifulsoup4 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (4.11.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (0.4)
    Requirement already satisfied: mistune<2,>=0.8.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbconvert->jupyter) (0.8.4)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jinja2->notebook) (2.1.1)
    Requirement already satisfied: jsonschema>=2.6 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbformat->notebook) (4.4.0)
    Requirement already satisfied: fastjsonschema in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from nbformat->notebook) (2.15.1)
    Requirement already satisfied: ptyprocess in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from terminado>=0.8.3->notebook) (0.7.0)
    Requirement already satisfied: argon2-cffi-bindings in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from argon2-cffi->notebook) (21.2.0)
    Requirement already satisfied: debugpy<2.0,>=1.0.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter) (1.5.1)
    Requirement already satisfied: matplotlib-inline<0.2.0,>=0.1.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter) (0.1.2)
    Requirement already satisfied: ipython>=7.23.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipykernel->jupyter) (8.2.0)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipywidgets->jupyter) (3.5.2)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipywidgets->jupyter) (1.0.0)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jupyter-console->jupyter) (3.0.20)
    Requirement already satisfied: qtpy>=2.0.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from qtconsole->jupyter) (2.0.1)
    Requirement already satisfied: decorator in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.18.1)
    Requirement already satisfied: setuptools>=18.5 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (63.4.1)
    Requirement already satisfied: pexpect>4.3 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (4.8.0)
    Requirement already satisfied: pickleshare in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.7.5)
    Requirement already satisfied: stack-data in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.2.0)
    Requirement already satisfied: backcall in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.2.0)
    Requirement already satisfied: attrs>=17.4.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jsonschema>=2.6->nbformat->notebook) (21.4.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jsonschema>=2.6->nbformat->notebook) (0.18.0)
    Requirement already satisfied: wcwidth in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->jupyter-console->jupyter) (0.2.5)
    Requirement already satisfied: six>=1.5 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.1->jupyter-client>=5.3.4->notebook) (1.16.0)
    Requirement already satisfied: packaging in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from qtpy>=2.0.1->qtconsole->jupyter) (21.3)
    Requirement already satisfied: cffi>=1.0.1 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from argon2-cffi-bindings->argon2-cffi->notebook) (1.15.1)
    Requirement already satisfied: soupsieve>1.2 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from beautifulsoup4->nbconvert->jupyter) (2.3.1)
    Requirement already satisfied: webencodings in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from bleach->nbconvert->jupyter) (0.5.1)
    Requirement already satisfied: pycparser in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook) (2.21)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from jedi>=0.16->ipython>=7.23.1->ipykernel->jupyter) (0.8.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from packaging->qtpy>=2.0.1->qtconsole->jupyter) (3.0.4)
    Requirement already satisfied: executing in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from stack-data->ipython>=7.23.1->ipykernel->jupyter) (0.8.3)
    Requirement already satisfied: asttokens in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from stack-data->ipython>=7.23.1->ipykernel->jupyter) (2.0.5)
    Requirement already satisfied: pure-eval in /home/qalmaqihir/anaconda3/lib/python3.9/site-packages (from stack-data->ipython>=7.23.1->ipykernel->jupyter) (0.2.2)
    Installing collected packages: notebook
      Attempting uninstall: notebook
        Found existing installation: notebook 6.4.8
        Uninstalling notebook-6.4.8:
          Successfully uninstalled notebook-6.4.8
    Successfully installed notebook-6.4.12
    Note: you may need to restart the kernel to use updated packages.
