================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 06 - Combine Dataset: Concate and Append
------------------------------------------------------------------------

- <a href="#combining-datasets-concat-and-append"
  id="toc-combining-datasets-concat-and-append">Combining Datasets: Concat
  and Append</a>
- <a href="#simple-concatenation-with-pd.concat"
  id="toc-simple-concatenation-with-pd.concat">Simple Concatenation with
  pd.concat</a>
  - <a href="#duplicate-indices" id="toc-duplicate-indices">Duplicate
    indices</a>
  - <a
    href="#catching-the-repeats-as-an-error.-if-youd-like-to-simply-verify-that-the-indices-in-the"
    id="toc-catching-the-repeats-as-an-error.-if-youd-like-to-simply-verify-that-the-indices-in-the">Catching
    the repeats as an error. If you’d like to simply verify that the indices
    in the</a>
  - <a
    href="#ignoring-the-index.-sometimes-the-index-itself-does-not-matter-and-you-would-prefer"
    id="toc-ignoring-the-index.-sometimes-the-index-itself-does-not-matter-and-you-would-prefer">Ignoring
    the index. Sometimes the index itself does not matter, and you would
    prefer</a>
  - <a
    href="#adding-multiindex-keys.-another-alternative-is-to-use-the-keys-option-to-specify-a-label"
    id="toc-adding-multiindex-keys.-another-alternative-is-to-use-the-keys-option-to-specify-a-label">Adding
    MultiIndex keys. Another alternative is to use the keys option to
    specify a label</a>
- <a href="#concatenation-with-joins"
  id="toc-concatenation-with-joins">Concatenation with joins</a>
  - <a href="#the-append-method" id="toc-the-append-method">The append()
    method</a>

------------------------------------------------------------------------

## Combining Datasets: Concat and Append

Some of the most interesting studies of data come from combining
different data sources. These operations can involve anything from very
straightforward concatena‐ tion of two different datasets, to more
complicated database-style joins and merges that correctly handle any
overlaps between the datasets. Series and DataFrames are built with this
type of operation in mind, and Pandas includes functions and methods
that make this sort of data wrangling fast and straightforward.

``` python
import numpy as np
import pandas as pd
```

``` python
def make_df(cols, ind):
    data={c:[str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,ind)
```

``` python
make_df('ABC',range(3))
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# Recall Numpy concatenation 
x=[1,2,3]
y=[4,3,6]
z=[9,8,0]
np.concatenate([x,y,z])
```

    array([1, 2, 3, 4, 3, 6, 9, 8, 0])

``` python
x=[[2,4],[3,5]]
x=np.concatenate([x,x], axis=1)
x
```

    array([[2, 4, 2, 4],
           [3, 5, 3, 5]])

## Simple Concatenation with pd.concat

Pandas has a function, pd.concat(), which has a similar syntax to
np.concatenate but contains a number of options that we’ll discuss
momentarily

`#Signature in Pandas v0.18 pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)`

``` python
ser1=pd.Series(['A','B','C'],index=[1,2,3])
ser2=pd.Series(['D','E','F'], index=[4,5,6])
pd.concat([ser1,ser2])
```

    1    A
    2    B
    3    C
    4    D
    5    E
    6    F
    dtype: object

``` python
df3=make_df("AB",[0,1])
df4=make_df("CD",[0,1])
```

``` python
df3
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df4
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
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C1</td>
      <td>D1</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.concat([df3,df4])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.concat([df3,df4],axis=1)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
  </tbody>
</table>
</div>

### Duplicate indices

One important difference between np.concatenate and pd.concat is that
Pandas concatenation preserves indices, even if the result will have
duplicate indices!

``` python
x=make_df("AB",[0,1])
y=make_df("AB",[2,3])
x
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
  </tbody>
</table>
</div>

``` python
y
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>

``` python
y.index
```

    Int64Index([2, 3], dtype='int64')

``` python
x.index=y.index
```

``` python
x.index
```

    Int64Index([2, 3], dtype='int64')

``` python
x
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
  </tbody>
</table>
</div>

``` python
y
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.concat([x,y])
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.concat([x,y],axis=1)
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
      <th>A</th>
      <th>B</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>A0</td>
      <td>B0</td>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A1</td>
      <td>B1</td>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>

### Catching the repeats as an error. If you’d like to simply verify that the indices in the

result of pd.concat() do not overlap, you can specify the
verify_integrity flag. With this set to True, the concatenation will
raise an exception if there are duplicate indices. Here is an example,
where for clarity we’ll catch and print the error message:

``` python
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)
```

    ValueError: Indexes have overlapping values: Int64Index([2, 3], dtype='int64')

### Ignoring the index. Sometimes the index itself does not matter, and you would prefer

it to simply be ignored. You can specify this option using the
ignore_index flag. With this set to True, the concatenation will create
a new integer index for the resulting Series:

``` python
print(x);print(y)
```

        A   B
    2  A0  B0
    3  A1  B1
        A   B
    2  A2  B2
    3  A3  B3

``` python
pd.concat([x,y],ignore_index=True)
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>

### Adding MultiIndex keys. Another alternative is to use the keys option to specify a label

for the data sources; the result will be a hierarchically indexed series
containing the data:

``` python
pd.concat([x,y], keys=['x','y'])
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">x</th>
      <th>2</th>
      <td>A0</td>
      <td>B0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">y</th>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
  </tbody>
</table>
</div>

## Concatenation with joins

In the simple examples we just looked at, we were mainly concatenating
DataFrames with shared column names. In practice, data from different
sources might have differ‐ ent sets of column names, and pd.concat
offers several options in this case. Consider the concatenation of the
following two DataFrames, which have some (but not all!) columns in
common:

``` python
df5=make_df('ABC',[1,2])
df6=make_df('BCD',[3,4])
df6
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
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df5
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.concat([df5,df6])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
  </tbody>
</table>
</div>

***By default, the entries for which no data is available are filled
with NA values. To change this, we can specify one of several options
for the join and join_axes param‐ eters of the concatenate function.***

``` python
pd.concat([df5,df6],join='inner')
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
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>B1</td>
      <td>C1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B2</td>
      <td>C2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B3</td>
      <td>C3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B4</td>
      <td>C4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.concat([df5,df6],join='outer')
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
  </tbody>
</table>
</div>

### The append() method

Because direct array concatenation is so common, Series and DataFrame
objects have an append method that can accomplish the same thing in
fewer keystrokes. For example, rather than calling pd.concat(\[df1,
df2\]), you can simply call df1.append(df2):

``` python
df1=make_df('AB',[1,2])
df2=make_df('AB',[3,4])
df1
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df2
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df1.append(df2)
```

    /tmp/ipykernel_36950/3062608662.py:1: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df1.append(df2)

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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A4</td>
      <td>B4</td>
    </tr>
  </tbody>
</table>
</div>
