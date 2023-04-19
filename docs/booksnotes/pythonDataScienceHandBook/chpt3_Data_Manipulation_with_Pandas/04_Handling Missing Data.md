Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 04 - Handling Missing Data
------------------------------------------------------------------------

- <a href="#none-pythonic-missing-data"
  id="toc-none-pythonic-missing-data">None: Pythonic missing data</a>
- <a href="#nan-missing-numerical-data"
  id="toc-nan-missing-numerical-data">NaN: Missing numerical data</a>
- <a href="#detecting-null-values"
  id="toc-detecting-null-values">Detecting null values</a>
- <a href="#filling-null-values" id="toc-filling-null-values">Filling null
  values</a>

------------------------------------------------------------------------

## None: Pythonic missing data

``` python
import numpy as np
import pandas as pd
```

``` python
vals1=np.array([1,None,3,4])
vals1
```

    array([1, None, 3, 4], dtype=object)

``` python
for dtype in ['object','int']:
    print("dtype = ", dtype)
    %timeit np.arange(1E6, dtype=dtype).sum()
    print()
```

    dtype =  object
    57.5 ms ± 707 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    dtype =  int
    1.09 ms ± 35.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

``` python
# My get error on arregate functions of numpy
vals1.sum()
```

    TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'

## NaN: Missing numerical data

``` python
vals2=np.array([1,np.nan, 3,4])
vals2.dtype
```

``` python
vals1.dtype
```

``` python
1+np.nan
```

``` python
0*np.nan
```

``` python
vals2.sum(), vals2.min(), vals2.max()
```

***NumPy does provide some special aggregations that will ignore these
missing values***

## Detecting null values

Pandas data structures have two useful methods for detecting null data:
isnull() and notnull(). Either one will return a Boolean mask over the
data.

``` python
data.isnull()
```

``` python
# Deleting all null values
data=pd.Series([1,np.nan, 'hello',None])
data
```

``` python
df[3]=np.nan
df
```

    NameError: name 'df' is not defined

``` python
rng=np.random.RandomState(42)
ser=pd.Series(rng.randint(0,10,4))
df=pd.DataFrame(rng.randint(0,10,(3,4)),columns=['A',"B",'C','D'])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df[3]=np.nan
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.columns=[1,2,3,4,5]
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.dropna(axis='columns',how='all')
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.dropna(axis='rows',thresh=3)
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

## Filling null values

Sometimes rather than dropping NA values, you’d rather replace them with
a valid value. This value might be a single number like zero, or it
might be some sort of imputation or interpolation from the good values.
You could do this in-place using the isnull() method as a mask, but
because it is such a common operation Pandas provides the fillna()
method, which returns a copy of the array with the null values replaced.

``` python
data=pd.Series([1,np.nan,2,None,3], index=list('abcde'))
data
```

    a    1.0
    b    NaN
    c    2.0
    d    NaN
    e    3.0
    dtype: float64

``` python
# filling na values with a single 0
sum(data.isnull())
```

    2

``` python
data.fillna(-1)
```

    a    1.0
    b   -1.0
    c    2.0
    d   -1.0
    e    3.0
    dtype: float64

``` python
#forward-fill --> propagates the previous value forward
data.fillna(method='ffill')
```

    a    1.0
    b    1.0
    c    2.0
    d    2.0
    e    3.0
    dtype: float64

``` python
# Back fill, to propgate the next value backward
data.fillna(method='bfill')
```

    a    1.0
    b    2.0
    c    2.0
    d    3.0
    e    3.0
    dtype: float64

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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>9</td>
      <td>2</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.fillna(method='ffill',axis=1)
# if the previous value is not available during a forward fill, the NA value remains
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>
