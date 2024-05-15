Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 03 -Operations on Data in Pandas
------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
```

``` python
rng=np.random.RandomState(42)
ser=pd.Series(rng.randint(0,10,4))
```

``` python
ser
```

    0    6
    1    3
    2    7
    3    4
    dtype: int64

``` python
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
np.exp(ser)
```

    0     403.428793
    1      20.085537
    2    1096.633158
    3      54.598150
    dtype: float64

``` python
np.sin(df*np.pi/4)
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
      <td>-1.000000</td>
      <td>7.071068e-01</td>
      <td>1.000000</td>
      <td>-1.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.707107</td>
      <td>1.224647e-16</td>
      <td>0.707107</td>
      <td>-7.071068e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.707107</td>
      <td>1.000000e+00</td>
      <td>-0.707107</td>
      <td>1.224647e-16</td>
    </tr>
  </tbody>
</table>
</div>

``` python
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127}, name='population')
```

``` python
area
```

    Alaska        1723337
    Texas          695662
    California     423967
    Name: area, dtype: int64

``` python
population
```

    California    38332521
    Texas         26448193
    New York      19651127
    Name: population, dtype: int64

``` python
population/area
```

    Alaska              NaN
    California    90.413926
    New York            NaN
    Texas         38.018740
    dtype: float64

``` python
area.index | population.index
```

    /tmp/ipykernel_15930/3572280633.py:1: FutureWarning: Index.__or__ operating as a set operation is deprecated, in the future this will be a logical operation matching Series.__or__.  Use index.union(other) instead.
      area.index | population.index

    Index(['Alaska', 'California', 'New York', 'Texas'], dtype='object')
