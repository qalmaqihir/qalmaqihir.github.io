Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# Example: Birthrate Data
------------------------------------------------------------------------
<center>
<a href=''>![Image](../../../assets/img/logo1.png)</a>
</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
------------------------------------------------------------------------

- <a href="#example-birthrate-data"
  id="toc-example-birthrate-data">Example: Birthrate Data</a>
  - <a href="#further-data-exploration"
    id="toc-further-data-exploration">Further data exploration</a>

------------------------------------------------------------------------

# Example: Birthrate Data

As a more interesting example, let’s take a look at the freely available
data on births in the United States, provided by the Centers for Disease
Control (CDC). This data can be found at
[link](https://raw.githubusercontent.com/jakevdp/data-CDCbirths/master/)
births.csv (this dataset has been analyzed rather extensively by Andrew
Gelman and his group; see, for example, this blog post)

``` python
# shell command to download the data:
# !curl -O https://raw.githubusercontent.com/jakevdp/data-CDCbirths/
# master/births.csv
```

``` python
import numpy as np
import pandas as pd
```

``` python
births = pd.read_csv("../data/births.csv")
births.head()
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
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>gender</th>
      <th>births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1969</td>
      <td>1</td>
      <td>1.0</td>
      <td>F</td>
      <td>4046</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1969</td>
      <td>1</td>
      <td>1.0</td>
      <td>M</td>
      <td>4440</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1969</td>
      <td>1</td>
      <td>2.0</td>
      <td>F</td>
      <td>4454</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1969</td>
      <td>1</td>
      <td>2.0</td>
      <td>M</td>
      <td>4548</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1969</td>
      <td>1</td>
      <td>3.0</td>
      <td>F</td>
      <td>4548</td>
    </tr>
  </tbody>
</table>
</div>

**We can start to understand this data a bit more by using a pivot
table. Let’s add a dec‐ ade column, and take a look at male and female
births as a function of decade:**

``` python
births['decade']=10* (births['year']//10)
births.head()
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
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>gender</th>
      <th>births</th>
      <th>decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1969</td>
      <td>1</td>
      <td>1.0</td>
      <td>F</td>
      <td>4046</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1969</td>
      <td>1</td>
      <td>1.0</td>
      <td>M</td>
      <td>4440</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1969</td>
      <td>1</td>
      <td>2.0</td>
      <td>F</td>
      <td>4454</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1969</td>
      <td>1</td>
      <td>2.0</td>
      <td>M</td>
      <td>4548</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1969</td>
      <td>1</td>
      <td>3.0</td>
      <td>F</td>
      <td>4548</td>
      <td>1960</td>
    </tr>
  </tbody>
</table>
</div>

``` python
births.pivot_table('births', index='decade',columns='gender', aggfunc='sum')
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
      <th>gender</th>
      <th>F</th>
      <th>M</th>
    </tr>
    <tr>
      <th>decade</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>1753634</td>
      <td>1846572</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>16263075</td>
      <td>17121550</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>18310351</td>
      <td>19243452</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>19479454</td>
      <td>20420553</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>18229309</td>
      <td>19106428</td>
    </tr>
  </tbody>
</table>
</div>

*We immediately see that male births outnumber female births in every
decade. To see this trend a bit more clearly, we can use the built-in
plotting tools in Pandas to visual‐ ize the total number of births by
year*

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
```

``` python
sns.set()
births.pivot_table('births', index='year',columns='gender', aggfunc='sum').plot()
plt.ylabel("total birth per year")
```

    Text(0, 0.5, 'total birth per year')

![](Example_%20Birthrate%20Data_files/figure-gfm/cell-9-output-2.png)

### Further data exploration

Though this doesn’t necessarily relate to the pivot table, there are a
few more interest‐ ing features we can pull out of this dataset using
the Pandas tools covered up to this point. We must start by cleaning the
data a bit, removing outliers caused by mistyped dates (e.g., June 31st)
or missing values (e.g., June 99th). One easy way to remove these all at
once is to cut outliers; we’ll do this via a robust sigma-clipping
operation

``` python
quartiles = np.percentile(births['births'],[25,50,75])
mu=quartiles[1]
sig=0.74*(quartiles[2]-quartiles[0])
```

``` python
mu
```

    4814.0

``` python
sig
```

    689.31

This final line is a robust estimate of the sample mean, where the 0.74
comes from the interquartile range of a Gaussian distribution.  
**With this we can use the query() method to filter out rows with births
outside these values:**

``` python
births=births.query('(births > @mu -5 * @sig) & (births<@mu + 5*@sig)')
```

``` python
births
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
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>gender</th>
      <th>births</th>
      <th>decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1969</td>
      <td>1</td>
      <td>1.0</td>
      <td>F</td>
      <td>4046</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1969</td>
      <td>1</td>
      <td>1.0</td>
      <td>M</td>
      <td>4440</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1969</td>
      <td>1</td>
      <td>2.0</td>
      <td>F</td>
      <td>4454</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1969</td>
      <td>1</td>
      <td>2.0</td>
      <td>M</td>
      <td>4548</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1969</td>
      <td>1</td>
      <td>3.0</td>
      <td>F</td>
      <td>4548</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15062</th>
      <td>1988</td>
      <td>12</td>
      <td>29.0</td>
      <td>M</td>
      <td>5944</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15063</th>
      <td>1988</td>
      <td>12</td>
      <td>30.0</td>
      <td>F</td>
      <td>5742</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15064</th>
      <td>1988</td>
      <td>12</td>
      <td>30.0</td>
      <td>M</td>
      <td>6095</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15065</th>
      <td>1988</td>
      <td>12</td>
      <td>31.0</td>
      <td>F</td>
      <td>4435</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15066</th>
      <td>1988</td>
      <td>12</td>
      <td>31.0</td>
      <td>M</td>
      <td>4698</td>
      <td>1980</td>
    </tr>
  </tbody>
</table>
<p>14610 rows × 6 columns</p>
</div>

**Next we set the day column to integers; previously it had been a
string because some columns in the dataset contained the value ‘null’:**

``` python
births['day']=births['day'].astype(int)
```

    /tmp/ipykernel_18639/3805690895.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      births['day']=births['day'].astype(int)

``` python
births
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
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>gender</th>
      <th>births</th>
      <th>decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1969</td>
      <td>1</td>
      <td>1</td>
      <td>F</td>
      <td>4046</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1969</td>
      <td>1</td>
      <td>1</td>
      <td>M</td>
      <td>4440</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1969</td>
      <td>1</td>
      <td>2</td>
      <td>F</td>
      <td>4454</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1969</td>
      <td>1</td>
      <td>2</td>
      <td>M</td>
      <td>4548</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1969</td>
      <td>1</td>
      <td>3</td>
      <td>F</td>
      <td>4548</td>
      <td>1960</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15062</th>
      <td>1988</td>
      <td>12</td>
      <td>29</td>
      <td>M</td>
      <td>5944</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15063</th>
      <td>1988</td>
      <td>12</td>
      <td>30</td>
      <td>F</td>
      <td>5742</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15064</th>
      <td>1988</td>
      <td>12</td>
      <td>30</td>
      <td>M</td>
      <td>6095</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15065</th>
      <td>1988</td>
      <td>12</td>
      <td>31</td>
      <td>F</td>
      <td>4435</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>15066</th>
      <td>1988</td>
      <td>12</td>
      <td>31</td>
      <td>M</td>
      <td>4698</td>
      <td>1980</td>
    </tr>
  </tbody>
</table>
<p>14610 rows × 6 columns</p>
</div>

***Finally, we can combine the day, month, and year to create a Date
index This allows us to quickly compute the weekday corresponding to
each row:***

``` python
# create a datetime index from the year, month, day
births.index = pd.to_datetime(10000 * births.year +
100 * births.month +
births.day, format='%Y%m%d')
```

``` python
births['dayofweek']=births.index.day_of_week
```

``` python
# Using this we can plot the births by weekday for several decades
import matplotlib.pyplot as plt
import matplotlib as mpl
births.pivot_table('births', index='dayofweek',
columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day');
```

    /tmp/ipykernel_18639/3967923407.py:6: UserWarning: FixedFormatter should only be used together with FixedLocator
      plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])

![](Example_%20Birthrate%20Data_files/figure-gfm/cell-19-output-2.png)

``` python
births_by_date=births.pivot_table('births',
                                 [births.index.month, births.index.day])
```

``` python
births_by_date
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
      <th>births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th>1</th>
      <td>4009.225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4247.400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4500.900</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4571.350</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4603.625</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">12</th>
      <th>27</th>
      <td>4850.150</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5044.200</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5120.150</td>
    </tr>
    <tr>
      <th>30</th>
      <td>5172.350</td>
    </tr>
    <tr>
      <th>31</th>
      <td>4859.200</td>
    </tr>
  </tbody>
</table>
<p>366 rows × 1 columns</p>
</div>

``` python
births_by_date.index=[pd.datetime(2012,month,day) for (month,day)in births_by_date.index]
```

    /tmp/ipykernel_18639/1749910599.py:1: FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.
      births_by_date.index=[pd.datetime(2012,month,day) for (month,day)in births_by_date.index]

``` python
births_by_date
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
      <th>births</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-01-01</th>
      <td>4009.225</td>
    </tr>
    <tr>
      <th>2012-01-02</th>
      <td>4247.400</td>
    </tr>
    <tr>
      <th>2012-01-03</th>
      <td>4500.900</td>
    </tr>
    <tr>
      <th>2012-01-04</th>
      <td>4571.350</td>
    </tr>
    <tr>
      <th>2012-01-05</th>
      <td>4603.625</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2012-12-27</th>
      <td>4850.150</td>
    </tr>
    <tr>
      <th>2012-12-28</th>
      <td>5044.200</td>
    </tr>
    <tr>
      <th>2012-12-29</th>
      <td>5120.150</td>
    </tr>
    <tr>
      <th>2012-12-30</th>
      <td>5172.350</td>
    </tr>
    <tr>
      <th>2012-12-31</th>
      <td>4859.200</td>
    </tr>
  </tbody>
</table>
<p>366 rows × 1 columns</p>
</div>

``` python
# Focusing on the month and day only, we now have a time series reflecting the average
#number of births by date of the year.
fig, ax = plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax)
```

    <AxesSubplot:>

![](Example_%20Birthrate%20Data_files/figure-gfm/cell-24-output-2.png)
