================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# Example: US States Data
------------------------------------------------------------------------

- <a href="#example-us-states-data"
  id="toc-example-us-states-data">Example: US States Data</a>

------------------------------------------------------------------------

# Example: US States Data

Merge and join operations come up most often when one is combining data
from dif‐ ferent sources. Here we will consider an example of some data
about US states and their populations.

``` python
import numpy as np
import pandas as pd
```

``` python
pop = pd.read_csv("../data/state-population.csv")
areas= pd.read_csv("../data/state-areas.csv")
abbrevs= pd.read_csv("../data/state-abbrevs.csv")
```

``` python
pop.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
    </tr>
  </tbody>
</table>
</div>

``` python
areas.head()
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
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>52423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>656425</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>114006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>53182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>163707</td>
    </tr>
  </tbody>
</table>
</div>

``` python
abbrevs.head()
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
      <th>state</th>
      <th>abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>

**Given this information, say we want to compute a relatively
straightforward result:rank US states and territories by their 2010
population density**

``` python
merged=pd.merge(pop,abbrevs, how='outer',left_on='state/region',right_on='abbreviation')
```

``` python
merged
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
      <td>AL</td>
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
      <th>2539</th>
      <td>USA</td>
      <td>total</td>
      <td>2010</td>
      <td>309326295.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2540</th>
      <td>USA</td>
      <td>under18</td>
      <td>2011</td>
      <td>73902222.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2541</th>
      <td>USA</td>
      <td>total</td>
      <td>2011</td>
      <td>311582564.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2542</th>
      <td>USA</td>
      <td>under18</td>
      <td>2012</td>
      <td>73708179.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2543</th>
      <td>USA</td>
      <td>total</td>
      <td>2012</td>
      <td>313873685.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2544 rows × 6 columns</p>
</div>

``` python
merged=merged.drop('abbreviation',1)
```

    /tmp/ipykernel_88168/2168094788.py:1: FutureWarning: In a future version of pandas all arguments of DataFrame.drop except for the argument 'labels' will be keyword-only.
      merged=merged.drop('abbreviation',1)

``` python
merged.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# CHeck if there is any mismatch,
merged.isnull().any()
```

    state/region    False
    ages            False
    year            False
    population       True
    state            True
    dtype: bool

``` python
## Some of the population and state info is null, lets check which one
merged[merged['population'].isnull()].head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2448</th>
      <td>PR</td>
      <td>under18</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2449</th>
      <td>PR</td>
      <td>total</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2450</th>
      <td>PR</td>
      <td>total</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2451</th>
      <td>PR</td>
      <td>under18</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2452</th>
      <td>PR</td>
      <td>total</td>
      <td>1993</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

``` python
merged[merged['state'].isnull()].head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2448</th>
      <td>PR</td>
      <td>under18</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2449</th>
      <td>PR</td>
      <td>total</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2450</th>
      <td>PR</td>
      <td>total</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2451</th>
      <td>PR</td>
      <td>under18</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2452</th>
      <td>PR</td>
      <td>total</td>
      <td>1993</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

``` python
merged.loc[merged['state'].isnull(), 'state/region'].unique()
```

    array(['PR', 'USA'], dtype=object)

``` python
# To fix the missing values of PR, USA
merged.loc[merged['state/region']=='PR', 'state']='Puerto Rico'
merged.loc[merged['state/region']=='USA', 'state']='United State'
merged.isnull().any()
```

    state/region    False
    ages            False
    year            False
    population       True
    state           False
    dtype: bool

``` python
#Now lets merged the result wiht the area dataset
final=pd.merge(merged, areas, on='state',how='left')
final.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
  </tbody>
</table>
</div>

``` python
final.isnull().any()
```

    state/region     False
    ages             False
    year             False
    population        True
    state            False
    area (sq. mi)     True
    dtype: bool

``` python
# Lets check the regions which areas is null
final['state'][final['area (sq. mi)'].isnull()].unique()
```

    array(['United State'], dtype=object)

``` python
# No area value for USA; we can either insert it by suming all the areas or just drop it
final.dropna(inplace=True)
final.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
  </tbody>
</table>
</div>

**Now we have all the data we need. To answer the question of interest,
let’s first select the portion of the data corresponding with the year
2000, and the total population. We’ll use the query() function to do
this quickly (this requires the numexpr package to be installed;**

``` python
data2000=final.query("year==2000 & ages=='total'")
```

``` python
data2000.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>AL</td>
      <td>total</td>
      <td>2000</td>
      <td>4452173.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>AK</td>
      <td>total</td>
      <td>2000</td>
      <td>627963.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>AZ</td>
      <td>total</td>
      <td>2000</td>
      <td>5160586.0</td>
      <td>Arizona</td>
      <td>114006.0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>AR</td>
      <td>total</td>
      <td>2000</td>
      <td>2678588.0</td>
      <td>Arkansas</td>
      <td>53182.0</td>
    </tr>
    <tr>
      <th>220</th>
      <td>CA</td>
      <td>total</td>
      <td>2000</td>
      <td>33987977.0</td>
      <td>California</td>
      <td>163707.0</td>
    </tr>
  </tbody>
</table>
</div>

Now let’s compute the population density and display it in order. We’ll
start by rein‐ dexing our data on the state, and then compute the
result:

``` python
data2000.set_index('state', inplace=True)
density=data2000['population']/data2000['area (sq. mi)']
```

``` python
density.sort_values(ascending=False, inplace=True)
density.head()
```

    state
    District of Columbia    8412.441176
    Puerto Rico             1084.098151
    New Jersey               966.592639
    Rhode Island             679.785113
    Connecticut              615.399892
    dtype: float64

``` python
density.tail()
```

    state
    South Dakota    9.800755
    North Dakota    9.080434
    Montana         6.146192
    Wyoming         5.053262
    Alaska          0.956641
    dtype: float64
