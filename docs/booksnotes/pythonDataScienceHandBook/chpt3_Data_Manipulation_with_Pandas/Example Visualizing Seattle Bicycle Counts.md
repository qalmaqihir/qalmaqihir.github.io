Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# Example: Visualzing Seattle Bucycle Counts
------------------------------------------------------------------------

- <a href="#example-visualizing-seattle-bicycle-counts"
  id="toc-example-visualizing-seattle-bicycle-counts">Example: Visualizing
  Seattle Bicycle Counts</a>
  - <a href="#visualizing-the-data"
    id="toc-visualizing-the-data">Visualizing the Data</a>
    - <a href="#digging-into-the-data" id="toc-digging-into-the-data">Digging
      into the data</a>

------------------------------------------------------------------------

# Example: Visualizing Seattle Bicycle Counts

As a more involved example of working with some time series data, let’s
take a look at bicycle counts on Seattle’s Fremont Bridge. This data
comes from an automated bicy‐ cle counter, installed in late 2012, which
has inductive sensors on the east and west sidewalks of the bridge.

``` python
!curl -o FremontBridge.csv https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 2679k    0 2679k    0     0   170k      0 --:--:--  0:00:15 --:--:--  296k

``` python
import pandas as pd
import numpy as np
```

``` python
data=pd.read_csv('FremontBridge.csv',index_col='Date',parse_dates=True)
data.head()
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
      <th>Fremont Bridge Total</th>
      <th>Fremont Bridge East Sidewalk</th>
      <th>Fremont Bridge West Sidewalk</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-10-03 00:00:00</th>
      <td>13.0</td>
      <td>4.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2012-10-03 01:00:00</th>
      <td>10.0</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2012-10-03 02:00:00</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2012-10-03 03:00:00</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2012-10-03 04:00:00</th>
      <td>7.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data.columns=['West','East','Total']
data['Total']=data.eval('West + East')
```

``` python
data.dropna().describe()
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
      <th>West</th>
      <th>East</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>86122.000000</td>
      <td>86122.000000</td>
      <td>86122.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>106.798449</td>
      <td>47.996238</td>
      <td>154.794687</td>
    </tr>
    <tr>
      <th>std</th>
      <td>134.926536</td>
      <td>61.795993</td>
      <td>192.517894</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.000000</td>
      <td>6.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>59.000000</td>
      <td>27.000000</td>
      <td>86.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>143.000000</td>
      <td>66.000000</td>
      <td>209.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1097.000000</td>
      <td>698.000000</td>
      <td>1569.000000</td>
    </tr>
  </tbody>
</table>
</div>

## Visualizing the Data

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set();
```

``` python
data.plot()
plt.ylabel('Hourly Bicycle Count');
```

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-9-output-1.png)

``` python
weekly = data.resample('W').sum()
weekly.plot(style=[':','--','-'])
plt.ylabel('Weekly bicycle count')
```

    Text(0, 0.5, 'Weekly bicycle count')

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-10-output-2.png)

**people bicycle more in the summer than in the winter, and even within
a particular season the bicy‐ cle use varies from week to week (likely
dependent on weather**

``` python
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':','--','-'])
plt.ylabel('Mean hourly count')
```

    Text(0, 0.5, 'Mean hourly count')

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-11-output-2.png)

**The jaggedness of the result is due to the hard cutoff of the window.
We can get a smoother version of a rolling mean using a window
function—for example, a Gaus‐ sian window.**

``` python
daily.rolling(50,center=True, win_type='gaussian').sum(std=10).plot(style=[':','--','-'])
```

    <AxesSubplot:xlabel='Date'>

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-12-output-2.png)

### Digging into the data

While the smoothed data views above are useful to get an idea of the
general trend in the data, they hide much of the interesting structure.
For example, we might want to look at the average traffic as a function
of the time of day. We can do this using the GroupBy functionality

``` python
by_time=data.groupby(data.index.time).mean()
hourly_ticks=4*60*np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':','--','-'])
```

    <AxesSubplot:xlabel='time'>

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-13-output-2.png)

``` python
by_weekday=data.groupby(data.index.day_of_week).mean()
by_weekday.index=['Mon','Tues','Wed','Thr','Fri','Sat','Sun']
by_weekday.plot(style=[':','--','-'])
```

    <AxesSubplot:>

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-14-output-2.png)

This shows a strong distinction between weekday and weekend totals, with
around twice as many average riders crossing the bridge on Monday
through Friday than on Saturday and Sunday. With this in mind, let’s do
a compound groupby and look at the hourly trend on weekdays versus
weekends. We’ll start by grouping by both a flag marking the week‐ end,
and the time of day:

``` python
weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()
```

``` python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays',
xticks=hourly_ticks, style=[':', '--', '-'])
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends',
xticks=hourly_ticks, style=[':', '--', '-']);
```

![](Example%20Visualizing%20Seattle%20Bicycle%20Counts_files/figure-gfm/cell-16-output-1.png)
