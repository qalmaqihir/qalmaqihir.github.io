Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 13 - Resampling, Shifting and Windowing
------------------------------------------------------------------------

- <a href="#resampling-shifting-and-windowing"
  id="toc-resampling-shifting-and-windowing">Resampling, Shifting, and
  Windowing</a>
  - <a href="#resampling-and-converting-frequencies"
    id="toc-resampling-and-converting-frequencies">Resampling and converting
    frequencies</a>
  - <a href="#time-shifts" id="toc-time-shifts">Time-shifts</a>
  - <a href="#rolling-windows" id="toc-rolling-windows">Rolling windows</a>

------------------------------------------------------------------------

# Resampling, Shifting, and Windowing

The ability to use dates and times as indices to intuitively organize
and access data is an important piece of the Pandas time series tools.
The benefits of indexed data in general (automatic alignment during
operations, intuitive data slicing and access, etc.) still apply, and
Pandas provides several additional time series–specific operations.

``` python
!conda install pandas-datareader -y
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done

    ## Package Plan ##

      environment location: /home/qalmaqihir/anaconda3

      added / updated specs:
        - pandas-datareader


    The following packages will be downloaded:

        package                    |            build
        ---------------------------|-----------------
        conda-4.14.0               |   py39h06a4308_0         915 KB
        pandas-datareader-0.10.0   |     pyhd3eb1b0_0          71 KB
        ------------------------------------------------------------
                                               Total:         987 KB

    The following NEW packages will be INSTALLED:

      pandas-datareader  pkgs/main/noarch::pandas-datareader-0.10.0-pyhd3eb1b0_0

    The following packages will be UPDATED:

      conda                               4.13.0-py39h06a4308_0 --> 4.14.0-py39h06a4308_0



    Downloading and Extracting Packages
    pandas-datareader-0. | 71 KB     | ##################################### | 100% 
    conda-4.14.0         | 915 KB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done

``` python
import pandas as pd
```

``` python
from pandas_datareader import data
```

``` python
google = data.DataReader('GOOG',start='2004',end='2017', data_source='yahoo')
```

``` python
google.head()
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
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2004-08-19</th>
      <td>2.591785</td>
      <td>2.390042</td>
      <td>2.490664</td>
      <td>2.499133</td>
      <td>897427216.0</td>
      <td>2.499133</td>
    </tr>
    <tr>
      <th>2004-08-20</th>
      <td>2.716817</td>
      <td>2.503118</td>
      <td>2.515820</td>
      <td>2.697639</td>
      <td>458857488.0</td>
      <td>2.697639</td>
    </tr>
    <tr>
      <th>2004-08-23</th>
      <td>2.826406</td>
      <td>2.716070</td>
      <td>2.758411</td>
      <td>2.724787</td>
      <td>366857939.0</td>
      <td>2.724787</td>
    </tr>
    <tr>
      <th>2004-08-24</th>
      <td>2.779581</td>
      <td>2.579581</td>
      <td>2.770615</td>
      <td>2.611960</td>
      <td>306396159.0</td>
      <td>2.611960</td>
    </tr>
    <tr>
      <th>2004-08-25</th>
      <td>2.689918</td>
      <td>2.587302</td>
      <td>2.614201</td>
      <td>2.640104</td>
      <td>184645512.0</td>
      <td>2.640104</td>
    </tr>
  </tbody>
</table>
</div>

``` python
google.Close
```

    Date
    2004-08-19     2.499133
    2004-08-20     2.697639
    2004-08-23     2.724787
    2004-08-24     2.611960
    2004-08-25     2.640104
                    ...    
    2016-12-23    39.495499
    2016-12-27    39.577499
    2016-12-28    39.252499
    2016-12-29    39.139500
    2016-12-30    38.591000
    Name: Close, Length: 3115, dtype: float64

``` python
#google['Close']
```

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
```

``` python
google['Close'].plot();
```

![](13_Resampling%20Shifting%20and%20Windowing_files/figure-gfm/cell-11-output-1.png)

### Resampling and converting frequencies

One common need for time series data is resampling at a higher or lower
frequency. You can do this using the resample() method, or the much
simpler asfreq() method. The primary difference between the two is that
resample() is fundamentally a data aggregation, while asfreq() is
fundamentally a data selection.

``` python
google["Close"].plot(alpha=0.5, style='-')
google["Close"].resample('BA').mean().plot(style=':')
google["Close"].asfreq('BA').plot(style='--');
plt.legend(['input','resample','asfreq'],loc='upper left')
```

    <matplotlib.legend.Legend at 0x7fe1d96367c0>

![](13_Resampling%20Shifting%20and%20Windowing_files/figure-gfm/cell-12-output-2.png)

``` python
goog=google['Close']
```

``` python
fig, ax = plt.subplots(2, sharex=True)
data = goog.iloc[:10]
data.asfreq('D').plot(ax=ax[0], marker='o')
data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
ax[1].legend(["back-fill", "forward-fill"]);
```

![](13_Resampling%20Shifting%20and%20Windowing_files/figure-gfm/cell-14-output-1.png)

### Time-shifts

Another common time series–specific operation is shifting of data in
time. Pandas has two closely related methods for computing this: shift()
and tshift(). In short, the difference between them is that shift()
shifts the data, while tshift() shifts the index. In both cases, the
shift is specified in multiples of the frequenc

``` python
fig, ax = plt.subplots(3, sharey=True)
# apply a frequency to the data
goog = goog.asfreq('D', method='pad')
goog.plot(ax=ax[0])
goog.shift(900).plot(ax=ax[1])
goog.tshift(900).plot(ax=ax[2])
# legends and annotations
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')
ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=0.3, color='red')
ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
ax[1].axvline(local_max + offset, alpha=0.3, color='red')
ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
ax[2].axvline(local_max + offset, alpha=0.3, color='red');
```

    /tmp/ipykernel_66268/1389856076.py:6: FutureWarning: tshift is deprecated and will be removed in a future version. Please use shift instead.
      goog.tshift(900).plot(ax=ax[2])

![](13_Resampling%20Shifting%20and%20Windowing_files/figure-gfm/cell-15-output-2.png)

``` python
ROI = 100 * (goog.tshift(-365) / goog - 1)
ROI.plot()
plt.ylabel('% Return on Investment');
```

    /tmp/ipykernel_66268/2632432407.py:1: FutureWarning: tshift is deprecated and will be removed in a future version. Please use shift instead.
      ROI = 100 * (goog.tshift(-365) / goog - 1)

![](13_Resampling%20Shifting%20and%20Windowing_files/figure-gfm/cell-16-output-2.png)

### Rolling windows

Rolling statistics are a third type of time series–specific operation
implemented by Pandas. These can be accomplished via the rolling()
attribute of Series and Data Frame objects, which returns a view similar
to what we saw with the groupby opera‐ tion

``` python
rolling = goog.rolling(365, center=True)
data = pd.DataFrame({'input': goog,
'one-year rolling_mean': rolling.mean(),
'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)
```

![](13_Resampling%20Shifting%20and%20Windowing_files/figure-gfm/cell-17-output-1.png)
