================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# Example: Exploring Marathon Finishing Times
------------------------------------------------------------------------


- <a href="#example-exploring-marathon-finishing-times"
  id="toc-example-exploring-marathon-finishing-times">Example: Exploring
  Marathon Finishing Times</a>

------------------------------------------------------------------------

# Example: Exploring Marathon Finishing Times

Here we’ll look at using Seaborn to help visualize and understand
finishing results from a marathon. I’ve scraped the data from sources on
the Web, aggregated it and removed any identifying information, and put
it on GitHub where it can be downloa‐ ded (if you are interested in
using Python for web scraping, I would recommend Web Scraping with
Python by Ryan Mitchell). We will start by downloading the data from the
Web, and loading it into Pandas:

``` python
# !curl -O https://raw.githubusercontent.com/jakevdp/marathon-data/master/marathon-data.csv
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  836k  100  836k    0     0   9808      0  0:01:27  0:01:27 --:--:-- 11084 13626      0  0:01:02  0:00:11  0:00:51 10234

``` python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd
```

``` python
!ls
```

    '01general Matplotlib tips.ipynb'
     02simple_lineplots.ipynb
    '03simple scatter plots.ipynb'
    '04visualizing errors.ipynb'
    '05density and contour plots.ipynb'
    '06Histograms Binnings and Density.ipynb'
    '07customized plot legends.ipynb'
    '08customizing colorbar.ipynb'
    '09multiple subplots.ipynb'
    '10text and annotation Example.ipynb'
    '11customizing ticks.ipynb'
    '12customizing matplotlib configuration and stylesheets.ipynb'
    '13threedimensional plotting.ipynb'
    '14_geographic data with basemap.ipynb'
    '15visualiztion with seaborn.ipynb'
     cos_sinplots.png
    'example California cities.ipynb'
    'Example Exploring Marathon Finishing times.ipynb'
    'Example Handwritten Digits.ipynb'
    'example surface temperature data.ipynb'
    'Example Visualizing a Mobius Strip.ipynb'
     gistemp250.nc.gz
     marathon-data.csv

``` python
data = pd.read_csv('marathon-data.csv')
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
      <th>age</th>
      <th>gender</th>
      <th>split</th>
      <th>final</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>M</td>
      <td>01:05:38</td>
      <td>02:08:51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>M</td>
      <td>01:06:26</td>
      <td>02:09:28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>M</td>
      <td>01:06:49</td>
      <td>02:10:42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>M</td>
      <td>01:06:16</td>
      <td>02:13:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>M</td>
      <td>01:06:32</td>
      <td>02:13:59</td>
    </tr>
  </tbody>
</table>
</div>

``` python
data.dtypes
```

    age        int64
    gender    object
    split     object
    final     object
    dtype: object

``` python
# lets convert split and final to times
def convert_time(s):
    h,m,s=map(int,s.split(':'))
    return pd.datetools.timedelta(hours=h, minutes=m, seconds=s)
```

``` python
data = pd.read_csv('marathon-data.csv',converters={'split':convert_time, 'final':convert_time})
data.head()
```

    AttributeError: module 'pandas' has no attribute 'datetools'
