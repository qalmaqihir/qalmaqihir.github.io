================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# Example: Surface Temperature Data
------------------------------------------------------------------------

- <a href="#example-surface-temperature-data"
  id="toc-example-surface-temperature-data">Example: Surface Temperature
  Data</a>
  - <a href="#unfortunately-no-dataset-found"
    id="toc-unfortunately-no-dataset-found">Unfortunately, no dataset found
    !</a>

------------------------------------------------------------------------

# Example: Surface Temperature Data

As an example of visualizing some more continuous geographic data, let’s
consider the “polar vortex” that hit the eastern half of the United
States in January 2014. A great source for any sort of climatic data is
NASA’s Goddard Institute for Space Stud‐ ies. Here we’ll use the GIS 250
temperature data, which we can download using shell commands (these
commands may have to be modified on Windows machines). The data used
here was downloaded on 6/12/2016, and the file size is approximately 9
MB:

``` python
!curl -O http://data.giss.nasa.gov/pub/gistemp/gistemp250.nc.gz
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   263  100   263    0     0    468      0 --:--:-- --:--:-- --:--:--   467

``` python
!gunzip gistemp250.nc.gz
```


    gzip: gistemp250.nc.gz: not in gzip format

The data comes in NetCDF format, which can be read in Python by the
netCDF4 library. You can install this library as shown here:

``` python
# ! conda install netcdf4
```

``` python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
```

## Unfortunately, no dataset found !

``` python
from netCDF4 import Dataset
data = Dataset('gistemp250.nc')
```

``` python
from netCDF4 import date2index
from datetime import datetime
timeindex = date2index(datetime(2014, 1, 15),
data.variables['time'])
```

``` python
lat = data.variables['lat'][:]

lon = data.variables['lon'][:]

lon, lat = np.meshgrid(lon, lat)

temp_anomaly = data.variables['tempanomaly'][timeindex]
```

``` python
fig = plt.figure(figsize=(10, 8))
m = Basemap(projection='lcc', resolution='c',
            width=8E6, height=8E6,
            lat_0=45, lon_0=-100,)

m.shadedrelief(scale=0.5)

m.pcolormesh(lon, lat, temp_anomaly,
             latlon=True, cmap='RdBu_r')

plt.clim(-8, 8)

m.drawcoastlines(color='lightgray')

plt.title('January 2014 Temperature Anomaly')

plt.colorbar(label='temperature anomaly (°C)');
```
