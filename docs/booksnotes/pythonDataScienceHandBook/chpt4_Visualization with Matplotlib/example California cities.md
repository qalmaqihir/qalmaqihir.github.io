Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# Example: California Cities
------------------------------------------------------------------------

- <a href="#example-california-cities"
  id="toc-example-california-cities">Example: California Cities</a>

------------------------------------------------------------------------

# Example: California Cities

Recall that in “Customizing Plot Legends”, we demonstrated the use of
size and color in a scatter plot to convey information about the
location, size, and population of California cities. Here, we’ll create
this plot again, but using Basemap to put the data in context.

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
```

``` python

cities=pd.read_csv('../data/california_cities.csv')
cities.head()
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
      <th>Unnamed: 0</th>
      <th>city</th>
      <th>latd</th>
      <th>longd</th>
      <th>elevation_m</th>
      <th>elevation_ft</th>
      <th>population_total</th>
      <th>area_total_sq_mi</th>
      <th>area_land_sq_mi</th>
      <th>area_water_sq_mi</th>
      <th>area_total_km2</th>
      <th>area_land_km2</th>
      <th>area_water_km2</th>
      <th>area_water_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Adelanto</td>
      <td>34.576111</td>
      <td>-117.432778</td>
      <td>875.0</td>
      <td>2871.0</td>
      <td>31765</td>
      <td>56.027</td>
      <td>56.009</td>
      <td>0.018</td>
      <td>145.107</td>
      <td>145.062</td>
      <td>0.046</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>AgouraHills</td>
      <td>34.153333</td>
      <td>-118.761667</td>
      <td>281.0</td>
      <td>922.0</td>
      <td>20330</td>
      <td>7.822</td>
      <td>7.793</td>
      <td>0.029</td>
      <td>20.260</td>
      <td>20.184</td>
      <td>0.076</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Alameda</td>
      <td>37.756111</td>
      <td>-122.274444</td>
      <td>NaN</td>
      <td>33.0</td>
      <td>75467</td>
      <td>22.960</td>
      <td>10.611</td>
      <td>12.349</td>
      <td>59.465</td>
      <td>27.482</td>
      <td>31.983</td>
      <td>53.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Albany</td>
      <td>37.886944</td>
      <td>-122.297778</td>
      <td>NaN</td>
      <td>43.0</td>
      <td>18969</td>
      <td>5.465</td>
      <td>1.788</td>
      <td>3.677</td>
      <td>14.155</td>
      <td>4.632</td>
      <td>9.524</td>
      <td>67.28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Alhambra</td>
      <td>34.081944</td>
      <td>-118.135000</td>
      <td>150.0</td>
      <td>492.0</td>
      <td>83089</td>
      <td>7.632</td>
      <td>7.631</td>
      <td>0.001</td>
      <td>19.766</td>
      <td>19.763</td>
      <td>0.003</td>
      <td>0.01</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# extract the data we're interested in 
lat=cities['latd'].values
lon=cities['longd'].values
population=cities['population_total'].values
area=cities['area_total_km2'].values
```

``` python
# 1. Draw the map background
fig=plt.figure(figsize=(8,8))
m=Basemap(projection='lcc', resolution='h',
         lat_0=37.5, lon_0=-119,
         width=1E6, height=1.2E6)

m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population and size reflecting area
m.scatter(lon, lat, latlon=True,
         c=np.log10(population),s=area,
         cmap='Reds',alpha=0.5)

# 3. Create colobar and legned
plt.colorbar(label=r'$log_{10}({\rm population})$')
plt.clim(3,7)


# 4. make legend with dummy points
for a in [100,300,500]:
    plt.scatter([],[],c='k',alpha=0.5,s=a,label=str(a)+'km$^2$')

plt.legend(scatterpoints=1, frameon=False,
          labelspacing=1, loc='lower left')

```

    <matplotlib.legend.Legend at 0x7f0308093b50>

![](example%20California%20cities_files/figure-gfm/cell-5-output-2.png)
