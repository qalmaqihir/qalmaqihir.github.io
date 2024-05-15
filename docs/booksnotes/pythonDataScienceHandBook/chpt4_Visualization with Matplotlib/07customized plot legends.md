================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 07 -  Customizing Plot Legends
------------------------------------------------------------------------

- <a href="#customizing-plot-legends"
  id="toc-customizing-plot-legends">Customizing Plot Legends</a>
  - <a href="#choosing-elements-for-the-legend"
    id="toc-choosing-elements-for-the-legend">Choosing Elements for the
    Legend</a>
  - <a href="#legend-for-size-of-points"
    id="toc-legend-for-size-of-points">Legend for Size of Points</a>
  - <a href="#multiple-legends" id="toc-multiple-legends">Multiple
    Legends</a>
------------------------------------------------------------------------

# Customizing Plot Legends

Plot legends give meaning to a visualization, assigning labels to the
various plot ele‐ ments. We previously saw how to create a simple
legend; here we’ll take a look at cus‐ tomizing the placement and
aesthetics of the legend in Matplotlib. The simplest legend can be
created with the plt.legend() command, which auto‐ matically creates a
legend for any labeled plot elements

``` python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
```

``` python
x=np.linspace(0,10,10000)
fig,ax=plt.subplots()
ax.plot(x, np.sin(x),'-b',label='Sine')
ax.plot(x, np.cos(x),'--r',label='Cosine')

ax.axis('equal')
leg=ax.legend();
```

![](07customized%20plot%20legends_files/figure-gfm/cell-3-output-1.png)

``` python
#many wasy we can customize the legend,
# we can specify the location and turn off the frame 
ax.legend(loc='upper left', frameon=False)
fig
```

![](07customized%20plot%20legends_files/figure-gfm/cell-4-output-1.png)

``` python
#We can use the ncol command to specify the number of columns in the legend

ax.legend(frameon=False, loc='lower center',ncol=2)
fig
```

![](07customized%20plot%20legends_files/figure-gfm/cell-5-output-1.png)

``` python
# We can use a rounded box (fancybox) or add a shadow, change the transparency
#(alpha value) of the frame, or change the padding around the text
ax.legend(fancybox=True, framealpha=0.8, shadow=True, borderpad=0.5)
fig
```

![](07customized%20plot%20legends_files/figure-gfm/cell-6-output-1.png)

## Choosing Elements for the Legend

As we’ve already seen, the legend includes all labeled elements by
default. If this is not what is desired, we can fine-tune which elements
and labels appear in the legend by using the objects returned by plot
commands. The plt.plot() command is able to create multiple lines at
once, and returns a list of created line instances. Passing any of these
to plt.legend() will tell it which to identify, along with the labels
we’d like to specify

``` python
y=np.sin(x[:,np.newaxis]+np.pi*np.arange(0,2,0.5))
lines=plt.plot(x,y)

# line is a list of plt.line2D instance
plt.legend(lines[:2],['first','second'])
```

    <matplotlib.legend.Legend at 0x7fba9d7a2c10>

![](07customized%20plot%20legends_files/figure-gfm/cell-7-output-2.png)

**I generally find in practice that it is clearer to use the first
method, applying labels to the plot elements you’d like to show on the
legend**

``` python
plt.plot(x,y[:,0],label='first')
plt.plot(x,y[:,1],label='second')
plt.plot(x,y[:,2:])
plt.legend(framealpha=1, frameon=True);
```

![](07customized%20plot%20legends_files/figure-gfm/cell-8-output-1.png)

## Legend for Size of Points

Sometimes the legend defaults are not sufficient for the given
visualization. For exam‐ ple, perhaps you’re using the size of points to
mark certain features of the data, and want to create a legend
reflecting this. Here is an example where we’ll use the size of points
to indicate populations of California cities. We’d like a legend that
specifies the scale of the sizes of the points, and we’ll accomplish
this by plotting some labeled data with no entries

``` python
import pandas as pd
cities=pd.read_csv('../data/california_cities.csv')
```

``` python
# extract the data we'er interestd in
lat, lon = cities['latd'], cities['longd']
population, area=cities['population_total'], cities['area_total_km2']
```

``` python
# scatter the points, using size and color but no label
plt.scatter(lon,lat,label=None,
           c=np.log10(population), cmap='viridis',
           s=area,linewidth=0, alpha=0.5)

#plt.axis(aspect='equal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3,7)
```

![](07customized%20plot%20legends_files/figure-gfm/cell-11-output-1.png)

``` python
# scatter the points, using size and color but no label
plt.scatter(lon,lat,label=None,
           c=np.log10(population), cmap='viridis',
           s=area,linewidth=0, alpha=0.5)

#plt.axis(aspect='equal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3,7)

# here we created a legend:
# we'll plot empty lists with the desired size and label
for area in [100,300,500]:
    plt.scatter([],[],c='k',alpha=0.3,s=area,
               label=str(area)+' km$^2$')
plt.legend(scatterpoints=1, frameon=False,
              title='City Area')
plt.title('California cities: Area and Population')    
```

    Text(0.5, 1.0, 'California cities: Area and Population')

![](07customized%20plot%20legends_files/figure-gfm/cell-12-output-2.png)

## Multiple Legends

Sometimes when designing a plot you’d like to add multiple legends to
the same axes. Unfortunately, Matplotlib does not make this easy: via
the standard legend interface, it is only possible to create a single
legend for the entire plot. If you try to create a second legend using
plt.legend() or ax.legend(), it will simply override the first one. We
can work around this by creating a new legend artist from scratch, and
then using the lower-level ax.add_artist() method to manually add the
second artist to the plot

``` python
fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)
for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color='black')
ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
loc='upper right', frameon=False)

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
loc='lower right', frameon=False)
ax.add_artist(leg);
```

![](07customized%20plot%20legends_files/figure-gfm/cell-13-output-1.png)
