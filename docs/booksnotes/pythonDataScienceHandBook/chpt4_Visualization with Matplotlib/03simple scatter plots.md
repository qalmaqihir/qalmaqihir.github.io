Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 03 -  Simple Scatter Plots
------------------------------------------------------------------------

- <a href="#simple-scatter-plots" id="toc-simple-scatter-plots">Simple
  Scatter Plots</a>
  - <a href="#scatter-plots-with-plt.plot"
    id="toc-scatter-plots-with-plt.plot">Scatter Plots with plt.plot</a>
  - <a href="#scatter-plots-with-plt.scatter"
    id="toc-scatter-plots-with-plt.scatter">Scatter Plots with
    plt.scatter</a>
------------------------------------------------------------------------

# Simple Scatter Plots

Another commonly used plot type is the simple scatter plot, a close
cousin of the line plot. Instead of points being joined by line
segments, here the points are represented individually with a dot,
circle, or other shape.

``` python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

## Scatter Plots with plt.plot

In the previous section, we looked at plt.plot/ax.plot to produce line
plots. It turns out that this same function can produce scatter plots as
well

``` python
x=np.linspace(0,10,30)
y=np.sin(x)
```

``` python
plt.plot(x,y,'o',color='black');
```

![](03simple%20scatter%20plots_files/figure-gfm/cell-4-output-1.png)

``` python
# possible markers are
rng=np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="marker='{0}'".format(marker))

plt.legend(numpoints=1)
plt.xlim(0,1.8);
```

![](03simple%20scatter%20plots_files/figure-gfm/cell-5-output-1.png)

For even more possibilities, these character codes can be used together
with line and color codes to plot points along with a line connecting
them

``` python
plt.plot(x,y,'-*r');
```

![](03simple%20scatter%20plots_files/figure-gfm/cell-6-output-1.png)

Additional keyword arguments to plt.plot specify a wide range of
properties of the lines and markers

``` python
plt.plot(x, y, '-p', color='gray',
markersize=15, linewidth=4,
markerfacecolor='white',
markeredgecolor='gray',
markeredgewidth=2)
plt.ylim(-1.2, 1.2)
```

    (-1.2, 1.2)

![](03simple%20scatter%20plots_files/figure-gfm/cell-7-output-2.png)

## Scatter Plots with plt.scatter

A second, more powerful method of creating scatter plots is the
plt.scatter func‐ tion, which can be used very similarly to the plt.plot
function

``` python
plt.scatter(x,y,marker='o')
```

    <matplotlib.collections.PathCollection at 0x7fe4dcc12040>

![](03simple%20scatter%20plots_files/figure-gfm/cell-8-output-2.png)

**The primary difference of plt.scatter from plt.plot is that it can be
used to create scatter plots where the properties of each individual
point (size, face color, edge color, etc.) can be individually
controlled or mapped to data.**

``` python
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
cmap='viridis')
plt.colorbar(); # show color scale
```

    /tmp/ipykernel_61416/429789784.py:8: MatplotlibDeprecationWarning: Auto-removal of grids by pcolor() and pcolormesh() is deprecated since 3.5 and will be removed two minor releases later; please call grid(False) first.
      plt.colorbar(); # show color scale

![](03simple%20scatter%20plots_files/figure-gfm/cell-9-output-2.png)

Notice that the color argument is automatically mapped to a color scale
(shown here by the colorbar() command), and the size argument is given
in pixels. In this way, the color and size of points can be used to
convey information in the visualization, in order to illustrate
multidimensional data.

**For example, we might use the Iris data from Scikit-Learn, where each
sample is one of three types of flowers that has had the size of its
petals and sepals carefully meas‐ ured**

``` python
from sklearn.datasets import load_iris
iris=load_iris()
features=iris.data.T
```

``` python
plt.scatter(features[0], feaatures[1],alpha=0.2,
           s=100*features[3],c=iris.target,cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);
```

![](03simple%20scatter%20plots_files/figure-gfm/cell-11-output-1.png)
