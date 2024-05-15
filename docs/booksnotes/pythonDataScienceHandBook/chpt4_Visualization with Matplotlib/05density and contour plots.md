Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 05 -  Density and Contour Plots
------------------------------------------------------------------------
- <a href="#density-and-contour-plots"
  id="toc-density-and-contour-plots">Density and Contour Plots</a>
  - <a href="#visualizing-a-three-dimensional-function"
    id="toc-visualizing-a-three-dimensional-function">Visualizing a
    Three-Dimensional Function</a>
------------------------------------------------------------------------

# Density and Contour Plots

Sometimes it is useful to display three-dimensional data in two
dimensions using contours or color-coded regions. There are three
Matplotlib functions that can be helpful for this task: plt.contour for
contour plots, plt.contourf for filled contour plots, and plt.imshow for
showing images. This section looks at several examples of using these.

``` python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
```

## Visualizing a Three-Dimensional Function

We’ll start by demonstrating a contour plot using a function z = f x, y
, using the fol‐ lowing particular choice for f

``` python
def f(x,y):
    return np.sin(x)**10 + np.cos(10+y*x)*np.cos(x)

```

**A contour plot can be created with the plt.contour function. It takes
three argu‐ ments: a grid of x values, a grid of y values, and a grid of
z values. The x and y values represent positions on the plot, and the z
values will be represented by the contour levels. Perhaps the most
straightforward way to prepare such data is to use the np.meshgrid
function, which builds two-dimensional grids from one-dimensional
arrays:**

``` python
x=np.linspace(0,5,50)
y=np.linspace(0,5,40)

X,Y = np.meshgrid(x,y)
```

``` python
Z=f(X,Y)
```

``` python
plt.contour(X,Y,Z, colors='black')
```

    <matplotlib.contour.QuadContourSet at 0x7f5d78321eb0>

![](05density%20and%20contour%20plots_files/figure-gfm/cell-6-output-2.png)

Notice that by default when a single color is used, negative values are
represented by dashed lines, and positive values by solid lines.
Alternatively, you can color-code the lines by specifying a colormap
with the cmap argument. Here, we’ll also specify that we want more lines
to be drawn—20 equally spaced intervals within the data range

``` python
plt.contour(X,Y,Z,20, cmap='RdGy')
```

    <matplotlib.contour.QuadContourSet at 0x7f5d78a0adc0>

![](05density%20and%20contour%20plots_files/figure-gfm/cell-7-output-2.png)

Here we chose the RdGy (short for Red-Gray) colormap, which is a good
choice for centered data. Matplotlib has a wide range of colormaps
available.

**Our plot is looking nicer, but the spaces between the lines may be a
bit distracting. We can change this by switching to a filled contour
plot using the `plt.contourf()` function (notice the f at the end),
which uses largely the same syntax as `plt.contour()`.Additionally,
we’ll add a `plt.colorbar()` command, which automatically creates an
additional axis with labeled color information for the plot**

``` python
plt.contourf(X,Y,Z,20,cmap='RdGy')
plt.colorbar()
```

    <matplotlib.colorbar.Colorbar at 0x7f5d7847d190>

![](05density%20and%20contour%20plots_files/figure-gfm/cell-8-output-2.png)

*The colorbar makes it clear that the black regions are “peaks,” while
the red regions are “valleys.” One potential issue with this plot is
that it is a bit “splotchy.” That is, the color steps are discrete
rather than continuous, which is not always what is desired. You could
remedy this by setting the number of contours to a very high number, but
this results in a rather inefficient plot: Matplotlib must render a new
polygon for each step in the level. A better way to handle this is to
use the plt.imshow() function, which inter‐ prets a two-dimensional grid
of data as an image.*

``` python
plt.imshow(Z, extent=[0,5,0,5],origin='lower',
          cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image')
```

    TypeError: axis() got an unexpected keyword argument 'aspect'

![](05density%20and%20contour%20plots_files/figure-gfm/cell-9-output-2.png)

There are a few potential gotchas with imshow(), however:  
• `plt.imshow()` doesn’t accept an x and y grid, so you must manually
specify the extent `[xmin, xmax, ymin, ymax]` of the image on the plot.

• `plt.imshow()` by default follows the standard image array definition
where the origin is in the upper left, not in the lower left as in most
contour plots. This must be changed when showing gridded data.

• `plt.imshow()` will automatically adjust the axis aspect ratio to
match the input data; you can change this by setting, for example,
plt.axis(aspect=‘image’) to make x and y units match.

``` python
# A combined contours plots and image plots
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
cmap='RdGy', alpha=0.5)
plt.colorbar();
```

![](05density%20and%20contour%20plots_files/figure-gfm/cell-10-output-1.png)
