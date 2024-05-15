================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 06 -  Histograms, Binnings, and Density
------------------------------------------------------------------------

- <a href="#histograms-binnings-and-density"
  id="toc-histograms-binnings-and-density">Histograms, Binnings, and
  Density</a>
  - <a href="#two-dimensional-histograms-and-binnings"
    id="toc-two-dimensional-histograms-and-binnings">Two-Dimensional
    Histograms and Binnings</a>
    - <a href="#plt.hexbin-hexagonal-binnings"
      id="toc-plt.hexbin-hexagonal-binnings">plt.hexbin: Hexagonal
      binnings</a>
    - <a href="#kernel-density-estimation"
      id="toc-kernel-density-estimation">Kernel density estimation</a>

------------------------------------------------------------------------

# Histograms, Binnings, and Density

A simple histogram can be a great first step in understanding a dataset.
Earlier, we saw a preview of Matplotlib’s histogram function

``` python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
```

``` python
data=np.random.randn(1000)
plt.hist(data);
```

![](06Histograms%20Binnings%20and%20Density_files/figure-gfm/cell-3-output-1.png)

``` python
# A more customized histogram
plt.hist(data, bins=30, alpha=0.5,
        histtype='stepfilled', color='steelblue',
        edgecolor='none');
```

![](06Histograms%20Binnings%20and%20Density_files/figure-gfm/cell-4-output-1.png)

**The plt.hist docstring has more information on other customization
options avail‐ able. I find this combination of histtype=‘stepfilled’
along with some transpar‐ ency alpha to be very useful when comparing
histograms of several distributions**

``` python
x1 = np.random.normal(0,0.8,1000)
x2 = np.random.normal(-2,1,1000)
x3 = np.random.normal(2,3,1000)
kwargs=dict(histtype='stepfilled',alpha=0.3, bins=40)

plt.hist(x1,**kwargs);
plt.hist(x2,**kwargs);
plt.hist(x3,**kwargs);
```

![](06Histograms%20Binnings%20and%20Density_files/figure-gfm/cell-5-output-1.png)

***If you would like to simply compute the histogram (that is, count the
number of points in a given bin) and not display it, the np.histogram()
function is available:***

``` python
counts, bin_edges=np.histogram(data,bins=5)
counts
```

    array([ 17, 292, 555, 132,   4])

## Two-Dimensional Histograms and Binnings

Just as we create histograms in one dimension by dividing the number
line into bins, we can also create histograms in two dimensions by
dividing points among two- dimensional bins. We’ll take a brief look at
several ways to do this here. We’ll start by defining some data—an x and
y array drawn from a multivariate Gaussian distribution:

``` python
mean=[0,0]
con=[[1,1],[1,2]]
x,y=np.random.multivariate_normal(mean,con, 10000).T
```

#### plt.hist2d: Two-dimensional histogram

One straightforward way to plot a two-dimensional histogram is to use
Matplotlib’s plt.hist2d function

``` python
plt.hist2d(x,y, bins=30, cmap='Blues')
cb=plt.colorbar()
cb.set_label('Counts in Bin')
```

![](06Histograms%20Binnings%20and%20Density_files/figure-gfm/cell-8-output-1.png)

Just as with plt.hist, plt.hist2d has a number of extra options to
fine-tune the plot and the binning, which are nicely outlined in the
function docstring. Further, just as plt.hist has a counterpart in
np.histogram, plt.hist2d has a counterpart in np.histogram2d, which can
be used as follows:

``` python
counts, xedges, yedges = np.histogram2d(x,y,bins=30)
xedges
```

    array([-4.42186998, -4.13921105, -3.85655212, -3.57389318, -3.29123425,
           -3.00857531, -2.72591638, -2.44325744, -2.16059851, -1.87793957,
           -1.59528064, -1.31262171, -1.02996277, -0.74730384, -0.4646449 ,
           -0.18198597,  0.10067297,  0.3833319 ,  0.66599084,  0.94864977,
            1.2313087 ,  1.51396764,  1.79662657,  2.07928551,  2.36194444,
            2.64460338,  2.92726231,  3.20992125,  3.49258018,  3.77523911,
            4.05789805])

### plt.hexbin: Hexagonal binnings

The two-dimensional histogram creates a tessellation of squares across
the axes. Another natural shape for such a tessellation is the regular
hexagon. For this purpose, Matplotlib provides the plt.hexbin routine,
which represents a two-dimensional dataset binned within a grid of
hexagons

``` python
plt.hexbin(x,y, gridsize=30, cmap='Blues')
cb=plt.colorbar(label='Count in Bins')
```

![](06Histograms%20Binnings%20and%20Density_files/figure-gfm/cell-10-output-1.png)

### Kernel density estimation

Another common method of evaluating densities in multiple dimensions is
kernel density estimation (KDE).

KDE can be thought of as a way to “smear out” the points in space and
add up the result to obtain a smooth function. One extremely quick and
simple KDE implementation exists in the scipy.stats package. Here is a
quick example of using the KDE on this data

``` python
from scipy.stats import gaussian_kde
```

``` python
# fit an arry of size nxn
data = np.vstack([x,y])
kde=gaussian_kde(data)

# evaluate on a regular grid
xgrid=np.linspace(-3.5,3.5,40)
ygrid=np.linspace(-6,6,40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

Z= kde.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
```

``` python
# plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
          origin='lower', aspect='auto',
          extent=[-3.5,3.5,-5,5],
          cmap='Blues')
cb=plt.colorbar()
cb.set_label('Density')
```

![](06Histograms%20Binnings%20and%20Density_files/figure-gfm/cell-13-output-1.png)

KDE has a smoothing length that effectively slides the knob between
detail and smoothness (one example of the ubiquitous bias–variance
trade-off). The literature on choosing an appropriate smoothing length
is vast: gaussian_kde uses a rule of thumb to attempt to find a nearly
optimal smoothing length for the input data.
