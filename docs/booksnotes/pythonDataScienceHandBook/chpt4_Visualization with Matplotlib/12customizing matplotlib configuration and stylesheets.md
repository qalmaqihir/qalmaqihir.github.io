Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 12 -  Customizing Matplotlib: Configurations and Stylesheets
------------------------------------------------------------------------

- <a href="#customizing-matplotlib-configurations-and-stylesheets"
  id="toc-customizing-matplotlib-configurations-and-stylesheets">Customizing
  Matplotlib: Configurations and Stylesheets</a>
  - <a href="#plot-customization-by-hand"
    id="toc-plot-customization-by-hand">Plot Customization by Hand</a>
  - <a href="#changing-the-defaults-rcparams"
    id="toc-changing-the-defaults-rcparams">Changing the Defaults:
    rcParams</a>
  - <a href="#stylesheets" id="toc-stylesheets">Stylesheets</a>
    - <a href="#default-style" id="toc-default-style">Default style</a>

------------------------------------------------------------------------

# Customizing Matplotlib: Configurations and Stylesheets

Matplotlib’s default plot settings are often the subject of complaint
among its users. While much is slated to change in the 2.0 Matplotlib
release, the ability to customize default settings helps bring the
package in line with your own aesthetic preferences. Here we’ll walk
through some of Matplotlib’s runtime configuration (rc) options, and
take a look at the newer stylesheets feature, which contains some nice
sets of default configurations.

## Plot Customization by Hand

Throughout this chapter, we’ve seen how it is possible to tweak
individual plot set‐ tings to end up with something that looks a little
bit nicer than the default. It’s possi‐ ble to do these customizations
for each individual plot. For example, here is a fairly drab default
histogram

``` python
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np
%matplotlib inline
```

``` python
x=np.random.randn(1000)
plt.hist(x);
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-3-output-1.png)

``` python
# We can adjust this by hand to make it a much more visually pleasing plot
# use  a gray backgoung
ax=plt.axes()
ax.set_axisbelow(True)

# draw solid white grid line
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
#hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

#lighten ticks and labels
ax.tick_params(color='gray',direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
    
for tick in ax.get_yticklabels():
    tick.set_color('gray')

# control face and edge color of histogram
ax.hist(x,edgecolor='#E6E6E6', color='#EE6666')

```

    (array([ 12.,  37.,  98., 173., 257., 213., 124.,  59.,  23.,   4.]),
     array([-2.90788595, -2.30201003, -1.69613411, -1.09025818, -0.48438226,
             0.12149367,  0.72736959,  1.33324551,  1.93912144,  2.54499736,
             3.15087329]),
     <BarContainer object of 10 artists>)

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-4-output-2.png)

This looks better, and you may recognize the look as inspired by the
look of the R language’s ggplot visualization package. But this took a
whole lot of effort! We defi‐ nitely do not want to have to do all that
tweaking each time we create a plot. Fortu‐ nately, there is a way to
adjust these defaults once in a way that will work for all plots.

## Changing the Defaults: rcParams

Each time Matplotlib loads, it defines a runtime configuration (rc)
containing the default styles for every plot element you create. You can
adjust this configuration at any time using the plt.rc convenience
routine. Let’s see what it looks like to modify the rc parameters so
that our default plot will look similar to what we did before. We’ll
start by saving a copy of the current rcParams dictionary, so we can
easily reset these changes in the current session:

``` python
Ipython_default=plt.rcParams.copy()
```

``` python
from matplotlib import cycler
colors = cycler('color',
['#EE6666', '#3388BB', '#9988DD',
'#EECC55', '#88BB44', '#FFBBBB'])

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
axisbelow=True, grid=True, prop_cycle=colors)

plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
```

``` python
# With these settings defined, we can now create a plot and see our settings in action
plt.hist(x);
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-7-output-1.png)

``` python
# Let’s see what simple line plots look like with these rc parameters
for i in range(4):
    plt.plot(np.random.rand(10))
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-8-output-1.png)

I find this much more aesthetically pleasing than the default styling.
If you disagree with my aesthetic sense, the good news is that you can
adjust the rc parameters to suit your own tastes! These settings can be
saved in a .matplotlibrc file, which you can read about in the
Matplotlib documentation. That said, I prefer to customize Mat‐ plotlib
using its stylesheets instead.

## Stylesheets

Even if you don’t create your own style, the stylesheets included by
default are extremely useful. The available styles are listed in
plt.style.available—here I’ll list only the first five for brevity:

``` python
plt.style.available[:5]
```

    ['Solarize_Light2',
     '_classic_test_patch',
     '_mpl-gallery',
     '_mpl-gallery-nogrid',
     'bmh']

``` python

plt.style.use('Solarize_Light2')
```

``` python
with plt.style.context('Solarize_Light2'):
    make_a_plot()
```

    NameError: name 'make_a_plot' is not defined

``` python
def hist_and_lines():
    np.random.seed(0)
    fig, ax= plt.subplots(1,2,figsize=(11,4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(100))
        ax[1].legend(['a','b','c'],loc='lower left')
```

### Default style

The default style is what we’ve been seeing so far throughout the book;
we’ll start with that. First, let’s reset our runtime configuration to
the notebook default:

``` python
# reset rcParams
plt.rcParams.update(Ipython_default);
```

``` python
# let's see how it looks
hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-14-output-1.png)

``` python
# let use one of the styles avialable
plt.style.available[:]
```

    ['Solarize_Light2',
     '_classic_test_patch',
     '_mpl-gallery',
     '_mpl-gallery-nogrid',
     'bmh',
     'classic',
     'dark_background',
     'fast',
     'fivethirtyeight',
     'ggplot',
     'grayscale',
     'seaborn',
     'seaborn-bright',
     'seaborn-colorblind',
     'seaborn-dark',
     'seaborn-dark-palette',
     'seaborn-darkgrid',
     'seaborn-deep',
     'seaborn-muted',
     'seaborn-notebook',
     'seaborn-paper',
     'seaborn-pastel',
     'seaborn-poster',
     'seaborn-talk',
     'seaborn-ticks',
     'seaborn-white',
     'seaborn-whitegrid',
     'tableau-colorblind10']

``` python
# Using fivethrityeight
with plt.style.context('fivethirtyeight'):
    hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-16-output-1.png)

``` python
# Using ggplot
with plt.style.context('ggplot'):
    hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-17-output-1.png)

**Bayesian Methods for Hackers style** There is a very nice short online
book called Probabilistic Programming and Bayesian Methods for Hackers;
it features figures created with Matplotlib, and uses a nice set of rc
parameters to create a consistent and visually appealing style
throughout the book. This style is reproduced in the bmh stylesheet

``` python
# Using bmh
with plt.style.context('bmh'):
    hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-18-output-1.png)

**Dark background** For figures used within presentations, it is often
useful to have a dark rather than light background. The dark_background
style provides this

``` python
with plt.style.context('dark_background'):
    hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-19-output-1.png)

**Grayscale** Sometimes you might find yourself preparing figures for a
print publication that does not accept color figures.

``` python
with plt.style.context('grayscale'):
    hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-20-output-1.png)

**Seaborn style** Matplotlib also has stylesheets inspired by the
Seaborn library (discussed more fully in “Visualization with Seaborn” on
page 311). As we will see, these styles are loaded automatically when
Seaborn is imported into a notebook.

``` python
import seaborn
hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-21-output-1.png)

``` python
with plt.style.context('seaborn'):
    hist_and_lines()
```

![](12customizing%20matplotlib%20configuration%20and%20stylesheets_files/figure-gfm/cell-22-output-1.png)
