Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 11 -  Customizing Ticks
------------------------------------------------------------------------

- <a href="#customizing-ticks" id="toc-customizing-ticks">Customizing
  Ticks</a>
  - <a href="#major-and-minor-ticks" id="toc-major-and-minor-ticks">Major
    and Minor Ticks</a>
  - <a href="#hiding-ticks-or-labels" id="toc-hiding-ticks-or-labels">Hiding
    Ticks or Labels</a>
  - <a href="#reducing-or-increasing-the-number-of-ticks"
    id="toc-reducing-or-increasing-the-number-of-ticks">Reducing or
    Increasing the Number of Ticks</a>
  - <a href="#fancy-tick-formats" id="toc-fancy-tick-formats">Fancy Tick
    Formats</a>
------------------------------------------------------------------------

# Customizing Ticks

Matplotlib’s default tick locators and formatters are designed to be
generally sufficient in many common situations, but are in no way
optimal for every plot.

## Major and Minor Ticks

Within each axis, there is the concept of a major tick mark and a minor
tick mark. As the names would imply, major ticks are usually bigger or
more pronounced, while minor ticks are usually smaller. By default,
Matplotlib rarely makes use of minor ticks, but one place you can see
them is within logarithmic plots

``` python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

``` python
ax = plt.axes(xscale='log', yscale='log')
```

![](11customizing%20ticks_files/figure-gfm/cell-3-output-1.png)

``` python
print(ax.xaxis.get_major_locator())
print(ax.xaxis.get_minor_locator())
```

    <matplotlib.ticker.LogLocator object at 0x7faa7bb8f640>
    <matplotlib.ticker.LogLocator object at 0x7faa7be3baf0>

``` python
print(ax.xaxis.get_major_formatter())
print(ax.xaxis.get_minor_formatter())
```

    <matplotlib.ticker.LogFormatterSciNotation object at 0x7faa7bb7ca60>
    <matplotlib.ticker.LogFormatterSciNotation object at 0x7faab054b6d0>

## Hiding Ticks or Labels

Perhaps the most common tick/label formatting operation is the act of
hiding ticks or labels. We can do this using plt.NullLocator() and
plt.NullFormatter(), as shown here

``` python
ax=plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_formatter(plt.NullFormatter())
```

![](11customizing%20ticks_files/figure-gfm/cell-6-output-1.png)

``` python
ax=plt.axes()
ax.plot(np.random.rand(50))
```

![](11customizing%20ticks_files/figure-gfm/cell-7-output-1.png)

``` python
fig, ax = plt.subplots(5,5,figsize=(5,5))
fig.subplots_adjust(hspace=0,wspace=0)

#Get some face images from scikit-learn
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces().images

for i in range(5):
    for j in range(5):
        ax[i,j].xaxis.set_major_locator(plt.NullLocator())
        ax[i,j].yaxis.set_major_locator(plt.NullLocator())
        ax[i,j].imshow(faces[10*i+j], cmap='bone')
```

![](11customizing%20ticks_files/figure-gfm/cell-8-output-1.png)

## Reducing or Increasing the Number of Ticks

One common problem with the default settings is that smaller subplots
can end up with crowded labels.

``` python
fig, ac =plt.subplots(4,4,sharex=True, sharey=True)
```

![](11customizing%20ticks_files/figure-gfm/cell-9-output-1.png)

Particularly for the x ticks, the numbers nearly overlap, making them
quite difficult to decipher. We can fix this with the plt.MaxNLocator(),
which allows us to specify the maximum number of ticks that will be
displayed. Given this maximum number, Mat‐ plotlib will use internal
logic to choose the particular tick locations

``` python
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    
fig
```

![](11customizing%20ticks_files/figure-gfm/cell-10-output-1.png)

## Fancy Tick Formats

Matplotlib’s default tick formatting can leave a lot to be desired; it
works well as a broad default, but sometimes you’d like to do something
more

``` python
# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);
```

![](11customizing%20ticks_files/figure-gfm/cell-11-output-1.png)

``` python
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig
```

![](11customizing%20ticks_files/figure-gfm/cell-12-output-1.png)

``` python
def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

    
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig
```

![](11customizing%20ticks_files/figure-gfm/cell-13-output-1.png)

**This is much better! Notice that we’ve made use of Matplotlib’s LaTeX
support, speci‐ fied by enclosing the string within dollar signs. This
is very convenient for display of mathematical symbols and formulae; in
this case, “$\pi$” is rendered as the Greek character π. The
plt.FuncFormatter() offers extremely fine-grained control over the
appearance of your plot ticks, and comes in very handy when you’re
preparing plots for presenta‐ tion or publication.**
