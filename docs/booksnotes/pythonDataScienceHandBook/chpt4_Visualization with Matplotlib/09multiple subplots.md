================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 09 -  Multiple Subplots
------------------------------------------------------------------------

- <a href="#multiple-subplots" id="toc-multiple-subplots">Multiple
  Subplots</a>
  - <a href="#plt.axes-subplots-by-hand"
    id="toc-plt.axes-subplots-by-hand">plt.axes: Subplots by Hand</a>
  - <a href="#plt.subplot-simple-grids-of-subplots"
    id="toc-plt.subplot-simple-grids-of-subplots">plt.subplot: Simple Grids
    of Subplots</a>
  - <a href="#plt.subplots-the-whole-grid-in-one-go"
    id="toc-plt.subplots-the-whole-grid-in-one-go">plt.subplots: The Whole
    Grid in One Go</a>
  - <a href="#plt.gridspec-more-complicated-arrangements"
    id="toc-plt.gridspec-more-complicated-arrangements">plt.GridSpec: More
    Complicated Arrangements</a>

------------------------------------------------------------------------

# Multiple Subplots

Sometimes it is helpful to compare different views of data side by side.
To this end, Matplotlib has the concept of subplots: groups of smaller
axes that can exist together within a single figure. These subplots
might be insets, grids of plots, or other more complicated layouts. In
this section, we’ll explore four routines for creating subplots in
Matplotlib. We’ll start by setting up the notebook for plotting and
importing the functions we will use:

``` python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
```

## plt.axes: Subplots by Hand

The most basic method of creating an axes is to use the plt.axes
function. As we’ve seen previously, by default this creates a standard
axes object that fills the entire fig‐ ure. plt.axes also takes an
optional argument that is a list of four numbers in the figure
coordinate system. These numbers represent \[bottom, left, width,
height\] in the figure coordinate system, which ranges from 0 at the
bottom left of the figure to 1 at the top right of the figure.

``` python
ax1=plt.axes()
ax2=plt.axes([0.65, 0.65, 0.2,0.2])
```

![](09multiple%20subplots_files/figure-gfm/cell-3-output-1.png)

**The equivalent of this command within the object-oriented interface is
fig.add_axes(). Let’s use this to create two vertically stacked axes**

``` python
fig = plt.figure()
ax1=fig.add_axes([0.1,0.5, 0.8,0.4],
                xticklabels=[],ylim=(-1.2,1.2))
ax2=fig.add_axes([0.1,0.1
                  , 0.8,0.4],
                xticklabels=[],ylim=(-1.2,1.2))

x=np.linspace(-0,10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));
```

![](09multiple%20subplots_files/figure-gfm/cell-4-output-1.png)

## plt.subplot: Simple Grids of Subplots

Aligned columns or rows of subplots are a common enough need that
Matplotlib has several convenience routines that make them easy to
create. The lowest level of these is plt.subplot(), which creates a
single subplot within a grid. As you can see, this command takes three
integer arguments—the number of rows, the number of col‐ umns, and the
index of the plot to be created in this scheme, which runs from the
upper left to the bottom right

``` python
for i in range(1,7):
    plt.subplot(2,3,i)
    plt.text(0.5,0.5,str((2,3,i)),
            fontsize=18,ha='center')
```

![](09multiple%20subplots_files/figure-gfm/cell-5-output-1.png)

*The command plt.subplots_adjust can be used to adjust the spacing
between these plots.*

**object-oriented command, fig.add_subplot():**

``` python
fig = plt.figure()
fig.subplots_adjust(hspace=0.4,wspace=0.4)
for i in range(1,7):
    ax=fig.add_subplot(2,3,i)
    ax.text(0.5,0.5, str((2,3,i)),
           fontsize=18, ha='center')
```

![](09multiple%20subplots_files/figure-gfm/cell-6-output-1.png)

## plt.subplots: The Whole Grid in One Go

The approach just described can become quite tedious when you’re
creating a large grid of subplots, especially if you’d like to hide the
x- and y-axis labels on the inner plots. For this purpose,
plt.subplots() is the easier tool to use (note the s at the end of
subplots). Rather than creating a single subplot, this function creates
a full grid of subplots in a single line, returning them in a NumPy
array. The arguments are the number of rows and number of columns, along
with optional keywords sharex and sharey, which allow you to specify the
relationships between different axes.

``` python
fig,ax=plt.subplots(2,3,sharex='col',sharey='row')
```

![](09multiple%20subplots_files/figure-gfm/cell-8-output-1.png)

``` python
for i in range(2):
    for j in range(3):
        ax[i,j].text(0.5,0.5, str((2,3,i)),
           fontsize=18, ha='center')
fig
```

![](09multiple%20subplots_files/figure-gfm/cell-9-output-1.png)

## plt.GridSpec: More Complicated Arrangements

To go beyond a regular grid to subplots that span multiple rows and
columns, plt.GridSpec() is the best tool. The plt.GridSpec() object does
not create a plot by itself; it is simply a convenient interface that is
recognized by the plt.subplot() command.

``` python
grid=plt.GridSpec(2,3,wspace=0.4, hspace=0.3)
```

``` python
plt.subplot(grid[0,0])
plt.subplot(grid[0,1:])
plt.subplot(grid[1,:2])
plt.subplot(grid[1,2]);
```

![](09multiple%20subplots_files/figure-gfm/cell-11-output-1.png)

``` python
# Create some normally distributed data
mean=[0,0]
cov=[[1,1],[1,2]]
x,y=np.random.multivariate_normal(mean,cov,3000).T
```

``` python
# Set up the axes with gridspec
fig = plt.figure(figsize=(6,6))
grid=plt.GridSpec(4,4, wspace=0.2, hspace=0.2)
main_ax=fig.add_subplot(grid[:-1,1:])
y_hist=fig.add_subplot(grid[:-1,0],xticklabels=[], sharey=main_ax)
x_hist=fig.add_subplot(grid[-1,1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.plot(x,y,'ok',markersize=3,alpha=0.2)

# historgram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled', 
           orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled',
           orientation='horizontal', color='gray')
y_hist.invert_xaxis()
```

![](09multiple%20subplots_files/figure-gfm/cell-13-output-1.png)
