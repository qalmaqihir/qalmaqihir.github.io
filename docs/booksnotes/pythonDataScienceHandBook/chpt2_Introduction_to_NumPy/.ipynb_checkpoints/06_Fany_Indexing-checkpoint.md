Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 1 - Introduction to Numpy**

# 06 - Comparisons, Masks, and Boolean Logic

------------------------------------------------------------------------
<center>
<a href=''>![Image](../../../assets/img/logo1.png)</a>
</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
----------------------------------------------------------------------------


- <a href="#fancy-indexing" id="toc-fancy-indexing">Fancy Indexing</a>
  - <a href="#exploring-fancy-indexing"
    id="toc-exploring-fancy-indexing">Exploring Fancy Indexing</a>
  - <a href="#combined-indexing" id="toc-combined-indexing">Combined
    Indexing</a>
  - <a href="#example-selecting-random-points"
    id="toc-example-selecting-random-points">Example: Selecting Random
    Points</a>
  - <a href="#modifying-values-with-fancy-indexing"
    id="toc-modifying-values-with-fancy-indexing">Modifying Values with
    Fancy Indexing</a>
  - <a href="#exmple-binning-data" id="toc-exmple-binning-data">Exmple
    Binning Data</a>

------------------------------------------------------------------------

# Fancy Indexing

In the previous sections, we saw how to access and modify portions of
arrays using simple indices (e.g., `arr[0]`), slices (e.g., `arr[:5]`),
and Boolean masks (e.g., arr`[arr> 0]`). In this section, we’ll look at
another style of array indexing, known as fancy indexing. **Fancy
indexing is like the simple indexing we’ve already seen, but we pass
arrays of indices in place of single scalars. This allows us to very
quickly access and modify complicated subsets of an array’s values.**

## Exploring Fancy Indexing

Fancy indexing is conceptually simple: it means passing an array of
indices to access multiple array elements at once.

``` python
import numpy as np
```

``` python
rand=np.random.RandomState(42)
x=rand.randint(100,size=10)
print(x)
```

    [51 92 14 71 60 20 82 86 74 74]

``` python
# three different elements
[x[3],x[7],x[2]]
```

    [71, 86, 14]

``` python
# alternatively, we could do like this
ind = [3, 7,4]
x[ind]
```

    array([71, 86, 60])

***With fancy indexing, the shape of the result reflects the shape of
the index arrays rather than the shape of the array being indexed:***

``` python
ind=np.array([[3,7],[4,5]])
x[ind]
```

    array([[71, 86],
           [60, 20]])

``` python
# Multi-dimensional fancy indexing
x=np.arange(12).reshape((3,4))
x
                        
```

    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

``` python
row=np.array([0,1,2])
col=np.array([2,1,3])
x[row, col]
```

    array([ 2,  5, 11])

``` python
x[row[:,np.newaxis],col]
```

    array([[ 2,  1,  3],
           [ 6,  5,  7],
           [10,  9, 11]])

``` python
##Here, each row value is matched with each column vector, exactly as we saw in broad‐
##casting of arithmetic operations

row[:,np.newaxis]
```

    array([[0],
           [1],
           [2]])

``` python
row[:,np.newaxis]*col
```

    array([[0, 0, 0],
           [2, 1, 3],
           [4, 2, 6]])

``` python
## It is always important to remember with fancy indexing that the return value reflects
##the broadcasted shape of the indices, rather than the shape of the array being indexed.
```

## Combined Indexing

For even more powerful operations, fancy indexing can be combined with
the other indexing schemes

``` python
x
```

    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

``` python
# Combing fancy and simple index
x[2,[2,0,1]]
```

    array([10,  8,  9])

``` python
# Combining fancy indexing with masking
mask=np.array([1,0,1,0], dtype=bool)
x[row[:,np.newaxis],mask]
```

    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]])

## Example: Selecting Random Points

One common use of fancy indexing is the selection of subsets of rows
from a matrix. For example, we might have an N by D matrix representing
N points in D dimen‐ sions, such as the following points drawn from a
two-dimensional normal distribu‐ tion

``` python
mean=[0,0]
conv=[[1,2],
     [2,5]]
x=rand.multivariate_normal(mean,conv,100)
x.shape
```

    (100, 2)

``` python
print(x)
```

    [[-0.644508   -0.46220608]
     [ 0.7376352   1.21236921]
     [ 0.88151763  1.12795177]
     [ 2.04998983  5.97778598]
     [-0.1711348  -2.06258746]
     [ 0.67956979  0.83705124]
     [ 1.46860232  1.22961093]
     [ 0.35282131  1.49875397]
     [-2.51552505 -5.64629995]
     [ 0.0843329  -0.3543059 ]
     [ 0.19199272  1.48901291]
     [-0.02566217 -0.74987887]
     [ 1.00569227  2.25287315]
     [ 0.49514263  1.18939673]
     [ 0.0629872   0.57349278]
     [ 0.75093031  2.99487004]
     [-3.0236127  -6.00766046]
     [-0.53943081 -0.3478899 ]
     [ 1.53817376  1.99973464]
     [-0.50886808 -1.81099656]
     [ 1.58115602  2.86410319]
     [ 0.99305043  2.54294059]
     [-0.87753796 -1.15767204]
     [-1.11518048 -1.87508012]
     [ 0.4299908   0.36324254]
     [ 0.97253528  3.53815717]
     [ 0.32124996  0.33137032]
     [-0.74618649 -2.77366681]
     [-0.88473953 -1.81495444]
     [ 0.98783862  2.30280401]
     [-1.2033623  -2.04402725]
     [-1.51101746 -3.2818741 ]
     [-2.76337717 -7.66760648]
     [ 0.39158553  0.87949228]
     [ 0.91181024  3.32968944]
     [-0.84202629 -2.01226547]
     [ 1.06586877  0.95500019]
     [ 0.44457363  1.87828298]
     [ 0.35936721  0.40554974]
     [-0.90649669 -0.93486441]
     [-0.35790389 -0.52363012]
     [-1.33461668 -3.03203218]
     [ 0.02815138  0.79654924]
     [ 0.37785618  0.51409383]
     [-1.06505097 -2.88726779]
     [ 2.32083881  5.97698647]
     [ 0.47605744  0.83634485]
     [-0.35490984 -1.03657119]
     [ 0.57532883 -0.79997124]
     [ 0.33399913  2.32597923]
     [ 0.6575612  -0.22389518]
     [ 1.3707365   2.2348831 ]
     [ 0.07099548 -0.29685467]
     [ 0.6074983   1.47089233]
     [-0.34226126 -1.10666237]
     [ 0.69226246  1.21504303]
     [-0.31112937 -0.75912097]
     [-0.26888327 -1.89366817]
     [ 0.42044896  1.85189522]
     [ 0.21115245  2.00781492]
     [-1.83106042 -2.91352836]
     [ 0.7841796   1.97640753]
     [ 0.10259314  1.24690575]
     [-1.91100558 -3.66800923]
     [ 0.13143756 -0.07833855]
     [-0.1317045  -1.64159158]
     [-0.14547282 -1.34125678]
     [-0.51172373 -1.40960773]
     [ 0.69758045  0.72563649]
     [ 0.11677083  0.88385162]
     [-1.16586444 -2.24482237]
     [-2.23176235 -2.63958101]
     [ 0.37857234  0.69112594]
     [ 0.87475323  3.400675  ]
     [-0.86864365 -3.03568353]
     [-1.03637857 -1.18469125]
     [-0.53334959 -0.37039911]
     [ 0.30414557 -0.5828419 ]
     [-1.47656656 -2.13046298]
     [-0.31332021 -1.7895623 ]
     [ 1.12659538  1.49627535]
     [-1.19675798 -1.51633442]
     [-0.75210154 -0.79770535]
     [ 0.74577693  1.95834451]
     [ 1.56094354  2.9330816 ]
     [-0.72009966 -1.99780959]
     [-1.32319163 -2.61218347]
     [-2.56215914 -6.08410838]
     [ 1.31256297  3.13143269]
     [ 0.51575983  2.30284639]
     [ 0.01374713 -0.11539344]
     [-0.16863279  0.39422355]
     [ 0.12065651  1.13236323]
     [-0.83504984 -2.38632016]
     [ 1.05185885  1.98418223]
     [-0.69144553 -1.56919875]
     [-1.2567603  -1.125898  ]
     [ 0.09619333 -0.64335574]
     [-0.99658689 -2.35038099]
     [-1.21405259 -1.77693724]]

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(x[:,0],x[:,1])
```

    <matplotlib.collections.PathCollection at 0x7f507340ffd0>

![](06_Fany%20Indexing_files/figure-gfm/cell-19-output-2.png)

Let’s use fancy indexing to select 20 random points. We’ll do this by
first choosing 20 random indices with no repeats, and use these indices
to select a portion of the origi‐ nal array

``` python
indices=np.random.choice(x.shape[0],20,replace=False)
indices
```

    array([80, 87, 70, 77, 18, 56, 41, 42, 84,  9, 85, 61,  6, 71, 24, 69,  8,
           22, 23, 54])

``` python
selection = x[indices]
```

``` python
selection
```

    array([[ 1.12659538,  1.49627535],
           [-2.56215914, -6.08410838],
           [-1.16586444, -2.24482237],
           [ 0.30414557, -0.5828419 ],
           [ 1.53817376,  1.99973464],
           [-0.31112937, -0.75912097],
           [-1.33461668, -3.03203218],
           [ 0.02815138,  0.79654924],
           [ 1.56094354,  2.9330816 ],
           [ 0.0843329 , -0.3543059 ],
           [-0.72009966, -1.99780959],
           [ 0.7841796 ,  1.97640753],
           [ 1.46860232,  1.22961093],
           [-2.23176235, -2.63958101],
           [ 0.4299908 ,  0.36324254],
           [ 0.11677083,  0.88385162],
           [-2.51552505, -5.64629995],
           [-0.87753796, -1.15767204],
           [-1.11518048, -1.87508012],
           [-0.34226126, -1.10666237]])

``` python
plt.scatter(x[:,0],x[:,1], alpha=0.3)
plt.scatter(selection[:,0],selection[:,1],facecolor='none',s=200);
```

![](06_Fany%20Indexing_files/figure-gfm/cell-23-output-1.png)

**This sort of strategy is often used to quickly partition datasets, as
is often needed in train/test splitting for validation of statistical
models**

## Modifying Values with Fancy Indexing

Just as fancy indexing can be used to access parts of an array, it can
also be used to modify parts of an array. For example, imagine we have
an array of indices and we’d like to set the corresponding items in an
array to some value:

``` python
x=np.arange(10)
i=np.array([2,1,8,4])
x[i]=-9
x
```

    array([ 0, -9, -9,  3, -9,  5,  6,  7, -9,  9])

***We can use any assignment-type operator for this*** Notice, though,
that repeated indices with these operations can cause some poten‐ tially
unexpected results

``` python
x[i]-=10
```

``` python
x
```

    array([  0, -19, -19,   3, -19,   5,   6,   7, -19,   9])

``` python
x=np.zeros(10)
```

``` python
x[[0,0]]=[4,6]
```

``` python
x
```

    array([6., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

Where did the 4 go? The result of this operation is to first assign
`x[0] = 4`, followed by `x[0] = 6`. The result, of course, is that
`x[0]` contains the value 6.

``` python
i=[2,3,3,4,4,4]
x[i]+=1
```

``` python
x
```

    array([6., 0., 1., 1., 1., 0., 0., 0., 0., 0.])

You might expect that x\[3\] would contain the value 2, and x\[4\] would
contain the value 3, as this is how many times each index is repeated.
Why is this not the case? Conceptually, this is because x\[i\] += 1 is
meant as a shorthand of x\[i\] = x\[i\] + 1. x\[i\] + 1 is evaluated,
and then the result is assigned to the indices in x. With this in mind,
it is not the augmentation that happens multiple times, but the
assignment, which leads to the rather nonintuitive results.

***So what if you want the other behavior where the operation is
repeated? For this, you can use the at() method of ufuncs (available
since NumPy 1.8), and do the following***

``` python
x=np.zeros(10)
np.add.at(x,i,1)
x
```

    array([0., 0., 1., 2., 3., 0., 0., 0., 0., 0.])

The at() method does an in-place application of the given operator at
the specified indices (here, i) with the specified value (here, 1).
Another method that is similar in spirit is the reduceat() method of
ufuncs, which you can read about in the NumPy documentation.

## Exmple Binning Data

``` python
np.random.seed(42)
x=np.random.randn(100)
```

``` python
# compute a histogram by hand
bins=np.linspace(-5,5,20)
counts=np.zeros_like(bins)
```

``` python
# find the approperiate bin for each x
i=np.searchsorted(bins,x)
```

``` python
# add 1 to each of these bins
np.add.at(counts,i,1)
```

``` python
# plot the results
plt.plot(bins, counts)  #linestyle='steps'
```

![](06_Fany%20Indexing_files/figure-gfm/cell-37-output-1.png)

Of course, it would be silly to have to do this each time you want to
plot a histogram. This is why Matplotlib provides the plt.hist()
routine, which does the same in a single line:

``` python
plt.hist(x,bins,histtype='step');
```

![](06_Fany%20Indexing_files/figure-gfm/cell-38-output-1.png)

``` python
print("NumPy routine")
%timeit counts, edges =np.histogram(x,bins)
```

    NumPy routine
    29.1 µs ± 4.23 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

``` python
print("custom routine")
%timeit np.add.at(counts, np.searchsorted(bins,x),1)
```

    custom routine
    13.2 µs ± 1.27 µs per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

***Our own one-line algorithm is several times faster than the optimized
algorithm in NumPy! How can this be? If you dig into the np.histogram
source code (you can do this in IPython by typing np.histogram??),
you’ll see that it’s quite a bit more involved than the simple
search-and-count that we’ve done; this is because NumPy’s algorithm is
more flexible, and particularly is designed for better performance when
the number of data points becomes large:***

``` python
x=np.random.randn(1000000)
print("NumPy routine: ")
%timeit counts, edges=np.histogram(x,bins)

print("Custom routine: ")
%timeit np.add.at(counts, np.searchsorted(bins,x),1)
```

    NumPy routine: 
    67.3 ms ± 5.67 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    Custom routine: 
