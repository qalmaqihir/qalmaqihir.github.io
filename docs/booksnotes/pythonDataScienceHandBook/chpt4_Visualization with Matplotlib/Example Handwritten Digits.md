================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# Example: Handwritten Digits
------------------------------------------------------------------------

- <a href="#example-handwritten-digits"
  id="toc-example-handwritten-digits">Example: Handwritten Digits</a>

------------------------------------------------------------------------

# Example: Handwritten Digits

For an example of where this might be useful, let’s look at an
interesting visualization of some handwritten digits data. This data is
included in Scikit-Learn, and consists of nearly 2,000 8×8 thumbnails
showing various handwritten digits. For now, let’s start by downloading
the digits data and visualizing several of the exam‐ ple images with
plt.imshow()

``` python
import matplotlib.pyplot as plt
```

``` python
# load images of the digit 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
```

``` python
digits=load_digits(n_class=6)
fig, ax=plt.subplots(8,8,figsize=(6,6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])
```

![](Example%20Handwritten%20Digits_files/figure-gfm/cell-4-output-1.png)

**Because each digit is defined by the hue of its 64 pixels, we can
consider each digit to be a point lying in 64-dimensional space: each
dimension represents the brightness of one pixel. But visualizing
relationships in such high-dimensional spaces can be extremely
difficult. One way to approach this is to use a dimensionality reduction
technique such as manifold learning to reduce the dimensionality of the
data while maintaining the relationships of interest. Dimensionality
reduction is an example of unsupervised machine learning**

Deferring the discussion of these details, let’s take a look at a
two-dimensional mani‐ fold learning projection of this digits data

``` python
from sklearn.manifold import Isomap
```

``` python
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)


# plot the results
plt.scatter(projection[:,0],projection[:,1],lw=0.1,
           c=digits.target, cmap=plt.cm.get_cmap('cubehelix',6))
plt.colorbar(ticks=range(6),label='digit value')
plt.clim(-0.5,5.5)
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_isomap.py:348: UserWarning: The number of connected components of the neighbors graph is 2 > 1. Completing the graph to fit Isomap might be slow. Increase the number of neighbors to avoid this issue.
      self._fit_transform(X)
    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/scipy/sparse/_index.py:82: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
      self._set_intXint(row, col, x.flat[0])

![](Example%20Handwritten%20Digits_files/figure-gfm/cell-6-output-2.png)

The projection also gives us some interesting insights on the
relationships within the dataset: for example, the ranges of 5 and 3
nearly overlap in this projection, indicating that some handwritten
fives and threes are difficult to distinguish, and therefore more likely
to be confused by an automated classification algorithm. Other values,
like 0 and 1, are more distantly separated, and therefore much less
likely to be con‐ fused. This observation agrees with our intuition,
because 5 and 3 look much more similar than do 0 and 1.
