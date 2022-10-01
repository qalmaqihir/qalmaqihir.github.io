Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 1 - Introduction to Numpy**

# 03 - Aggregations: Min, Max, and Everything in Between

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


- <a href="#aggregations-min-max-and-everything-in-between"
  id="toc-aggregations-min-max-and-everything-in-between">Aggregations:
  Min, Max, and Everything in Between</a>
  - <a href="#multidimensional-aggregates"
    id="toc-multidimensional-aggregates">Multidimensional aggregates</a>
- <a href="#some-aggregation-functions"
  id="toc-some-aggregation-functions">Some aggregation functions</a>
  - <a href="#example-what-is-the-average-height-of-us-presidents"
    id="toc-example-what-is-the-average-height-of-us-presidents">Example:
    What Is the Average Height of US Presidents?</a>

------------------------------------------------------------------------


## Aggregations: Min, Max, and Everything in Between

Often when you are faced with a large amount of data, a first step is to
compute sum‐ mary statistics for the data in question. Perhaps the most
common summary statistics are the mean and standard deviation, which
allow you to summarize the “typical” val‐ ues in a dataset, but other
aggregates are useful as well (the sum, product, median, minimum and
maximum, quantiles, etc.).

``` python
l=np.random.random(100)
sum(l)
```

    47.51294159911191

``` python
np.sum(l)
```

    47.5129415991119

``` python
big_array=np.random.rand(100000)
%timeit sum(big_array)
%timeit np.sum(big_array)
```

    6.73 ms ± 88.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    31.5 µs ± 326 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

***Be careful, though: the sum function and the np.sum function are not
identical, which can sometimes lead to confusion! In particular, their
optional arguments have differ‐ ent meanings, and np.sum is aware of
multiple array dimensions***

``` python
min(big_array), max(big_array)
```

    (1.4444103570987465e-06, 0.9999881721555508)

``` python
np.min(big_array), np.max(big_array)
```

    (1.4444103570987465e-06, 0.9999881721555508)

``` python
%timeit min(big_array)
%timeit np.min(big_array)
```

    5.44 ms ± 49.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    35.5 µs ± 64.6 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

### Multidimensional aggregates

``` python
m=np.random.random((3,4))
print(m)
```

    [[0.15322668 0.10058762 0.85504471 0.19779527]
     [0.24515716 0.61526756 0.9677193  0.0045308 ]
     [0.57826711 0.49512073 0.68039294 0.43271134]]

``` python

m.sum()
```

    5.325821234912031

``` python
# Aggregation functions take an additional argument specifying the axis along which the aggregate is computed.
m.min(axis=0)
```

    array([0.15322668, 0.10058762, 0.68039294, 0.0045308 ])

``` python
np.max(m)
```

    0.9677193034993363

``` python
m.max(axis=0)
```

    array([0.57826711, 0.61526756, 0.9677193 , 0.43271134])

``` python
m.min(axis=1)
```

    array([0.10058762, 0.0045308 , 0.43271134])

**The way the axis is specified here can be confusing to users coming
from other lan‐ guages. The axis keyword specifies the dimension of the
array that will be collapsed, rather than the dimension that will be
returned. So specifying axis=0 means that the The way the axis is
specified here can be confusing to users coming from other lan‐ guages.
The axis keyword specifies the dimension of the array that will be
collapsed, rather than the dimension that will be returned. So
specifying axis=0 means that the first axis will be collapsed: for
two-dimensional arrays, this means that values within each column will
be aggregated.**

## Some aggregation functions

![](3.png)

### Example: What Is the Average Height of US Presidents?

Aggregates available in NumPy can be extremely useful for summarizing a
set of val‐ ues. As a simple example, let’s consider the heights of all
US presidents.

``` python
!head -4 ../data/president_heights.csv
```

    order,name,height(cm)
    1,George Washington,189
    2,John Adams,170
    3,Thomas Jefferson,189

``` python
import pandas as pd
data=pd.read_csv("../data/president_heights.csv")
height=np.array(data['height(cm)'])
print(height)
```

    [189 170 189 163 183 171 185 168 173 183 173 173 175 178 183 193 178 173
     174 183 183 168 170 178 182 180 183 178 182 188 175 179 183 193 182 183
     177 185 188 188 182 185]

``` python
data.head()
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
      <th>order</th>
      <th>name</th>
      <th>height(cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>George Washington</td>
      <td>189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>John Adams</td>
      <td>170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Thomas Jefferson</td>
      <td>189</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>James Madison</td>
      <td>163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>James Monroe</td>
      <td>183</td>
    </tr>
  </tbody>
</table>
</div>

``` python
print(f"Mean Height = ",np.mean(height))
print(f"St.dv Height = ",np.std(height))
```

    Mean Height =  179.73809523809524
    St.dv Height =  6.931843442745892

``` python
print(f"Mean Height = ",height.mean())
print(f"St.dv Height = ",height.std())
```

    Mean Height =  179.73809523809524
    St.dv Height =  6.931843442745892

``` python
print(f"Max Height = ",np.max(height))
print(f"Min Height = ",np.min(height))
```

    Max Height =  193
    Min Height =  163

``` python
print(f"25th precentile Height = ",np.percentile(height,25))
print(f"Median Height = ",np.median(height))
print(f"75th Percentile = ",np.percentile(height,75))
```

    25th precentile Height =  174.25
    Median Height =  182.0
    75th Percentile =  183.0

**We see that the median height of US presidents is 182 cm, or just shy
of six feet**

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
```

``` python
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');
```

![](03_Aggregation%20Min%20Max_files/figure-gfm/cell-23-output-1.png)


# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`


<a href=''>Top</a>