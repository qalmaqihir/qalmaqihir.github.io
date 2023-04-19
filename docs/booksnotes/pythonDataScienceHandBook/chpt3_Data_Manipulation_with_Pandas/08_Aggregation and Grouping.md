Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 07 - Aggregation and Grouping
------------------------------------------------------------------------

- <a href="#aggregation-and-grouping"
  id="toc-aggregation-and-grouping">Aggregation and Grouping</a>
  - <a href="#aggregate-filter-transform-apply"
    id="toc-aggregate-filter-transform-apply">Aggregate, filter, transform,
    apply</a>

------------------------------------------------------------------------


# Aggregation and Grouping

An essential piece of analysis of large data is efficient summarization:
computing aggregations like sum(), mean(), median(), min(), and max(),
in which a single num‐ ber gives insight into the nature of a
potentially large dataset. In this section, we’ll explore aggregations
in Pandas, from simple operations akin to what we’ve seen on NumPy
arrays, to more sophisticated operations based on the concept of a
groupby. \## Planets Data

``` python
import seaborn as sns
import numpy as np
import pandas as pd
```

``` python
planets= sns.load_dataset('planets')
planets.shape
```

    (1035, 6)

``` python
planets.head()
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
      <th>method</th>
      <th>number</th>
      <th>orbital_period</th>
      <th>mass</th>
      <th>distance</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>269.300</td>
      <td>7.10</td>
      <td>77.40</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>874.774</td>
      <td>2.21</td>
      <td>56.95</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>763.000</td>
      <td>2.60</td>
      <td>19.84</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>326.030</td>
      <td>19.40</td>
      <td>110.62</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>516.220</td>
      <td>10.50</td>
      <td>119.47</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# simple aggregation in Pandas
rng=np.random.RandomState(42)
ser=pd.Series(rng.rand(5))
ser
```

    0    0.374540
    1    0.950714
    2    0.731994
    3    0.598658
    4    0.156019
    dtype: float64

``` python
ser.sum()
```

    2.811925491708157

``` python
ser.mean()
```

    0.5623850983416314

``` python
df=pd.DataFrame({'A':rng.rand(5),'B':rng.rand(5)})
df
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.155995</td>
      <td>0.020584</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.058084</td>
      <td>0.969910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.866176</td>
      <td>0.832443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.601115</td>
      <td>0.212339</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.708073</td>
      <td>0.181825</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.mean()
```

    A    0.477888
    B    0.443420
    dtype: float64

``` python
df.sum()
```

    A    2.389442
    B    2.217101
    dtype: float64

``` python
planets.dropna().describe()
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
      <th>number</th>
      <th>orbital_period</th>
      <th>mass</th>
      <th>distance</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>498.00000</td>
      <td>498.000000</td>
      <td>498.000000</td>
      <td>498.000000</td>
      <td>498.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.73494</td>
      <td>835.778671</td>
      <td>2.509320</td>
      <td>52.068213</td>
      <td>2007.377510</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.17572</td>
      <td>1469.128259</td>
      <td>3.636274</td>
      <td>46.596041</td>
      <td>4.167284</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.328300</td>
      <td>0.003600</td>
      <td>1.350000</td>
      <td>1989.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.00000</td>
      <td>38.272250</td>
      <td>0.212500</td>
      <td>24.497500</td>
      <td>2005.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.00000</td>
      <td>357.000000</td>
      <td>1.245000</td>
      <td>39.940000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.00000</td>
      <td>999.600000</td>
      <td>2.867500</td>
      <td>59.332500</td>
      <td>2011.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.00000</td>
      <td>17337.500000</td>
      <td>25.000000</td>
      <td>354.000000</td>
      <td>2014.000000</td>
    </tr>
  </tbody>
</table>
</div>

``` python
planets.count()
```

    method            1035
    number            1035
    orbital_period     992
    mass               513
    distance           808
    year              1035
    dtype: int64

These are all methods of DataFrame and Series objects. To go deeper into
the data, however, simple aggregates are often not enough. The next
level of data summarization is the groupby operation, which allows you
to quickly and efficiently compute aggregates on subsets of data. \##
GroupBy: Split, Apply, Combine Simple aggregations can give you a flavor
of your dataset, but often we would prefer to aggregate conditionally on
some label or index: this is implemented in the so- called groupby
operation. The name “group by” comes from a command in the SQL database
language, but it is perhaps more illuminative to think of it in the
terms first coined by Hadley Wickham of Rstats fame: split, apply,
combine. \### Split, apply, combine A canonical example of this
split-apply-combine operation, where the “apply” is a summation
aggregation:  
**• The split step involves breaking up and grouping a DataFrame
depending on the value of the specified key.  
• The apply step involves computing some function, usually an aggregate,
transformation, or filtering, within the individual groups.  
• The combine step merges the results of these operations into an output
array.**

While we could certainly do this manually using some combination of the
masking, aggregation, and merging commands covered earlier, it’s
important to realize that the intermediate splits do not need to be
explicitly instantiated. Rather, the GroupBy can (often) do this in a
single pass over the data, updating the sum, mean, count, min, or other
aggregate for each group along the way. The power of the GroupBy is that
it abstracts away these steps: the user need not think about how the
computation is done under the hood, but rather thinks about the
operation as a whole.

``` python
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data': range(6)}, columns=['key', 'data'])
df
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
      <th>key</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.groupby('key') # a groupby object is created
```

    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f9e6c0b6a30>

*Notice that what is returned is not a set of DataFrames, but a
DataFrameGroupBy object. This object is where the magic is: you can
think of it as a special view of the DataFrame, which is poised to dig
into the groups but does no actual computation until the aggregation is
applied. This “lazy evaluation” approach means that common aggregates
can be implemented very efficiently in a way that is almost transparent
to the user.*

``` python
df.groupby('key').sum()
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
      <th>data</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>3</td>
    </tr>
    <tr>
      <th>B</th>
      <td>5</td>
    </tr>
    <tr>
      <th>C</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>

Let’s introduce some of the other func‐ tionality that can be used with
the basic GroupBy operation.  
**Column indexing.** The GroupBy object supports column indexing in the
same way as the DataFrame, and returns a modified GroupBy object.

``` python
planets.groupby('method')
```

    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f9e6e0c71f0>

``` python
planets.groupby('method')['orbital_period']
```

    <pandas.core.groupby.generic.SeriesGroupBy object at 0x7f9e6c56f9a0>

Here we’ve selected a particular Series group from the original
DataFrame group by reference to its column name. As with the GroupBy
object, no computation is done until we call some aggregate on the
object:

``` python
planets.groupby('method')['orbital_period'].median()
```

    method
    Astrometry                         631.180000
    Eclipse Timing Variations         4343.500000
    Imaging                          27500.000000
    Microlensing                      3300.000000
    Orbital Brightness Modulation        0.342887
    Pulsar Timing                       66.541900
    Pulsation Timing Variations       1170.000000
    Radial Velocity                    360.200000
    Transit                              5.714932
    Transit Timing Variations           57.011000
    Name: orbital_period, dtype: float64

**Iteration over groups**. The GroupBy object supports direct iteration
over the groups, returning each group as a Series or DataFrame

``` python
for (method,group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))
```

    Astrometry                     shape=(2, 6)
    Eclipse Timing Variations      shape=(9, 6)
    Imaging                        shape=(38, 6)
    Microlensing                   shape=(23, 6)
    Orbital Brightness Modulation  shape=(3, 6)
    Pulsar Timing                  shape=(5, 6)
    Pulsation Timing Variations    shape=(1, 6)
    Radial Velocity                shape=(553, 6)
    Transit                        shape=(397, 6)
    Transit Timing Variations      shape=(4, 6)

**Dispatch methods.** Through some Python class magic, any method not
explicitly implemented by the GroupBy object will be passed through and
called on the groups, whether they are DataFrame or Series objects. For
example, you can use the describe() method of DataFrames to perform a
set of aggregations that describe each group in the data:

``` python
planets.groupby('method')['year'].describe().unstack()
```

           method                       
    count  Astrometry                          2.0
           Eclipse Timing Variations           9.0
           Imaging                            38.0
           Microlensing                       23.0
           Orbital Brightness Modulation       3.0
                                             ...  
    max    Pulsar Timing                    2011.0
           Pulsation Timing Variations      2007.0
           Radial Velocity                  2014.0
           Transit                          2014.0
           Transit Timing Variations        2014.0
    Length: 80, dtype: float64

## Aggregate, filter, transform, apply

The preceding discussion focused on aggregation for the combine
operation, but there are more options available. In particular, GroupBy
objects have aggregate(), filter(), transform(), and apply() methods
that efficiently implement a variety of useful operations before
combining the grouped data.

``` python
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data1': range(6),
'data2': rng.randint(0, 10, 6)},
columns = ['key', 'data1', 'data2'])
```

``` python
df
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

**Aggregation.** We’re now familiar with GroupBy aggregations with
sum(), median(), and the like, but the aggregate() method allows for
even more flexibility. It can take a string, a function, or a list
thereof, and compute all the aggregates at once. Here is a quick example
combining all these:

``` python
df.groupby('key').aggregate(['min',np.median, max])
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">data1</th>
      <th colspan="3" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>min</th>
      <th>median</th>
      <th>max</th>
      <th>min</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>1.5</td>
      <td>3</td>
      <td>3</td>
      <td>4.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>2.5</td>
      <td>4</td>
      <td>0</td>
      <td>3.5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>3.5</td>
      <td>5</td>
      <td>3</td>
      <td>6.0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.groupby('key').aggregate({'data1':'min','data2':'max'})
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

**Filtering.** A filtering operation allows you to drop data based on
the group proper‐ ties. For example, we might want to keep all groups in
which the standard deviation is larger than some critical value:

``` python
def filter_func(x):
    return x['data2'].std()>4

print(df)
```

      key  data1  data2
    0   A      0      5
    1   B      1      0
    2   C      2      3
    3   A      3      3
    4   B      4      7
    5   C      5      9

``` python
df.groupby('key').std()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.12132</td>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>B</th>
      <td>2.12132</td>
      <td>4.949747</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2.12132</td>
      <td>4.242641</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.groupby('key').filter(filter_func)
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

**Transformation.** While aggregation must return a reduced version of
the data, trans‐ formation can return some transformed version of the
full data to recombine. For such a transformation, the output is the
same shape as the input. A common example is to center the data by
subtracting the group-wise mean:

``` python
df.groupby('key').transform(lambda s: s - s.mean())
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.5</td>
      <td>-3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.5</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.5</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>

**The apply() method.** The apply() method lets you apply an arbitrary
function to the group results. The function should take a DataFrame, and
return either a Pandas object (e.g., DataFrame, Series) or a scalar; the
combine operation will be tailored to the type of output returned.

``` python
def norm_by_data2(x):
    x['data1']/=x['data2'].sum()
    return x
```

``` python
df
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.groupby('key').apply(norm_by_data2)
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0.000000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>0.142857</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>0.166667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>0.375000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>0.571429</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>0.416667</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>

**Specifying the split key** In the simple examples presented before, we
split the DataFrame on a single column name. This is just one of many
options by which the groups can be defined, and we’ll go through some
other options for group specification here. **A list, array, series, or
index providing the grouping keys.** The key can be any series or list
with a length matching that of the DataFrame.

``` python
l=[0,1,0,1,2,0]
```

``` python
df.groupby(l).sum()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.groupby('key').sum()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>B</th>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>C</th>
      <td>7</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>
