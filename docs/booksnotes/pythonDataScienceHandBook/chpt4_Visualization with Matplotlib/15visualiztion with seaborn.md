================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 15 -  Visualization with Seaborn
------------------------------------------------------------------------

- <a href="#visualization-with-seaborn"
  id="toc-visualization-with-seaborn">Visualization with Seaborn</a>
  - <a href="#seaborn-versus-matplotlib"
    id="toc-seaborn-versus-matplotlib">Seaborn Versus Matplotlib</a>
  - <a href="#exploring-seaborn-plots"
    id="toc-exploring-seaborn-plots">Exploring Seaborn Plots</a>
    - <a href="#histograms-kde-and-densities"
      id="toc-histograms-kde-and-densities">Histograms, KDE, and densities</a>
    - <a href="#pair-plots" id="toc-pair-plots">Pair plots</a>
    - <a href="#faceted-histograms" id="toc-faceted-histograms">Faceted
      histograms</a>
    - <a href="#factor-plots" id="toc-factor-plots">Factor plots</a>
    - <a href="#joint-distributions" id="toc-joint-distributions">Joint
      distributions</a>
    - <a href="#bar-plots" id="toc-bar-plots">Bar plots</a>
------------------------------------------------------------------------

# Visualization with Seaborn

Matplotlib has proven to be an incredibly useful and popular
visualization tool, but even avid users will admit it often leaves much
to be desired. There are several valid complaints about Matplotlib that
often come up:

• Prior to version 2.0, Matplotlib’s defaults are not exactly the best
choices. It was based off of MATLAB circa 1999, and this often shows.  
• Matplotlib’s API is relatively low level. Doing sophisticated
statistical visualiza‐ tion is possible, but often requires a lot of
boilerplate code.

• Matplotlib predated Pandas by more than a decade, and thus is not
designed for use with Pandas DataFrames. In order to visualize data from
a Pandas DataFrame, you must extract each Series and often concatenate
them together into the right format. It would be nicer to have a
plotting library that can intelligently use the DataFrame labels in a
plot.

An answer to these problems is Seaborn. Seaborn provides an API on top
of Matplot‐ lib that offers sane choices for plot style and color
defaults, defines simple high-level functions for common statistical
plot types, and integrates with the functionality pro‐ vided by Pandas
DataFrames.

## Seaborn Versus Matplotlib

Here is an example of a simple random-walk plot in Matplotlib, using its
classic plot formatting and colors.

``` python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd
```

``` python
# create some data
rng=np.random.RandomState(0)
x=np.linspace(0,10,500)
y=np.cumsum(rng.randn(500,6),0)
```

``` python
x
```

    array([ 0.        ,  0.02004008,  0.04008016,  0.06012024,  0.08016032,
            0.1002004 ,  0.12024048,  0.14028056,  0.16032064,  0.18036072,
            0.2004008 ,  0.22044088,  0.24048096,  0.26052104,  0.28056112,
            0.3006012 ,  0.32064128,  0.34068136,  0.36072144,  0.38076152,
            0.4008016 ,  0.42084168,  0.44088176,  0.46092184,  0.48096192,
            0.501002  ,  0.52104208,  0.54108216,  0.56112224,  0.58116232,
            0.6012024 ,  0.62124248,  0.64128257,  0.66132265,  0.68136273,
            0.70140281,  0.72144289,  0.74148297,  0.76152305,  0.78156313,
            0.80160321,  0.82164329,  0.84168337,  0.86172345,  0.88176353,
            0.90180361,  0.92184369,  0.94188377,  0.96192385,  0.98196393,
            1.00200401,  1.02204409,  1.04208417,  1.06212425,  1.08216433,
            1.10220441,  1.12224449,  1.14228457,  1.16232465,  1.18236473,
            1.20240481,  1.22244489,  1.24248497,  1.26252505,  1.28256513,
            1.30260521,  1.32264529,  1.34268537,  1.36272545,  1.38276553,
            1.40280561,  1.42284569,  1.44288577,  1.46292585,  1.48296593,
            1.50300601,  1.52304609,  1.54308617,  1.56312625,  1.58316633,
            1.60320641,  1.62324649,  1.64328657,  1.66332665,  1.68336673,
            1.70340681,  1.72344689,  1.74348697,  1.76352705,  1.78356713,
            1.80360721,  1.82364729,  1.84368737,  1.86372745,  1.88376754,
            1.90380762,  1.9238477 ,  1.94388778,  1.96392786,  1.98396794,
            2.00400802,  2.0240481 ,  2.04408818,  2.06412826,  2.08416834,
            2.10420842,  2.1242485 ,  2.14428858,  2.16432866,  2.18436874,
            2.20440882,  2.2244489 ,  2.24448898,  2.26452906,  2.28456914,
            2.30460922,  2.3246493 ,  2.34468938,  2.36472946,  2.38476954,
            2.40480962,  2.4248497 ,  2.44488978,  2.46492986,  2.48496994,
            2.50501002,  2.5250501 ,  2.54509018,  2.56513026,  2.58517034,
            2.60521042,  2.6252505 ,  2.64529058,  2.66533066,  2.68537074,
            2.70541082,  2.7254509 ,  2.74549098,  2.76553106,  2.78557114,
            2.80561122,  2.8256513 ,  2.84569138,  2.86573146,  2.88577154,
            2.90581162,  2.9258517 ,  2.94589178,  2.96593186,  2.98597194,
            3.00601202,  3.0260521 ,  3.04609218,  3.06613226,  3.08617234,
            3.10621242,  3.12625251,  3.14629259,  3.16633267,  3.18637275,
            3.20641283,  3.22645291,  3.24649299,  3.26653307,  3.28657315,
            3.30661323,  3.32665331,  3.34669339,  3.36673347,  3.38677355,
            3.40681363,  3.42685371,  3.44689379,  3.46693387,  3.48697395,
            3.50701403,  3.52705411,  3.54709419,  3.56713427,  3.58717435,
            3.60721443,  3.62725451,  3.64729459,  3.66733467,  3.68737475,
            3.70741483,  3.72745491,  3.74749499,  3.76753507,  3.78757515,
            3.80761523,  3.82765531,  3.84769539,  3.86773547,  3.88777555,
            3.90781563,  3.92785571,  3.94789579,  3.96793587,  3.98797595,
            4.00801603,  4.02805611,  4.04809619,  4.06813627,  4.08817635,
            4.10821643,  4.12825651,  4.14829659,  4.16833667,  4.18837675,
            4.20841683,  4.22845691,  4.24849699,  4.26853707,  4.28857715,
            4.30861723,  4.32865731,  4.34869739,  4.36873747,  4.38877756,
            4.40881764,  4.42885772,  4.4488978 ,  4.46893788,  4.48897796,
            4.50901804,  4.52905812,  4.5490982 ,  4.56913828,  4.58917836,
            4.60921844,  4.62925852,  4.6492986 ,  4.66933868,  4.68937876,
            4.70941884,  4.72945892,  4.749499  ,  4.76953908,  4.78957916,
            4.80961924,  4.82965932,  4.8496994 ,  4.86973948,  4.88977956,
            4.90981964,  4.92985972,  4.9498998 ,  4.96993988,  4.98997996,
            5.01002004,  5.03006012,  5.0501002 ,  5.07014028,  5.09018036,
            5.11022044,  5.13026052,  5.1503006 ,  5.17034068,  5.19038076,
            5.21042084,  5.23046092,  5.250501  ,  5.27054108,  5.29058116,
            5.31062124,  5.33066132,  5.3507014 ,  5.37074148,  5.39078156,
            5.41082164,  5.43086172,  5.4509018 ,  5.47094188,  5.49098196,
            5.51102204,  5.53106212,  5.5511022 ,  5.57114228,  5.59118236,
            5.61122244,  5.63126253,  5.65130261,  5.67134269,  5.69138277,
            5.71142285,  5.73146293,  5.75150301,  5.77154309,  5.79158317,
            5.81162325,  5.83166333,  5.85170341,  5.87174349,  5.89178357,
            5.91182365,  5.93186373,  5.95190381,  5.97194389,  5.99198397,
            6.01202405,  6.03206413,  6.05210421,  6.07214429,  6.09218437,
            6.11222445,  6.13226453,  6.15230461,  6.17234469,  6.19238477,
            6.21242485,  6.23246493,  6.25250501,  6.27254509,  6.29258517,
            6.31262525,  6.33266533,  6.35270541,  6.37274549,  6.39278557,
            6.41282565,  6.43286573,  6.45290581,  6.47294589,  6.49298597,
            6.51302605,  6.53306613,  6.55310621,  6.57314629,  6.59318637,
            6.61322645,  6.63326653,  6.65330661,  6.67334669,  6.69338677,
            6.71342685,  6.73346693,  6.75350701,  6.77354709,  6.79358717,
            6.81362725,  6.83366733,  6.85370741,  6.87374749,  6.89378758,
            6.91382766,  6.93386774,  6.95390782,  6.9739479 ,  6.99398798,
            7.01402806,  7.03406814,  7.05410822,  7.0741483 ,  7.09418838,
            7.11422846,  7.13426854,  7.15430862,  7.1743487 ,  7.19438878,
            7.21442886,  7.23446894,  7.25450902,  7.2745491 ,  7.29458918,
            7.31462926,  7.33466934,  7.35470942,  7.3747495 ,  7.39478958,
            7.41482966,  7.43486974,  7.45490982,  7.4749499 ,  7.49498998,
            7.51503006,  7.53507014,  7.55511022,  7.5751503 ,  7.59519038,
            7.61523046,  7.63527054,  7.65531062,  7.6753507 ,  7.69539078,
            7.71543086,  7.73547094,  7.75551102,  7.7755511 ,  7.79559118,
            7.81563126,  7.83567134,  7.85571142,  7.8757515 ,  7.89579158,
            7.91583166,  7.93587174,  7.95591182,  7.9759519 ,  7.99599198,
            8.01603206,  8.03607214,  8.05611222,  8.0761523 ,  8.09619238,
            8.11623246,  8.13627255,  8.15631263,  8.17635271,  8.19639279,
            8.21643287,  8.23647295,  8.25651303,  8.27655311,  8.29659319,
            8.31663327,  8.33667335,  8.35671343,  8.37675351,  8.39679359,
            8.41683367,  8.43687375,  8.45691383,  8.47695391,  8.49699399,
            8.51703407,  8.53707415,  8.55711423,  8.57715431,  8.59719439,
            8.61723447,  8.63727455,  8.65731463,  8.67735471,  8.69739479,
            8.71743487,  8.73747495,  8.75751503,  8.77755511,  8.79759519,
            8.81763527,  8.83767535,  8.85771543,  8.87775551,  8.89779559,
            8.91783567,  8.93787575,  8.95791583,  8.97795591,  8.99799599,
            9.01803607,  9.03807615,  9.05811623,  9.07815631,  9.09819639,
            9.11823647,  9.13827655,  9.15831663,  9.17835671,  9.19839679,
            9.21843687,  9.23847695,  9.25851703,  9.27855711,  9.29859719,
            9.31863727,  9.33867735,  9.35871743,  9.37875752,  9.3987976 ,
            9.41883768,  9.43887776,  9.45891784,  9.47895792,  9.498998  ,
            9.51903808,  9.53907816,  9.55911824,  9.57915832,  9.5991984 ,
            9.61923848,  9.63927856,  9.65931864,  9.67935872,  9.6993988 ,
            9.71943888,  9.73947896,  9.75951904,  9.77955912,  9.7995992 ,
            9.81963928,  9.83967936,  9.85971944,  9.87975952,  9.8997996 ,
            9.91983968,  9.93987976,  9.95991984,  9.97995992, 10.        ])

``` python
y
```

    array([[  1.76405235,   0.40015721,   0.97873798,   2.2408932 ,
              1.86755799,  -0.97727788],
           [  2.71414076,   0.2488    ,   0.87551913,   2.6514917 ,
              2.01160156,   0.47699563],
           [  3.47517849,   0.37047502,   1.31938237,   2.98516603,
              3.50568063,   0.27183736],
           ...,
           [-34.82533536, -44.37245964, -32.86660099,  31.93843765,
              9.67250307,  -9.16537805],
           [-35.4875268 , -45.95006671, -33.20716103,  30.63521756,
             10.13925372,  -9.00427173],
           [-35.16749487, -43.87089005, -34.11462701,  30.44281336,
              8.92673797,  -9.08487024]])

``` python
# plot the data usng matplotlib defaults
plt.plot(x,y)
plt.legend("ABCDEG", ncol=2, loc='upper left')
```

    <matplotlib.legend.Legend at 0x7f9257fd7e50>

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-6-output-2.png)

Although the result contains all the information we’d like it to convey,
it does so in a way that is not all that aesthetically pleasing, and
even looks a bit old-fashioned in the context of 21st-century data
visualization.

``` python
import seaborn as sns
sns.set()
# some plotting code as above

plt.plot(x,y)
plt.legend("ABCDEF", ncol=2, loc='upper left')
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.2
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"

    <matplotlib.legend.Legend at 0x7f9253d88220>

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-7-output-3.png)

## Exploring Seaborn Plots

The main idea of Seaborn is that it provides high-level commands to
create a variety of plot types useful for statistical data exploration,
and even some statistical model fitting. Let’s take a look at a few of
the datasets and plot types available in Seaborn. Note that all of the
following could be done using raw Matplotlib commands (this is, in fact,
what Seaborn does under the hood), but the Seaborn API is much more
convenient.

### Histograms, KDE, and densities

Often in statistical data visualization, all you want is to plot
histograms and joint dis‐ tributions of variables. We have seen that
this is relatively straightforward in Matplot‐ lib (

``` python
data = np.random.multivariate_normal([0,0],[[5,2],[2,2]], size=2000)
data =pd.DataFrame(data, columns=['x','y'])
for col in 'xy':
    plt.hist(data[col], alpha=0.5)
```

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-8-output-1.png)

**Rather than a histogram, we can get a smooth estimate of the
distribution using a kernel density estimation, which Seaborn does with
sns.kdeplot**

``` python
for col in 'xy':
    sns.kdeplot(data[col], shade=True)
```

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-9-output-1.png)

**Histograms and KDE can be combined using displot**

``` python
sns.distplot(data['x'])
sns.distplot(data['y']);
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)
    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-10-output-2.png)

**If we pass the full two-dimensional dataset to kdeplot, we will get a
two-dimensional visualization of the data**

``` python
sns.kdeplot(data);
```

    ValueError: If using all scalar values, you must pass an index

**We can see the joint distribution and the marginal distributions
together using sns.jointplot. For this plot, we’ll set the style to a
white background**

``` python
with sns.axes_style('white'):
    sns.jointplot('x','y',data,kind='kde')
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y, data. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-12-output-2.png)

**There are other parameters that can be passed to jointplot—for
example, we can use a hexagonally based histogram instead**

``` python
with sns.axes_style('white'):
    sns.jointplot("x","y",data,kind='hex')
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y, data. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-13-output-2.png)

### Pair plots

When you generalize joint plots to datasets of larger dimensions, you
end up with pair plots. This is very useful for exploring correlations
between multidimensional data, when you’d like to plot all pairs of
values against each other.

We’ll demo this with the well-known Iris dataset, which lists
measurements of petals and sepals of three iris species:

``` python
iris=sns.load_dataset('iris')
iris.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# visulizing the multidimensional relationships among the samples is 
# as easy 
sns.pairplot(iris,hue='species', size=2.5)
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/axisgrid.py:2076: UserWarning: The `size` parameter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)

    <seaborn.axisgrid.PairGrid at 0x7f923c0e5910>

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-15-output-3.png)

### Faceted histograms

Sometimes the best way to view data is via histograms of subsets.
Seaborn’s FacetGrid makes this extremely simple. We’ll take a look at
some data that shows the amount that restaurant staff receive in tips
based on various indicator data

``` python
tips = sns.load_dataset('tips')
tips.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
tips['tip_pct']=100*tips['tip']/tips['total_bill']
tips['tip_pct']
```

    0       5.944673
    1      16.054159
    2      16.658734
    3      13.978041
    4      14.680765
             ...    
    239    20.392697
    240     7.358352
    241     8.822232
    242     9.820426
    243    15.974441
    Name: tip_pct, Length: 244, dtype: float64

``` python
grid=sns.FacetGrid(tips, row='sex',col='time', margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0,40,15));
```

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-18-output-1.png)

### Factor plots

Factor plots can be useful for this kind of visualization as well. This
allows you to view the distribution of a parameter within bins defined
by any other parameter

``` python
with sns.axes_style(style='ticks'):
    g=sns.factorplot('day','total_bill','sex', data=tips, kind='box')
    g.set_axis_labels('Day','Total Bill');
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y, hue. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-19-output-2.png)

### Joint distributions

Similar to the pair plot we saw earlier, we can use sns.jointplot to
show the joint distribution between different datasets, along with the
associated marginal distribu‐ tions

``` python
with sns.axes_style('white'):
    sns.jointplot('total_bill','tip', data=tips,kind='hex')
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-20-output-2.png)

``` python
# the joint plot can also do some automatic kernel density nd regression
sns.jointplot('total_bill','tip', data=tips,kind='reg');
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-21-output-2.png)

### Bar plots

Time series can be plotted with sns.factorplot

``` python
planets = sns.load_dataset('planets')
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
with sns.axes_style('white'):
    g=sns.factorplot('year', data=planets,aspect=2,
                    kind='count',color='steelblue')
    g.set_xticklabels(step=5)
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-23-output-2.png)

``` python
# we can learn more by looking at the method of discovery of each of these planets
with sns.axes_style('white'):
    g=sns.factorplot('year', data=planets,aspect=2,
                    kind='count',color='steelblue',hue='method', order=range(2001,2015))
    g.set_ylabels('Number of Planets Discovered')
```

    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/categorical.py:3717: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.
      warnings.warn(msg)
    /home/qalmaqihir/anaconda3/lib/python3.9/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(

![](15visualiztion%20with%20seaborn_files/figure-gfm/cell-24-output-2.png)
