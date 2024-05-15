Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 1 - Introduction to Numpy**

# 05 - Comparisons, Masks, and Boolean Logic

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
--
- <a href="#comparisons-masks-and-boolean-logic"
  id="toc-comparisons-masks-and-boolean-logic">Comparisons, Masks, and
  Boolean Logic</a>
  - <a href="#example-counting-rainy-days"
    id="toc-example-counting-rainy-days">Example: Counting Rainy Days</a>
    - <a href="#digging-into-the-data" id="toc-digging-into-the-data">Digging
      into the data</a>
  - <a href="#comparison-operators-as-ufuncs"
    id="toc-comparison-operators-as-ufuncs">Comparison Operators as
    ufuncs</a>
  - <a href="#working-with-boolean-arrays"
    id="toc-working-with-boolean-arrays">Working with Boolean Arrays</a>
    - <a href="#counting-entries" id="toc-counting-entries">Counting
      entries</a>
    - <a href="#boolean-operators" id="toc-boolean-operators">Boolean
      Operators</a>
  - <a href="#boolean-arrays-as-masks"
    id="toc-boolean-arrays-as-masks">Boolean Arrays as Masks</a>

------------------------------------------------------------------------

# Comparisons, Masks, and Boolean Logic

This section covers the use of Boolean masks to examine and manipulate
values within NumPy arrays. Masking comes up when you want to extract,
modify, count, or otherwise manipulate values in an array based on some
criterion: for example, you might wish to count all values greater than
a certain value, or perhaps remove all out‐ liers that are above some
threshold. In NumPy, Boolean masking is often the most efficient way to
accomplish these types of tasks.

## Example: Counting Rainy Days

``` python
import numpy as np
import pandas as pd

rainfall = pd.read_csv("../data/Seattle2014.csv")['PRCP'].values
rainfall
```

    array([  0,  41,  15,   0,   0,   3, 122,  97,  58,  43, 213,  15,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   5,   0,   0,   0,   0,
             0,  89, 216,   0,  23,  20,   0,   0,   0,   0,   0,   0,  51,
             5, 183, 170,  46,  18,  94, 117, 264, 145, 152,  10,  30,  28,
            25,  61, 130,   3,   0,   0,   0,   5, 191, 107, 165, 467,  30,
             0, 323,  43, 188,   0,   0,   5,  69,  81, 277,   3,   0,   5,
             0,   0,   0,   0,   0,  41,  36,   3, 221, 140,   0,   0,   0,
             0,  25,   0,  46,   0,   0,  46,   0,   0,   0,   0,   0,   0,
             5, 109, 185,   0, 137,   0,  51, 142,  89, 124,   0,  33,  69,
             0,   0,   0,   0,   0, 333, 160,  51,   0,   0, 137,  20,   5,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  38,
             0,  56,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,  18,  64,   0,   5,  36,  13,   0,
             8,   3,   0,   0,   0,   0,   0,   0,  18,  23,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   3, 193,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   5,   0,   0,   0,   0,   0,   0,   0,
             0,   5, 127, 216,   0,  10,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,  84,  13,   0,  30,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,
             3,   0,   0,   0,   3, 183, 203,  43,  89,   0,   0,   8,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  74,   0,  76,
            71,  86,   0,  33, 150,   0, 117,  10, 320,  94,  41,  61,  15,
             8, 127,   5, 254, 170,   0,  18, 109,  41,  48,  41,   0,   0,
            51,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36, 152,
             5, 119,  13, 183,   3,  33, 343,  36,   0,   0,   0,   0,   8,
            30,  74,   0,  91,  99, 130,  69,   0,   0,   0,   0,   0,  28,
           130,  30, 196,   0,   0, 206,  53,   0,   0,  33,  41,   0,   0,
             0])

``` python
inches=rainfall/254
inches.shape
```

    (365,)

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.hist(inches,40);
```

![](05_Comparisons%20Masks%20and%20Boolean%20Logic_files/figure-gfm/cell-5-output-1.png)

### Digging into the data

## Comparison Operators as ufuncs

``` python
x=np.array([0,0,0,1,2,3,4,5])
x<3
```

    array([ True,  True,  True,  True,  True, False, False, False])

``` python
x>2
```

    array([False, False, False, False, False,  True,  True,  True])

``` python
x>=2
```

    array([False, False, False, False,  True,  True,  True,  True])

``` python
x<=3
```

    array([ True,  True,  True,  True,  True,  True, False, False])

``` python
x!=3
```

    array([ True,  True,  True,  True,  True, False,  True,  True])

``` python
x==3
```

    array([False, False, False, False, False,  True, False, False])

``` python
(2*x)==(x**2)
```

    array([ True,  True,  True, False,  True, False, False, False])

``` python
rng=np.random.RandomState(0)
x=rng.randint(10, size=(3,4))
```

``` python
x
```

    array([[5, 0, 3, 3],
           [7, 9, 3, 5],
           [2, 4, 7, 6]])

``` python
x<4
```

    array([[False,  True,  True,  True],
           [False, False,  True, False],
           [ True, False, False, False]])

## Working with Boolean Arrays

### Counting entries

``` python
print(x)
```

    [[5 0 3 3]
     [7 9 3 5]
     [2 4 7 6]]

``` python
np.count_nonzero(x<4)
```

    5

``` python
np.sum(x>3)
```

    7

``` python
np.sum(x<3,axis=1) #checking each row
```

    array([1, 0, 1])

``` python
np.sum(x<3,axis=0) #checking each col
```

    array([1, 1, 0, 0])

***If we’re interested in quickly checking whether any or all the values
are true, we can use (you guessed it) np.any() or np.all():***

``` python
np.any(x>8)
```

    True

``` python
np.any(x<5)
```

    True

``` python
np.all(x==6)
```

    False

``` python
np.all(x<10)
```

    True

``` python
np.all(x<8, axis=1)
```

    array([ True, False,  True])

### Boolean Operators

We’ve already seen how we might count, say, all days with rain less than
four inches, or all days with rain greater than two inches. But what if
we want to know about all days with rain less than four inches and
greater than one inch? This is accomplished through Python’s bitwise
logic operators, &, \|, ^, and \~. Like with the standard arith‐ metic
operators, NumPy overloads these as ufuncs that work element-wise on
(usu‐ ally Boolean) arrays.

``` python
np.sum((inches>0.5)&(inches<1))
```

    29

``` python
inches>(0.5 & inches)<1
```

    TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

``` python
print("Number of days without rain: ", np.sum(inches==0))
print("Number of days with rain: ", np.sum(inches!=0))
print("Days with more than 0.5 inches: ", np.sum(inches>0.5))
print("Rainy days with <0.1 inches: ", np.sum((inches>0) & (inches<0.2)))
```

    Number of days without rain:  215
    Number of days with rain:  150
    Days with more than 0.5 inches:  37
    Rainy days with <0.1 inches:  75

## Boolean Arrays as Masks

In the preceding section, we looked at aggregates computed directly on
Boolean arrays. A more powerful pattern is to use Boolean arrays as
masks, to select particular subsets of the data themselves.

``` python
x
```

    array([[5, 0, 3, 3],
           [7, 9, 3, 5],
           [2, 4, 7, 6]])

``` python
x<5
```

    array([[False,  True,  True,  True],
           [False, False,  True, False],
           [ True,  True, False, False]])

``` python
x[x<5]
```

    array([0, 3, 3, 3, 2, 4])

``` python
# Construct a mask of all rainy days
rainy=(inches>0)

# A construct of all summer days 
summer = (np.arange(365)-172<90)&(np.arange(365)-172>0)
```

``` python
rainy
```

    array([False,  True,  True, False, False,  True,  True,  True,  True,
            True,  True,  True, False, False, False, False, False, False,
           False, False, False,  True, False, False, False, False, False,
            True,  True, False,  True,  True, False, False, False, False,
           False, False,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True, False, False, False,  True,  True,  True,  True,
            True,  True, False,  True,  True,  True, False, False,  True,
            True,  True,  True,  True, False,  True, False, False, False,
           False, False,  True,  True,  True,  True,  True, False, False,
           False, False,  True, False,  True, False, False,  True, False,
           False, False, False, False, False,  True,  True,  True, False,
            True, False,  True,  True,  True,  True, False,  True,  True,
           False, False, False, False, False,  True,  True,  True, False,
           False,  True,  True,  True, False, False, False, False, False,
           False, False, False, False, False, False, False,  True, False,
            True, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
            True,  True, False,  True,  True,  True, False,  True,  True,
           False, False, False, False, False, False,  True,  True, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False,  True,  True, False, False, False,
           False, False, False, False, False, False,  True, False, False,
           False, False, False, False, False, False,  True,  True,  True,
           False,  True, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True,  True,
           False,  True, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True,  True,
           False, False, False,  True,  True,  True,  True,  True, False,
           False,  True, False, False, False, False, False, False, False,
           False, False, False,  True,  True, False,  True,  True,  True,
           False,  True,  True, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,
            True,  True,  True,  True, False, False,  True, False, False,
           False, False, False, False, False, False, False, False,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False, False, False, False,  True,  True,  True, False,  True,
            True,  True,  True, False, False, False, False, False,  True,
            True,  True,  True, False, False,  True,  True, False, False,
            True,  True, False, False, False])

``` python
summer
```

    array([False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False])

``` python
print("Median precip on rainy days in 2014 (inches): ",np.median(inches[rainy]))
print("Median precip on summber days in 2014 (inches): ",np.median(inches[summer]))
print("Maximum precip on summer in 2014 (inches): ", np.max(inches[summer]))
print("Median precip on summer in 2014 (inches) on non-summer rainy days: ",np.median(inches[rainy & ~summer]) )
```

    Median precip on rainy days in 2014 (inches):  0.19488188976377951
    Median precip on summber days in 2014 (inches):  0.0
    Maximum precip on summer in 2014 (inches):  0.8503937007874016
    Median precip on summer in 2014 (inches) on non-summer rainy days:  0.20078740157480315

**By combining Boolean operations, masking operations, and aggregates,
we can very quickly answer these sorts of questions for our dataset.**\_

> When you use & and \| on integers, the expression operates on the bits
> of the element, applying the and or the or to the individual bits
> making up the number:  
> When you use and or or, it’s equivalent to asking Python to treat the
> object as a single Boolean entity. In Python, all nonzero integers
> will evaluate as True. Thus
