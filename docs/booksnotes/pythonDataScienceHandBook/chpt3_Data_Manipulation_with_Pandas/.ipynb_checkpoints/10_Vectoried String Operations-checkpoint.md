Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 10 - Vectorized String Operations
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
------------------------------------------------------------------------
- <a href="#vectorized-string-operations"
  id="toc-vectorized-string-operations">Vectorized String Operations</a>
  - <a href="#tables-of-pandas-string-methods"
    id="toc-tables-of-pandas-string-methods">Tables of Pandas String
    Methods</a>

------------------------------------------------------------------------

# Vectorized String Operations

One strength of Python is its relative ease in handling and manipulating
string data. Pandas builds on this and provides a comprehensive set of
vectorized string operations that become an essential piece of the type
of munging required when one is working with (read: cleaning up)
real-world data. \## Introducing Pandas String Operations We saw in
previous sections how tools like NumPy and Pandas generalize arithmetic
operations so that we can easily and quickly perform the same operation
on many array elements.

``` python
import numpy as np
x=np.array([2,3,5,7,11,13])
x*2
```

    array([ 4,  6, 10, 14, 22, 26])

**This vectorization of operations simplifies the syntax of operating on
arrays of data: we no longer have to worry about the size or shape of
the array, but just about what operation we want done. For arrays of
strings, NumPy does not provide such simple access, and thus you’re
stuck using a more verbose loop syntax:**

``` python
data=['peteR','khan','Haider','KILLY','GuIDol',' ']
[s.capitalize() for s in data]
```

    ['Peter', 'Khan', 'Haider', 'Killy', 'Guidol', ' ']

``` python
data=['peteR','khan','Haider',None,'KILLY','GuIDol']
[s.capitalize() for s in data]
```

    AttributeError: 'NoneType' object has no attribute 'capitalize'

*Pandas includes features to address both this need for vectorized
string operations and for correctly handling missing data via the str
attribute of Pandas Series and Index objects containing strings.*

``` python
import pandas as pd
names=pd.Series(data)
names
```

    0     peteR
    1      khan
    2    Haider
    3      None
    4     KILLY
    5    GuIDol
    dtype: object

``` python
# Now we can capitalize All without any error
names.str.capitalize()
```

    0     Peter
    1      Khan
    2    Haider
    3      None
    4     Killy
    5    Guidol
    dtype: object

## Tables of Pandas String Methods

If you have a good understanding of string manipulation in Python, most
of Pandas’ string syntax is intuitive enough that it’s probably
sufficient to just list a table of avail‐ able methods; we will start
with that here, before diving deeper into a few of the sub‐ tleties.

``` python
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
'Eric Idle', 'Terry Jones', 'Michael Palin'])
```

``` python
monte
```

    0    Graham Chapman
    1       John Cleese
    2     Terry Gilliam
    3         Eric Idle
    4       Terry Jones
    5     Michael Palin
    dtype: object

Methods similar to Python string methods Nearly all Python’s built-in
string methods are mirrored by a Pandas vectorized string method. Here
is a list of Pandas str methods that mirror Python string methods:

|          |          |              |             |             |              |            |                |
|----------|----------|--------------|-------------|-------------|--------------|------------|----------------|
| len()    | lower()  | translate()  | ljust()     | upper()     | startswith() | isupper()  | islower()      |
| rjust()  | find()   | endswith()   | isnumeric() | center()    | rfind()      | isalnum()  |                |
| rsplit() | rstrip() | capitalize() | isspace()   | partition() | lstrip()     | swapcase() | rpartition()\` |
