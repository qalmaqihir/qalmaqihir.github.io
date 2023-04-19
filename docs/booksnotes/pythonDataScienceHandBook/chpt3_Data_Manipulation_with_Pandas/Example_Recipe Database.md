Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# Example: Recipe Database
------------------------------------------------------------------------

- <a href="#example-recipe-database"
  id="toc-example-recipe-database">Example: Recipe Database</a>
  - <a
    href="#unfortunately-the-dataset-is-not-present-to-do-all-the-other-operations"
    id="toc-unfortunately-the-dataset-is-not-present-to-do-all-the-other-operations">Unfortunately
    the dataset is not present to do all the other operations :(</a>

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
```

# Example: Recipe Database

These vectorized string operations become most useful in the process of
cleaning up messy, real-world data. Here I’ll walk through an example of
that, using an open recipe database compiled from various sources on the
Web. Our goal will be to parse the recipe data into ingredient lists, so
we can quickly find a recipe based on some ingredients we have on hand

``` python
try:
    recipes=pd.read_json('../data/recipeitems-latest.json')
except ValueError as e:
    print("Value Error: ", e)
```

    Value Error:  Expected object or value

``` python
with open('../data/recipeitems-latest.json') as f:
    line=f.readline()
pd.read_json(line).shape
```

    ValueError: Expected object or value

## Unfortunately the dataset is not present to do all the other operations :(
