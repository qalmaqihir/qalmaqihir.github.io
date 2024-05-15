Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 08 - Pivot Table
------------------------------------------------------------------------

- <a href="#pivot-tables" id="toc-pivot-tables">Pivot Tables</a>
  - <a href="#motivating-pivot-tables"
    id="toc-motivating-pivot-tables">Motivating pivot tables</a>
  - <a href="#pivot-tables-by-hand" id="toc-pivot-tables-by-hand">Pivot
    Tables by Hand</a>
  - <a href="#pivot-table-syntax" id="toc-pivot-table-syntax">Pivot Table
    Syntax</a>
  - <a href="#multilevel-pivot-tables"
    id="toc-multilevel-pivot-tables">Multilevel pivot tables</a>

------------------------------------------------------------------------

# Pivot Tables

We have seen how the GroupBy abstraction lets us explore relationships
within a data‐ set. A pivot table is a similar operation that is
commonly seen in spreadsheets and other programs that operate on tabular
data. The pivot table takes simple column- wise data as input, and
groups the entries into a two-dimensional table that provides a
multidimensional summarization of the data. The difference between pivot
tables and GroupBy can sometimes cause confusion; it helps me to think
of pivot tables as essentially a multidimensional version of GroupBy
aggregation. That is, you split- apply-combine, but both the split and
the combine happen across not a one- dimensional index, but across a
two-dimensional grid.

## Motivating pivot tables

``` python
import numpy as np
import pandas as pd
import seaborn as sns
```

``` python
titanic = sns.load_dataset('titanic')
titanic.head()
```

    URLError: <urlopen error [Errno -3] Temporary failure in name resolution>

## Pivot Tables by Hand

To start learning more about this data, we might begin by grouping it
according to gender, survival status, or some combination thereof. If
you have read the previous section, you might be tempted to apply a
GroupBy operation—for example, let’s look at survival rate by gender:

``` python
titanic.groupby('sex')['[survived]'].mean()
```

    NameError: name 'titanic' is not defined

**This immediately gives us some insight: overall, three of every four
females on board survived, while only one in five males survived! This
is useful, but we might like to go one step deeper and look at survival
by both sex and, say, class. Using the vocabulary of GroupBy, we might
proceed using something like this: we group by class and gender, select
survival, apply a mean aggregate, com‐ bine the resulting groups, and
then unstack the hierarchical index to reveal the hidden
multidimensionality. In code:**

``` python
titanic.groupby(['sex','class'])['survived'].aggregate('mean').unstack()
```

    NameError: name 'titanic' is not defined

## Pivot Table Syntax

Here is the equivalent to the preceding operation using the pivot_table
method of DataFrames

``` python
titanic.pivot_table('survived',index='sex', column='class')
```

    NameError: name 'titanic' is not defined

## Multilevel pivot tables

Just as in the GroupBy, the grouping in pivot tables can be specified
with multiple lev‐ els, and via a number of options. For example, we
might be interested in looking at age as a third dimension. We’ll bin
the age using the pd.cut function:

``` python
age = pd.cut(titanic['age'],[0,18,80])
titanic.pivot_table('survived',['sex',ge],'class')
```

    NameError: name 'titanic' is not defined

``` python
fare=pd.cut(titanic['fare'],2)
titanic.pivot_table('survived',['sex',ge],[fare,'class'])
```

    NameError: name 'titanic' is not defined

``` python
### There are some additional options for the pivot tables
# call signature as of Pandas 0.18
DataFrame.pivot_table(data, values=None, index=None, columns=None,
aggfunc='mean', fill_value=None, margins=False,
dropna=True, margins_name='All')
```

    NameError: name 'DataFrame' is not defined
