Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 07 - Combining Datasets: Merge and Join
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

- <a href="#combining-datasets-merge-and-join"
  id="toc-combining-datasets-merge-and-join">Combining Datasets: Merge and
  Join</a>
  - <a href="#categories-of-joins" id="toc-categories-of-joins">Categories
    of Joins</a>
  - <a href="#one-to-one-joins" id="toc-one-to-one-joins">One-to-one
    joins</a>
  - <a href="#many-to-one-joins" id="toc-many-to-one-joins">Many-to-one
    joins</a>
  - <a href="#many-to-many-joins" id="toc-many-to-many-joins">Many-to-many
    joins</a>
  - <a href="#specification-of-the-merge-key"
    id="toc-specification-of-the-merge-key">Specification of the Merge
    Key</a>
    - <a href="#the-on-keyword" id="toc-the-on-keyword">The on keyword</a>
    - <a href="#the-left_on-and-right_on-keywords"
      id="toc-the-left_on-and-right_on-keywords">The left_on and right_on
      keywords</a>
  - <a href="#specifying-set-arithmetic-for-joins"
    id="toc-specifying-set-arithmetic-for-joins">Specifying Set Arithmetic
    for Joins</a>
  - <a href="#overlapping-column-names-the-suffixes-keyword"
    id="toc-overlapping-column-names-the-suffixes-keyword">Overlapping
    Column Names: The suffixes Keyword</a>

------------------------------------------------------------------------

``` python
import numpy as  np
import pandas as pd
```

# Combining Datasets: Merge and Join

One essential feature offered by Pandas is its high-performance,
in-memory join and merge operations. If you have ever worked with
databases, you should be familiar with this type of data interaction.
The main interface for this is the pd.merge func‐ tion, and we’ll see a
few examples of how this can work in practice. \### Relational Algebra
The behavior implemented in pd.merge() is a subset of what is known as
relational algebra, which is a formal set of rules for manipulating
relational data, and forms the conceptual foundation of operations
available in most databases. The strength of the relational algebra
approach is that it proposes several primitive operations, which become
the building blocks of more complicated operations on any dataset. With
this lexicon of fundamental operations implemented efficiently in a
database or other pro‐ gram, a wide range of fairly complicated
composite operations can be performed. Pandas implements several of
these fundamental building blocks in the pd.merge() function and the
related join() method of Series and DataFrames. As we will see, these
let you efficiently link data from different sources.

### Categories of Joins

The pd.merge() function implements a number of types of joins: the
one-to-one, many-to-one, and many-to-many joins. All three types of
joins are accessed via an identical call to the pd.merge() interface;
the type of join performed depends on the form of the input data.

### One-to-one joins

Perhaps the simplest type of merge expression is the one-to-one join,
which is in many ways very similar to the column-wise concatenation

``` python
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
'hire_date': [2004, 2008, 2012, 2014]})
df1
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df2
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
      <th>employee</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lisa</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df3=pd.merge(df1,df2)
```

``` python
df3
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>

### Many-to-one joins

Many-to-one joins are joins in which one of the two key columns contains
duplicate entries. For the many-to-one case, the resulting DataFrame
will preserve those dupli‐ cate entries as appropriate.

``` python
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
'supervisor': ['Carly', 'Guido', 'Steve']})
df4
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
      <th>group</th>
      <th>supervisor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accounting</td>
      <td>Carly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Engineering</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HR</td>
      <td>Steve</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df3
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df3,df4)
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
      <th>supervisor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
      <td>Carly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
      <td>Steve</td>
    </tr>
  </tbody>
</table>
</div>

### Many-to-many joins

Many-to-many joins are a bit confusing conceptually, but are
nevertheless well defined. If the key column in both the left and right
array contains duplicates, then the result is a many-to-many merge. This
will be perhaps most clear with a concrete example

``` python
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
'Engineering', 'Engineering', 'HR', 'HR'],'skills': ['math', 'spreadsheets', 'coding', 'linux',
'spreadsheets', 'organization']})
df5
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
      <th>group</th>
      <th>skills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accounting</td>
      <td>math</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Accounting</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Engineering</td>
      <td>coding</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Engineering</td>
      <td>linux</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HR</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HR</td>
      <td>organization</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df1
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df1,df5)
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
      <th>employee</th>
      <th>group</th>
      <th>skills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>math</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>coding</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>linux</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>coding</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>linux</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sue</td>
      <td>HR</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sue</td>
      <td>HR</td>
      <td>organization</td>
    </tr>
  </tbody>
</table>
</div>

## Specification of the Merge Key

We’ve already seen the default behavior of pd.merge(): it looks for one
or more matching column names between the two inputs, and uses this as
the key. However, often the column names will not match so nicely, and
pd.merge() provides a variety of options for handling this.

### The on keyword

Most simply, you can explicitly specify the name of the key column using
the on key‐ word, which takes a column name or a list of column names:

``` python
df1
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df2
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
      <th>employee</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lisa</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df1,df2, on='employee')
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>

### The left_on and right_on keywords

At times you may wish to merge two datasets with different column names;
for exam‐ ple, we may have a dataset in which the employee name is
labeled as “name” rather than “employee”. In this case, we can use the
left_on and right_on keywords to specify the two column names:

``` python
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'salary': [70000, 80000, 120000, 90000]})
df3
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
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df1
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df1,df3, left_on='employee', right_on='name')
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
      <th>employee</th>
      <th>group</th>
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>Sue</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>

``` python
# to drop the redundant column
pd.merge(df1,df3, left_on='employee', right_on='name').drop('name',axis=1)
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
      <th>employee</th>
      <th>group</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>

## Specifying Set Arithmetic for Joins

In all the preceding examples we have glossed over one important
consideration in performing a join: the type of set arithmetic used in
the join. This comes up when a value appears in one key column but not
the other.

``` python
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
'food': ['fish', 'beans', 'bread']},
columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
'drink': ['wine', 'beer']},
columns=['name', 'drink'])
df6
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
      <th>name</th>
      <th>food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df7
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
      <th>name</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joseph</td>
      <td>beer</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df6,df7,how='inner') # Interection
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df6,df7,how='outer') # union
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Joseph</td>
      <td>NaN</td>
      <td>beer</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df6,df7,how='left')
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df6,df7,how='right')
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joseph</td>
      <td>NaN</td>
      <td>beer</td>
    </tr>
  </tbody>
</table>
</div>

## Overlapping Column Names: The suffixes Keyword

Finally, you may end up in a case where your two input DataFrames have
conflicting column names.

``` python
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [1, 2, 3, 4]})
df8
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
      <th>name</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [3, 1, 4, 2]})
df9
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
      <th>name</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

``` python
pd.merge(df8,df9, on='name')
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
      <th>name</th>
      <th>rank_x</th>
      <th>rank_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

\*\*Because the output would have two conflicting column names, the
merge function automatically appends a suffix \_x or \_y to make the
output columns unique. If these defaults are inappropriate, it is
possible to specify a custom suffix using the suffixes keyword:\*\*

``` python
pd.merge(df8,df9,on='name',suffixes=['_L','_R'])
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
      <th>name</th>
      <th>rank_L</th>
      <th>rank_R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
