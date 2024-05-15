================
by Jawad Haider

# **05 - Operations**
------------------------------------------------------------------------
<center>
<a href=''>![Image](../../assets/img/logo1.png)</a>
</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
------------------------------------------------------------------------

- <a href="#operations" id="toc-operations"><span
  class="toc-section-number">1</span> Operations</a>
  - <a href="#info-on-unique-values" id="toc-info-on-unique-values"><span
    class="toc-section-number">1.0.1</span> Info on Unique Values</a>
  - <a href="#selecting-data" id="toc-selecting-data"><span
    class="toc-section-number">1.0.2</span> Selecting Data</a>
  - <a href="#applying-functions" id="toc-applying-functions"><span
    class="toc-section-number">1.0.3</span> Applying Functions</a>
  - <a href="#permanently-removing-a-column"
    id="toc-permanently-removing-a-column"><span
    class="toc-section-number">1.0.4</span> Permanently Removing a
    Column</a>
  - <a href="#get-column-and-index-names"
    id="toc-get-column-and-index-names"><span
    class="toc-section-number">1.0.5</span> Get column and index names:</a>
  - <a href="#sorting-and-ordering-a-dataframe"
    id="toc-sorting-and-ordering-a-dataframe"><span
    class="toc-section-number">1.0.6</span> Sorting and Ordering a
    DataFrame:</a>
- <a href="#great-job-thats-the-end-of-this-part."
  id="toc-great-job-thats-the-end-of-this-part."><span
  class="toc-section-number">2</span> Great Job! Thats the end of this
  part.</a>

------------------------------------------------------------------------

# Operations

There are lots of operations with pandas that will be really useful to
you, but don’t fall into any distinct category. Let’s show them here in
this lecture:

``` python
import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>

### Info on Unique Values

``` python
df['col2'].unique()
```

    array([444, 555, 666])

``` python
df['col2'].nunique()
```

    3

``` python
df['col2'].value_counts()
```

    444    2
    555    1
    666    1
    Name: col2, dtype: int64

### Selecting Data

``` python
#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]
```

``` python
newdf
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>

### Applying Functions

``` python
def times2(x):
    return x*2
```

``` python
df['col1'].apply(times2)
```

    0    2
    1    4
    2    6
    3    8
    Name: col1, dtype: int64

``` python
df['col3'].apply(len)
```

    0    3
    1    3
    2    3
    3    3
    Name: col3, dtype: int64

``` python
df['col1'].sum()
```

    10

### Permanently Removing a Column

``` python
del df['col1']
```

``` python
df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>

### Get column and index names:

``` python
df.columns
```

    Index(['col2', 'col3'], dtype='object')

``` python
df.index
```

    RangeIndex(start=0, stop=4, step=1)

### Sorting and Ordering a DataFrame:

``` python
df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>666</td>
      <td>ghi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>xyz</td>
    </tr>
  </tbody>
</table>
</div>

``` python
df.sort_values(by='col2') #inplace=False by default
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col2</th>
      <th>col3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>444</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>xyz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>555</td>
      <td>def</td>
    </tr>
    <tr>
      <th>2</th>
      <td>666</td>
      <td>ghi</td>
    </tr>
  </tbody>
</table>
</div>

# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`
