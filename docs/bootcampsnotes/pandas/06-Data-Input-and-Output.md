================
by Jawad Haider

# **06 - Input Output**
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


- <a href="#data-input-and-output" id="toc-data-input-and-output"><span
  class="toc-section-number">1</span> Data Input and Output</a>
  - <a href="#csv" id="toc-csv"><span class="toc-section-number">1.1</span>
    CSV</a>
    - <a href="#csv-input" id="toc-csv-input"><span
      class="toc-section-number">1.1.1</span> CSV Input</a>
    - <a href="#csv-output" id="toc-csv-output"><span
      class="toc-section-number">1.1.2</span> CSV Output</a>
  - <a href="#excel" id="toc-excel"><span
    class="toc-section-number">1.2</span> Excel</a>
    - <a href="#excel-input" id="toc-excel-input"><span
      class="toc-section-number">1.2.1</span> Excel Input</a>
    - <a href="#excel-output" id="toc-excel-output"><span
      class="toc-section-number">1.2.2</span> Excel Output</a>
  - <a href="#html" id="toc-html"><span
    class="toc-section-number">1.3</span> HTML</a>
    - <a href="#html-input" id="toc-html-input"><span
      class="toc-section-number">1.3.1</span> HTML Input</a>
- <a href="#great-job-thats-the-end-of-this-part."
  id="toc-great-job-thats-the-end-of-this-part."><span
  class="toc-section-number">2</span> Great Job! Thats the end of this
  part.</a>

------------------------------------------------------------------------

<div class="alert alert-info">

<strong>NOTE:</strong> Typically we will just be either reading csv
files directly or using pandas-datareader to pull data from the web.
Consider this lecture just a quick overview of what is possible with
pandas (we won’t be working with SQL or Excel files in this course)

</div>

# Data Input and Output

This notebook is the reference code for getting input and output, pandas
can read a variety of file types using its pd.read\_ methods. Let’s take
a look at the most common data types:

``` python
import numpy as np
import pandas as pd
```

## CSV

Comma Separated Values files are text files that use commas as field
delimeters.<br> Unless you’re running the virtual environment included
with the course, you may need to install <tt>xlrd</tt> and
<tt>openpyxl</tt>.<br> In your terminal/command prompt run:

    conda install xlrd
    conda install openpyxl

Then restart Jupyter Notebook. (or use pip install if you aren’t using
the Anaconda Distribution)

### CSV Input

``` python
df = pd.read_csv('example.csv')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>

### CSV Output

``` python
df.to_csv('example.csv',index=False)
```

## Excel

Pandas can read and write MS Excel files. However, this only imports
data, not formulas or images. A file that contains images or macros may
cause the <tt>.read_excel()</tt>method to crash.

### Excel Input

``` python
pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>

### Excel Output

``` python
df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
```

## HTML

Pandas can read table tabs off of HTML.<br> Unless you’re running the
virtual environment included with the course, you may need to install
<tt>lxml</tt>, <tt>htmllib5</tt>, and <tt>BeautifulSoup4</tt>.<br> In
your terminal/command prompt run:

    conda install lxml
    conda install html5lib
    conda install beautifulsoup4

Then restart Jupyter Notebook. (or use pip install if you aren’t using
the Anaconda Distribution)

### HTML Input

Pandas read_html function will read tables off of a webpage and return a
list of DataFrame objects:

``` python
df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
```

``` python
df[0].head()
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
      <th>Bank Name</th>
      <th>City</th>
      <th>ST</th>
      <th>CERT</th>
      <th>Acquiring Institution</th>
      <th>Closing Date</th>
      <th>Updated Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Washington Federal Bank for Savings</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>30570</td>
      <td>Royal Savings Bank</td>
      <td>December 15, 2017</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Farmers and Merchants State Bank of Argonia</td>
      <td>Argonia</td>
      <td>KS</td>
      <td>17719</td>
      <td>Conway Bank</td>
      <td>October 13, 2017</td>
      <td>February 21, 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fayette County Bank</td>
      <td>Saint Elmo</td>
      <td>IL</td>
      <td>1802</td>
      <td>United Fidelity Bank, fsb</td>
      <td>May 26, 2017</td>
      <td>July 26, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Guaranty Bank, (d/b/a BestBank in Georgia &amp; Mi...</td>
      <td>Milwaukee</td>
      <td>WI</td>
      <td>30003</td>
      <td>First-Citizens Bank &amp; Trust Company</td>
      <td>May 5, 2017</td>
      <td>March 22, 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First NBC Bank</td>
      <td>New Orleans</td>
      <td>LA</td>
      <td>58302</td>
      <td>Whitney Bank</td>
      <td>April 28, 2017</td>
      <td>December 5, 2017</td>
    </tr>
  </tbody>
</table>
</div>

------------------------------------------------------------------------

# Great Job! Thats the end of this part.

`Don't forget to give a star on github and follow for more curated Computer Science, Machine Learning materials`
