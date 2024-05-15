Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 11 - Working with Time Series
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

- <a href="#working-with-time-series"
  id="toc-working-with-time-series">Working with Time Series</a>
  - <a href="#dates-and-times-in-python"
    id="toc-dates-and-times-in-python">Dates and Times in Python</a>
    - <a href="#native-python-dates-and-times-datetime-and-dateutil"
      id="toc-native-python-dates-and-times-datetime-and-dateutil">Native
      Python dates and times: datetime and dateutil</a>
    - <a href="#dates-and-times-in-pandas-best-of-both-worlds"
      id="toc-dates-and-times-in-pandas-best-of-both-worlds">Dates and times
      in Pandas: Best of both worlds</a>
  - <a href="#pandas-time-series-indexing-by-time"
    id="toc-pandas-time-series-indexing-by-time">Pandas Time Series:
    Indexing by Time</a>
  - <a href="#pandas-time-series-data-structures"
    id="toc-pandas-time-series-data-structures">Pandas Time Series Data
    Structures</a>
    - <a href="#regular-sequences-pd.date_range"
      id="toc-regular-sequences-pd.date_range">Regular sequences:
      pd.date_range()</a>

------------------------------------------------------------------------

# Working with Time Series

Pandas was developed in the context of financial modeling, so as you
might expect, it contains a fairly extensive set of tools for working
with dates, times, and time- indexed data. Date and time data comes in a
few flavors, which we will discuss here:

• Time stamps reference particular moments in time (e.g., July 4th,
2015, at 7:00a.m.).  
• Time intervals and periods reference a length of time between a
particular beginning and end point—for example, the year 2015. Periods
usually reference a special case of time intervals in which each
interval is of uniform length and does not overlap (e.g., 24 hour-long
periods constituting days).  
• Time deltas or durations reference an exact length of time (e.g., a
duration of 22.56 seconds).

## Dates and Times in Python

The Python world has a number of available representations of dates,
times, deltas, and timespans. While the time series tools provided by
Pandas tend to be the most useful for data science applications, it is
helpful to see their relationship to other packages used in Python.

### Native Python dates and times: datetime and dateutil

Python’s basic objects for working with dates and times reside in the
built-in date time module. Along with the third-party dateutil module,
you can use it to quickly perform a host of useful functionalities on
dates and times. For example, you can manually build a date using the
datetime type:

``` python
from datetime import datetime
datetime(year=2022, month=7,day=30)
```

    datetime.datetime(2022, 7, 30, 0, 0)

``` python
# Using the dateutil moduel, we can parse dates from different string formats
from dateutil import parser
date=parser.parse("30th of August 2022")
date
```

    datetime.datetime(2022, 8, 30, 0, 0)

``` python
# once we have a datetime object, we can do things like printng the day of the week
date.strftime('%A')

#In the final line, we’ve used one of the standard string format codes for printing dates
#("%A"),
```

    'Tuesday'

**A related package to be aware of is pytz, which contains tools for
working with the most migraine-inducing piece of time series data: time
zones.**  
The power of datetime and dateutil lies in their flexibility and easy
syntax: you can use these objects and their built-in methods to easily
perform nearly any operation you might be interested in. Where they
break down is when you wish to work with large arrays of dates and
times: just as lists of Python numerical variables are subopti‐ mal
compared to NumPy-style typed numerical arrays, lists of Python datetime
objects are suboptimal compared to typed arrays of encoded dates.  
\### Typed arrays of times: NumPy’s datetime64 The weaknesses of
Python’s datetime format inspired the NumPy team to add a set of native
time series data type to NumPy. The datetime64 dtype encodes dates as
64-bit integers, and thus allows arrays of dates to be represented very
compactly. The date time64 requires a very specific input format:

``` python
import numpy as np
date=np.array('2022-07-30',dtype=np.datetime64)
date
```

    array('2022-07-30', dtype='datetime64[D]')

``` python
# Now we can do vectorized operation on it
date + np.arange(12)
```

    array(['2022-07-30', '2022-07-31', '2022-08-01', '2022-08-02',
           '2022-08-03', '2022-08-04', '2022-08-05', '2022-08-06',
           '2022-08-07', '2022-08-08', '2022-08-09', '2022-08-10'],
          dtype='datetime64[D]')

*Because of the uniform type in NumPy datetime64 arrays, this type of
operation can be accomplished much more quickly than if we were working
directly with Python’s datetime objects, especially as arrays get large
(we introduced this type of vectoriza‐ tion in “Computation on NumPy
Arrays: Universal Functions” on page 50). One detail of the datetime64
and timedelta64 objects is that they are built on a fun‐ damental time
unit. Because the datetime64 object is limited to 64-bit precision, the
range of encodable times is 264 times this fundamental unit. In other
words, date time64 imposes a trade-off between time resolution and
maximum time span.*

**For example, if you want a time resolution of one nanosecond, you only
have enough information to encode a range of 264 nanoseconds, or just
under 600 years. NumPy will infer the desired unit from the input; for
example, here is a day-based datetime:**

``` python
np.datetime64('2022-07-30 12:00')
```

    numpy.datetime64('2022-07-30T12:00')

Notice that the time zone is automatically set to the local time on the
computer exe‐ cuting the code. You can force any desired fundamental
unit using one of many for‐ mat codes; for example, here we’ll force a
nanosecond-based time:

``` python
np.datetime64('2022-7-30 12:59.50','ns')
```

    ValueError: Error parsing datetime string "2022-7-30 12:59.50" at position 5

### Dates and times in Pandas: Best of both worlds

Pandas builds upon all the tools just discussed to provide a Timestamp
object, which combines the ease of use of datetime and dateutil with the
efficient storage and vectorized interface of numpy.datetime64. From a
group of these Timestamp objects, Pandas can construct a DatetimeIndex
that can be used to index data in a Series or DataFrame; we’ll see many
examples of this below. For example, we can use Pandas tools to repeat
the demonstration from above. We can parse a flexibly formatted string
date, and use format codes to output the day of the week:

``` python
import pandas as pd
```

``` python
date=pd.to_datetime('30th August, 2022')
```

``` python
date
```

    Timestamp('2022-08-30 00:00:00')

``` python
date.strftime('%A')
```

    'Tuesday'

``` python
# Additionaly we can do Numpy-style Vectorized operations directly on this object
date+pd.to_timedelta(np.arange(12),'D')
```

    DatetimeIndex(['2022-08-30', '2022-08-31', '2022-09-01', '2022-09-02',
                   '2022-09-03', '2022-09-04', '2022-09-05', '2022-09-06',
                   '2022-09-07', '2022-09-08', '2022-09-09', '2022-09-10'],
                  dtype='datetime64[ns]', freq=None)

## Pandas Time Series: Indexing by Time

Where the Pandas time series tools really become useful is when you
begin to index data by timestamps. For example, we can construct a
Series object that has time- indexed data:

``` python
index = pd.DatetimeIndex(['2022-07-04','2022-08-04','2023-07-04','2023-08-04'])
data=pd.Series([0,1,2,3], index=index)
data
```

    2022-07-04    0
    2022-08-04    1
    2023-07-04    2
    2023-08-04    3
    dtype: int64

``` python
data['2022-07-04':'2023-07-04']
```

    2022-07-04    0
    2022-08-04    1
    2023-07-04    2
    dtype: int64

``` python
# just passing year to obtain a slice
data['2022']
```

    2022-07-04    0
    2022-08-04    1
    dtype: int64

``` python
data['2023']
```

    2023-07-04    2
    2023-08-04    3
    dtype: int64

## Pandas Time Series Data Structures

• For time stamps, Pandas provides the Timestamp type. As mentioned
before, it is essentially a replacement for Python’s native datetime,
but is based on the more efficient numpy.datetime64 data type. The
associated index structure is DatetimeIndex.

• For time periods, Pandas provides the Period type. This encodes a
fixed frequency interval based on numpy.datetime64. The associated index
structure is PeriodIndex.

• For time deltas or durations, Pandas provides the Timedelta type.
Timedelta is a more efficient replacement for Python’s native
datetime.timedelta type, and is based on numpy.timedelta64. The
associated index structure is TimedeltaIndex.

**The most fundamental of these date/time objects are the Timestamp and
DatetimeIn dex objects. While these class objects can be invoked
directly, it is more common to use the `pd.to_datetime()` function,
which can parse a wide variety of formats. Passing a single date to
`pd.to_datetime()` yields a Timestamp; passing a series of dates by
default yields a DatetimeIndex:**

``` python
dates=pd.to_datetime([datetime(2022,7,30),'30th of July, 2022','2022-7-30','20220730'])
dates
```

    DatetimeIndex(['2022-07-30', '2022-07-30', '2022-07-30', '2022-07-30'], dtype='datetime64[ns]', freq=None)

Any DatetimeIndex can be converted to a PeriodIndex with the to_period()
func‐ tion with the addition of a frequency code; here we’ll use ‘D’ to
indicate daily frequency:

``` python
dates.to_period('D')
```

    PeriodIndex(['2022-07-30', '2022-07-30', '2022-07-30', '2022-07-30'], dtype='period[D]')

``` python
# The above is a TimedeltaIndex, we can subtract one date from another
dates-dates[0]
```

    TimedeltaIndex(['0 days', '0 days', '0 days', '0 days'], dtype='timedelta64[ns]', freq=None)

``` python
dates=pd.to_datetime([datetime(2022,7,28),'29th of July, 2022','2022-7-30','20220731'])
dates
```

    DatetimeIndex(['2022-07-28', '2022-07-29', '2022-07-30', '2022-07-31'], dtype='datetime64[ns]', freq=None)

``` python
dates.to_period('D')
```

    PeriodIndex(['2022-07-28', '2022-07-29', '2022-07-30', '2022-07-31'], dtype='period[D]')

``` python
dates-dates[0]
```

    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days'], dtype='timedelta64[ns]', freq=None)

``` python
dates-dates[1]
```

    TimedeltaIndex(['-1 days', '0 days', '1 days', '2 days'], dtype='timedelta64[ns]', freq=None)

### Regular sequences: pd.date_range()

To make the creation of regular date sequences more convenient, Pandas
offers a few functions for this purpose: pd.date_range() for timestamps,
pd.period_range() for periods, and pd.timedelta_range() for time
deltas.  
*we have seen that Python range() and NumPy’s np.arange() turn a
startpoint, endpoint, and optional stepsize into a sequence. Similarly,
pd.date_range() accepts a start date, an end date, and an optional
frequency code to create a regular sequence of dates. By default, the
fre‐ quency is one day:*

``` python
pd.date_range('2022-07-30','2023-07-30')
```

    DatetimeIndex(['2022-07-30', '2022-07-31', '2022-08-01', '2022-08-02',
                   '2022-08-03', '2022-08-04', '2022-08-05', '2022-08-06',
                   '2022-08-07', '2022-08-08',
                   ...
                   '2023-07-21', '2023-07-22', '2023-07-23', '2023-07-24',
                   '2023-07-25', '2023-07-26', '2023-07-27', '2023-07-28',
                   '2023-07-29', '2023-07-30'],
                  dtype='datetime64[ns]', length=366, freq='D')

**Alternatively, the date range can be specified not with a start- and
endpoint, but with a startpoint and a number of periods:**

``` python

pd.date_range('2022-07-30', periods=8)
```

    DatetimeIndex(['2022-07-30', '2022-07-31', '2022-08-01', '2022-08-02',
                   '2022-08-03', '2022-08-04', '2022-08-05', '2022-08-06'],
                  dtype='datetime64[ns]', freq='D')

**You can modify the spacing by altering the freq argument, which
defaults to D. For example, here we will construct a range of hourly
timestamps:**

``` python
pd.date_range('2022-07-30', periods=8, freq='H')
```

    DatetimeIndex(['2022-07-30 00:00:00', '2022-07-30 01:00:00',
                   '2022-07-30 02:00:00', '2022-07-30 03:00:00',
                   '2022-07-30 04:00:00', '2022-07-30 05:00:00',
                   '2022-07-30 06:00:00', '2022-07-30 07:00:00'],
                  dtype='datetime64[ns]', freq='H')

**To create regular sequences of period or time delta values, the very
similar pd.period_range() and pd.timedelta_range() functions are useful.
Here are some monthly periods:**

``` python
pd.period_range('2022-07', periods=8, freq='M')
```

    PeriodIndex(['2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                 '2023-01', '2023-02'],
                dtype='period[M]')

``` python
# and a sequence of durations increasing by an hour
pd.timedelta_range(0, periods=10, freq='H')
```

    TimedeltaIndex(['0 days 00:00:00', '0 days 01:00:00', '0 days 02:00:00',
                    '0 days 03:00:00', '0 days 04:00:00', '0 days 05:00:00',
                    '0 days 06:00:00', '0 days 07:00:00', '0 days 08:00:00',
                    '0 days 09:00:00'],
                   dtype='timedelta64[ns]', freq='H')
