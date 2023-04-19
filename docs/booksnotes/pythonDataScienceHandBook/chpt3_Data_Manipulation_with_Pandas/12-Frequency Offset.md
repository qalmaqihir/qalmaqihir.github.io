Notes [book] Data Science Handbook
================
by Jawad Haider
# **Chpt 2 - Data Manipulation with Pandas**

# 12 - Frequency Offset
------------------------------------------------------------------------

- <a href="#frequencies-and-offsets"
  id="toc-frequencies-and-offsets">Frequencies and Offsets</a>

------------------------------------------------------------------------


# Frequencies and Offsets

Fundamental to these Pandas time series tools is the concept of a
frequency or date offset. Just as we saw the D (day) and H (hour) codes
previously, we can use such codes to specify any desired frequency
spacing.

| Code | Description            |
|------|------------------------|
| MS   | Month start            |
| BMS  | Business month start   |
| QS   | Quarter start          |
| BQS  | Business quarter start |
| AS   | Year start             |
| BAS  | Business year start    |

| Code | Description  | Code | Description          |
|------|--------------|------|----------------------|
| D    | Calendar day | B    | Business day         |
| W    | Weekly       |      |                      |
| M    | Month end    | BM   | Business month end   |
| Q    | Quarter end  | BQ   | Business quarter end |
| A    | Year end     | BA   | Business year end    |
| H    | Hours        | BH   | Business hours       |
| T    | Minutes      |      |                      |
| S    | Seconds      |      |                      |
| L    | Milliseonds  |      |                      |
| U    | Microseconds |      |                      |
| N    | Nanoseconds  |      |                      |

**Additionally, you can change the month used to mark any quarterly or
annual code by adding a three-letter month code as a suffix:  
• Q-JAN, BQ-FEB, QS-MAR, BQS-APR, etc.  
• A-JAN, BA-FEB, AS-MAR, BAS-APR, etc.**  
**In the same way, you can modify the split-point of the weekly
frequency by adding a three-letter weekday code:  
• W-SUN, W-MON, W-TUE, W-WED, etc.**

On top of this, codes can be combined with numbers to specify other
frequencies. For example, for a frequency of 2 hours 30 minutes, we can
combine the hour (H) and minute (T) codes as follows:

``` python
import pandas as pd
```

``` python
pd.timedelta_range(0,periods=9, freq='2H30T')
```

    TimedeltaIndex(['0 days 00:00:00', '0 days 02:30:00', '0 days 05:00:00',
                    '0 days 07:30:00', '0 days 10:00:00', '0 days 12:30:00',
                    '0 days 15:00:00', '0 days 17:30:00', '0 days 20:00:00'],
                   dtype='timedelta64[ns]', freq='150T')

All of these short codes refer to specific instances of Pandas time
series offsets, which can be found in the pd.tseries.offsets module. For
example, we can create a busi‐ ness day offset directly as follows:

``` python
from pandas.tseries.offsets  import BDay
pd.date_range('2022-07-28', periods=5, freq=BDay())
```

    DatetimeIndex(['2022-07-28', '2022-07-29', '2022-08-01', '2022-08-02',
                   '2022-08-03'],
                  dtype='datetime64[ns]', freq='B')
