Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 10 -  Text and Annotation
------------------------------------------------------------------------

- <a href="#text-and-annotation" id="toc-text-and-annotation">Text and
  Annotation</a>
- <a href="#transforms-and-text-position"
  id="toc-transforms-and-text-position">Transforms and Text Position</a>
- <a href="#arrows-and-annotation" id="toc-arrows-and-annotation">Arrows
  and Annotation</a>

------------------------------------------------------------------------

# Text and Annotation

Creating a good visualization involves guiding the reader so that the
figure tells a story. In some cases, this story can be told in an
entirely visual manner, without the need for added text, but in others,
small textual cues and labels are necessary. Perhaps the most basic
types of annotations you will use are axes labels and titles, but the
options go beyond this. Let’s take a look at some data and how we might
visualize and annotate it to help convey interesting information.

``` python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
```

``` python
births = pd.read_csv('../data/births.csv')

quartiles = np.percentile(births['births'],[25,50,75])
mu, sig= quartiles[1], 0.74*(quartiles[2]-quartiles[0])
births=births.query('(births > @mu - 5*@sig)& (births<@mu + 5* @sig)')

births['day']=births['day'].astype(int)
births.index=pd.to_datetime(10000*births.year + 
                           100*births.month+
                           births.day, format="%Y%m%d")

births_by_date=births.pivot_table('births',
                                 [births.index.month, births.index.day])

births_by_date.index=[pd.datetime(2012, month, day) for (month, day) in births_by_date.index]
```

    /tmp/ipykernel_132909/914953373.py:15: FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.
      births_by_date.index=[pd.datetime(2012, month, day) for (month, day) in births_by_date.index]

``` python
fig, ax=plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax)
```

    <AxesSubplot:>

![](10text%20and%20annotation%20Example_files/figure-gfm/cell-4-output-2.png)

``` python
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
# Add labels to the plot
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)
# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
ylabel='average daily births')
# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
```

![](10text%20and%20annotation%20Example_files/figure-gfm/cell-5-output-1.png)

# Transforms and Text Position

In the previous example, we anchored our text annotations to data
locations. Some‐ times it’s preferable to anchor the text to a position
on the axes or figure, independent of the data. In Matplotlib, we do
this by modifying the transform.

Any graphics display framework needs some scheme for translating between
coordinate systems. For example, a data point at x, y = 1, 1 needs to
somehow be represented at a certain location on the figure, which in
turn needs to be represented in pixels on the screen.

Mathematically, such coordinate transformations are relatively
straightforward, and Matplotlib has a well-developed set of tools that
it uses internally to perform them (the tools can be explored in the
Matplotlib.transforms submodule).

The average user rarely needs to worry about the details of these
transforms, but it is helpful knowledge to have when considering the
placement of text on a figure. There are three predefined transforms
that can be useful in this situation:

*ax.transData*  
Transform associated with data coordinates  
*ax.transAxes*  
Transform associated with the axes (in units of axes dimensions)  
*fig.transFigure*  
Transform associated with the figure (in units of figure dimensions)

``` python
# # %matplotlib inline
%matplotlib notebook
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# plt.style.use('seaborn-whitegrid')
# import numpy as np
# import pandas as pd
```

``` python
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0,10,0,10])

# transform=ax.transData is the default,
ax.text(1,5,". Data: (1,5)", transform=ax.transData)
ax.text(5,3,". Data: (2,1.5)", transform=ax.transData)
ax.text(0.2,0.2,". Data: (0.2,0.2)", transform=ax.transData)
```

    <IPython.core.display.Javascript object>

<img src="data:image/png;base64,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ymuvvSYnn3wyAlYgewZdO7iwtGOJgNmyRMDsebay+MQTT0i7du3k2GOPLXs3KjDbQDDo2vGEpR1LBMyWJQJmz7OVxXvuuUe6dOkiixcvlq+++kpOP/106dWrFxVYgewZdO3gwtKOJQJmyxIBM+Y5d+5c0X8tjxUrVsiee+4pZ599trz33nvywAMPyM0339xKwIxdwRwEIACBbwSBhoYGF8/pchPHjBkzZOutt5Yf/ehHAfLAgQODiHFAAAIQgMA3h4BLAXv//ffl2WeflSFDhsjHH38st99+u9x6663fnKjxpBCAAAQgIC4FTOP26KOPyjvvvCOrV68Oa2A77bQT4YQABCAAgW8QAbcCplOGWonpbsQzzjij5PfA3nrrLfnlL38p7du3l7322ktOOOGEENJK1xQR80r3W7BggUyfPj34qNOhgwcPln/84x9y4403ylZbbRXc2W677eSss84qwrUSm5X8HD58uHTt2rW5/dChQ+Vb3/pW1VlWit+yZctkwoQJzT4uWbJETj31VOnZs+cG4am/5qFx/OlPfypHHXVUCetayU11qpKftZSflfyspfws52et5efUqVNFNxKtXbtW+vXrJ/vvv39zjtZSfsYMfC4FTCsv/UXm0aNHy8KFC2Xy5MlhwGg6dJAdM2aMbLHFFnLFFVcEcfj8888rXhMDK6VNlo/nn3++jB07NoiBTn/27ds37Kx89dVXgyBX68jyUweI9adns64pwvfYe2qnvPLKK0PcP/zww6rzXLVqlVx//fXhJWT77bdvJWC1kJsanyw/ayU/s/yslfzM8rOpT2zo/NQXEx07tX988cUXovx0V3ctjZ0p44dLAXvkkUekR48ecuihh4ZnveCCC8IuRK0UdGv9pEmT5LrrrgvnHn/88fBz3blY7poUYLFtK/moNhobG5srm3vvvVe+973vySabbCJvv/12VQUsy08dyPQFoeWRdU0so5R2sffUHav6qxVHH320vP7661XnqQPUmjVr5Mknn5Ru3bqVCFit5KZyr+RnLeVnlp+1kp9Zfjblei3kpy676MvyunXrwga4KVOmhC8a1VJ+xo4NLgXsrrvukt69e8u+++4bnlPfJs4777wwZaTb6vUNY8SIEeGcJoxOKWkFVu6aWFgp7Sr52NLOZ599FvzXClLL96efflo23XTT8IZ80kknyQ9+8IOU2ya3zfLzF7/4RZiGXbp0qey6665yyimnyN13311VlvpQWX42PfjIkSNDBaYvLfPnz686zyY/dI12fQGrldxsmSRt+VlL+VmJp56rlfzM8rPW8lP9mTNnTphK1FkBPWoxP7MGtLoQsMsvv1z0TSxFwFpekwUpz/n1B9y27qdVoVaKP//5z2WPPfYIOyo/+eQT2WeffWTRokVyzTXXhE9mderUKY8LUddk+Tl79mw56KCDpHPnzkFkf/zjH8ubb75ZImBFs2xLwNq651//+tfQKXV3qh4bgqeFgFWDZ8yAWwv5meVnreRnlp96vpbyUz+/N2vWrLDUsvHGGycJWDXzM2sQcylg+ta4+eaby+GHHx6eT9e4xo0bJxtttFGoFMaPHx++k6jHzJkzw9SczveWuyYLUp7zlXxUezqFeNVVV4WKRiuctg6tIocNGyZbbrllHheirsnys6URHSyWL18eNs5Uk6X6EOOnborZZptt5OCDD95gPCsNZLWSmy3hlKvAaiU/Y4Shqc2GzM8YP2slP9944w3R36XVmR+dJWg6ajE/swYxlwKmpa52PBUAXai///77m9e89IG1JNa3BN0gMWrUKLnooovCFGKla7JApZ7P8lErn1122aVksP3d734Xpg5/8pOfiE4t6nRY0RVYJT+V2cSJE4MfHTt2lNtuu00OOOCAsDmmmiyVfRZPbaObYgYMGBA2T+ixIXhmDWS1kJsxAlYr+VmJZy3lZ1bcayU/V65cGYTr6quvlu7du7ca1motP7PGXZcCpg/18MMPh/lbrQb0k1IqZFoK77fffuH3w6ZNmxaeXbeIHnPMMeG/179m/e8nZsFKPV/OR50u1Ln7lr+79sMf/lD69OkTfilbRUw3Auj2f123K/qoxFI3I7z00ktBwHbYYQc588wzw9b/arPMirmev/jii8NLzWabbRaQffnll1Xn+fe//10eeuihMBOgC+P6ErX33nuHKrqWcrOSn7WUn1k8ayU/s/yslfzUKXadldLllqZD17b1pa+W8jN2zHMrYLEPSDsIQAACEKhPAghYfcaVp4IABCBQ9wQQsLoPMQ8IAQhAoD4JIGD1GVeeCgIQgEDdE0DA6j7EPCAEIACB+iSAgNVnXHkqCEAAAnVPAAGr+xDzgBCAAATqkwACVp9x5akgAAEI1D0BBKzuQ8wDQgACEKhPAghYfcaVp4IABCBQ9wQQsLoPMQ8IAQhAoD4JIGD1GVeeCgIQgEDdE0DA6j7EPCAEIACB+iSAgNVnXHkqCEAAAnVPAAGr+xDzgBCAAATqkwACVp9x5akgAAEI1D0BBKzuQ8wDQgACEKhPAghYfcaVp4IABCBQ9wQQsLoPMQ8IAQhAoD4JIGD1GVeeCgIQgEDdE0DA6j7EPCAEIACB+iSAgNVnXHkqCEAAAnVPAAGr+xDzgBCAAATqkwACVp9x5akgAAEI1D2B/wPZGPVMR2w8nwAAAABJRU5ErkJggg==" width="432">

    Text(0.2, 0.2, '. Data: (0.2,0.2)')

``` python
ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
fig
```

    <IPython.core.display.Javascript object>

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAAAXNSR0IArs4c6QAAIABJREFUeF7tnQmQFdX1/w/rIIqoKcqIG1ZMjJZxwxUTLYxbmRgVcUtpEBcUFVFEWRQXLHcFAXEtUREiosGtkiASohZxKxMXojHGmIoYBAwlqIwkLPnVuf+a+c9j5r2+tz39eOf56SqqdPr26dOfc+799rn3Tk+7BQsW/E84IAABCEAAAs4ItEPAnEUMdyEAAQhAIBBAwEgECEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwSQABcxk2nIYABCAAAQSMHIAABCAAAZcEEDCXYcNpCEAAAhBAwMgBCEAAAhBwScCtgL344ovy5JNPSocOHeTkk0+W3r17uwwATkMAAhCAQD4CLgXsiy++kFGjRsktt9wiq1atkhkzZsjgwYPzEeAqCEAAAhBwScClgP3hD3+Qd955RwYNGuQSOk5DAAIQgMDXJ+BSwJ544glZunSpfPnll/LZZ5/JiSeeKLvtttvXp4EFCEAAAhBwQ8ClgM2aNUvee+89GTFihHz66ady1VVXyd133y3t2rVrBv+f//zHTRBwFAIQgEAtEWhoaKgld8r64lLA5s2bJ8uXL5d+/fqFBxs6dKiMHTtWunfvXiJgHjZ2/OUvf5Gdd9655pMFP+1CBEs7lmoJnrY8//jHPwoCZsu0xNqyZcvkjjvukDFjxoRpxEsvvVTuuusuad++PQJWEHcGCTuwsLRjiYDZslRrCJg901YW58yZI/PnzxedKuzfv7/ss88+JW3051RgdoFg0IWlHQFbS+SmLU8EzJZnLmsIWC5sZS9ikLDjCUs7llRgtiypwOx55rKIgOXChoDZYmvTGgJmCxmetjypwGx55rKGgOXChoDZYkPA4NlMwIvQImBVSNqsWyBgWYTSznvpfB789OAjU3Np/SOmtZe4I2Ax0Sy4DQJmC9hL5/PgpwcfETDb/uOJJwJmH/tkiwhYMrKKFzDo2vGEpR1LT8LgJe4ImG1+5rKGgOXCxhqYLbakNbCRI0eGb3xuttlmsnr1atl3333l/PPPr/hLpbNnz5Yjjzwy2eu1a9fKueeeK1deeaVsvfXWMn78eHnsscfklVdeabVm8+yzz8qtt94q3/72t8O5Pn36yHHHHSejR4+We+65Rzp16pR8f8sLvAiDFz8RMMvszGkLAcsJrsxlXjqfBz/L+agCdsQRR0jfvn1l3bp1cuedd4ZvfupXZsod+jUa/bRa6jFt2jRZuXKlnHPOOeEzbN26dZOJEyfKq6++2krA9NujK1askNNPP73kNg8++KCoEJ555pmptzdt7yHmnipFBMw0PfMZQ8DycSt3FYOEHc8YAdO7qYhpdfXwww+Hj1Zfc8010rFjx/DFmQkTJsjjjz8eKicVvNtvvz18G3TJkiXS2NgoQ4YMCT9XcVNxOuyww0oeQIXy0UcfDdWefs1mk002kf32269NAVOxW7NmTSsB0/scc8wx8txzz9nByWGJ3MwBrcIlCJgtz1zWELBc2MpexCBhxzNWwPSOOoV46qmnBjHbfPPNZZdddgnitcUWW8hpp53WLDr6eTX9I686tbdw4cLwfdByldmiRYtC5fXMM8+UPFQ5Abv33nvlhRdeCFOZ//vf/4JQfv/73w/X6h+TvfHGG6VXr152gBItkZuJwDKaI2C2PHNZQ8ByYUPAbLG1aS1FwFRozjjjjPChal2H0j/gqtOKRx99dKiymkRH18xuuOEGeffdd0OFtnjxYtGPXrd1vPnmmzJ58mS57777ogTs7bffDlXd/vvvL6+//nqoBJvET4VSReyAAw6oArm2b4GA2aJHwGx55rKGgOXChoDZYvtaAqbTdocffniYKlShOPvss+Wggw6S+++/v3masEnAdJ1Kvw2qf6Vc/1KDfh+0koDp+ppWVi2PchXY+g9x4IEHhmqvQ4cOwS8ELC5pvAgtAhYXz0JbIWC2eL10Pg9+xlZgur71+eefh793pxXXpEmTpGfPnnLWWWfJHnvsIcOGDQsfrNYBZ8qUKWGjxcUXXxzWtrStClpbh04h6g7Ep59+OkrAtFrbcccdwwaT999/Xy655JLmCkzFSyu/HXbYwTbhEqx5iLk+jhc/EbCE5CuqKQJmS9ZL5/Pg50svvSS6/X393YUtt9Fr/qpIDR8+XDp37hxEaerUqbLtttvK8ccfL9dee22ooFQ8dDehbuIYPHhwWBvT89pWN3FstdVWZTdxzJw5M0xNqi0Vpj/96U+y1157ySGHHCIDBw4Ma2y6geTjjz+WUaNGhfUvrQp1+7z+BfSvvvoqCOvcuXNtky3RmoeYI2CJQY1s7vIPWsY8GwIWQym+DYNEPKuslrXAUgVO19MGDRpU1l3drHHTTTeVPf/QQw/Jf//73zC1uSGPWuAZ8/xe/KQCi4lmwW0QMFvAXjqfBz9rwUetpLRi019k1qqurUMFasCAAW2e0+36WjHqLzJrhbghj1rgGfP8XvxEwGKiWXAbBMwWsJfO58FPDz56mvKCp21fR8BseeayhoDlwlb2IgYJO56wtGOJ0NqyVGsImD3TZIsIWDKyihcw6NrxhKUdSwTMliUCZs8zl0UELBc2KjBbbG1aQ8BsIcPTlicVmC3PXNYQsFzYEDBbbAgYPJsJeBFaBKwKSZt1CwQsi1DaeS+dz4OfHnxkai6tf8S09hJ3BCwmmgW3QcBsAXvpfB789OAjAmbbfzzxRMDsY59sEQFLRlbxAgZdO56wtGPpSRi8xB0Bs83PXNYQsFzYWAOzxcYaGDxZAyswB/iUVIFwY0x7eSvDz5hoxrWBZRyn2FbwjCUV144KLI5Toa2owGzxMkjY8YSlHUumEG1ZqjUEzJ5pskUELBkZa2C2yJiOhWcJAS8vLghYlRJXReqiiy6SE044IfwJiJYHAmYbBC+dz4OfHnyksrHtP554ImD2sW/T4vTp0+Wtt96SI488EgErmDmDrh1gWNqx9CQMXuKOgNnmZ5vW9I/sqYD16tVLevTogYAVzNxL5/PgpwcfEQb7DuUl7giYfexbWbzuuuvCn1Z//vnnEbAq8PbS+Tz46cFHBMy+U3mJOwJmH/sSiypa//73v6V///7hT62Xq8C6du1asCdf37z+VdwuXbp8fUMFW8BPO8CwtGOpluBpy7OxsVEaGhpsjRZkzeXvgd12222ifxG2ffv2smzZMunUqZOcc845svvuuzdjYhOHbcZ4eXv04KcHH6nAbPuPJ55UYPaxL2uxUgXWu3fvKnqS71YMZvm4lbvKA08PPnoacOFp24cQMFueFa0hYNWBzSBhxxmWdiwRWluWag0Bs2eabJEpxGRkFS9g0LXjCUs7lgiYLUsEzJ5nLosIWC5sZS9i0LXjCUs7lgiYLUsEzJ5nLosIWC5sCJgttjatIWC2kOFpy5MpRFueuawhYLmwIWC22BAweDYT8CK0CFgVkjbrFghYFqG08146nwc/PfjI1Fxa/4hp7SXuCFhMNAtug4DZAvbS+Tz46cFHBMy2/3jiiYDZxz7ZIgKWjKziBQy6djxhacfSkzB4iTsCZpufuawhYLmwsQZmi401MHiyBlZgDrj8lFQMDwQshlJ8Gy9vjx789OAjlU1834ht6SXuVGCxES2wHQJmC9dL5/PgpwcfETDb/uOJJwJmH/tkiwhYMjLWwGyRMR0LzxICXl5cELAqJW6l2yBgtkHw0vk8+OnBR08VAzxt+zoCZsszlzUELBc2qgZbbG1aY8C1hQxPW54ImC3PXNYQsFzYEDBbbAgYPJsJeBFaBKwKSZt1CwQsi1DaeS+dz4OfHnxkCjGtf8S09hJ3BCwmmgW3QcBsAXvpfB789OAjAmbbfzzxRMDsY59sEQFLRlbxAgZdO56wtGPpSRi8xB0Bs83PXNYQsFzYWAOzxcYaGDxZAyswB/gSR4FwY0x7eSvDz5hoxrWBZRyn2FbwjCUV144KLI5Toa2owGzxMkjY8YSlHUumEG1ZqjUEzJ5pskUELBkZa2C2yJiOhWcJAS8vLghYlRK30m0QMNsgeOl8Hvz04COVjW3/8cQTAbOPfbJFBCwZGRWYLTIqMHhSgRWcA2ziKBhwlnnexrMIpZ33wNODj54qBnim9ZGs1lRgWYSqcJ4KzBYyg4QdT1jasURobVmqNQTMnmmyRQQsGRlTiLbImEKEJ1OIBecAU4gFA84yz9t4FqG08x54evCRyiYt72Jae4k7FVhMNAtuQwVmC9hL5/PgpwcfETDb/uOJJwJmH/tWFqdOnSo6EKxdu1b69esn+++/f0kbBMw2CAy6djxhacfSkzB4iTsCZpufrawtWLBAnnrqKbniiivkiy++kOHDh8s999yDgBXI3Uvn8+CnBx8RBvvO5CXuCJh97EssatW1evVq6dKli6xbt04GDhwoU6ZMkQ4dOjS3owKzDYKXzufBTw8+ImC2/ccTTwTMPvZlLc6ZMydMJQ4dOpQKrEDuDLp2cGFpx9KTMHiJOwJmm59lrb322msya9YsGTNmjGy88catBKxr165V8iT/bVatWhUqyVo/8NMuQrC0Y6mW4GnLs7GxURoaGmyNFmTN7Tb6N954Q2bMmBHWwbp169YKD1OIthnj5e3Rg58efKSyse0/nnhSgdnHvsTiypUrg3BdffXV0r179zbvhoDZBoFB144nLO1YehIGL3FHwGzzs5U1XfeaOXOm9OzZs/nckCFDpEePHs3/j4DZBsFL5/PgpwcfEQbb/uOJJwJmH/tkiwhYMrKKFzDo2vGEpR1LT8LgJe4ImG1+5rKGgOXCVvYiL53Pg58efEQYbPuPJ54ImH3sky0iYMnIqMBskfEyAM8SAl5eXBCwKiVupdsgYLZB8NL5PPjpwUdPFQM8bfs6AmbLM5c1BCwXNqoGW2xtWmPAtYUMT1ueCJgtz1zWELBc2BAwW2wIGDybCXgRWgSsCkmbdQsELItQ2nkvnc+Dnx58ZAoxrX/EtPYSdwQsJpoFt0HAbAF76Xwe/PTgIwJm23888UTA7GOfbBEBS0ZW8QIGXTuesLRj6UkYvMQdAbPNz1zWELBc2FgDs8XGGhg8WQMrMAfcfsw3iwkClkUo7byXt0cPfnrwkcomrX/EtPYSdyqwmGgW3AYBswXspfN58NODjwiYbf/xxBMBs499skUELBkZa2C2yJiOhWcJAS8vLghYlRK30m0QMNsgeOl8Hvz04KOnigGetn0dAbPlmcsaApYLG1WDLbY2rTHg2kKGpy1PBMyWZy5rCFgubAiYLTYEDJ7NBLwILQJWhaTNugUClkUo7byXzufBTw8+MoWY1j9iWnuJOwIWE82C2yBgtoC9dD4PfnrwEQGz7T+eeCJg9rFPtoiAJSOreAGDrh1PWNqx9CQMXuKOgNnmZy5rCFgubKyB2WJjDQyerIEVmAN8iaNAuDGmvbyV4WdMNOPawDKOU2wreMaSimtHBRbHqdBWtVyBjRw5Ut555x3ZbLPN5PPPP5eDDz5Yzj//fGloaCjLZPbs2XLkkUcmM1u7dq2ce+65cuWVV8rKlSvl6quvDjZ22mknueaaa0rsrVmzRi6//HJZuHCh6H9fdtllsvfee4c2TYPE008/LQ899JC0b99eTjrpJOnfv3+JjU8++URGjRoVru/YsaPccsstsmDBAnnllVdk9OjRyf6nXuBhMPPgI1NzqZmX3d5L3BGw7FgW3qLWBeyII46Qvn37BiH7/e9/L0uXLpWxY8eW5dKvXz+ZNWtWMrdp06YF4TrnnHPktNNOk0svvVR22203GTp0qKhNFc+m41e/+lUQGxW5v/3tb0GIHn/88WYB23777eW4444LP+vUqZMce+yxMmPGjCDETceIESOCzaOOOkqmT58u//rXv4IQnnfeeUFI9d5FHh4GCQ8+ImD2Weol7giYfeyTLXoRME1qrYa0unr44Yfls88+C5WRVi9a5UyYMCEIxvjx44Pg3X777aIisWTJEmlsbJQhQ4aEn6u4devWTQ477LASViqUjz76qHTt2jXcY968eeG8VlLvvvuuaDXYdKxevVrWrVsXKsFly5aFCmvu3LnNArZ8+XJRkbv11lvDz6644go55JBDwr+mQ33S6zt06CC/+c1vZP78+XL99dfLSy+9FHxsujY5oJEXeBgkPPiIgEUmXEIzL3FHwBKCWlRTTwK28847hynEU089NQjI5ptvLrvssksQry222CJUTvvtt5+8+uqrQVhefPHFUAnpVJ9WUuUqs0WLFoXK65lnngmCp//95JNPBuQvv/xyEMbbbrutzRCMGzcuCOhFF13ULGAffPBBqNCapgJVTLfaaqsgdOsfOnU5YMCA8FwHHHCArFq1Kgjo888/X1TIm/1UnrV8eBnI8NM2i7zwRMBs457LmjcBU3E544wzpHv37qFK0QFfpxWPPvroUGU1CZhWSTfccEOonlRgFi9e3FxVrQ/qzTfflMmTJ8t9993XSsC0ItJqqi0B06k/rdTuvvvuMFXY9Dau04p//vOfmwVMq8Ktt95aTjzxxJJbq3jptOEOO+wgF1xwQfO5Aw88MIivVmdFHR4GCQ8+UoHZZ6iXuCNg9rFPtuhJwL773e/K4YcfHioirajOPvtsOeigg+T+++9vniZsErAnnngiTMvp5gid0tNNFE3Tgm0J2J133in33nuvqPDp9GJTBaR23n///TAd2fJ47LHHRDeM6HUtN5Vo59MNJzodqdWZHrpGpn7rFGbLQ8Vrm222kQsvvLDk5wjY/8PhZSDDz+Rhp+IFXngiYLZxz2XNk4CpYKg4XHXVVaHimjRpkvTs2VPOOuss2WOPPWTYsGHSu3dv0cSaMmWKrFixQi6++OIgJtpWBa2tQ6cQdeOErnfpoRWebqbQnYWDBw8OU5N9+vRpvlSnJHXKUDd+bLTRRiUmtfNpRaX+aeWmVZRuAlHR1bW3pkPvpVOd1113Xcn1Gg8VuxdeeCFXPGMv8jBIePARoY3NuPh2XuKOgMXHNHfLBx54IFQQ7dq1CwPzjjvu2GrA1EF/Qx6ffvppEJj1dxe23EavVZSuEQ0fPlw6d+4cRGnq1Kmy7bbbyvHHHy/XXnttqKB02lB3E+q6k4qPro3peW2rFZCuRZXbxDFz5swwNalrWLqdXtfZdt9991BB6aH27rrrrlBZ/frXvw7i2XRoFfjggw9Kjx49wrqbiq3+TLnrmt3PfvazUFE899xzoeI6+eSTRcVqk002CSa+853vhF2NWWtuVnHyMEh48BEBs8rI/2/HS9wRMPvYl1jUredPPfVUWIvRqkHXeW688caaE7AYDEUntQqcrqcNGjQoxp022+i0o67Hrb/WlWJQ18LUB7bRM4WYkjcxbYvuQzE+xLTx4icCFhPNr9HmkUceCRXBoYceGqzo4HjzzTeHreJNRy1PIbZ89KKTWn+hWCssrby0qstz6IaPjTfeOFRteQ4VQJ3m1G33RR9F87Tw34OPVGAWkS614SXuCJh97Ess6nSXTg/uu+++4ec6MOraTsupLwTMNgheOp8HPz34iIDZ9h9PPBEw+9hXFDD9/JH+vtH6AtayIivYpdzmdXqvS5cuua+v1oX4aUcalnYs1RI8bXk2fYzA1mox1lx+zFc3Ougv++quNj10ikw3ILTcOUcFZpswVA12PGFpx9JTZeMl7lRgtvnZytp7770XduvptvMPP/ww7Ipra9v2ht6FGIPBS1LjZ0w049rAMo5TbCt4xpKKa4eAxXH6Wq30u4GauLqdW3/xt1evXiX2qMC+Ft5WFzNI2PGEpR1LKjBblmoNAbNnmmwRAUtGVvECBl07nrC0Y4mA2bJEwOx55rKIgOXCVvYiBl07nrC0Y4mA2bJEwOx55rKIgOXChoDZYmvTGgJmCxmetjyZQrTlmcsaApYLGwJmiw0Bg2czAS9Ci4BVIWmzboGAZRFKO++l83nw04OPTM2l9Y+Y1l7ijoDFRLPgNgiYLWAvnc+Dnx58RMBs+48nngiYfeyTLSJgycgqXsCga8cTlnYsPQmDl7gjYLb5mcsaApYLG2tgtthYA4Mna2AF5oDLT0nF8EDAYijFt/Hy9ujBTw8+UtnE943Yll7iTgUWG9EC2yFgtnC9dD4PfnrwEQGz7T+eeCJg9rFPtoiAJSNjDcwWGdOx8Cwh4OXFBQGrUuJWug0CZhsEL53Pg58efPRUMcDTtq8jYLY8c1lDwHJho2qwxdamNQZcW8jwtOWJgNnyzGUNAcuFDQGzxYaAwbOZgBehRcCqkLRZt0DAsgilnffS+Tz46cFHphDT+kdMay9xR8BiollwGwTMFrCXzufBTw8+ImC2/ccTTwTMPvbJFhGwZGQVL2DQteMJSzuWnoTBS9wRMNv8zGUNAcuFjTUwW2ysgcGTNbACc4AvcRQIN8a0l7cy/IyJZlwbWMZxim0Fz1hSce2owOI4FdqKCswWL4OEHU9Y2rFkCtGWpVpDwOyZJltEwJKRsQZmi4zpWHiWEPDy4oKAVSlxK90GAbMNgpfO58FPDz5S2dj2H088ETD72CdbRMCSkVGB2SKjAoMnFVjBOcAmjoIBZ5nnbTyLUNp5Dzw9+OipYoBnWh/Jak0FlkWoCuepwGwhM0jY8YSlHUuE1palWkPA7JkmW0TAkpExhWiLjClEeDKFWHAOMIVYMOAs87yNZxFKO++BpwcfqWzS8i6mtZe4U4HFRLPgNlRgtoC9dD4PfnrwEQGz7T+eeCJg9rFPtoiAJSNjCtEWGVOI8GQKseAccDmFuHbtWrnzzjtlyZIlsmbNGhkwYIDsvPPOJagQMNvMoWqw4wlLO5aeKhsvcacCs83PVtbmzZsnH3zwgQwaNEg++ugjmTx5stx0000IWIHcvXQ+D3568BFhsO9MXuKOgNnHvsSiVl3r1q2Tzp07y4oVK2TUqFGhImt5UIHZBsFL5/PgpwcfETDb/uOJJwJmH/uyFqdPny7t27eXU045BQErkDuDrh1cWNqx9CQMXuKOgBnm59y5c0X/tTxOOukk2XPPPeW3v/2tvP7666EC69ixYysB69q1q6EnxZhatWqVdOnSpRjjhlbx0w4mLO1YqiV42vJsbGyUhoYGW6MFWXO5iUNZqKi9/PLLMmLEiDCVuP7BFKJtxnh5e/TgpwcfqWxs+48nnlRg9rEvsbh48WIZN26cXHvttWXfFBAw2yAw6NrxhKUdS0/C4CXuCJhtfraypute8+fPlx49ejSfGzNmjHTq1Kn5/xEw2yB46Xwe/PTgI8Jg23888UTA7GOfbBEBS0ZW8QIGXTuesLRj6UkYvMQdAbPNz1zWELBc2Mpe5KXzefDTg48Ig23/8cQTAbOPfbJFBCwZGRWYLTJeBuBZQsDLiwsCVqXErXQbBMw2CF46nwc/PfjoqWKAp21fR8BseeayhoDlwkbVYIutTWsMuLaQ4WnLEwGz5ZnLGgKWCxsCZosNAYNnMwEvQouAVSFps26BgGURSjvvpfN58NODj0whpvWPmNZe4o6AxUSz4DYImC1gL53Pg58efETAbPuPJ54ImH3sky0iYMnIKl7AoGvHE5Z2LD0Jg5e4I2C2+ZnLGgKWCxtrYLbYWAODJ2tgBeaA24/5ZjFBwLIIpZ338vbowU8PPlLZpPWPmNZe4k4FFhPNgtsgYLaAvXQ+D3568BEBs+0/nngiYPaxT7aIgCUjYw3MFhnTsfAsIeDlxQUBq1LiVroNAmYbBC+dz4OfHnz0VDHA07avI2C2PHNZQ8ByYaNqsMXWpjUGXFvI8LTliYDZ8sxlDQHLhQ0Bs8WGgMGzmYAXoUXAqpC0WbdAwLIIpZ330vk8+OnBR6YQ0/pHTGsvcUfAYqJZcBsEzBawl87nwU8PPiJgtv3HE08EzD72yRYRsGRkFS9g0LXjCUs7lp6EwUvcETDb/MxlDQHLhY01MFtsrIHBkzWwAnOAL3EUCDfGtJe3MvyMiWZcG1jGcYptBc9YUnHtqMDiOBXaigrMFi+DhB1PWNqxZArRlqVaQ8DsmSZbRMCSkbEGZouM6Vh4lhDw8uKCgFUpcSvdBgGzDYKXzufBTw8+UtnY9h9PPBEw+9gnW0TAkpFRgdkiowKDJxVYwTnAJo6CAWeZ5208i1DaeQ88PfjoqWKAZ1ofyWpNBZZFqArnqcBsITNI2PGEpR1LhNaWpVpDwOyZJltEwJKRMYVoi4wpRHgyhVhwDrieQly+fLlceOGFctlll8muu+5aggoBs80cqgY7nrC0Y0kFZsuSCsyeZ1mLEydOlIULF8qAAQMQsIK5M+jaAYalHUsEzJYlAmbPs02LCxYskFdeeUVWrVolffv2RcAK5s6gawcYlnYsETBblgiYPc9WFlevXi1jx46VkSNHypQpUxCwKjBn0LWDDEs7lgiYLUsEzJjn3LlzRf+1PPbaay/Zcsst5eCDD5ZJkyaVFbCuXbsae2NvTivILl262Bs2toifdkBhacdSLcHTlmdjY6M0NDTYGi3ImstNHKNHj5Z169YFJEuWLJFNN91ULrnkEtluu+2aMbGJwzZjqBrseMLSjiUVmC1LKjB7nhUtVqrAevfuXWVv0m/HYJbOrNIVHnh68BFhsM1LTzz5PTD72Je1iIBVBzaDrh1nWNqx9CQMXuKOgNnmZy5rTCHmwlb2Ii+dz4OfHnzjBxpNAAAKsUlEQVREGGz7jyeeCJh97JMtImDJyCpewKBrxxOWdiw9CYOXuCNgtvmZyxoClgsbFZgttjateRnI8NM2GbzwRMBs457LGgKWCxsCZosNAYNnMwEEzD4ZXG6jj8GAgMVQim/jpfN58NODj0zNxfeN2JZe4k4FFhvRAtshYLZwvXQ+D3568BEBs+0/nngiYPaxT7aIgCUjq3gBg64dT1jasfQkDF7ijoDZ5mcuawhYLmysgdliYw0MnqyBFZgDrIEVCDfGtJe3MvyMiWZcG1jGcYptBc9YUnHtqMDiOBXaigrMFi+DhB1PWNqxZArRlqVaQ8DsmSZbRMCSkbEGZouM6Vh4lhDw8uKCgFUpcSvdBgGzDYKXzufBTw8+UtnY9h9PPBEw+9gnW0TAkpFRgdkiowKDJxVYwTnAJo6CAWeZ5208i1DaeQ88PfjoqWKAZ1ofyWpNBZZFqArnqcBsITNI2PGEpR1LhNaWpVpDwOyZJltEwJKRMYVoi4wpRHgyhVhwDjCFWDDgLPO8jWcRSjvvgacHH6ls0vIuprWXuFOBxUSz4DZUYLaAvXQ+D3568BEBs+0/nngiYPaxT7aIgCUjYwrRFhlTiPBkCrHgHGAKsWDAWeZ5G88ilHbeA08PPnqqGOCZ1keyWlOBZRGqwnkqMFvIDBJ2PGFpxxKhtWWp1hAwe6bJFhGwZGRMIdoiYwoRnkwhFpwDTCEWDDjLPG/jWYTSznvg6cFHKpu0vItp7SXuVGAx0Sy4DRWYLWAvnc+Dnx58RMBs+48nngiYfeyTLSJgyciYQrRFxhQiPJlCLDgHmEIsGHCWed7GswilnffA04OPnioGeKb1kazWVGBZhKpwngrMFjKDhB1PWNqxRGhtWao1BMyeabJFBCwZGVOItsiYQoQnU4gF54DbKcSnnnpKXnzxRenQoYMMGjRIdtxxxxJUCJht5lA12PGEpR1LKjBbllRg9jxbWfzoo4/kjjvukBtuuEH++c9/ymuvvSYnn3wyAlYgewZdO7iwtGOJgNmyRMDsebay+MQTT0i7du3k2GOPLXs3KjDbQDDo2vGEpR1LBMyWJQJmz7OVxXvuuUe6dOkiixcvlq+++kpOP/106dWrFxVYgewZdO3gwtKOJQJmyxIBM+Y5d+5c0X8tjxUrVsiee+4pZ599trz33nvywAMPyM0339xKwIxdwRwEIACBbwSBhoYGF8/pchPHjBkzZOutt5Yf/ehHAfLAgQODiHFAAAIQgMA3h4BLAXv//ffl2WeflSFDhsjHH38st99+u9x6663fnKjxpBCAAAQgIC4FTOP26KOPyjvvvCOrV68Oa2A77bQT4YQABCAAgW8QAbcCplOGWonpbsQzzjij5PfA3nrrLfnlL38p7du3l7322ktOOOGEENJK1xQR80r3W7BggUyfPj34qNOhgwcPln/84x9y4403ylZbbRXc2W677eSss84qwrUSm5X8HD58uHTt2rW5/dChQ+Vb3/pW1VlWit+yZctkwoQJzT4uWbJETj31VOnZs+cG4am/5qFx/OlPfypHHXVUCetayU11qpKftZSflfyspfws52et5efUqVNFNxKtXbtW+vXrJ/vvv39zjtZSfsYMfC4FTCsv/UXm0aNHy8KFC2Xy5MlhwGg6dJAdM2aMbLHFFnLFFVcEcfj8888rXhMDK6VNlo/nn3++jB07NoiBTn/27ds37Kx89dVXgyBX68jyUweI9adns64pwvfYe2qnvPLKK0PcP/zww6rzXLVqlVx//fXhJWT77bdvJWC1kJsanyw/ayU/s/yslfzM8rOpT2zo/NQXEx07tX988cUXovx0V3ctjZ0p44dLAXvkkUekR48ecuihh4ZnveCCC8IuRK0UdGv9pEmT5LrrrgvnHn/88fBz3blY7poUYLFtK/moNhobG5srm3vvvVe+973vySabbCJvv/12VQUsy08dyPQFoeWRdU0so5R2sffUHav6qxVHH320vP7661XnqQPUmjVr5Mknn5Ru3bqVCFit5KZyr+RnLeVnlp+1kp9Zfjblei3kpy676MvyunXrwga4KVOmhC8a1VJ+xo4NLgXsrrvukt69e8u+++4bnlPfJs4777wwZaTb6vUNY8SIEeGcJoxOKWkFVu6aWFgp7Sr52NLOZ599FvzXClLL96efflo23XTT8IZ80kknyQ9+8IOU2ya3zfLzF7/4RZiGXbp0qey6665yyimnyN13311VlvpQWX42PfjIkSNDBaYvLfPnz686zyY/dI12fQGrldxsmSRt+VlL+VmJp56rlfzM8rPW8lP9mTNnTphK1FkBPWoxP7MGtLoQsMsvv1z0TSxFwFpekwUpz/n1B9y27qdVoVaKP//5z2WPPfYIOyo/+eQT2WeffWTRokVyzTXXhE9mderUKY8LUddk+Tl79mw56KCDpHPnzkFkf/zjH8ubb75ZImBFs2xLwNq651//+tfQKXV3qh4bgqeFgFWDZ8yAWwv5meVnreRnlp96vpbyUz+/N2vWrLDUsvHGGycJWDXzM2sQcylg+ta4+eaby+GHHx6eT9e4xo0bJxtttFGoFMaPHx++k6jHzJkzw9SczveWuyYLUp7zlXxUezqFeNVVV4WKRiuctg6tIocNGyZbbrllHheirsnys6URHSyWL18eNs5Uk6X6EOOnborZZptt5OCDD95gPCsNZLWSmy3hlKvAaiU/Y4Shqc2GzM8YP2slP9944w3R36XVmR+dJWg6ajE/swYxlwKmpa52PBUAXai///77m9e89IG1JNa3BN0gMWrUKLnooovCFGKla7JApZ7P8lErn1122aVksP3d734Xpg5/8pOfiE4t6nRY0RVYJT+V2cSJE4MfHTt2lNtuu00OOOCAsDmmmiyVfRZPbaObYgYMGBA2T+ixIXhmDWS1kJsxAlYr+VmJZy3lZ1bcayU/V65cGYTr6quvlu7du7ca1motP7PGXZcCpg/18MMPh/lbrQb0k1IqZFoK77fffuH3w6ZNmxaeXbeIHnPMMeG/179m/e8nZsFKPV/OR50u1Ln7lr+79sMf/lD69OkTfilbRUw3Auj2f123K/qoxFI3I7z00ktBwHbYYQc588wzw9b/arPMirmev/jii8NLzWabbRaQffnll1Xn+fe//10eeuihMBOgC+P6ErX33nuHKrqWcrOSn7WUn1k8ayU/s/yslfzUKXadldLllqZD17b1pa+W8jN2zHMrYLEPSDsIQAACEKhPAghYfcaVp4IABCBQ9wQQsLoPMQ8IAQhAoD4JIGD1GVeeCgIQgEDdE0DA6j7EPCAEIACB+iSAgNVnXHkqCEAAAnVPAAGr+xDzgBCAAATqkwACVp9x5akgAAEI1D0BBKzuQ8wDQgACEKhPAghYfcaVp4IABCBQ9wQQsLoPMQ8IAQhAoD4JIGD1GVeeCgIQgEDdE0DA6j7EPCAEIACB+iSAgNVnXHkqCEAAAnVPAAGr+xDzgBCAAATqkwACVp9x5akgAAEI1D0BBKzuQ8wDQgACEKhPAghYfcaVp4IABCBQ9wQQsLoPMQ8IAQhAoD4JIGD1GVeeCgIQgEDdE0DA6j7EPCAEIACB+iSAgNVnXHkqCEAAAnVPAAGr+xDzgBCAAATqkwACVp9x5akgAAEI1D2B/wPZGPVMR2w8nwAAAABJRU5ErkJggg==" width="432">

# Arrows and Annotation

Along with tick marks and text, another useful annotation mark is the
simple arrow. Drawing arrows in Matplotlib is often much harder than you
might hope. While there is a plt.arrow() function available, I wouldn’t
suggest using it; the arrows it creates are SVG objects that will be
subject to the varying aspect ratio of your plots, and the result is
rarely what the user intended. Instead, I’d suggest using the plt.anno
tate() function. This function creates some text and an arrow, and the
arrows can be very flexibly specified.

``` python
%matplotlib inline
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
arrowprops=dict(arrowstyle="->",
connectionstyle="angle3,angleA=0,angleB=-90"));
```

![](10text%20and%20annotation%20Example_files/figure-gfm/cell-9-output-1.png)

The arrow style is controlled through the arrowprops dictionary, which
has numer‐ ous options available. These options are fairly well
documented in Matplotlib’s online documentation,

``` python
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
# Add labels to the plot
ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data',
xytext=(50, -30), textcoords='offset points',
arrowprops=dict(arrowstyle="->",
connectionstyle="arc3,rad=-0.2"))
ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data',
bbox=dict(boxstyle="round", fc="none", ec="gray"),xytext=(10, -40), textcoords='offset points', ha='center',
arrowprops=dict(arrowstyle="->"))
ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
xytext=(0, -20), textcoords='offset points')
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
xycoords='data', textcoords='data',
arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data',
xytext=(-80, -40), textcoords='offset points',
arrowprops=dict(arrowstyle="fancy",
fc="0.6", ec="none",
connectionstyle="angle3,angleA=0,angleB=-90"))
ax.annotate('Thanksgiving', xy=('2012-11-25', 4500), xycoords='data',
xytext=(-120, -60), textcoords='offset points',
bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
arrowprops=dict(arrowstyle="->",
connectionstyle="angle,angleA=0,angleB=80,rad=20"))
ax.annotate('Christmas', xy=('2012-12-25', 3850), xycoords='data',
xytext=(-30, 0), textcoords='offset points',
size=13, ha='right', va="center",
bbox=dict(boxstyle="round", alpha=0.1),
arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));
# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
ylabel='average daily births')
# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
ax.set_ylim(3600, 5400);
```

![](10text%20and%20annotation%20Example_files/figure-gfm/cell-10-output-1.png)