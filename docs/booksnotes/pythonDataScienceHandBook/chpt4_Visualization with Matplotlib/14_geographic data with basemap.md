Notes \[Book\] Data Science Handbook
================
by Jawad Haider

# **Chpt 4 - Visualization with Matplotlib**

# 14 -  Geographic Data with Basemap
------------------------------------------------------------------------

- <a href="#geographic-data-with-basemap"
  id="toc-geographic-data-with-basemap">Geographic Data with Basemap</a>
  - <a href="#map-projections" id="toc-map-projections">Map Projections</a>
    - <a
      href="#lets-start-by-a-convenience-routine-to-draw-our-world-map-along-with-the-longitude-and-latitude-lines"
      id="toc-lets-start-by-a-convenience-routine-to-draw-our-world-map-along-with-the-longitude-and-latitude-lines">Let’s
      start by a convenience routine to draw our world map along with the
      longitude and latitude lines:</a>
    - <a href="#cylindrical-projections"
      id="toc-cylindrical-projections">Cylindrical projections</a>
    - <a href="#pseudo-cylindrical-projections"
      id="toc-pseudo-cylindrical-projections">Pseudo-cylindrical
      projections</a>
    - <a href="#perspective-projections"
      id="toc-perspective-projections">Perspective projections</a>
    - <a href="#conic-projections" id="toc-conic-projections">Conic
      projections</a>
    - <a href="#other-projections" id="toc-other-projections">Other
      projections</a>
- <a href="#drawing-a-map-background"
  id="toc-drawing-a-map-background">Drawing a Map Background</a>
- <a href="#plotting-data-on-maps" id="toc-plotting-data-on-maps">Plotting
  Data on Maps</a>
------------------------------------------------------------------------

# Geographic Data with Basemap

One common type of visualization in data science is that of geographic
data. Matplot‐ lib’s main tool for this type of visualization is the
Basemap toolkit, which is one of several Matplotlib toolkits that live
under the mpl_toolkits namespace. Admittedly, Basemap feels a bit clunky
to use, and often even simple visualizations take much longer to render
than you might hope. More modern solutions, such as leaflet or the
Google Maps API, may be a better choice for more intensive map
visualizations. Still, Basemap is a useful tool for Python users to have
in their virtual toolbelts. In this sec‐ tion, we’ll show several
examples of the type of map visualization that is possible with this
toolkit.

Installation of Basemap is straightforward; if you’re using conda you
can type this and the package will be downloaded:

`$ conda install basemap`

``` python
# !conda install basemap
```

``` python
# !python -m pip install basemap
```

``` python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from PIL import Image
```

``` python
plt.figure(figsize=(8,8))
m=Basemap(projection='ortho', resolution='i', lat_0=50, lon_0=-100)
# m.bluemarble(scale=0.9);
# plt.show(m)

m.bluemarble()
m.drawcoastlines()
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-7-output-2.png)

``` python
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

ma = Basemap(llcrnrlon=-10.5,llcrnrlat=33,urcrnrlon=10.,urcrnrlat=46.,
             resolution='i', projection='cass', lat_0 = 39.5, lon_0 = 0.)

ma.bluemarble()

ma.drawcoastlines()

plt.show()
```

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-8-output-1.png)

**The useful thing is that the globe shown here is not a mere image; it
is a fully func‐ tioning Matplotlib axes that understands spherical
coordinates and allows us to easily over-plot data on the map! For
example, we can use a different map projection, zoom in to North
America, and plot the location of Seattle. We’ll use an etopo image
(which shows topographical features both on land and under the ocean) as
the map back‐ ground**

``` python
fig  = plt.figure(figsize=(8,8))
m=Basemap(projection='ortho', resolution=None,
         width=8E6, height=8E6,
         lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

# Map (long, lat) to (x,y) for plotting
x,y=m(-122.3,47.6)
plt.plot(x,y,'ok',markersize=5)
plt.text(x,y,'Seattle',fontsize=12);
plt.show()
```

    warning: width and height keywords ignored for Orthographic projection

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-9-output-3.png)

## Map Projections

The first thing to decide when you are using maps is which projection to
use. You’re probably familiar with the fact that it is impossible to
project a spherical map, such as that of the Earth, onto a flat surface
without somehow distorting it or breaking its continuity. These
projections have been developed over the course of human history, and
there are a lot of choices! Depending on the intended use of the map
projection, there are certain map features (e.g., direction, area,
distance, shape, or other consider‐ ations) that are useful to maintain.

### Let’s start by a convenience routine to draw our world map along with the longitude and latitude lines:

``` python
# from itertools import chain

# def draw_map(m, scale=0.2):
#     #draw a shadded-relief image
#     m.shadedrelief(scale=scale)
    
#     # lats and longs are returned as a dictionary
#     lats = m.drawparallels(np.linspace(-90,90,13))
#     lons = m.drawparallels(np.linspace(-180,180,13))
    
#     # keys contain the plt.Line2D instances
#     lat_lines=chain(*(tup[1][0] for tup in lats.items()))
#     lon_lines = chain(*(tup[1][0] for tup in lons.items()))
#     all_lines = chain(lat_lines, lon_lines)
    
#     # cycle throught these lines and set the desird style
#     for line in all_lines:
#         line.set(linestyle='-', alpha=0.3, color='w')
   
from itertools import chain
def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))
    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
```

### Cylindrical projections

The simplest of map projections are cylindrical projections, in which
lines of constant latitude and longitude are mapped to horizontal and
vertical lines, respectively. This type of mapping represents equatorial
regions quite well, but results in extreme dis‐ tortions near the poles.
The spacing of latitude lines varies between different cylindri‐ cal
projections, leading to different conservation properties, and different
distortion near the poles.  
**we show an example of the equidistant cylindrical pro‐ jection, which
chooses a latitude scaling that preserves distances along meridians.
Other cylindrical projections are the Mercator (projection=‘merc’) and
the cylin‐ drical equal-area (projection=‘cea’) projections.**

``` python
# An equidistant cylinderical projection
fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='cyl', resolution=None,
llcrnrlat=-90, urcrnrlat=90,
llcrnrlon=-180, urcrnrlon=180, )
draw_map(m)

# The additional arguments to Basemap for this view specify the latitude (lat) and lon‐
# gitude (lon) of the lower-left corner (llcrnr) and upper-right corner (urcrnr) for the
# desired map, in units of degrees
```

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-11-output-1.png)

``` python
# # An equidistant cylinderical projection
# fig = plt.figure(figsize=(8, 6), edgecolor='w')
# m = Basemap(projection='merc', resolution=None,
# llcrnrlat=-90, urcrnrlat=90,
# llcrnrlon=-180, urcrnrlon=180, )
# draw_map(m)

# The additional arguments to Basemap for this view specify the latitude (lat) and lon‐
# gitude (lon) of the lower-left corner (llcrnr) and upper-right corner (urcrnr) for the
# desired map, in units of degrees
```

### Pseudo-cylindrical projections

Pseudo-cylindrical projections relax the requirement that meridians
(lines of constant longitude) remain vertical; this can give better
properties near the poles of the projec‐ tion. The Mollweide projection
(projection=‘moll’) is one common example of this, in which all
meridians are elliptical arcs.  
**It is constructed so as to preserve area across the map: though there
are distortions near the poles, the area of small patches reflects the
true area. Other pseudo-cylindrical projections are the sinusoidal
(projection=‘sinu’) and Robinson (projection=‘robin’) projections.**

``` python
fig=plt.figure(figsize=(8,6),edgecolor='w')

m=Basemap(projection='moll',resolution=None,
         lat_0=0, lon_0=0)

draw_map(m)

# The extra arguments to Basemap here refer to the central latitude (lat_0) and longi‐
# tude (lon_0) for the desired map.
```

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-13-output-1.png)

### Perspective projections

Perspective projections are constructed using a particular choice of
perspective point, similar to if you photographed the Earth from a
particular point in space (a point which, for some projections,
technically lies within the Earth!). One common exam‐ ple is the
orthographic projection (projection=‘ortho’), which shows one side of
the globe as seen from a viewer at a very long distance. Thus, it can
show only half the globe at a time. Other perspective-based projections
include the gnomonic projection (projection=‘gnom’) and stereographic
projection (projection=‘stere’). These are often the most useful for
showing small portions of the map.

``` python
fig = plt.figure(figsize=(8, 8))
m=Basemap(projection='ortho',resolution=None,
         lat_0=50, lon_0=0)
draw_map(m)
```

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-14-output-1.png)

### Conic projections

A conic projection projects the map onto a single cone, which is then
unrolled. This can lead to very good local properties, but regions far
from the focus point of the cone may become very distorted. One example
of this is the Lambert conformal conic projection (projection=‘lcc’),
which we saw earlier in the map of North America. It projects the map
onto a cone arranged in such a way that two standard parallels
(specified in Basemap by lat_1 and lat_2) have well-represented
distances, with scale decreasing between them and increasing outside of
them. Other useful conic projec‐ tions are the equidistant conic
(projection=‘eqdc’) and the Albers equal-area (pro jection=‘aea’)
projection.

**Conic projections, like perspective projections, tend to be good
choices for representing small to medium patches of the globe**

``` python
fig = plt.figure(figsize=(8, 8))
m=Basemap(projection='lcc', resolution=None,
         lon_0=0, lat_0=50, lat_1=45, lat_2=55,
         width=1.6E7, height=1.2E7)
draw_map(m)

```

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-15-output-1.png)

### Other projections

If you’re going to do much with map-based visualizations, I encourage
you to read up on other available projections, along with their
properties, advantages, and disadvan‐ tages. Most likely, they are
available in the Basemap package. If you dig deep enough into this
topic, you’ll find an incredible subculture of geo-viz geeks who will be
ready to argue fervently in support of their favorite projection for any
given application!

# Drawing a Map Background

Earlier we saw the bluemarble() and shadedrelief() methods for
projecting global images on the map, as well as the drawparallels() and
drawmeridians() methods for drawing lines of constant latitude and
longitude. The Basemap package contains a range of useful functions for
drawing borders of physical features like continents, oceans, lakes, and
rivers, as well as political boundaries such as countries and US states
and counties. The following are some of the available drawing functions
that you may wish to explore using IPython’s help features:

• Physical boundaries and bodies of water  
`drawcoastlines()` Draw continental coast lines  
`drawlsmask()` Draw a mask between the land and sea, for sea with
projecting images on one or the other  
`drawmapboundary()` Draw the map boundary, including the fill color for
oceans  
`drawrivers()` Draw rivers on the map  
`fillcontinents()`

Fill the continents with a given color; optionally fill lakes with
another color

• Political boundaries  
`drawcountries()`  
Draw country boundaries  
`drawstates()`  
Draw US state boundaries  
`drawcounties()`  
Draw US county boundaries

• Map features  
`drawgreatcircle()`  
Draw a great circle between two points  
`drawparallels()`  
Draw lines of constant latitude  
`drawmeridians()`  
Draw lines of constant longitude  
`drawmapscale()`  
Draw a linear scale on the map

• Whole-globe images  
`bluemarble()`  
Project NASA’s blue marble image onto the map  
`shadedrelief()`  
Project a shaded relief image onto the map  
`etopo()`  
Draw an etopo relief image onto the map  
`warpimage()`  
Project a user-provided image onto the map

For the boundary-based features, you must set the desired resolution
when creating a Basemap image. The resolution argument of the Basemap
class sets the level of detail in boundaries, either ‘c’ (crude), ‘l’
(low), ‘i’ (intermediate), ‘h’ (high), ‘f’ (full), or None if no
boundaries will be used. This choice is important: setting high-
resolution boundaries on a global map, for example, can be very slow.

``` python
fig, ax = plt.subplots(1,2, figsize=(12,8))

for i, res in enumerate(['l','h']):
    m=Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2,
             width=90000, height=120000, resolution=res, ax=ax[i])
    
    m.fillcontinents(color='#FFDDCC', lake_color='#DDEEFF')
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    ax[i].set_title("resolution = '{0}''".format(res));
```

![](14_geographic%20data%20with%20basemap_files/figure-gfm/cell-16-output-1.png)

Notice that the low-resolution coastlines are not suitable for this
level of zoom, while high-resolution works just fine. The low level
would work just fine for a global view, however, and would be much
faster than loading the high-resolution border data for the entire
globe! It might require some experimentation to find the correct
resolution parameter for a given view; the best route is to start with a
fast, low-resolution plot and increase the resolution as needed.

# Plotting Data on Maps

Perhaps the most useful piece of the Basemap toolkit is the ability to
over-plot a variety of data onto a map background. For simple plotting
and text, any plt function works on the map; you can use the Basemap
instance to project latitude and longitude coordinates to (x, y)
coordinates for plotting with plt, as we saw earlier in the Seattle
example.

In addition to this, there are many map-specific functions available as
methods of the Basemap instance. These work very similarly to their
standard Matplotlib counterparts, but have an additional Boolean
argument latlon, which if set to True allows you to pass raw latitudes
and longitudes to the method, rather than projected (x, y) coordinates.

Some of these map-specific methods are:

`contour()/contourf()`  
Draw contour lines or filled contours  
`imshow()`  
Draw an image  
`pcolor()/pcolormesh()`  
Draw a pseudocolor plot for irregular/regular meshes  
`plot()`  
Draw lines and/or markers  
`scatter()`  
Draw points with markers  
`quiver()`  
Draw vectors  
`barbs()`  
Draw wind barbs  
`drawgreatcircle()`  
Draw a great circle
