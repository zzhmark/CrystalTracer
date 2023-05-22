# HexaVidCounter

Hexagon Video Counter, to measure the hexagon crystals growth in a biological imaging video.
Actually there isn't any hexagon detection but maxima finding and flooding to see their 
area sizes.

The hexagons can have dim borders and shape and may not look like a hexagon. The image can
contain cells even when it's GFP that is supposed to only have the crystals. So there is a
trade-off between recall and accuracy.

## To install and build the cython modules

```bash
pip install numpy==1.19 Cython findmaxima2d opencv-python matplotlib

python setup.py build_ext --inplace
```

The findmaxima2d may contain some deprecated numpy codes, if you use the higher version of
python, you can modify the package code manually to make it work.

I forked a third party source code `gwdt` for distance transforming. This one and some modules I
wrote need cython building before you run anything. The build options are located in `setup.py`

## User Guide

### Basic steps

The main processing functions are in `flake_detection`. Some functions are parameterized while
some others are not for faster development. Feel free to change the code because it's not a package.

Note that pyx files are those cythonized. You must rebuild the code to make your change work.

With local maxima found with a measure same with imageJ, `find_circles.pyx` contains the code and measure the radius 
supposing they are circles, and the code to flood the maxima to calculate areas. The maxima found
here may not be the center of the hexagon sometimes, and the radius is typically not accurate.
Areas measured are more often stable.

`filter.py` integrates the above steps to solve the inaccuracy. The center of the hexagon
becomes the average coordination of the flooding and radius is recalculated. But there may
still be many misdetection of nonhexagon structures like cells, but usually this is ok
because the following steps have some robustness against such situation.

`trace.py` connects hexagon annotations to create an evolving stream. It backtracks from the
last frame usually with easy to see hexagons to go way down to find their smaller ancestors.
To make it more robust, it finds an ancestor across several frames in case one of them has
a serious detection error. So the time frame of each trace are not continuous.
It uses a KDTree to retrieve nearest detections and compare their distance,
sizes and time difference altogether to find the best precursor.

`draw.py` is used to plot the circles and flooding. They can be used to make videos.

### Integrated sample code

The sample processing steps and test codes are written in `experiment`. 
They are more detailed procedures
that integrate what's in `flake_detection` and use multiprocessing features to speed up. You
can modify and run these codes for an image series.

`hexa_extract.py` contains the image processing step, which I used to test single images and
adjust the parameter. It's not used in the whole pipeline.

`sample_img_filter.py` is a more formal version to batch processing the images. It gives you
the detection dataframes and can output some images for circles and flooding(for memory reasons I commented them).

`sample_img_trace.py` finds traces, but since multiprocessing is implemented in `trace.py`,
it's rather simple.

`sample_plot.py` plots the growth of area for each hexagon trace.

`sample_video_gen.py` generates videos for each hexagon trace.

As you see, the `sample*.py` functions together and uses a zip file of all frames of imaging
frames as its input. It decompresses the zipped files in a temporary folder to make it more
efficient for memory usage, but may be slowed down a bit. When I tested, the filtering step
takes the most time.