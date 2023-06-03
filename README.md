# BioimageChipTracer

Measure any illuminating targets a biological images and track their movements.
Actually there isn't any hexagon detection but maxima finding and flooding to see their 
area sizes.

The hexagons can have dim borders and shape and may not look like a hexagon. The image can
contain cells even when it's GFP that is supposed to only have the crystals. So there is a
trade-off between recall and accuracy.

## To install and build the cython modules

```bash
pip install -r requirements.txt

python setup.py build_ext --inplace
```

The `findmaxima2d` may contain some deprecated numpy codes, if you use the higher version of
python, you can modify the package code manually to make it work.

I forked a third party source code `gwdt` for distance transforming. This one and some modules I
wrote need cython building before you run anything. The build options are located in `setup.py`

## User Guide

### Basic steps

The main processing functions are in `tracer`. Some functions are parameterized while
some others are not for faster development. Feel free to change the code because it's not a package.

Note that pyx files are those cythonized. You must rebuild the code to make your change work.

First, the chips are detected as local maxima using a measure same with imageJ, which is available as a pypi package
named `findmaxima2d`

With local maxima found, `detection.pyx` detects measures the shape and area of the chips. It estimates the radius of
the chip and uses active contour to fit the border to get the segmentation.

`tracking.py` connects detected chips to create tracks. It backtracks from the
last frame usually with highly confident chips to go back to find their smaller ancestors.

`draw.pyx` is used to plot circles, segmentations and contours. They are used to make videos.

The module `gwdt` is forked for the image processing before maxima finding.

### Pipeline script for an image series archive

The above steps are integrated for a bunch of real data in `pipeline` in a parallel computing manner. 

* `filter_stacks.py` is for batch detection for all the image frames in parallel.
* `track_detections.py` just call the tracking.
* `area_plot.py` plots the growth of area for each track.
* `filming.py` generates growth videos for each track.

Because I put the image series in multiple zip files, this pipeline does a lot of unzipping.