# CrystalTracer

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

The `findmaxima2d` contains deprecated numpy implementations. When using newer versions of
python, it becomes necessary to hack the package code to solve the errors.

I forked a third party source code `gwdt` for distance transforming. This one and some modules I
wrote need cython building before you run anything. The build options are located in `setup.py`

## User Guide

### GUI

Run `crystal_tracer/gui/main.py` to use the GUI tool for the pipeline.

Features:

1. multiprocessing with low memory cost
2. highly interactive visualization of intermediate results for tuning parameters
3. save/load configuration (of paths & parameters)
4. automatic navigation & file storage in a designated working directory
5. exporting tracking video and animated plots
6. easy-to-use shortcuts
7. accepts only CZI image as input

### Modules

`crystal_tracer.algorithm` contains the CV methods for crystal detection and path tracking. Use cython
and opencv to speed up computation.

`crystal_tracer.visual` provides basic visualization methods, including masking and video generation.

`crystal_tracer.gui` contains the GUI modules.

## Pipeline explanation

### Step 1: crystal detection

Frame by frame detection of crystals. For each frame, generating a table of crystal properties (coords, size, etc.)
and masks. They are numbered according to their timestamp.

There are currently 2 modes for masking. One is flooding, the other is active contour. The latter is more
stable and a bit more costly, and it can use the bright field image (but totally fine w/o it).

Parameters are highly adaptive. The block size should be high enough to cover the crystals, the
tolerance should be high when the background is packed with illuminant stuff.

### Step 2: path tracking

Starting from the crystals in the last frame, making tracks for each of them independently.
It's highly possible that different tracks can share common crystals along the path, especially
in dense regions. A neighborhood matching technique is adopted to compare the crystal candidates
more robustly.

To allow for more aggressive tracking, you can increase the distance threshold and time gap, as well as
relaxing the area restrictions. Typically, the distance threshold should be no more
than the biggest motion distance of the crystals. The time gap allows the tracking to be continued
when the current frame contains no useable candidate for a track. A big gap can easily let crystals bounce
to other tracks. The candidate crystals for each track are dynamically judged by prediction
from previous values, so the area can be kept more stable and invalid crystals are excluded.

### Step 3: plotting

Plotting the animated line plot of crystal sizes and making videos for each track.
