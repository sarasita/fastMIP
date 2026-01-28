# Regions

This folder contains information and tools for aggregating data from the grid-cell level to regional scales.

The region definitions provided here are intended to support consistent spatial aggregation across modelling teams and analysis workflows.

In short, we rely on the AR6 land regions provided through the regionmask package and work with area-weighted regional averages if not states otherwise. A function for generating these aggregates is given in ./aggregate_to_regions.py