# TrackML Particle Tracking Challenge
Rank 19 solution of the Kaggle featured competition [TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification)

This repository only contains the code for predicting a single event. See [CFlow](https://github.com/liuxiao/CFlow) for scaling the code to all 125 testing events.

## Basic procedures
1. Cluster hits using DBSCAN on transformed data using various parameters to generate a pool of track candidates. This step is under `/mymodule/cluster2.py`
2. Remove duplicated tracks from the pool of candidates. Create a track object for each remaining candidate. Perform helix fitting and outlier removal. This step is under `/mymodule/track.py`
3. Merge tracks. This step is under `/mymodule/merger.py`
4. Above procedures are repeated several times with various parameters, focusing on clustering tracks with different length with longer track first.

The entire procedure is in `pipeline.ipynb`
