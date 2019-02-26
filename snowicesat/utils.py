"""
A collection of some useful miscellaneous functions.
"""

from __future__ import absolute_import, division

from joblib import Memory
import posixpath
import salem
import os
import pandas as pd
import numpy as np
import logging
import paramiko as pm
import xarray as xr
import rasterio
import subprocess
from rasterio.merge import merge as merge_tool
from rasterio.warp import transform as transform_tool
from rasterio.mask import mask as riomask
import geopandas as gpd
import shapely
import datetime as dt
from configobj import ConfigObj, ConfigObjError
from itertools import product
import dask
import sys
import glob
import fnmatch
import netCDF4
from scipy import stats
from salem import lazy_property, read_shapefile
from functools import partial, wraps
from oggm.utils import *
# Locals
import snowicesat.cfg as cfg
from pathlib import Path
from osgeo import ogr


log = logging.getLogger(__name__)


def outline_envelope(geom):
    (minX, maxX, minY, maxY) = geom.GetEnvelope()

        # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minX, minY)
    ring.AddPoint(maxX, minY)
    ring.AddPoint(maxX, maxY)
    ring.AddPoint(minX, maxY)
    ring.AddPoint(minX, minY)

    # Create polygon
    poly_envelope = ogr.Geometry(ogr.wkbPolygon)
    poly_envelope.AddGeometry(ring)
    return poly_envelope