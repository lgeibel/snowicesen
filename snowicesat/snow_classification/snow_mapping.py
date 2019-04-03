from __future__ import absolute_import, division

from distutils.version import LooseVersion
from salem import Grid, wgs84
import os
import numpy as np
import numpy.ma as ma
import pyproj
import logging
import xarray as xr
from crampon import entity_task
from crampon import utils
import snowicesat.cfg as cfg
from skimage import filters
from skimage import exposure
from skimage.io import imread
import matplotlib.pyplot as plt
import math
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
import rasterio
from rasterio.plot import show


# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def otsu_tresholding(gdir):
    """
    Performs Otsu_tresholding on sentinel-image
    of glacier. Returns snow cover map in asmag_snow_cover variable in
    snow_cover.nc
       :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return:
    """
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))

    #get NIR band as np array
    nir = sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values

    val = filters.threshold_otsu(nir)
#    hist, bins_center = exposure.histogram(nir)
    snow = nir > val
    snow = snow * 1

    plt.subplot(1,2,1)
    plt.imshow(nir, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(nir, cmap='gray')
    plt.imshow(snow, alpha=0.5)
    plt.show()

    #write to netcdf: copy xarray_dataset structure, drop bands -1, squeeze
    snow_xr = sentinel.drop([band_id for band_id in sentinel['band'].values][:-1], dim='band').squeeze('band', drop=True)
    snow_xr['asmag_snow_cover'] = snow_xr['img_values'] # assign new variable
    snow_xr = snow_xr.drop(['img_values'])
    snow_xr['asmag_snow_cover'].loc[
        (dict(time=cfg.PARAMS['date'][0]))] \
        = snow
    snow_xr.to_netcdf(gdir.get_filepath('snow_cover'), 'w')


@entity_task(log)
def naegeli_snow_mapping(gdir):
    """
    Performs snow cover mapping on sentinel-image
    of glacier as described in Naegeli, 2019- Change detection
     of bare-ice albedo in the Swiss Alps
    Returns snow cover map in naegeli_snow_cover variable in
    snow_cover.nc
       :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return:
    """
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))

    #Albedo shortwave to broadband conversion after Knap:
    alpha_knapp = 0.726 * sentinel.sel(band='B03', time=cfg.PARAMS['date'][0]).img_values.values +\
                  0.322 * sentinel.sel(band='B03', time=cfg.PARAMS['date'][0]).img_values.values ** 2 + \
                  0.015 * sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values + \
                  0.581 * sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values ** 2
    #Albedo conversion after Liang
    alpha_Liang = 0.356*sentinel.sel(band='B02', time=cfg.PARAMS['date'][0]).img_values.values +\
     0.130*sentinel.sel(band='B04', time=cfg.PARAMS['date'][0]).img_values.values +\
     0.373*sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values +\
     0.085*sentinel.sel(band='B11', time=cfg.PARAMS['date'][0]).img_values.values +\
     0.072*sentinel.sel(band='B12', time=cfg.PARAMS['date'][0]).img_values.values + 0.0018
    #
    # nir = sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values
    # 2,4,6,11,12

