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
from snowicesat.preprocessing.image_corrections import assign_bc
from skimage import filters
from skimage import exposure
from skimage.io import imread
import matplotlib.pyplot as plt
import math
import pandas as pd
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
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
    # TODO: Außenseite as np.nan

    # get NIR band as np array
    nir = sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values/10000
    val = filters.threshold_otsu(nir[nir != 0])
    hist, bins_center = exposure.histogram(nir[nir != 0])
    snow = nir > val
    snow = snow * 1

#    plt.figure(figsize=(9, 4))
#    plt.plot(bins_center, hist, lw=2)
#    plt.axvline(val, color='k', ls='--')
#    plt.title('Histogram and Otsu-Treshold')
#    plt.show()

#    plt.subplot(1,2,1)
#    plt.imshow(nir, cmap='gray')
#    plt.title('NIR Band')
#    plt.subplot(1,2,2)
#    plt.imshow(nir, cmap='gray')
#    plt.imshow(snow, alpha=0.5)
#    plt.title('Snow Covered Area after Ostu-Tresholding')
#    plt.show()

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
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
    dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
    elevation_grid = dem_ts.isel(time=0, band=0).height_in_m.values

    #Albedo shortwave to broadband conversion after Knap:
    albedo_k = 0.726 * sentinel.sel(band='B03', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
               + 0.322 * (sentinel.sel(band='B03', time=cfg.PARAMS['date'][0]).img_values.values/10000) ** 2 \
               + 0.015 * sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values/10000  \
               + 0.581 * (sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values/10000) ** 2

    # #Albedo conversion after Liang
    # albedo_l = 0.356 * sentinel.sel(band='B02', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.130 * sentinel.sel(band='B04', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.373 * sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.085 * sentinel.sel(band='B11', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.072 * sentinel.sel(band='B12', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.0018

    albedo = [albedo_k]

    # Peform primary suface type evaluation: albedo > 0.55 = snow,
    # albedo < 0.25 = ice, 0.25 < albedo < 0.55 = ambigous range,
    # secondary evaluation is performed

    for albedo_ind in albedo:
        if albedo_ind.shape != elevation_grid.shape: # account for 1 pixel offset in DEM
            elevation_grid = elevation_grid[0:albedo_ind.shape[0], 0:albedo_ind.shape[1]]
        snow = albedo_ind > 0.55
        ambig = (albedo_ind < 0.55) & (albedo_ind > 0.25)

        # plt.subplot(1, 2, 1)
        # plt.imshow(albedo_ind)
        # plt.imshow(snow, alpha = 0.5)
        # plt.title("Snow Area")
        # plt.subplot(1, 2, 2)
        # plt.imshow(albedo_ind)
        # plt.imshow(ambig, alpha = 0.5)
        # plt.title("Ambigous Area")
        # plt.show()

        # Find critical albedo: albedo at location with highest albedo slope
        # (assumed to be snow line altitude)

        # Albedo slope: get DEM and albedo of ambigous range, transform into vector
        dem_amb = elevation_grid[ambig]
        albedo_amb = albedo_ind[ambig]

        # Write dem and albedo into pandas DataFrame:
        df = pd.DataFrame({'dem_amb': dem_amb.tolist(),
                           'albedo_amb': albedo_amb.tolist()})
        # Sort values by elevation, drop negative values:
        df = df.sort_values(by=['dem_amb'])

        # Sort into 10 bands over entire range:
        dem_min = int(round(df[df.dem_amb > 0].dem_amb.min()))
        dem_max = int(round(df.dem_amb.max()))
        delta_h = int(round((dem_max - dem_min)/15))
        dem_avg = range(dem_min, dem_max, delta_h)
        albedo_avg = []
        for height_20 in dem_avg:
            # Index of df.dem that is between the current and the next elevation band into list:
            albedo_in_band = df.albedo_amb[(df.dem_amb > height_20) & (df.dem_amb < height_20+20)].tolist()
            # Average over all albedo values in one band:
            # what if band is empty? Skip to next step:
            if not albedo_in_band: # if list is empty append 0
                albedo_avg.append(0)
            else: # if not append average albedo of elevation band
                albedo_avg.append(sum(albedo_in_band)/len(albedo_in_band))

        # Interpolate if value = 0
        

        plt.plot(dem_avg, albedo_avg)
        plt.xlabel("Altitude in m")
        plt.ylabel("Albedo")
        plt.show()








