from __future__ import absolute_import, division

import os
import numpy as np
import logging
import xarray as xr
from crampon import entity_task
import snowicesat.cfg as cfg
from snowicesat import utils, snow_mapping
from skimage import filters
from skimage import exposure
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
def plot_results(gdir):
    """
    Display Results:

    - Cloud Mask
    - Snow Maps ASMAG
    - Snow Maps Naegeli
    - Snow Maps Naegeli_Improved

    :param  gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return: None
    """
    # Open Sentinel File to get Background Image:

    try:
        sentinel_temp = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
        sentinel_cloud_masked = xr.open_dataset(gdir.get_filepath('cloud_masked'))
        sentinel_ekstrand_corrected = xr.open_dataset(gdir.get_filepath('ekstrand'))
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
        dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
        snow_xr = xr.open_dataset(gdir.get_filepath('snow_cover'))
    except FileNotFoundError:
        return


    b04 = sentinel_temp.sel(band='B04', time=cfg.PARAMS['date'][0]).img_values.values / 10000
    b03 = sentinel_temp.sel(band='B03', time=cfg.PARAMS['date'][0]).img_values.values / 10000
    b02 = sentinel_temp.sel(band='B02', time=cfg.PARAMS['date'][0]).img_values.values / 10000

    rgb_image = np.array([b04, b03, b02]).transpose((1, 2, 0))
    # Cut value > 1 to 1:
    rgb_image [rgb_image>1] = 1

    plot_cloud_cover(gdir, sentinel_ekstrand_corrected, sentinel_cloud_masked)
    plot_snow_cover_ASMAG(gdir, sentinel_temp, dem_ts, snow_xr, rgb_image)

def plot_cloud_cover(gdir, ekstrand_corrected, cloud_masked):
    """
    Plot Thermal Band 12 and Cloud Mask retrieved by the s2cloudless
    algorithm on top of the thermal band
    Parameters:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    ekstrand_corrected: Xarray DataSet: ekstrand corrected scene
                for all bands & dates for this glacier
    cloud_masked: Xarray DataSet: cloud cover masked scene for all bands
            & dates for this glacier
    """
    # Cloud Mask = Ekstrand_correcte array - Cloud_masked array
    cloud_mask = \
        ekstrand_corrected.sel(band='B08', time=cfg.PARAMS['date'][0]).\
            img_values.values \
        - cloud_masked.sel(band='B08', time=cfg.PARAMS['date'][0])\
            .img_values.values

    cloud_mask = np.ma.array(cloud_mask)
    # mask values that are zero, plot only clouds:
    cloud_mask_masked = np.ma.masked_where(cloud_mask == 0, cloud_mask)

    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(ekstrand_corrected.sel(band='B12', time=cfg.PARAMS['date'][0]).
               img_values.values, cmap='gray')
    plt.title('Thermal Band')

    plt.subplot(1,2,2)
    plt.imshow(ekstrand_corrected.sel(band='B12', time=cfg.PARAMS['date'][0]).
               img_values.values, cmap='gray')
    plt.imshow(cloud_mask_masked, cmap='inferno')
    plt.title("Thermal Band 12 + Cloud Mask")
    plt.suptitle(str(gdir.name + " - " + gdir.id) + ' on '+
                 str(utils.int_to_datetime(cfg.PARAMS['date'])[0]), fontsize=12)
    plt.savefig(gdir.get_filepath('plt_cloud_mask'), bbox_inches='tight')

def plot_snow_cover_ASMAG(gdir, sentinel, dem_ts, snow_xr, rgb_image):
    """Plot Snow Mask and SLA as retrieved with ASMAG-
    Algorithm

    Parameters:
    -----------
    gdir:     gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    sentinel_temp: Xarray Dataset: Sentinel Images after all
        preprocessing steps
    dem_ts: Xarray Dataset: Dem of scene in local grid
    snow_xr: Xarray Dataset: Snow cover Maps and SLA of all algorithms

    Returns:
    -----------
    None
    """
    # get NIR band as np array

    nir = sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values / 10000
    if nir[nir > 0].size > 0:
        try:
            val = filters.threshold_otsu(nir[nir > 0])
        except ValueError:
            # All pixels cloud-covered and not detected correctly
            # Manually set as no snow
            val = 1
            bins_center = 0
            hist = 0

        hist, bins_center = exposure.histogram(nir[nir > 0])
    else:
        val = 1
        bins_center = 0
        hist = 0

    snow = snow_xr.sel(model='asmag', time=cfg.PARAMS['date'][0]).snow_map.values
    snow = np.ma.array(snow)
    # mask values that are not = 1, plot only snow:
    snow_masked = np.ma.masked_where(snow != 1, snow)

    fig1 = plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')
    plt.title('Histogram and Otsu-Treshold')

    plt.subplot(1, 3, 2)
    plt.imshow(nir, cmap='gray')
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.subplot(1, 3, 3)
    plt.imshow(nir, cmap='gray')
    plt.imshow(snow_masked, cmap='Greys')
    plt.title('Snow Covered Area after Ostu-Tresholding')
    plt.suptitle(str(gdir.name + " - " + gdir.id), fontsize=18)
    plt.show(fig1)
    plt.savefig(gdir.get_filepath('plt_otsu'), bbox_inches='tight')

def plot_snow_cover_naegeli(gdir, sentinel, dem_ts, snow_xr, rgb_image):
    """ Plot Snow cover mask as retrieved after
    Method by Naegeli """

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")

    # Albedo shortwave to broadband conversion after Knap:
    albedo_k = snow_mapping.albedo_knap(sentinel)

    # TODO: try with nir band only

    # Limit Albedo to 1
    albedo_k[albedo_k > 1] = 1
    albedo = [albedo_k]

    # Peform primary suface type evaluation: albedo > 0.55 = snow,
    # albedo < 0.25 = ice, 0.25 < albedo < 0.55 = ambiguous range,
    # Pixel-wise
    for albedo_ind in albedo:
        if albedo_ind.shape != elevation_grid.shape:
            if elevation_grid.shape[0] > albedo_ind.shape[0] or \
                    elevation_grid.shape[1] > albedo_ind.shape[1]:  # Shorten elevation grid
                elevation_grid = elevation_grid[0:albedo_ind.shape[0], 0:albedo_ind.shape[1]]
            if elevation_grid.shape[0] < albedo_ind.shape[0]:  # Extend elevation grid: append row:
                elevation_grid = np.append(elevation_grid,
                                           [elevation_grid[
                                            (elevation_grid.shape[0] -
                                             albedo_ind.shape[0]), :]], axis=0)
            if elevation_grid.shape[1] < albedo_ind.shape[1]:  # append column
                b = elevation_grid[:, (elevation_grid.shape[1] -
                                       albedo_ind.shape[1])]. \
                    reshape(elevation_grid.shape[0], 1)
                elevation_grid = np.hstack((elevation_grid, b))
                # Expand grid on boundaries to obtain raster in same shape

        snow = albedo_ind > 0.55
        ambig = (albedo_ind < 0.55) & (albedo_ind > 0.2)

        plt.subplot(2, 2, 2)
        plt.imshow(albedo_ind)
        plt.imshow(snow * 2 + 1 * ambig, cmap="Blues_r")
        plt.contour(elevation_grid, cmap="hot",
                    levels=list(
                        range(int(elevation_grid[elevation_grid > 0].min()),
                              int(elevation_grid.max()),
                              int((elevation_grid.max() -
                                   elevation_grid[elevation_grid > 0].min()) / 10)
                              )))
        plt.colorbar()
        plt.title("Snow and Ambig. Area")

