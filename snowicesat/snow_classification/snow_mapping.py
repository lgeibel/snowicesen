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
from scipy import stats
from scipy.optimize import leastsq, curve_fit
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

    sentinel.close()


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
    albedo_k = 0.726 * sentinel.sel(band='B03',
                                    time=cfg.PARAMS['date'][0]).img_values.values/10000 \
               + 0.322 * (sentinel.sel(band='B03',
                                    time=cfg.PARAMS['date'][0]).img_values.values/10000) ** 2 \
               + 0.015 * sentinel.sel(band='B08',
                                    time=cfg.PARAMS['date'][0]).img_values.values/10000  \
               + 0.581 * (sentinel.sel(band='B08',
                                    time=cfg.PARAMS['date'][0]).img_values.values/10000) ** 2

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
    # Pixel-wise
    for albedo_ind in albedo:
        if albedo_ind.shape != elevation_grid.shape:  # account for 1 pixel offset in DEM
            elevation_grid = elevation_grid[0:albedo_ind.shape[0], 0:albedo_ind.shape[1]]
        snow = albedo_ind > 0.55
        ambig = (albedo_ind < 0.55) & (albedo_ind > 0.25)

        plt.subplot(1, 3, 1)
        plt.imshow(albedo_ind)
        plt.imshow(snow, alpha = 0.5)
        plt.contour(elevation_grid)
        plt.title("Snow Area after 1st evaluation")

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

        # Try two ways to obatin critical albedo:
        # 1. Fitting to step function:
        albedo_crit_fit, SLA_fit = max_albedo_slope_fit(df)

        # 2. Iterate over elevation bands with increasing resolution
        #albedo_crit_it, SLA_it = max_albedo_slope_iterate(df)

        # Result: both have very similar results, but fitting
        # function seems more stable --> will use this value

        # Derive corrected albedo with outlier suppression:
        albedo_corr = albedo_ind
        r_crit = 
        for i in range(0,ambig.shape[0]):
            for j in range(0,ambig.shape[1]):
                if ambig[i,j]:
                    albedo_corr[i,j] = albedo_ind[i,j] - (SLA_fit - elevation_grid[i,j]) * 0.005
                    # Secondary surface type evaluation on ambiguous range:
                    if albedo_corr[i,j] > albedo_crit_fit:
                        snow[i,j] = True
                # Probability test to eliminate extreme outliers:
                if elevation_grid[i,j] < (SLA_fit - 400):
                    snow[i,j] = False
                if elevation_grid[i,j] > (SLA_fit + 400):
                    snow[i,j] = True

        print(SLA_fit)
        plt.subplot(1, 3, 2)
        plt.imshow(albedo_ind)
        plt.imshow(ambig, alpha = 0.5)
        plt.contour(elevation_grid)
        plt.title("Ambigous Area")

        plt.subplot(1, 3, 3)
        plt.imshow(albedo_ind)
        plt.imshow(snow, alpha=0.5)
        plt.contour(elevation_grid)
        plt.title("Final snow mask")
        plt.show()

    sentinel.close()

def max_albedo_slope_iterate(df):
    """Finds elevation and value of highest
    albedo/elevation slope while iterating over elevation bins of
    decreasing height extend
    ---------
    Input: df: Dataframe  containing the variable dem_amb (elevations of ambiguous range)
    and albedo_amb (albedo values in ambiguous range)
    Return: alb_max_slope, max_loc: Albedo at maximum albedo slope and location
    of maximum albedo slope
    """

    #Smart minimum finding:
    # Iterate over decreasing elevation bands: (2 bands, 4 bands, 8 bands, etc.)
    # Sort into bands over entire range:
    df = df[df.dem_amb > 0]
    dem_min = int(round(df[df.dem_amb > 0].dem_amb.min()))
    dem_max = int(round(df.dem_amb.max()))
    for i in range(0, int(np.log(df.dem_amb.size))):
        delta_h = int(round((dem_max - dem_min) / (2 ** (i + 1))))
        if delta_h > 20: # only look at height bands with h > 20 Meters
            dem_avg = range(dem_min, dem_max, delta_h)
            albedo_avg = []
            # Sort array into height bands:
            for num, height_20 in enumerate(dem_avg):
                # Write index of df.dem that is between the
                # current and the next elevation band into list:
                albedo_in_band = df.albedo_amb[(df.dem_amb > height_20) &
                                               (df.dem_amb < height_20 + 20)].tolist()
                # Average over all albedo values in one band:
                if not albedo_in_band:  # if list is empty append 0
                    albedo_avg.append(0)
                else:  # if not append average albedo of elevation band
                    albedo_avg.append(sum(albedo_in_band) / len(albedo_in_band))
            for num, local_alb in enumerate(albedo_avg):
                if albedo_avg[num] is 0:  # Interpolate if value == 0 as
                    if num > 0:
                        # interpolate between neighbours
                        albedo_avg[num] = (albedo_avg[num - 1] + albedo_avg[num + 1]) / 2
                    elif num == len(albedo_avg): # nearest neighbor:
                        albedo_avg[num] = albedo_avg[num - 1]
                    else:
                        albedo_avg[num] = albedo_avg[num+1]



            # Find elevation/location with steepest albedo slope in the proximity of max values from
            # previous iteration:
            if i > 1:
                if max_loc > 0:
                    max_loc_sub = np.argmax(np.abs(np.gradient
                                               (albedo_avg[(2*max_loc - 2):(2*max_loc + 2)])))
                    max_loc = 2*max_loc - 2 + max_loc_sub
                else:
                    max_loc = np.argmax(np.abs(np.gradient(albedo_avg)))
            else:
                max_loc = np.argmax(np.abs(np.gradient(albedo_avg)))
            if max_loc < (len(albedo_avg)-1):
                alb_max_slope = (albedo_avg[max_loc]+albedo_avg[max_loc+1])/2
                height_max_slope = (dem_avg[max_loc]+dem_avg[max_loc+1])/2

    #plt.plot(dem_avg, albedo_avg)
    #plt.axvline(height_max_slope, color='k', ls='--')
    #plt.xlabel("Altitude in m")
    #plt.ylabel("Albedo")
    #plt.show()

    return alb_max_slope, height_max_slope

def max_albedo_slope_fit(df):
    """
    Finds albedo slope with fitting to step function
    :param df:  Dataframe  containing the variable dem_amb (elevations of ambiguous range)
    and albedo_amb (albedo values in ambiguous range)
    Returns: alb_max_slope, max_loc: Albedo at maximum albedo slope and location
    of maximum albedo slope
    """
    df = df[df.dem_amb > 0]
    dem_min = int(round(df[df.dem_amb > 0].dem_amb.min()))
    dem_max = int(round(df.dem_amb.max()))

    delta_h = int(round((dem_max - dem_min) / (30)))
    dem_avg = range(dem_min, dem_max, delta_h)
    albedo_avg = []        # Sort array into height bands:
    for num, height_20 in enumerate(dem_avg):
       # Write index of df.dem that is between the
       # current and the next elevation band into list:
        albedo_in_band = df.albedo_amb[(df.dem_amb > height_20) &
                                           (df.dem_amb < height_20 + 20)].tolist()
        # Average over all albedo values in one band:
        if not albedo_in_band:  # if list is empty append 0
            albedo_avg.append(0)
        else:  # if not append average albedo of elevation band
            albedo_avg.append(sum(albedo_in_band) / len(albedo_in_band))
    for num, local_alb in enumerate(albedo_avg):
        if albedo_avg[num] is 0:  # Interpolate if value == 0 as
            #  central difference (Boundaries cant be zero)
            albedo_avg[num] = (albedo_avg[num - 1] + albedo_avg[num + 1]) / 2

    # curve fitting: bounds for inital model:
    # bounds:
    # a: step size of heaviside function: 0.1-0.3
    # b: elevation of snow - ice transition: dem_min -dem_max
    # c: average albedo of bare ice: 0.25-0.4

    popt, pcov = curve_fit(model, dem_avg, albedo_avg, bounds=([0.1, dem_min, 0.25], [0.3, dem_max, 0.4]))

    #plt.plot(dem_avg, albedo_avg, dem_avg, model(dem_avg, popt[0], popt[1], popt[2]))
    #plt.show()

    #get index of elevation of albedo- transition:
    max_loc = (np.abs(dem_avg - popt[1])).argmin()
    if max_loc < (len(albedo_avg) - 1):
        alb_max_slope = (albedo_avg[max_loc] + albedo_avg[max_loc + 1]) / 2
        height_max_slope = (dem_avg[max_loc] + dem_avg[max_loc + 1]) / 2
    else:
        alb_max_slope = albedo_avg[max_loc]
        height_max_slope = dem_avg[max_loc]
    return alb_max_slope, height_max_slope


def model(alti, a, b, c):
    """ Create model for step-function
    Input: alti: Altitude distribution of glacier
            a: step size of heaviside function
            b: elevation of snow-ice transition
            c: average albedo of bare ice
    Return: step-function model
    """
    return (0.5 * (np.sign(alti - b) + 1))*a + c  # Heaviside fitting function










