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
def asmag_snow_mapping(gdir):
    """
    Performs Otsu_tresholding on sentinel-image
    of glacier. Returns snow cover map in asmag_snow_cover variable in
    snow_cover.nc
       :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return: None
    """
    try:
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
    except FileNotFoundError:
        print("Exiting asmag_snow_mapping")
        return
    # TODO: Außenseite as np.nan

    # get NIR band as np array
    nir = sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values/10000
    if nir[nir > 0.2].size > 0:
        val = filters.threshold_otsu(nir[nir > 0.2])
        hist, bins_center = exposure.histogram(nir[nir > 0.2])
    else:
        val = 1
        # no pixels are snow covered

    snow = nir > val
    snow = snow * 1

    # Get Snow Line Altitude :
    SLA = get_SLA_asmag(gdir, snow)
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')
    plt.title('Histogram and Otsu-Treshold')

    plt.subplot(1,3,2)
    plt.imshow(nir, cmap='gray')
    plt.title('NIR Band')
    plt.subplot(1,3,3)
    plt.imshow(nir, cmap='gray')
    plt.imshow(snow, alpha=0.5)
    plt.title('Snow Covered Area after Ostu-Tresholding')
    plt.show()

    #write to netcdf: copy xarray_dataset structure, drop bands -1, squeeze
    snow_xr = sentinel.drop([band_id for band_id in sentinel['band'].values][:-1],
                            dim='band').squeeze('band', drop=True)
    snow_xr['asmag_snow_cover'] = snow_xr['img_values'] # assign new variable
    snow_xr = snow_xr.drop(['img_values'])
    snow_xr['asmag_snow_cover'].loc[
        (dict(time=cfg.PARAMS['date'][0]))] \
        = snow
    snow_xr.to_netcdf(gdir.get_filepath('snow_cover'), 'w')

    sentinel.close()

def get_SLA_asmag(gdir, snow):
    """Snow line altitude retrieval as described in the ASMAG algorithm.
    Returns None if there is no 20m elevation band with over 50% snow cover
    :param: gdir: :py:class:`crampon.GlacierDirectory`
                    A GlacierDirectory instance.
            snow: binary snow cover map as np-Array
    :return: SLA in meters
    """
    # Get DEM:
    dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
    elevation_grid = dem_ts.isel(time=0, band=0).height_in_m.values
    # Convert DEM to 20 Meter elevation bands:
    cover = []
    for num, height in enumerate(range(int(elevation_grid[elevation_grid > 0].min()),
                        int(elevation_grid.max()), 20)):
        if num > 0:
            #starting at second iteration:
            if snow.shape != elevation_grid.shape:
                print("Solar Angles and DEM grid have different size. "
                      "Pixel difference:", snow.shape[0] - elevation_grid.shape[0], ", ",
                      snow.shape[1] - elevation_grid.shape[1],
                      "Will be adapted to same grid size")
                if elevation_grid.shape[0] > snow.shape[0] or \
                        elevation_grid.shape[1] > snow.shape[1]:  # Shorten elevation grid
                    elevation_grid = elevation_grid[0:snow.shape[0], 0:snow.shape[1]]
                if elevation_grid.shape[0] < snow.shape[0]:  # Extend elevation grid: append row:
                    elevation_grid = np.append(elevation_grid,
                                               [elevation_grid[(elevation_grid.shape[0] -
                                                                snow.shape[0]), :]], axis=0)
                if elevation_grid.shape[1] < snow.shape[1]:  # append column
                    print('Adding column')
                    b = elevation_grid[:, (elevation_grid.shape[1] -
                                           snow.shape[1])].reshape(elevation_grid.shape[0], 1)
                    elevation_grid = np.hstack((elevation_grid, b))
                    # Expand grid on boundaries to obtain raster in same shape after

            # find all pixels with same elevation between "height" and "height-20":
            band_height = 20
            while band_height > 0:
                snow_band = snow[(elevation_grid > (height-band_height)) & (elevation_grid < height)]
                if snow_band.size == 0:
                        band_height -= 1
                else:
                    break
            # Snow cover on 20 m elevation band:
            cover.append(snow_band[snow_band == 1].size/snow_band.size)

    bands = 5
    num = 0
    if any(loc_cover > 0.5 for loc_cover in cover):
        while num < len(cover):
            # check if there are 5 continuous bands with snow cover > 50%
            if all(bins > 0.5 for bins in cover[num:(num+bands)]):
                # select lowest band as
                SLA = range(int(elevation_grid[elevation_grid > 0].min()),
                            int(elevation_grid.max()), 20)[num]
                print(SLA)
                break  # stop loop
            if num == (len(cover)-bands-1):
                # if end of glacier is reached and no SLA found:
                bands = bands - 1
                # start search again
                num = 0
            num += 1
    else:
        return
    dem_ts.close()
    print(SLA)
    return SLA

@entity_task(log)
def naegeli_snow_mapping(gdir):
    """
    Performs snow cover mapping on sentinel-image
    of glacier as described in Naegeli, 2019- Change detection
     of bare-ice albedo in the Swiss Alps
    Creates snow cover map in naegeli_snow_cover variable in
    snow_cover.nc
       :param gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return:
    """
    try:
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
    except FileNotFoundError:
        print("Exiting snow mapping 2", gdir)
        return
    print(gdir)
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

    # TODO: try with nir band only
    # #Albedo conversion after Liang
    # albedo_l = 0.356 * sentinel.sel(band='B02', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.130 * sentinel.sel(band='B04', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.373 * sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.085 * sentinel.sel(band='B11', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.072 * sentinel.sel(band='B12', time=cfg.PARAMS['date'][0]).img_values.values/10000 \
    #            + 0.0018

    albedo = [albedo_k]
    fig = plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(albedo_k, cmap='gray')
    plt.title("Albedo")    # Peform primary suface type evaluation: albedo > 0.55 = snow,
    # albedo < 0.25 = ice, 0.25 < albedo < 0.55 = ambigous range,
    # Pixel-wise
    for albedo_ind in albedo:
        if albedo_ind.shape != elevation_grid.shape:
            print("Solar Angles and DEM grid have different size. "
                  "Pixel difference:", albedo_ind.shape[0] - elevation_grid.shape[0], ", ",
                  albedo_ind.shape[1] - elevation_grid.shape[1],
                  "Will be adapted to same grid size")
            if elevation_grid.shape[0] > albedo_ind.shape[0] or \
                    elevation_grid.shape[1] > albedo_ind.shape[1]:  # Shorten elevation grid
                elevation_grid = elevation_grid[0:albedo_ind.shape[0], 0:albedo_ind.shape[1]]
            if elevation_grid.shape[0] < albedo_ind.shape[0]:  # Extend elevation grid: append row:
                print('Adding row')
                elevation_grid = np.append(elevation_grid,
                                           [elevation_grid[(elevation_grid.shape[0] -
                                                            albedo_ind.shape[0]), :]], axis=0)
            if elevation_grid.shape[1] < albedo_ind.shape[1]:  # append column
                print('Adding column')
                b = elevation_grid[:, (elevation_grid.shape[1] -
                                       albedo_ind.shape[1])].reshape(elevation_grid.shape[0], 1)
                elevation_grid = np.hstack((elevation_grid, b))
                # Expand grid on boundaries to obtain raster in same shape after
        snow = albedo_ind > 0.55
        ambig = (albedo_ind < 0.55) & (albedo_ind > 0.2)

        plt.subplot(2, 3, 2)
        plt.imshow(albedo_ind)
        plt.imshow(snow*1, alpha = 0.5)
        plt.contour(elevation_grid, origin='lower', cmap='flag',
                linewidths=2)
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
        albedo_crit_it, SLA_it = max_albedo_slope_iterate(df)

        # Result: both have very similar results, but fitting
        # function seems more stable --> will use this value
        SLA = SLA_it
        albedo_crit = albedo_crit_it

        # Derive corrected albedo with outlier suppression:
        albedo_corr = albedo_ind
        # TODO: make r_crit dynamical:
        r_crit = 400
        for i in range(0,ambig.shape[0]):
            for j in range(0,ambig.shape[1]):
                if ambig[i,j]:
                    albedo_corr[i,j] = albedo_ind[i,j] - \
                                       (SLA - elevation_grid[i,j]) * 0.005
                    # Secondary surface type evaluation on ambiguous range:
                    if albedo_corr[i,j] > albedo_crit:
                        snow[i,j] = True
                # Probability test to eliminate extreme outliers:
                if elevation_grid[i,j] < (SLA_fit - r_crit):
                    snow[i,j] = False
                if elevation_grid[i,j] > (SLA_fit + r_crit):
                    snow[i,j] = True

        print(SLA)
        plt.subplot(2, 3, 5)
        plt.imshow(albedo_k)
        plt.imshow(ambig*1)
        plt.contour(elevation_grid)
        plt.title("Ambigous Area")

        plt.subplot(2, 3, 6)
        plt.imshow(albedo_k)
        plt.imshow(snow*1)
        plt.contour(elevation_grid)
        plt.title("Final snow mask")
        plt.show()
        fig.savefig(gdir.get_filepath('plt_snowcover'))

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
        if delta_h > 30: # only look at height bands with h > 20 Meters
            dem_avg = range(dem_min, dem_max, delta_h)
            albedo_avg = []
            # Sort array into height bands:
            for num, height_20 in enumerate(dem_avg):
                # Write index of df.dem that is between the
                # current and the next elevation band into list:
                albedo_in_band = df.albedo_amb[(df.dem_amb > height_20) &
                                               (df.dem_amb < height_20 + delta_h)].tolist()
                # Average over all albedo values in one band:
                if not albedo_in_band:  # if list is empty append 0
                    albedo_avg.append(0)
                else:  # if not append average albedo of elevation band
                    albedo_avg.append(sum(albedo_in_band) / len(albedo_in_band))
            for num, local_alb in enumerate(albedo_avg):
                if albedo_avg[num] is 0:  # Interpolate if value == 0 as
                    if num > 0:
                        if num == (len(albedo_avg)-1): # nearest neighbor:
                            albedo_avg[num] = albedo_avg[num - 1]
                        # interpolate between neighbours
                        else:
                            albedo_avg[num] = (albedo_avg[num - 1] +
                                               albedo_avg[num + 1]) / 2
                    else:
                        albedo_avg[num] = albedo_avg[num+1]

            # Find elevation/location with steepest albedo slope in the proximity of max values from
            # previous iteration:
            if i > 1:
                if max_loc > 0:
                    max_loc_sub = np.argmax(np.abs(np.gradient
                                               (albedo_avg
                                                [(2*max_loc - 2):(2*max_loc + 2)])))
                    max_loc = 2*max_loc - 2 + max_loc_sub
                else:
                    max_loc = np.argmax(np.abs(np.gradient(albedo_avg)))
            else:
                max_loc = np.argmax(np.abs(np.gradient(albedo_avg)))
            if max_loc < (len(albedo_avg)-1):
                alb_max_slope = (albedo_avg[max_loc]+albedo_avg[max_loc+1])/2
                height_max_slope = (dem_avg[max_loc]+dem_avg[max_loc+1])/2

    plt.subplot(2, 3, 4)
    plt.plot(dem_avg, albedo_avg)
    plt.axvline(height_max_slope, color='k', ls='--')
    plt.xlabel("Altitude in m")
    plt.ylabel("Albedo")


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

    delta_h = int(round((dem_max - dem_min) / 30))
    delta_h = 1
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

    popt, pcov = curve_fit(model, dem_avg, albedo_avg,
                           bounds=([0.1, dem_min, 0.3], [0.3, dem_max, 0.45]))

    plt.subplot(2,3,3)
    plt.plot(dem_avg, albedo_avg, dem_avg, model(dem_avg, popt[0], popt[1], popt[2]))

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










