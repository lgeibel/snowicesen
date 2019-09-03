"""
=====
plots
=====

A collection of some useful plotting routines (some as entity tasks,
some as functions)

"""

from __future__ import absolute_import, division

import os
import numpy as np
import logging
import xarray as xr
from crampon import entity_task
import snowicesen.cfg as cfg
from snowicesen import utils, snow_mapping
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

    Parameters:
    ----------
    gdirs: :py:class
        `crampon.GlacierDirectory`
        A GlacierDirectory instance.
    Returns:
    --------
    None
    """
    # Open Sentinel File to get Background Image:

    try:
        sentinel_temp = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
        sentinel_cloud_masked = xr.open_dataset(gdir.get_filepath('cloud_masked'))
#        sentinel_ekstrand_corrected = xr.open_dataset(gdir.get_filepath('ekstrand'))
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
        dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
        snow_xr = xr.open_dataset(gdir.get_filepath('snow_cover'))
    except FileNotFoundError:
        log.error("One of the files necessary for plotting is not found")
        return
    ######  TODO: loop over snow_xr!!! #####
    for date in sentinel_cloud_masked.time.values:
        print(date)
        #### TODO: ------ USE SENTINEL_TMP and snow_xr for loop!!!!! ------
        b04 = sentinel_temp.sel(band='B02', time=date).img_values.values / 10000
        b03 = sentinel_temp.sel(band='B08', time=date).img_values.values / 10000
        b02 = sentinel_temp.sel(band='B12', time=date).img_values.values / 10000

        rgb_image = np.array([b04, b03, b02]).transpose((1, 2, 0))
    # Cut value > 1 to 1:
        rgb_image [rgb_image>1] = 1
    #    plt.imshow(rgb_image)
    #    plt.title(date)
    #    plt.show()

#        plot_cloud_cover(gdir, sentinel_ekstrand_corrected, sentinel_cloud_masked, date)
       # print('Plotting', date.values)
        print(date)
      #  plot_snow_cover_all(gdir, sentinel_temp, dem_ts, snow_xr, rgb_image, date)
        plot_cloud_cover(gdir,rgb_image, sentinel, sentinel_cloud_masked, date)


def plot_snow_cover_all(gdir, sentinel, dem_ts, snow_xr, rgb_image, date):
    """Plot Snow Mask and SLA as retrieved with ASMAG-
    Algorithm

    Parameters:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    sentinel_temp: Xarray Dataset
        Sentinel Images after all
        preprocessing steps
    dem_ts: Xarray Dataset
        Dem of scene in local grid
    snow_xr:  Xarray Dataset
        Snow cover Maps and SLA of all algorithms
    rgb_image: np.array
        RGB Image of tile as 3- Band np.parray
    date: int
        Current date for plotting

    Returns:
    -----------
    None
    """
    # Read snow maps from snow_xr:
    fig1 = plt.figure(figsize=(15, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(rgb_image)
    plt.title('RBG Image')
    
    try:
        plt.subplot(1, 4, 2)
        snow_xr.sel(time=date, model='asmag').snow_map.plot()
        plt.contour(dem_ts.isel(time=0,band=0 ).height_in_m.values, cmap='Greens')
            #   levels=[snow_xr.sel(time=date, model='asmag').SLA.values])
        plt.title('Asmag Snow Map & SLA')
        plt.subplot(1, 4, 3)
        snow_xr.sel(time=date, model='naegeli_orig').snow_map.plot()
        plt.contour(dem_ts.isel(time=0,band=0).height_in_m.values, cmap='Greens')
            #   levels=[snow_xr.sel(time=date, model='naegeli_orig').SLA.values])
        plt.title('Naegeli Original')
        plt.subplot(1, 4, 4)
        snow_xr.sel(time=date, model='naegeli_improv').snow_map.plot()
        plt.contour(dem_ts.isel(time=0,band=0 ).height_in_m.values, cmap='Greens')
             #  levels=[snow_xr.sel(time=date, model='naegeli_improv').SLA.values])
        plt.title('Naegeli Improved Method')
        plt.suptitle(str(gdir.name + " - " + gdir.id + " on " + str(date.values) ), fontsize=18)
        plt.show(fig1)
        plt.savefig(gdir.get_filepath('plt_all'), bbox_inches='tight')
        print(date)
    except ValueError:
        return




def plot_cloud_cover(gdir, rgb, sentinel, cloud_masked, index):
    """
    Plot Thermal Band 12 and Cloud Mask retrieved by the s2cloudless
    algorithm on top of the thermal band

    Parameters:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    ekstrand_corrected: Xarray DataSet
        ekstrand corrected scene
        for all bands & dates for this glacier
    cloud_masked: Xarray DataSet
        cloud cover masked scene for all bands
        & dates for this glacier

    Returns:
    -------
    None
    """
    # Cloud Mask = Ekstrand_correcte array - Cloud_masked array
    _,i = np.unique(sentinel['time'], return_index= True)
    sentinel = sentinel.isel(time=i)
    _,i = np.unique(cloud_masked['time'], return_index= True)
    cloud_masked = cloud_masked.isel(time=i)


    cloud_mask = \
        sentinel.sel(band='B08', time = index).\
            img_values.values \
        - cloud_masked.sel(band='B08', time = index)\
            .img_values.values

    cloud_mask = np.ma.array(cloud_mask)
    # mask values that are zero, plot only clouds:
    cloud_mask_masked = np.ma.masked_where(cloud_mask == 0, cloud_mask)
    print(cloud_mask_masked.shape)


    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(rgb)
    plt.imshow(cloud_mask_masked, cmap='inferno')
    plt.title("RGB + Masked Cloud Area")
    plt.axis('off')
    plt.suptitle(str(gdir.name + " - " + gdir.id) + ' on '+
                 str(index), fontsize=12)
    plt.savefig(gdir.get_filepath('plt_cloud_mask'), bbox_inches='tight')
    plt.show()

def plot_snow_cover_ASMAG(gdir, sentinel, dem_ts, rgb_image):
    """Plot Snow Mask and SLA as retrieved with ASMAG-
    Algorithm

    Parameters:
    -----------
    gdir:   gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    sentinel_temp: Xarray Dataset
        Sentinel Images after all
        preprocessing steps
    dem_ts: Xarray Dataset
        Dem of scene in local grid
    snow_xr: Xarray Dataset
        Snow cover Maps and SLA of all algorithms

    Returns:
    -----------
    None
    """
    # get NIR band as np array

    nir = sentinel.isel(band=7, time=0).img_values.values / 10000
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

    #snow = snow_xr.isel(model=0,time=0).snow_map.values
    #snow = np.ma.array(snow)
    snow = nir > val
    snow = snow * 1
    
    # mask values that are not = 1, plot only snow:
    snow_masked = np.ma.masked_where(snow != 1, snow)
    print(snow_masked.shape)

    print(val)
    fig1 = plt.figure(figsize=(15, 10))
    plt.rcParams.update({"font.size":16})
    ax0 =  plt.subplot2grid((1, 4),(0,0), colspan = 2)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')
    plt.title('Histogram and Otsu-Treshold')
    plt.xlabel('Data Value')
    plt.ylabel('Frequency')
    plt.title('Otsu Treshold = %1.3f' % val)
    plt.tick_params(labelsize=16)
    plt.grid()

    ax1 = plt.subplot2grid((1, 4), (0,2))
    plt.imshow(nir, cmap='gray')
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.axis("off")
    ax2 = plt.subplot2grid((1, 4), (0,3))
    plt.tight_layout()
    plt.imshow(nir, cmap='gray')
    plt.imshow(snow_masked, cmap='Greys')
    plt.title('Snow Cover after ASMAG')
    plt.axis("off")
    fig1.tight_layout(rect = [0, 0.03, 1, 0.95])
  #  fig1.suptitle(str(gdir.name + " - " + gdir.id), fontsize=18)
    plt.show(fig1)
    fig1.savefig(gdir.get_filepath('plt_otsu'), bbox_inches='tight')

def plot_snow_cover_naegeli(gdir, sentinel, dem_ts,  rgb_image):
    """ Plot Snow cover mask as retrieved after
    Method by Naegeli ------  NOT FINISEHD YET """

    plt.figure(figsize=(10, 10))
    plt.rcParams.update({"font.size":16})
    plt.subplot(1, 3, 1)
    plt.rcParams.update({"font.size":16})
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.title("RGB Image")

    # Albedo shortwave to broadband conversion after Knap:
    #albedo_k = snow_mapping.albedo_knap(sentinel)
    
    albedo_k = 0.726 * sentinel.isel(band=2,
                                    time=0).\
                                    img_values.values / 10000 \
               + 0.322 * (sentinel.isel(band=2,
                                       time=0).
                                    img_values.values / 10000) ** 2 \
               + 0.015 * sentinel.isel(band=7,
                                      time=0).\
                                        img_values.values / 10000 \
               + 0.581 * (sentinel.isel(band=7,
                                       time=0).
                                    img_values.values / 10000) ** 2


    # TODO: try with nir band only

    # Limit Albedo to 1
    albedo_k[albedo_k > 1] = 1
    albedo = [albedo_k]

    # Peform primary suface type evaluation: albedo > 0.55 = snow,
    # albedo < 0.25 = ice, 0.25 < albedo < 0.55 = ambiguous range,
    # Pixel-wise
    elevation_grid = dem_ts.isel(time=0).height_in_m.values[0,:,:]
    print(elevation_grid.shape, albedo[0].shape)
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

        snow = albedo_ind > 0.8
        ambig = (albedo_ind < 0.8) & (albedo_ind > 0.2)

        plt.subplot(1, 3, 2)
        plt.rcParams.update({"font.size":16})
        plt.imshow(rgb_image)
        plt.imshow(snow * 2, cmap="Blues_r")
        plt.imshow(ambig*1+ snow*2, cmap = "Blues_r")
        plt.contour(elevation_grid, colors= 'black',linewidths=0.3, 
                    levels=list(
                        range(int(elevation_grid[elevation_grid > 0].min()),
                              int(elevation_grid.max()),
                              100)
                              ))
        plt.axis('off')
        plt.title("Snow and Ambig. Area")


############     Iteration:
            # Albedo slope: get DEM and albedo of ambigous range, transform into vector
        if ambig[ambig==1].size > 3:  # only use if ambigious area is bigger than 3 pixels:
            try:
                dem_amb = elevation_grid[ambig]
            except IndexError:
                # FIXME: on RGI50-11:B5630n: Mismatch of boolean index
                log.error('BUG: (known on RGI50-11.B5630n: Mismatch of boolean index')
                return

            albedo_amb = albedo_ind[ambig]
             # Write dem and albedo into pandas DataFrame:
            df = pd.DataFrame({'dem_amb': dem_amb.tolist(),
                               'albedo_amb': albedo_amb.tolist()})
            # Sort values by elevation, drop negative values:
            df = df.sort_values(by=['dem_amb'])

            # Try two ways to obatin critical albedo:
            # 1. Fitting to step function:
            # albedo_crit_fit, SLA_fit = max_albedo_slope_fit(df)
            # 2. Iterate over elevation bands with increasing resolution
            try: 
               albedo_crit_it, SLA_it, r_square = snow_mapping.max_albedo_slope_iterate(df)
            except TypeError:
                # Function returns None: iteration could not be performed
                return
            # Result: both have very similar results, but fitting
            # function seems more stable --> will use this value
            SLA = SLA_it
            albedo_crit = albedo_crit_it
            # Derive corrected albedo with outlier suppression:
            albedo_corr = albedo_ind
            r_crit = 400
             # Make r_crit dependant on r_squared value (how well
             # does a step function model fit the elevation-albedo-profile?
             # Maximum for r_crit: maximum of elevation distance between SLA
             # and either lowest or highest snow-covered pixel
            if snow[snow * 1 == 1].size > 1:
                try:
                    r_crit_max = max(SLA - elevation_grid[snow * 1 == 1][
                    elevation_grid[snow * 1 == 1] > 0].min(),
                                 elevation_grid[snow * 1 == 1].max() - SLA)
                except ValueError: # Some Bug that I dont understand
                    return
            else:
                r_crit_max = elevation_grid[elevation_grid > 0].max() - SLA
            r_crit = - r_square * r_crit_max + r_crit_max
            r_crit = min(r_crit_max, r_crit)

            #Secondary Surface type Evaluation:
            for i in range(0, ambig.shape[0]):
                for j in range(0, ambig.shape[1]):
                    if ambig[i, j]:
                        albedo_corr[i, j] = albedo_ind[i, j] - \
                                            (SLA - elevation_grid[i, j]) * 0.005
                        # Secondary surface type evaluation on ambiguous range:
                        if albedo_corr[i, j] > albedo_crit:
                            snow[i, j] = True
                    # Probability test to eliminate extreme outliers:
                    if elevation_grid[i, j] < (SLA - r_crit):
                        snow[i, j] = False
                    if elevation_grid[i, j] > (SLA + r_crit):
                        snow[i, j] = True
        else:  # if no values in ambiguous area -->
            print(' No values in ambiguous area')
            r_crit = 400
            # either all now covered or no snow at all
            if snow[snow == 1].size / snow.size > 0.9:  # high snow cover:
                # Set SLA to lowest limit
                log.error('...because snow cover is very high - set SLA low')
                SLA = elevation_grid[elevation_grid > 0].min()
            elif snow[snow == 1].size / snow.size < 0.1:  # low/no snow cover:
                log.error('...because of no/very low snow cover - set SLA to max')
                SLA = elevation_grid.max()

        print(r_crit)
        plt.subplot(1, 3, 3)
        plt.rcParams.update({"font.size":16})
        plt.imshow(albedo_ind)
        plt.imshow(snow * 1, cmap="Blues_r")
        plt.contour(elevation_grid, colors= 'black',linewidths=0.3,
                    levels=list(
                       range(int(elevation_grid[elevation_grid > 0].min()),
                             int(elevation_grid.max()),
                             100)
                              ))
     #   plt.colorbar()
        crit_ar = np.ma.array(np.ones(elevation_grid.shape))
        upper = np.ma.masked_where((elevation_grid < SLA+r_crit) &   \
                (elevation_grid > SLA-r_crit),3*np.ones(elevation_grid.shape))
#        lower = np.ma.masked_where(elevation_grid < SLA-r_crit,np.ones(elevation_grid.shape))
        upper = elevation_grid
        upper[elevation_grid > SLA+r_crit] = None
        upper[elevation_grid < SLA-r_crit] = None
#        albedo_ind[elevation_grid < SLA+r_crit] = 1
#        albedo_ind[elevation_grid > SLA-r_crit] = 2
#        plt.imshow(albedo_ind)

#        plt.imshow(lower, 'Greens')
        plt.imshow(upper,'RdYlGn', alpha = 0.3)
#        plt.plot(elevation_grid(elevation_grid > SLA))
        plt.contour(elevation_grid, color='red',
                    levels=[SLA])
        plt.title("Final snow mask")
        plt.suptitle(str(gdir.name + " - " + gdir.id), fontsize=18)
        plt.axis('off')
    #    plt.show()
        plt.savefig(gdir.get_filepath('plt_impr_naegeli'), bbox_inches='tight')






        plt.show()

