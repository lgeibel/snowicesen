""" 
============
snow_mapping
============

Three snow cover and snow line altitude mapping algorithms
are implemented here (as entity tasks)
  -  ASMAG Snow Mapping based on Otsu- Thresholding
  - Snows mapping after Naegeli, 2019 (see function description for full reference)
  - A modified/ more flexible version of the algorithm by Naegeli, 2019

So far it is implemented in a way that it needs to run all 3 methods in 
the order shown above in order to create a proper output file.
(Eventually a better way to create the snow_cover.nc file needs 
to be implemented)

The output is stored in snow_cover.nc with the 3 Models 
as dimensions, a time dimension and the variables snow_map and SLA

"""

from __future__ import absolute_import, division

import os
import numpy as np
import logging
import xarray as xr
from crampon import entity_task
import snowicesen.cfg as cfg
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
def asmag_snow_mapping(gdir):
    """ Snow Mapping after ASMAG-Algorithm

    Performs Otsu-Thresholding on Sentinel-image
    of glacier and SLA retrieval as described by
    P. Rastner:
    "Automated mapping of snow cover on glaciers and
    calculation of snow line altitudes from multi-
    temporal Landsat data" (in Review)

    Stores snow cover map in "asmag" dimension of
    snow_cover.nc in variables snow_map and SLA

    Parameters
    ----------
    gdirs: :py:class:`crampon.GlacierDirectory`
       A GlacierDirectory instance.
    Returns
    ------
    None

    """
    try:
        try:
            sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))    
        except FileNotFoundError:
            # no data for 1st day
            log.error('No data for first day - Abort ASMAG_snow_mapping')
            return

        # get NIR band as np array
        try:
            nir = sentinel.sel(band='B08', time=cfg.PARAMS['date'][0]).img_values.values / 10000
        except KeyError:
            # no data fir this glacier & file
            log.error('No data for this glacier and file- Abort ASMAG_snow_mapping')
            return
        if nir[nir > 0].size > 0:
            try:
                val = filters.threshold_otsu(nir[nir > 0])
            except ValueError:
                # All pixels cloud-covered and not detected correctly
                # Manually set as no snow
                log.error('Exceptional case: all pixel cloud-covered but not detected correctly')
                val = 1
                bins_center = 0
                hist = 0

            hist, bins_center = exposure.histogram(nir[nir > 0])
        else:
            log.error('No pixels snow covered')
            val = 1
            bins_center = 0
            hist = 0
            # no pixels are snow covered

        snow = nir > val
        snow = snow * 1

        # Get Snow Line Altitude :
        SLA = get_SLA_asmag(gdir, snow)
        if SLA is None:
            SLA = 0

        # write snow map to netcdf:
        if not os.path.exists(gdir.get_filepath('snow_cover')):
            # create new dataset:
            snow_xr = sentinel.drop([band_id for band_id in sentinel['band'].values][:-1],
                                    dim='band').squeeze('band', drop=True)
                # When no snow_cover.nc but sentinel.nc file 
                # has been created for previous time steps:
            snow_xr = snow_xr.drop([time_id for time_id in sentinel['time'].values][:-1],
                                    dim='time').squeeze('time', drop= True)
            snow_xr = snow_xr.assign_coords(time = cfg.PARAMS['date'][0])
            snow_xr = snow_xr.expand_dims('time')
            snow_xr = snow_xr.drop(['img_values'])
            # add dimension: "Model" with entries: asmag, naegeli_orig, naegeli_improv
            snow_xr['model'] = ('model', ['asmag', 'naegeli_orig', 'naegeli_improv'])
            snow_xr['snow_map'] = (['model','time', 'y', 'x'],
                                     np.zeros((3,1,snow.shape[0], snow.shape[1]), dtype=np.uint16))
            # new variables "snow_map" and "SLA" (snow line altitude)
            snow_xr['SLA'] = (['model', 'time'], np.zeros((3, 1), dtype=np.uint16))



        else:
            # nc. already exists, create new xarray Dataset and concat
            # to obtain new time dimension
            snow_xr = xr.open_dataset(gdir.get_filepath('snow_cover'))
            snow_new_ds = snow_xr.copy()
            snow_new_ds = snow_new_ds.isel(time=0)
            snow_new_ds.coords['time'] = np.array([cfg.PARAMS['date'][0]])
            snow_xr = xr.concat([snow_xr, snow_new_ds], dim='time')
        try:
            # Assign snow map to ASMAG model snow_map variable
            snow_xr['snow_map'].loc[dict(model='asmag', time=cfg.PARAMS['date'][0])] = \
                    np.zeros([snow.shape[0], snow.shape[1]])
                    # weird bug, first need to assign zeros array...
            snow_xr['snow_map'].loc[dict(model='asmag', time=cfg.PARAMS['date'][0])] = \
                    snow

            # Assign NaN Value:
            snow_xr = snow_xr.where(snow_xr['snow_map']!=0)
        except ValueError:
            # TODO: very strange bug on RGI50-11A14C03-1, after 3rd day:
            # dimensions of "snow" and "sentinel" are different?
            log.error('Bug (only known for GRI50-11.A14C03-1)- Abort Snow Mapping')
            return
        # Assign SLA:
        snow_xr['SLA'].loc[dict(model='asmag', time=cfg.PARAMS['date'][0])] = SLA

        # Assign np.zeros for other two models (to make sure the dimensions are correct)
        snow_xr['snow_map'].loc[dict(model='naegeli_orig', time = cfg.PARAMS['date'][0])] = \
                    np.zeros([snow.shape[0], snow.shape[1]])
        snow_xr['SLA'].loc[dict(model = 'naegeli_orig', time=cfg.PARAMS['date'][0])]= 0
        snow_xr['snow_map'].loc[dict(model='naegeli_improv',time=cfg.PARAMS['date'][0])] = \
                    np.zeros([snow.shape[0], snow.shape[1]])
        snow_xr['SLA'].loc[dict(model='naegeli_improv', time = cfg.PARAMS['date'][0])]= 0
       

        snow_new = snow_xr.copy()
        #try:
        #    # safe dataset to file
        #    snow_new.to_netcdf(gdir.get_filepath('snow_cover'), 'w')
        #except PermissionError:
        snow_xr.close()
        # remove old file:
        if os.path.isfile(gdir.get_filepath('snow_cover')):
            os.remove(gdir.get_filepath('snow_cover'))
        snow_new.to_netcdf(gdir.get_filepath('snow_cover'), 'w')
        log.info('Asmag snow map written for {}'.format(cfg.PARAMS['date'][0]))

        sentinel.close()
        snow_new.close()
        log.debug('SLA 1 = {}'.format(SLA))
        print("SLA1 = ",SLA)

    except:
        print("Error occured, quitting")
        return

def get_SLA_asmag(gdir, snow, band_height = 20, bands = 5):
    """Snow line altitude retrieval as described in the
    ASMAG algorithm.

    This function is used to calculate an estimate for 
    the Snow Line Altitude (SLA) from a binary snow cover map.
    It is used for the ASMAG-Algorithm and both versions of
    the algorithm by Naegeli, 2019.

    Checks if 5 continuous 20m elevation bands have a snow
    cover higher than 50%.
    If so, the lowest elevation value is used as SLA. If no 5
    continuous elevation bands satisfy this criteria, the search
    continues for 4, then 3, etc.

    If there is no 20m elevation band with over 50% snow cover, the
    functions returns "None"

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
            A GlacierDirectory instance.
    snow: np_array:
            binary snow cover map of area in
            GlacierDirectory
    band_height: int:(optional, default = 20 m)
            height of elevation band to look at for
            SLA determination in m
    bands: int: (optional, default = 5)
            amount of continuous bands to start looking at to
            determine SLA


    Returns
    -------
    SLA: float
            SLA in meters (None if no snow covered bands)

    """
    try:
        # Get DEM:
        dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
        elevation_grid = dem_ts.isel(time=0, band=0).height_in_m.values
        # Convert DEM to 20 Meter elevation bands:
        cover = []
        for num, height in enumerate(range(int(elevation_grid[elevation_grid > 0].min()),
                                           int(elevation_grid.max()), 20)):
            if num > 0:                        
                # starting at second iteration:
                while snow.shape != elevation_grid.shape:
                    if elevation_grid.shape[0] > snow.shape[0] or \
                            elevation_grid.shape[1] > snow.shape[1]:  # Shorten elevation grid
                        elevation_grid = elevation_grid[0:snow.shape[0], 0:snow.shape[1]]
                    if elevation_grid.shape[0] < snow.shape[0]:  # Extend elevation grid: append row:
                        try: 
                            elevation_grid = np.append(elevation_grid,
                                                   [elevation_grid[(elevation_grid.shape[0] -
                                                                    snow.shape[0]), :]], axis=0)
                        except IndexError:
                            # BUG: very exeptionally, the snow_map is broken --> 
                            log.error('Snow map is broken - BUG!')
                            return
                    if elevation_grid.shape[1] < snow.shape[1]:  # append column
                        b = elevation_grid[:, (elevation_grid.shape[1] -
                            snow.shape[1]):elevation_grid.shape[1]]
                        elevation_grid = np.hstack((elevation_grid, b))
                        # Expand grid on boundaries to obtain raster in same shape after

                # find all pixels with same elevation between "height" and "height-20":
                while band_height > 0:
                    try:
                        snow_band = snow[(elevation_grid > (height - band_height))
                                     & (elevation_grid < height)]
                    except IndexError:
                        log.error(' Index Error:', elevation_grid.shape, snow.shape)
                    if snow_band.size == 0:
                        band_height -= 1
                    else:
                        break
                # Snow cover on 20 m elevation band:
                if snow_band.size == 0:
                    cover.append(0)
                else:
                    cover.append(snow_band[snow_band == 1].size / snow_band.size)

        num = 0
        if any(loc_cover > 0.5 for loc_cover in cover):
            while num < len(cover):
                # check if there are 5 continuous bands with snow cover > 50%
                if all(bins > 0.5 for bins in cover[num:(num + bands)]):
                    # select lowest band as SLA
                    SLA = range(int(elevation_grid[elevation_grid > 0].min()),
                                int(elevation_grid.max()), 20)[num]
                    break  # stop loop
                if num == (len(cover) - bands - 1):
                    # if end of glacier is reached and no SLA found:
                    bands = bands - 1
                    # start search again
                    num = -1
                if len(cover)<=bands:
                    bands = bands-1
                    num = -1
                num += 1
        else:
            SLA = elevation_grid.max()
        dem_ts.close()
        return SLA
    except:
        return

@entity_task(log)
def naegeli_snow_mapping(gdir):
    """ Snow Cover Mapping

    Performs snow cover mapping on Sentinel2-Image
    of glacier as described in
    Naegeli, 2019
    "Change detection
    of bare-ice albedo in the Swiss Alps"
    https://www.the-cryosphere.net/13/397/2019/tc-13-397-2019.html

    SLA is calculated with get_SLA_asmag() function.

    Saves snow cover and SLA in naegeli_orig dimension in
    snow_cover.nc netCDF file

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns
    -------
    None

    """
    try:
        # Primary surface type evaluation:
        
        try:
            snow, ambig, elevation_grid, albedo_ind = primary_surface_type_evaluation(gdir)
        except TypeError:
            # Function Returns None: No image for this glacier available
            log.error('No image for this glacier available - Abort Naegeli_snow_mapping')
            return

        # Albedo slope: get DEM and albedo of ambigous range,
        # transform into vector
        if ambig.any():  # only use if ambigious area contains any True values
            try:
               dem_amb = elevation_grid[ambig]
            except IndexError: 
                log.error(' FIXME: weird bug (known only oon 2015-08-06 on RGI50-11.B5630n:',
                ' dimensions dont match. Maybe related to overlap of tiles')
                return
            albedo_amb = albedo_ind[ambig]

            # Write dem and albedo into pandas DataFrame:
            df = pd.DataFrame({'dem_amb': dem_amb.tolist(),
                               'albedo_amb': albedo_amb.tolist()})
            # Sort values by elevation, drop negative values:
            df = df.sort_values(by=['dem_amb'])
            df = df[df.dem_amb > 0]
            #if not df.isnull.values.any()
             # 2. find location with maximum albedo slope
            try:
                albedo_crit, SLA = max_albedo_slope_orig(df)
            # Result: both have very similar results, but fitting
            # function seems more stable --> will use this value
            except TypeError:
                #BUG:  function returns None: some NaNs if d
                return

            # Derive corrected albedo with outlier suppression:
            albedo_corr = albedo_ind
            r_crit = 400
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
            r_crit = 400
            # either all snow covered or no snow at all
            if snow[snow == 1].size / snow.size > 0.9:  # high snow cover:
                # Set SLA to lowest limit
                SLA = elevation_grid[elevation_grid > 0].min()
            elif snow[snow == 1].size / snow.size < 0.1:  # low/no snow cover:
                SLA = elevation_grid.max()
        #plt.subplot(2, 2, 4)
        #plt.imshow(albedo_ind)
        #plt.imshow(snow * 1, cmap="Blues_r")
        #plt.contour(elevation_grid, cmap="hot",
        #            levels=list(
        #                range(int(elevation_grid[elevation_grid > 0].min()),
        #                    int(elevation_grid.max()),
        #                    int((elevation_grid.max() -
        #                         elevation_grid[elevation_grid > 0].min()) / 10)
        #                     )))
        #plt.colorbar()
        #plt.contour(elevation_grid, cmap='Greens',
        #            levels=[SLA - r_crit, SLA, SLA + r_crit])
        #plt.title("Final snow mask Orig.")
        #plt.suptitle(str(gdir.name + " - " + gdir.id), fontsize=18)
        #plt.show()
       # plt.savefig(gdir.get_filepath('plt_naegeli'), bbox_inches='tight')


        # Save snow cover map to .nc file:
        snow_xr = xr.open_dataset(gdir.get_filepath('snow_cover'))

        # calculate SLA from snow cover map:
        SLA_new = get_SLA_asmag(gdir, snow)
        if SLA_new is None:
            log.error(' SLA could not be detected with ASMAG: use SLA from Naegeli Method -',
                   ' POTENTIAL BUG! ')
            SLA_new = SLA
        SLA = SLA_new

        # write variables into dataset:
        try:
            snow_xr['snow_map'].loc[dict(model='naegeli_orig', time=cfg.PARAMS['date'][0])] = snow
        except KeyError:
            # Asmag snow mapping did not work, snow map is broken
            return
        # set NaN value:
        snow_xr = snow_xr.where(snow_xr['snow_map']!=0)
        snow_xr['SLA'].loc[dict(model='naegeli_orig', time=cfg.PARAMS['date'][0])] = SLA
        # safe to file
        snow_new = snow_xr

        snow_xr.close()
            # remove old file:
        os.remove(gdir.get_filepath('snow_cover'))
        snow_new.to_netcdf(gdir.get_filepath('snow_cover'), 'w')
        snow_new.close()
        print("SLA 2= ", SLA)
        log.info('SLA 2= {}'.format(SLA))
    except:
        return



@entity_task(log)
def naegeli_improved_snow_mapping(gdir):
    """ Snow Cover Mapping

    Performs snow cover mapping on Sentinel2-Image
    of glacier as described in
    Naegeli, 2019
    "Change detection
    of bare-ice albedo in the Swiss Alps"
    https://www.the-cryosphere.net/13/397/2019/tc-13-397-2019.html

    with some improvements:
    - detection of the critical SLA is performed with an iterative
    method to find step between snow-ice albedo change (data often
    too noisy to take steepest descend as the location for the transition)
    - r_crit for outlier suppresion is implemented dynamically, based on
    r_squared value (goodness of fit) of a step function onto the
    albedo- elevation profile (good fit = smaller r_crit, poor fit =
    large r_crit)

    SLA is calculated with get_SLA_asmag() function.

    Saves snow cover and SLA in naegeli_impr dimension in
    snow_cover.nc netCDF file

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns
    -------
    None

    """
    try:

        # Primary surface type evaluation:

        try:
            snow, ambig, elevation_grid, albedo_ind = primary_surface_type_evaluation(gdir)
        except TypeError:
            # Function Returns None: No image for this glacier available
            log.error('No image for glacier available- Abort naegeli_improved_snow_mapping')
            return

        # Find critical albedo: albedo at location with highest albedo slope
        # (assumed to be snow line altitude)

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
               albedo_crit_it, SLA_it, r_square = max_albedo_slope_iterate(df)
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
            log.error(' No values in ambiguous area')
            r_crit = 400
            # either all now covered or no snow at all
            if snow[snow == 1].size / snow.size > 0.9:  # high snow cover:
                # Set SLA to lowest limit
                log.error('...because snow cover is very high - set SLA low')
                SLA = elevation_grid[elevation_grid > 0].min()
            elif snow[snow == 1].size / snow.size < 0.1:  # low/no snow cover:
                log.error('...because of no/very low snow cover - set SLA to max')
                SLA = elevation_grid.max()

        #plt.subplot(2, 2, 4)
        #plt.imshow(albedo_ind)
        #plt.imshow(snow * 1, cmap="Blues_r")
        #plt.contour(elevation_grid, cmap="hot",
        #            levels=list(
        #               range(int(elevation_grid[elevation_grid > 0].min()),
        #                     int(elevation_grid.max()),
        #                     int((elevation_grid.max() -
        #                          elevation_grid[elevation_grid > 0].min()) / 10)
        #                      )))
        #plt.colorbar()
        #plt.contour(elevation_grid, cmap='Greens',
        #            levels=[SLA - r_crit, SLA, SLA + r_crit])
        #plt.title("Final snow mask")
        #plt.suptitle(str(gdir.name + " - " + gdir.id), fontsize=18)
    #   # plt.show()
        #plt.savefig(gdir.get_filepath('plt_impr_naegeli'), bbox_inches='tight')


      # Save snow cover map to .nc file:
        snow_xr = xr.open_dataset(gdir.get_filepath('snow_cover'))
        # calculate SLA from snow cover map:
        SLA_new = get_SLA_asmag(gdir, snow)
        if SLA_new is None:
            log.error('SLA by ASMAG is None: use SLA from improved mapping by Naegeli')
            SLA_new = SLA

        SLA = SLA_new
        # write variables into dataset:
        try:
            snow_xr['snow_map'].loc[dict(model='naegeli_improv', time=cfg.PARAMS['date'][0])] = snow
        except KeyError:
            # BUG: Broken snow map for very few cases
            return
        # Set NaN value:
        snow_xr = snow_xr.where(snow_xr['snow_map']!=0)
        # Set SLA
        snow_xr['SLA'].loc[dict(model='naegeli_improv', time=cfg.PARAMS['date'][0])] = SLA
        # safe to file
        snow_new = snow_xr
        snow_xr.close()
            # remove old file:
        os.remove(gdir.get_filepath('snow_cover'))
        snow_new.to_netcdf(gdir.get_filepath('snow_cover'), 'w')
        print("SLA 3=", SLA)
        log.info('SLA 3= {}'.format(SLA))
        snow_new.close()
        snow_xr.close()
    except:
        return

def max_albedo_slope_iterate(df):
    """ Finds elevation of SLA_crit and corresponding albedo

    To detect to location of the ice-snow transition, an iterative method
    that searchs for the maximum slope in the Albedo-elevation profile
    only in the proximity of the previous value. For every iteration, the number of
    elveation bins is increased by factor 2/ the height extend of each band
    is decreased by factor 2 until a minimum of 20 Meters height extend.

    Then a step function is fitted onto the elevation-ALbedo profile,
    assuming a step between snow and ice albedo at the transition.
    The goodness-fit (r_squared) value of the model to the data is
    then saved to later implement in the retrieval of a dynamic r_crit value

    Parameters
    ----------
    df: Dataframe  containing the variable dem_amb (elevations of ambiguous range)
    and albedo_amb (albedo values in ambiguous range) of ambiguous area

    Returns
    -------
    alb_max_slope, max_loc: float
            Albedo at maximum albedo slope and location
            of maximum albedo slope
    r_square: float
            r_square value to determine the fit of a step function onto the
            elevation-albedo profile

    """
    try:
        # Smart minimum finding:
        # Iterate over decreasing elevation bands: (2 bands, 4 bands, 8 bands, etc.)
        # Sort into bands over entire range:
        df = df[df.dem_amb > 0]
        try:
            dem_min = int(round(df[df.dem_amb > 0].dem_amb.min()))
        except ValueError:
                # Dataframe contains only NaN values...
            return
        dem_max = int(round(df.dem_amb.max()))
        alb_min = int(round(df[df.albedo_amb > 0].albedo_amb.min()))
        alb_max = int(round(df.albedo_amb.max()))
        delta_h = int(round((dem_max - dem_min) / 2))
        for i in range(0, int(np.log(df.dem_amb.size))):
            try:
                delta_h = int(round((dem_max - dem_min) / (2 ** (i + 1))))
            except ValueError:
                #some weird Bug
                return
            if (delta_h > 20) or (delta_h <=20 and i <3) : 
                # only look at height bands with h > 20 Meters for glaciers bigger
                # than 2* 20 meter
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
                            if num == (len(albedo_avg) - 1):  # nearest neighbor:
                                albedo_avg[num] = albedo_avg[num - 1]
                            # interpolate between neighbours
                            else:
                                albedo_avg[num] = (albedo_avg[num - 1] +
                                                   albedo_avg[num + 1]) / 2
                        else:
                            albedo_avg[num] = albedo_avg[num + 1]

                # Find elevation/location with steepest albedo slope
                # in the proximity of max values from
                # previous iteration:
                if i > 1:
                    if max_loc > 0:
                        try:
                            max_loc_sub = np.argmax((np.gradient
                            (albedo_avg[(2 * max_loc - 2):(2 * max_loc + 2)])))
                            max_loc = 2 * max_loc - 2 + max_loc_sub
                        # new location of maximum albedo slope
                        except ValueError:
                            # Array too small or whatever
                            return
                    else:
                        max_loc = np.argmax((np.gradient(albedo_avg)))
                        # first definition of max. albedo slope
                else:
                    max_loc = np.argmax((np.gradient(albedo_avg)))
                if max_loc < (len(albedo_avg) - 1):
                    # find location between two values, set as final value for albedo
                    # at SLA and SLA
                    alb_max_slope = (albedo_avg[max_loc] + albedo_avg[max_loc + 1]) / 2
                    height_max_slope = (dem_avg[max_loc] + dem_avg[max_loc + 1]) / 2
                else:
                    # if SLA is at highest elevation, pick this valiue
                    alb_max_slope = (albedo_avg[max_loc])
                    height_max_slope = (dem_avg[max_loc])


        # Fitting Step function to Determine fit with R^2:
        # curve fitting: bounds for inital model:
        # bounds:
        # a: step size of heaviside function: 0.1-0.3
        # b: elevation of snow - ice transition: dem_min + (SLA - dem_min)/2
        #                                    to  dem_max - (dem_max -SLA)/2
        # c: average albedo of bare ice: 0.25-0.55
        try:
            r_squared = get_r_squared(step_function_model, dem_avg, albedo_avg,
                                  bounds = ([0.1, dem_min + (height_max_slope - dem_min)/2, 0.3],
                                       [0.3, dem_max - (dem_max -height_max_slope)/2, 0.45]))
        except UnboundLocalError:
            # Glaier smaller than 25 m
            if delta_h == 0:
               delta_h =1
            dem_avg = range(dem_min, dem_max, delta_h)
            albedo_avg = range(dem_min, dem_max, delta_h)
            # Take middle as first guess:
            alb_max_slope = (alb_max - alb_min)/2
            height_max_slope = (dem_max - dem_min)/2
            r_squared = get_r_squared(step_function_model, dem_avg, albedo_avg,
                    bound = ([0.1, dem_min + (height_max_slope- dem_min)/2, 0.3],
                        [0.3, dem_max - (dem_max - height_max_slope)/2, 0.45]))


        return alb_max_slope, height_max_slope, r_squared
    except:
        return

def max_albedo_slope_orig(df):
    """Finds elevation and value of highest
    albedo/elevation slope as the maximum gradient
    of the elevation albedo profile with 20 Meter bins

    Parameters
    ----------
    df: PandasDataFrame
        containing the variable dem_amb (elevations of ambiguous range)
        and albedo_amb (albedo values in ambiguous range)


    Returns
    -------
    alb_max_slope, height_max_slope: Int
            Albedo at maximum albedo slope and altitude
                of maximum albedo slope

    """
    try:
        # Smart minimum finding:
        # Iterate over decreasing elevation bands: (2 bands, 4 bands, 8 bands, etc.)
        # Sort into bands over entire range:
        df = df[df.dem_amb > 0]
        try: 
            dem_min = int(round(df[df.dem_amb > 0].dem_amb.min()))
        except ValueError:
            # df contains only NaNs...
            return
        dem_max = int(round(df.dem_amb.max()))
        alb_min = int(round(df[df.albedo_amb > 0].albedo_amb.min()))
        alb_max = int(round(df.albedo_amb.max()))
        delta_h = 20
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
                    if num == (len(albedo_avg) - 1):  # nearest neighbor:
                        albedo_avg[num] = albedo_avg[num - 1]
                        # interpolate between neighbours
                    else:
                        albedo_avg[num] = (albedo_avg[num - 1] +
                                           albedo_avg[num + 1]) / 2
                else:
                    albedo_avg[num] = albedo_avg[num + 1]

        # Find elevation/location with steepest albedo slope
        try:
            max_loc = np.argmax((np.gradient(albedo_avg)))
        except ValueError:
            log.error('Can not dtermine location of maximum Gradient')
            max_loc = 0
        if max_loc < (len(albedo_avg) - 1):
            # find location between two values, set as final value for albedo
            # at SLA and SLA
            alb_max_slope = (albedo_avg[max_loc] + albedo_avg[max_loc + 1]) / 2
            height_max_slope = (dem_avg[max_loc] + dem_avg[max_loc + 1]) / 2
        else:
            # if SLA is at highest elevation, pick this valiue
            try:
                log.error('SLA is at highest elevation')
                alb_max_slope = albedo_avg[max_loc-1]
                height_max_slope = dem_avg[max_loc-1]
            except IndexError:
                log.error('Glacier smaller than 20 Meters')
                # Glacier smaller than 20 Meters
                alb_max_slope = df.albedo_amb.max()
                height_max_slope = df.dem_amb.max()

        #plt.subplot(2, 2, 3)
        #try:
        #    plt.plot(dem_avg, albedo_avg)
        #except UnboundLocalError:
        #    # Glacier smaller than 25 meters
        #    if delta_h == 0:
        #        delta_h = 1
        #    dem_avg = range(dem_min, dem_max, delta_h)
        #    albedo_avg = range(dem_min, dem_max, delta_h)
        #    # Take middle as a first guess:
        #    alb_max_slope = (alb_max - alb_min) / 2
        #    height_max_slope = (dem_max - dem_min) / 2
        #    plt.plot(dem_avg, albedo_avg)
        #plt.axvline(height_max_slope, color='k', ls='--')
        #plt.xlabel("Altitude in m")
        #plt.ylabel("Albedo")
        #plt.show()

        return alb_max_slope, height_max_slope
    except:
        return

def step_function_model(alti, a, b, c):
    """ Create model for step-function

    Parameters
    ----------
    alti: np.array
        Altitude distribution of glacier
    a:  float
        step size of heaviside function
    b:  float
        elevation of snow-ice transition
    c:  float
        average albedo of bare ice

    Returns
    -------
    model: np.array
        model of step-function
    """
    return (0.5 * (np.sign(alti - b) + 1)) * a + c 
        # Heaviside fitting function

def get_r_squared(step_function_model, dem_avg, albedo_avg, bounds):
    """
    Retrieves r_squared value from fitting step function to
    elevation-albedo profile:


    Parameters
    ----------
    step_function_model: function
            function in snow_mapping.py that creates step function
            with input parameters
    dem_avg: np.array
        height values in  elevation bands on
        glacier with a resolution
        determined by the iteration method
    albedo_avg: np.array
        albedo values in elevation bands
    bounds: tuple of two lists: 
        upper and lower parameter range
                    ([a_min],[b_min], [c_min], [a_max, b_max, c_max])
        with
            a: step size of heaviside function: 0.1-0.3
            b: elevation of snow - ice transition: dem_min + (SLA - dem_min)/2
                                        to  dem_max - (dem_max -SLA)/2
            c: average albedo of bare ice: 0.25-0.55

    Returns
    ------- 
    r_squared: float
        r_squared value (goodness of fit) of fitted step function model
                to elevation-albedo-profile

    """

    try:
        popt, pcov = curve_fit(step_function_model, dem_avg, albedo_avg,
                           bounds=bounds)
        residuals = abs(albedo_avg - step_function_model(dem_avg, popt[0], popt[1], popt[2]))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((albedo_avg - np.mean(albedo_avg)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
    except ValueError:
        print("What's the problem here?")
        log.error('BUG: Unkown problem with r_squared')
        r_squared = -1

    return r_squared


def primary_surface_type_evaluation(gdir):
    """
    Performs Primary surface type evaluation after K. Naegeli 2019:
    - Converts reflectance to broadband-Albedo after Knapp et al.
    - ALbedo > 0.55 is classified as snow, albedo < 0.2 as ice and values in
    between are the ambiguous range where secondary surface type evaluation is
    performed on

    Parameters
    ----------
    gdir: GlacierDirectory
        pyClass

    Returns
    -------
    snow: np.array, binary
        Snow Covered Area after primary surface type evaluation
    ambig: np.array, binary
        Amiguous Area for secondary surface type evaluation:
    elevation_grid: np.array
        Height of DEM on Glacier in m
    albedo_ind: np.array
        Broadband Surface Albedo 

    """
    try:
        try:
            sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))
        except FileNotFoundError:
            # no data for this glacier on 1st date
            log.error('No data for this glacier on 1st date- exit primary surface type evaulation')
            return
        try: 
            test = sentinel.sel(band='B03',time=cfg.PARAMS['date'][0])
        except KeyError:
            # no data for this glacier on current date
            log.error('No data for this glacier on this date - exit primary surface type evaluation')
            return

        if not sentinel.sel(band='B03', time=cfg.PARAMS['date'][0]). \
                img_values.values.any():  # check if all non-zero values in array
            # Cloud cover too high for a good classification
            log.error('All zero values (clouds)- No primary classification possible')
            return

        dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
        elevation_grid = dem_ts.isel(time=0, band=0).height_in_m.values

        # Albedo shortwave to broadband conversion after Knap:
        albedo_k = 0.726 * sentinel.sel(band='B03',
                                        time=cfg.PARAMS['date'][0]).img_values.values / 10000 \
                   + 0.322 * (sentinel.sel(band='B03',
                                           time=cfg.PARAMS['date'][0]).img_values.values / 10000) ** 2 \
                   + 0.015 * sentinel.sel(band='B08',
                                          time=cfg.PARAMS['date'][0]).img_values.values / 10000 \
                   + 0.581 * (sentinel.sel(band='B08',
                                           time=cfg.PARAMS['date'][0]).img_values.values / 10000) ** 2

        # TODO: try with nir band only

        # Limit Albedo to 1
        albedo_k[albedo_k > 1] = 1
        albedo = [albedo_k]

       # plt.figure(figsize=(15, 10))
       # plt.subplot(2, 2, 1)
        
       # sentinel = xr.open_dataset(gdir.get_filepath('sentinel')) 

       # b04 = sentinel.sel(band='B04', time=cfg.PARAMS['date'][0]).img_values.values / 10000
       # b03 = sentinel.sel(band='B03', time=cfg.PARAMS['date'][0]).img_values.values / 10000
       # b02 = sentinel.sel(band='B02', time=cfg.PARAMS['date'][0]).img_values.values / 10000

       # print('B04 = ', b04)

       # rgb_image = np.array([b04, b03, b02]).transpose((1, 2, 0))
       # plt.imshow(albedo_k, cmap='gray')
       # plt.imshow(rgb_image)
       # plt.title("RGB Image")

        # Peform primary suface type evaluation: albedo > 0.55 = snow,
        # albedo < 0.25 = ice, 0.25 < albedo < 0.55 = ambiguous range,
        # Pixel-wise
        for albedo_ind in albedo:
            if albedo_ind.shape != elevation_grid.shape:
                if elevation_grid.shape[0] > albedo_ind.shape[0] or \
                        elevation_grid.shape[1] > albedo_ind.shape[1]:  # Shorten elevation grid
                    elevation_grid = elevation_grid[0:albedo_ind.shape[0], 0:albedo_ind.shape[1]]
                if elevation_grid.shape[0] < albedo_ind.shape[0]:  # Extend elevation grid: append row:
                    try: 
                        elevation_grid = np.append(elevation_grid,
                                               [elevation_grid[
                                                (elevation_grid.shape[0] -
                                                 albedo_ind.shape[0]), :]], axis=0)
                    except IndexError:
                        # BUG: something went wrong with sentinel, only 1 case in 100000 files
                        log.error('BUG- Something wrong with Sentinel-Imagery - Index Error')
                        return
                if elevation_grid.shape[1] < albedo_ind.shape[1]:  # append column
                    b = elevation_grid[:, (elevation_grid.shape[1] -
                                           albedo_ind.shape[1]):elevation_grid.shape[1]]
                    elevation_grid = np.hstack((elevation_grid, b))
                    # Expand grid on boundaries to obtain raster in same shape
            #plt.imshow(elevation_grid)
            #plt.imshow(albedo_ind, cmap = "gray", alpha = 0.5)
            
            #plt.show()

            snow = albedo_ind > 0.55
            ambig = (albedo_ind < 0.55) & (albedo_ind > 0.2)

           # plt.subplot(2, 2, 2)
           # plt.imshow(albedo_ind)
           # plt.imshow(snow * 2 + 1 * ambig, cmap="Blues_r")
           # plt.contour(elevation_grid, cmap="hot",
           #             levels=list(
           #                 range(int(elevation_grid[elevation_grid > 0].min()),
           #                       int(elevation_grid.max()),
           #                       int((elevation_grid.max() -
           #                            elevation_grid[elevation_grid > 0].min()) / 10)
           #                       )))
           # plt.colorbar()
           # plt.title("Snow and Ambig. Area")
           # plt.show()

            sentinel.close()
            dem_ts.close()
            # Find critical albedo: albedo at location with highest albedo slope
            # (assumed to be snow line altitude)

        return snow, ambig, elevation_grid, albedo_ind
    except:
        return

def albedo_knap(sentinel):
    """
    Narrow to broadband albedo conversion after
    Knap, W.H. et al:
    Narrowband to broadband conversion
    of Landsat TM glacier albedos.
    Int. J. Remote Sens. 1999, 20, 2091–2110

    as adapted for Sentinel-2 Images by
    Naegeli, Kathrin et al (2017):
    Cross-comparison of albedo products for glacier surfaces derived from airborne and
    satellite (Sentinel-2 and Landsat 8) optical data. Remote Sensing, 9(2):110

    Parameters
    ----------
    sentinel: GeoPandas DataSet
            all Sentinel Bands for all dates of this scene

    Returns
    -------
    albedo_k: np.array
            Broadband Surfac Albedo

    """
    albedo_k = 0.726 * sentinel.sel(band='B03',
                                    time=cfg.PARAMS['date'][0]).\
                                    img_values.values / 10000 \
               + 0.322 * (sentinel.sel(band='B03',
                                       time=cfg.PARAMS['date'][0]).
                                    img_values.values / 10000) ** 2 \
               + 0.015 * sentinel.sel(band='B08',
                                      time=cfg.PARAMS['date'][0]).\
                                        img_values.values / 10000 \
               + 0.581 * (sentinel.sel(band='B08',
                                       time=cfg.PARAMS['date'][0]).
                                    img_values.values / 10000) ** 2
    return albedo_k
