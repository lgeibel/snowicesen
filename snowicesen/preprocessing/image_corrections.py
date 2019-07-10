"""
=================
image_corrections
=================
Entity Tasks for  Ekstrand Terain Correction, 
cloud masking with s2cloudless and removal of dark
(debris covered/shaded) sides of glacier

"""

from __future__ import absolute_import, division

import numpy as np
import logging
import xarray as xr
from crampon import entity_task
from crampon import utils
import snowicesen.cfg as cfg
import snowicesen.utils as utils
from scipy import stats
from s2cloudless import S2PixelCloudDetector
import matplotlib.pyplot as plt
import rasterio
import os
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool


# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def ekstrand_correction(gdir):
    """ Performs Ekstrand Terrain correction of
    scene in Glacier Directory

    Parameters:
    -----------
    gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns:
    --------
    None 

    """
    # Open Satellite image: 
    try: 
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
    except FileNotFoundError:
        # Glacier not located in tile:
        log.error('Glacier not located in tile- aborting Ekstrand-Correction')
        return
    # Get slope, aspect and hillshade of Glacier Scene:
    try:
        slope, aspect, hillshade, solar_azimuth, solar_zenith =\
            calc_slope_aspect_hillshade(gdir)
    except TypeError:
        # Function returns None: Glacier not located in tile
        log.error('Bug in Solar Angle File - Skip Ekstrand Correction')
        return
    
    b_sub = []
    # Loop over all bands:
    for band_id in sentinel['band'].values:
        time_id = cfg.PARAMS['date'][0]
        try:
            band_arr = sentinel.sel(band=band_id, time=time_id).img_values.values
        except KeyError:
            log.info('Glacier not located in tile- Aborting Ekstrand Correction')
            return
        # Prepare data for linear regression after Ekstrand
        # (for equation see Gao et. al, 2015:
        #  "An improved topographic correction model based on Minnaert"
        # y = k*x +b with k being the Ekstrand/Minnaert constant
        # axis values regression:
        x = np.log(hillshade * np.cos(solar_zenith))
        # y values for regression
        try:
            y = np.log(band_arr * np.cos(solar_zenith))
            # Reshape x, y to be vector instead of tensor:
            x_vec = np.reshape(x, x.size)
            y_vec = np.reshape(y, y.size)

            # Remove NaN values, only keep relevant pixels for regression:
            x_vec_rel = x_vec[np.isfinite(y_vec) & np.isfinite(x_vec)]
            y_vec_rel = y_vec[np.isfinite(y_vec) & np.isfinite(x_vec)]

            # Linear regression:
            k_ekstrand, intercept, r_value, p_value, std_err = \
                stats.linregress(x_vec_rel, y_vec_rel)

            # Different equations available - which one is correct?
            # Bippus (also used by Rastner:)
            band_arr_correct_bippus = band_arr * (np.cos(solar_zenith) /
                                                  np.cos(hillshade)) ** \
                                      (k_ekstrand * np.cos(hillshade))

            # Ekstrand (also used by Goa - most likely correct):
            band_arr_correct_ekstrand = band_arr * np.cos(slope) * (
                    np.cos(solar_zenith) / np.cos(hillshade) /
                    np.cos(slope)) ** k_ekstrand
        except ValueError:
            # very few data seem to have wrong solar angle data
            # --> simply using uncorrected values for this for now...
            log.info('Incorrect solar angles - using uncorrected image')
            band_arr_correct_ekstrand = band_arr
            band_arr_correct_bippus = band_arr

        #write corrected values to netcdf: update values
        sentinel['img_values'].loc[(dict(band=band_id,
                                         time=cfg.PARAMS['date'][0]))]\
            = band_arr_correct_ekstrand
        #plot to test
        # plt.figure(1)
        # plt.subplot(131)
        # plt.imshow(band_arr, cmap= 'gray')
        # plt.title("Band")
        # plt.colorbar()
        # plt.subplot(132)
        # plt.imshow(band_arr_correct_ekstrand, cmap= 'gray')
        # plt.colorbar()
        # plt.title("Ekstrand corrected band")
        # plt.subplot(133)
        # plt.imshow(band_arr-band_arr_correct_bippus, cmap='gray')
        # plt.colorbar()
        # plt.title("Difference Band - Ekstrand Correction")
        # plt.show()
    sentinel.to_netcdf(gdir.get_filepath('ekstrand'))
    sentinel.close()

def calc_slope_aspect_hillshade(gdir):
    """ Calculates Slope, Aspect and Hillshade

    Reads dem_ts group('alti') into xarray, then
    converts to a data_array.
    There, slope, aspect and
    hillshade of the scene in the gdir is derived

    Parameters:
    -----------
    gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns:
    --------
    slope, aspect, hillshade, azimuth_rad, zenith_rad: np.arrays
        3-D numpy arrays containing slope, aspect, hillshade
        and solar azimiuth and solar zenith angles in radians
    """

    dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
    elevation_grid = dem_ts.isel(time=0, band=0).height_in_m.values
    dx = dem_ts.attrs['res'][0]

    # hillshade requires solar angles:
    try:
        solar_angles = xr.open_dataset(gdir.get_filepath('solar_angles'))
        solar_azimuth = solar_angles.sel(time=cfg.PARAMS['date'][0], band='solar_azimuth')\
             .angles_in_deg.values
        solar_zenith = solar_angles.sel(time=cfg.PARAMS['date'][0], band='solar_zenith')\
             .angles_in_deg.values
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
    except FileNotFoundError:
        # Solar Angles have problem with Georeferencing
        log.info('Faulty Solar Angle file')
        # Workaround: Use average of SOlar Azimuth and Zenith angle:
        solar_zenith = np.full(elevation_grid.shape, cfg.PARAMS['zenith_mean'])
        solar_azimuth = np.full(elevation_grid.shape, cfg.PARAMS['azimuth_mean'])
    except KeyError:
        # No information for this date & glacier
        log.info('Faulty Solar Angle file')
        solar_zenith = np.full(elevation_grid.shape, cfg.PARAMS['zenith_mean'])
        solar_azimuth = np.full(elevation_grid.shape, cfg.PARAMS['azimuth_mean'])

    while solar_zenith.shape != elevation_grid.shape:
        print(solar_zenith.shape, elevation_grid.shape)
        if elevation_grid.shape[0] > solar_zenith.shape[0] or \
                elevation_grid.shape[1] > solar_zenith.shape[1]: # Shorten elevation grid
            elevation_grid = elevation_grid[0:solar_zenith.shape[0], 0:solar_zenith.shape[1]]
        if elevation_grid.shape[0] < solar_zenith.shape[0]: # Extend elevation grid: append row:
            try:
                elevation_grid = np.append(elevation_grid,
                            [elevation_grid[(elevation_grid.shape[0]-
                                  solar_zenith.shape[0]), :]], axis = 0)
            except IndexError: 
                # Bug: Someone is having a problem here...
                return
        if elevation_grid.shape[1] < solar_zenith.shape[1]: # append column
            b = elevation_grid[:, (elevation_grid.shape[1]-
                solar_zenith.shape[1]):elevation_grid.shape[1]]
            elevation_grid = np.hstack((elevation_grid, b))
                # Expand grid on boundaries to obtain raster in same shape after
        print(solar_zenith.shape, elevation_grid.shape)
    
    # differentiating
    z_bc = utils.assign_bc(elevation_grid)
    
    # Compute finite differences
    slope_x = (z_bc[1:-1, 2:] - z_bc[1:-1, :-2]) / (2 * dx)
    slope_y = (z_bc[2:, 1:-1] - z_bc[:-2, 1:-1]) / (2 * dx)
    # Magnitude of slope
    slope = np.arctan(np.sqrt(slope_x ** 2 + slope_y ** 2))
    # Aspect ratio in radians
    aspect = np.arctan2(slope_y, slope_x)

    # Convert solar angles from deg to rad:
    azimuth_rad = np.radians(solar_azimuth)
    zenith_rad = np.radians(solar_zenith)

    hillshade = (np.cos(zenith_rad) * np.cos(slope)) + \
                (np.sin(zenith_rad) * np.sin(slope) *
                 np.cos(azimuth_rad - aspect))
   # plt.imshow(hillshade)
   # plt.show()

    return slope, aspect, hillshade, azimuth_rad, zenith_rad



@entity_task(log)
def cloud_masking(gdir):
    """
    Masks cloudy pixels with s2cloudless algorithm

    - Reads necessary bands from ekstrand corrected netCDF file
    - Applies s2cloudless algorithm to scene to get cloud mask
    (https://github.com/sentinel-hub/sentinel2-cloud-detector)
    - Applies cloud mask to scene
    - saves processed scene with all bands in cloud_masked.nc file

    Parameters:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns:
    --------
    None

    """

    cloud_detector = S2PixelCloudDetector(threshold=0.6, average_over=1, dilation_size=2)

    try:
        sentinel = xr.open_dataset(gdir.get_filepath('ekstrand'))
    except FileNotFoundError:
        # Check if uncorrected sentinel file exists: 
        try:
            sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
        except FileNotFoundError: # No data exists at all
            # 1st time step: no data for this glacier available
            log.info('No data for this glacier - Abort Cloud Masking') 
            return

    try:
        wms_bands = sentinel.sel(
        band=['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
        time=cfg.PARAMS['date'][0])\
        .img_values.values
    except KeyError:
        # Check if uncorrected Sentinel File exists/has data: 
        try: 
            sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
            wms_bands = sentinel.sel(
            band=['B01', 'B02', 'B04','B05','B08','B8A','B09', 'B10', 'B11', 'B12'],
            time=cfg.PARAMS['date'][0])\
                    .img_values.values
        except KeyError:
            # No data for this glacier & date
            log.info('No data for this glacier and date - Abort Cloud Masking')
            return

    # rearrange dimensions, goal :[height, width, channels].
    #  now: [channels, height, width] and correct into to float (factor 10000)
    wms_bands = [np.transpose(wms_bands/10000, (1,2,0)) for _ in range(1)]
    cloud_masks = cloud_detector.get_cloud_masks(np.array(wms_bands))

#    cloud_probability = cloud_detector.get_cloud_probability_maps(np.array(wms_bands))
#    plot_cloud_mask(cloud_probability ,wms_bands)

    # Apply cloudmask to scene:
    for band_id in sentinel['band'].values:
        band_array = sentinel.sel(band=[band_id],
                time = cfg.PARAMS['date'][0]).img_values.values
        # Set threshold to exclude glaciers with more than 60% cloud cover
        #  -> no useful classification possible
        image = sentinel.sel(band=[band_id],
                time = cfg.PARAMS['date'][0]).img_values.values
        band_array[cloud_masks == 1] = 0
        try:
            cloud_cover = 1 - len(band_array[band_array>0])/len(image[image>0])
        except ZeroDivisionError:
            # for 100 % cloud cover:
            log.info:('Cloud Cover is 100%- abort Cloud Masking')
            return

        if cloud_cover > 0.7:
            # -> set all pixels to 0
            log.info('Cloud COver over 70% - abort Cloud Masking')
            return
        band_array = band_array[0,:,:]
        sentinel['img_values'].loc[(dict(band=band_id, time=cfg.PARAMS['date'][0]))] = band_array

    print("Cloud cover: ", cloud_cover)


    # Write Updated DataSet to file
    sentinel.to_netcdf(gdir.get_filepath('cloud_masked'))
    sentinel.close()

def plot_cloud_mask(mask, bands, figsize=(15, 15), fig=None):
    """
    Utility function for plotting a binary cloud mask.

    """
    if fig == None:
        plt.figure(figsize=figsize)
    plt.imshow(bands[0][0:, :, 8], cmap='gray')
    plt.imshow(mask[0, :, :], cmap=plt.cm.inferno, alpha=0.5)
    plt.colorbar()
    plt.show()

@entity_task(log)
def remove_sides(gdir):
    """ Thresholding to remove dark sides

    Removes dark sides of glaciers that the glacier mask sometimes does
    not consider and that would lead to misclassification. Thresholding
    of is based on NDSI, all values with NDSI < 0 are not considered in
    the further classification

    Processed bands stored in sentinel_temp.nc file

    Parameters:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns:
    --------
    None

    """
    try:
        sentinel = xr.open_dataset(gdir.get_filepath('cloud_masked'))
    except FileNotFoundError:
        log.info('No data for this glacier and date - Abort remove_sides')
        return
    band_array = {}
    for band_id in sentinel['band'].values: # Read all bands as np arrays
        try:
            band_array[band_id] = sentinel.sel(
            band=band_id,
            time=cfg.PARAMS['date'][0]) \
            .img_values.values/10000
        except KeyError:
            # no data for this date & glacier
            log.info('No data for this glacier and date - Abort remove_sides')
            return

    # Calculate NDSI - normalized differencial snow index
    NDSI = (band_array['B03']-band_array['B11'])/\
           (band_array['B03']+band_array['B11'])
    # Apply glacier tresholding as described in Paul et al. 2016
    # for Sentinel Data
    mask = (NDSI > 0.2)
    # Write into netCDF file again
    for band_id in sentinel['band'].values:
        band_array[band_id][mask == False] = 0
        sentinel['img_values'].loc[
            (dict(band=band_id, time=cfg.PARAMS['date'][0]))] \
            = band_array[band_id]*10000

        time_id = cfg.PARAMS['date'][0]


    # Write Updated DataSet to file
    sentinel.to_netcdf(gdir.get_filepath('sentinel_temp'), 'w')
    sentinel.close()
#    shutil.move(gdir.get_filepath('sentinel_temp'), gdir.get_filepath('sentinel'))


