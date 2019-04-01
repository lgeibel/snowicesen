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
from functools import partial
import geopandas as gpd
import shapely
import salem
from scipy import stats
import shutil
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
from oggm.core.gis import gaussian_blur, multi_to_poly,\
    _interp_polygon, _polygon_to_pix, define_glacier_region, glacier_masks
from oggm.utils import get_topo_file
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
def ekstrand_correction(gdir):
    """
    Performs Ekstrand Terrain correction of
    scene in Glacier Directory
       :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return:
    """
    # Get slope, aspect and hillshade of Glacier Scene:
    slope, aspect, hillshade, solar_azimuth, solar_zenith =\
        calc_slope_aspect_hillshade(gdir)

    # Open satellite image:
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
    b_sub = []
    # Loop over all bands:
    for band_id in sentinel['band'].values:
        time_id = cfg.PARAMS['date'][0]
        band_arr = sentinel.sel(band=band_id, time=time_id).img_values.values

        # Prepare data for linear regression after Ekstrand
        # (for equation see Gao et. al, 2015:
        #  "An improved topographic correction model based on Minnaert"
        # y = k*x +b with k being the Ekstrand/Minnaert constant
        # axis values regression:
        x = np.log(hillshade * np.cos(solar_zenith))
        # y values for regression
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

        # TODO: check which version is correct
        # Different equations available - which one is correct?
        # Bippus (also used by Rastner:)
        band_arr_correct_bippus = band_arr*(np.cos(solar_zenith)/
                                          np.cos(hillshade))**\
                                 (k_ekstrand*np.cos(hillshade))

        # Ekstrand (also used by Goa - most likely correct):
        band_arr_correct_ekstrand = band_arr * np.cos(slope) * (
                    np.cos(solar_zenith) / np.cos(hillshade) /
                    np.cos(slope)) ** k_ekstrand

        #write corrected values to netcdf: update values
        sentinel['img_values'].loc[(dict(band=band_id))] = band_arr_correct_ekstrand

     # Write Updated DataSet to file

    sentinel.to_netcdf(gdir.get_filepath('ekstrand'))
    shutil.move(gdir.get_filepath('ekstrand'), gdir.get_filepath('sentinel'))

        # #plot to test
        # plt.figure(1)
        # plt.subplot(121)
        # plt.imshow(band_arr, cmap= 'gray')
        # plt.title("Band")
        # plt.colorbar()
        #
        # plt.subplot(122)
        # plt.imshow(band_arr_correct_ekstrand, cmap= 'gray')
        # plt.colorbar()
        # plt.title("Ekstrand corrected band")
        #
        # plt.show()

        # # TODO: sensitivity analysis for scene with k between 0 and 1
        # k_range = list(np.arange(0.01,1,0.01))
        # band_arr_correct = np.zeros((len(k_range), band_arr.shape[0], band_arr.shape[1]))
        # print(k_range)
        # for k in k_range:
        #     print(k)
        #     band_arr_correct[k*100, :, :] = band_arr * np.cos(slope) * np.float_power(
        #         np.cos(solar_zenith) / np.cos(hillshade) / np.cos(slope), k)

def calc_slope_aspect_hillshade(gdir):
    """
    Reads dem_ts group('alti') to xarray, then
    converts to data_array, calculate slope, aspect and
    hillshade

    :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return: slope, aspect, hillshade, azimuth_rad, zenith_rad:
                3-D numpy arrays, angles in radians
    """

    dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
    elevation_grid = dem_ts.isel(time=0, band=0).height_in_m.values
    # Resample DEM to 10 Meter Resolution:
    dx = dem_ts.attrs['res'][0]

    # hillshade requires solar angles:
    solar_angles = xr.open_dataset(gdir.get_filepath('solar_angles'))
    solar_azimuth = solar_angles.sel(time=cfg.PARAMS['date'][0], band='solar_azimuth')\
        .angles_in_deg.values
    solar_zenith = solar_angles.sel(time=cfg.PARAMS['date'][0], band='solar_zenith')\
        .angles_in_deg.values

    if solar_zenith.shape != elevation_grid.shape:
        print("Solar Angles and DEM grid have different size. "
              "Pixel difference:", solar_zenith.shape[0]-elevation_grid.shape[0],", ",
              solar_zenith.shape[1]-elevation_grid.shape[1],
              "Will be adapted to same grid size")
        elevation_grid = elevation_grid[0:solar_zenith.shape[0], 0:solar_zenith.shape[1]]

    # Expand grid on boundaries to obtain raster in same shape after
    # differentiating
    z_bc = assign_bc(elevation_grid)
    # Compute finite differences
    slope_x = (z_bc[1:-1, 2:] - z_bc[1:-1, :-2]) / (2 * dx)
    slope_y = (z_bc[2:, 1:-1] - z_bc[:-2, 1:-1]) / (2 * dx)
    # Magnitude of slope in radians
    slope = np.arctan(np.sqrt(slope_x ** 2 + slope_y ** 2))
    # Aspect ratio in radians
    aspect = np.arctan2(slope_y, slope_x)

    # Convert solar angles from deg to rad:
    azimuth_rad = np.radians(solar_azimuth)
    zenith_rad = np.radians(solar_zenith)

    hillshade = (np.cos(zenith_rad) * np.cos(slope)) + \
                (np.sin(zenith_rad)* np.sin(slope) *
                 np.cos(azimuth_rad - aspect))
    #plt.imshow(hillshade)
    #plt.show()

    return slope, aspect, hillshade, azimuth_rad, zenith_rad



def assign_bc(elev_grid):

    """ Pads the boundaries of a grid
     Boundary condition pads the boundaries with equivalent values
     to the data margins, e.g. x[-1,1] = x[1,1]
     This creates a grid 2 rows and 2 columns larger than the input
    """

    ny, nx = elev_grid.shape  # Size of array
    z_bc = np.zeros((ny + 2, nx + 2))  # Create boundary condition array
    z_bc[1:-1,1:-1] = elev_grid  # Insert old grid in center

    #Assign boundary conditions - sides
    z_bc[0, 1:-1] = elev_grid[0, :]
    z_bc[-1, 1:-1] = elev_grid[-1, :]
    z_bc[1:-1, 0] = elev_grid[:, 0]
    z_bc[1:-1, -1] = elev_grid[:,-1]

    #Assign boundary conditions - corners
    z_bc[0, 0] = elev_grid[0, 0]
    z_bc[0, -1] = elev_grid[0, -1]
    z_bc[-1, 0] = elev_grid[-1, 0]
    z_bc[-1, -1] = elev_grid[-1, 0]

    return z_bc

@entity_task(log)
def cloud_masking(gdir):
    """
    Masks cloudy pixels with s2cloudless algorithm

    :param gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return:
    """

    cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=3, dilation_size=2)
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
    wms_bands = sentinel.sel(
        band=['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
        time=cfg.PARAMS['date'][0])\
        .img_values.values
    # rearrange dimensions, goal :[height, width, channels].
    #  now: [channels, height, width] and correct into to float (factor 10000)
    wms_bands = [np.transpose(wms_bands/10000, (1,2,0)) for _ in range(1)]
    cloud_masks = cloud_detector.get_cloud_masks(np.array(wms_bands))

    # Apply cloudmask to scene:
    for band_id in sentinel['band'].values:
        band_array = sentinel.sel(band=[band_id],
        time = cfg.PARAMS['date'][0]).img_values.values
        band_array[cloud_masks == 1] = 0
        sentinel['img_values'].loc[(dict(band=band_id))] = band_array

    # Write Updated DataSet to file
    sentinel.to_netcdf(gdir.get_filepath('cloud_masked'))
    shutil.move(gdir.get_filepath('cloud_masked'), gdir.get_filepath('sentinel'))

def plot_cloud_mask(mask, bands, prob_map, figsize=(15, 15), fig=None):
    """
    Utility function for plotting a binary cloud mask.
    """
    if fig == None:
        plt.figure(figsize=figsize)
    plt.imshow(bands[:, : , 8], cmap='gray')
    plt.imshow(mask, cmap='gray', alpha=0.9)
#    plt.imshow(prob_map, cmap=plt.cm.inferno, alpha=0.9)
    plt.show()

@entity_task(log)
def remove_sides(gdir):
    """
    Removes dark sides of glaciers that the glacier mask sometimes does
    not consider and that would lead to misclassification. Thresholding
    suggested by Paul et al. , 2016: " Glacier remote sensing using sentinel-2.
    part II: Mapping glacier extents and
    surface facies and comparison to landsat 8"
    :param gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return:
    """
    sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
    for band_id in sentinel['band'].values:
        band_array = sentinel.sel(
            band=band_id,
            time=cfg.PARAMS['date'][0]) \
            .img_values.values

