from __future__ import absolute_import, division

from distutils.version import LooseVersion
from salem import Grid, wgs84
import os
import numpy as np
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
from oggm.core.gis import gaussian_blur, multi_to_poly,\
    _interp_polygon, _polygon_to_pix, define_glacier_region, glacier_masks
from oggm.utils import get_topo_file
import matplotlib.pyplot as plt
import math

import rasterio
from rasterio.warp import reproject, Resampling
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
from scipy.ndimage.interpolation import map_coordinates


# Module logger
log = logging.getLogger(__name__)

@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_glacier_region_snowicesat(gdir, entity=None, reset_dems=False):
    """
    Define the local grid for a glacier entity.

    Very first task: define the glacier's local grid.
    Defines the local projection (Transverse Mercator), centered on the
    glacier. There is some options to set the resolution of the local grid.
    It can be adapted depending on the size of the glacier with:

        dx (m) = d1 * AREA (km) + d2 ; clipped to dmax

    or be set to a fixed value. See the params file for setting these options.

    After defining the grid, the topography and the outlines of the glacier
    are transformed into the local projection. The default interpolation for
    the topography is `cubic`.
    This function is mainly taken over from OGGM, some modification to handle
    multitemporal data have been made.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        Where to write the data
    entity: :py:class:`geopandas.GeoSeries`
        The glacier geometry to process
    reset_dems: bool
        Whether to reassemble DEMs from sources or not (time-consuming!). If
        DEMs are not yet present, they will be assembled anyway. Default:
        False.

    Returns
    -------
    None
    """
    area = gdir.area_km2
    dx = utils.dx_from_area(area)
    log.debug('(%s) area %.2f km, dx=%.1f', gdir.id, area, dx)

    # Make a local glacier map
    proj_params = dict(name='tmerc', lat_0=0., lon_0=gdir.cenlon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**proj_params)
    proj_in = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
    proj_out = pyproj.Proj(proj4_str, preserve_units=True)
    project = partial(pyproj.transform, proj_in, proj_out)

    # TODO: crampon. Here single outlines should be transformed and their union used
    # for defining the grid
    # transform geometry to map
    geometry = shapely.ops.transform(project, entity['geometry'])
    geometry = multi_to_poly(geometry, gdir=gdir)
    xx, yy = geometry.exterior.xy

    # Corners, incl. a buffer of N pix
    ulx = np.min(xx) - cfg.PARAMS['border']
    lrx = np.max(xx) + cfg.PARAMS['border']
    uly = np.max(yy) + cfg.PARAMS['border']
    lry = np.min(yy) - cfg.PARAMS['border']
    # n pixels
    nx = np.int((lrx - ulx) / dx)
    ny = np.int((uly - lry) / dx)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=proj_out, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)
    # save transformed geometry to disk
    entity = entity.copy()
    entity['geometry'] = geometry
    # Avoid fiona bug: https://github.com/Toblerity/Fiona/issues/365
    for k, s in entity.iteritems():
        if type(s) in [np.int32, np.int64]:
            entity[k] = int(s)
    towrite = gpd.GeoDataFrame(entity).T
    towrite.crs = proj4_str
    # Delete the source before writing
    if 'DEM_SOURCE' in towrite:
        del towrite['DEM_SOURCE']
    towrite.to_file(gdir.get_filepath('outlines'))


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
    slope, aspect, hillshade, solar_azimuth, solar_zenith = calc_slope_aspect_hillshade(gdir)

    # Peform linear regression after Ekstrand:
    # x = ln(hillshade* cos(solar_zenith))

def calc_slope_aspect_hillshade(gdir):
    """
    Reads dem_ts group('alti') to xarray, then
    converts to data_array, calculate slope, aspect and
    hillshade

    :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return: slope, aspect, hillshade, azimuth_rad, zenith_rad:
                3-D numpy arrays of angles in radians

    """

    dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'))
    #print("dem_ts = ", dem_ts)
    elevation_grid = dem_ts.isel(time=0, height_rm=0).height_in_m.values
    # Resample DEM to 10 Meter Resolution:
    dx = dem_ts.attrs['res'][0]

    # hillshade requires solar angles:
    solar_angles = xr.open_dataset(gdir.get_filepath('solar_angles'))
    solar_azimuth = solar_angles.sel(time=cfg.PARAMS['date'][0], angles='solar_azimuth').angles_in_deg.values
    solar_zenith = solar_angles.sel(time=cfg.PARAMS['date'][0], angles='solar_zenith').angles_in_deg.values
    print("solar angles", solar_azimuth.shape, solar_zenith.shape)

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

    print("slope", slope.shape)

    # TODO: is slope in deg or in radians?
    # TODO: double check elevation/Zenith!

    # Convert solar angles from deg to rad:
    azimuth_rad = math.radians(solar_azimuth)
    zenith_rad = math.radians(solar_zenith)
    hillshade = ((np.cos(zenith_rad) * np.cos(slope)) + (np.sin(zenith_rad)* np.sin(slope) * np.cos(azimuth_rad - aspect)))
    plt.imshow(hillshade)
    plt.show()

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

