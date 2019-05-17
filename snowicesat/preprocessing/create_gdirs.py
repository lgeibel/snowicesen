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
from scipy import stats
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

