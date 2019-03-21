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

#    # TODO: intersect outline.shp with TILE ID, create worksheet --> safe which glacier is on which tile --> safe in PARAMETER: TileID
   # print('Testing get_zones_from_worksheet')
    # similar to get_local_dems and get_zones_from_worksheet
    w_gdf = gpd.read_file(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\data\sentinel_tiles\sentinel2_tiles_switzerland.shp")
#    print(w_gdf['Name'].head())
    w_gdf_2 = gpd.read_file(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\data\DEM\worksheets\worksheet_SWISSALTI3D_2010.shp")
#    print(w_gdf_2['zone'].head())

    gdf = gpd.read_file(gdir.get_filepath('outlines'))
    gdf = gdf.to_crs(w_gdf.crs)
    res_is = gpd.overlay(w_gdf, gdf, how='intersection')

    # zones might be duplicates if the glacier shape is 'winding'
    list_tile_id = np.unique(res_is['Name_1'].tolist())


    # TODO: crampon. This needs rework if it should work also for SGI
    # Also transform the intersects if necessary
    gdf = cfg.PARAMS['intersects_gdf']
    if len(gdf) > 0:
        gdf = gdf.loc[((gdf.RGIId_1 == gdir.id) |
                       (gdf.RGIId_2 == gdir.id))]
        if len(gdf) > 0:
            gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
            if hasattr(gdf.crs, 'srs'):
                # salem uses pyproj
                gdf.crs = gdf.crs.srs
            gdf.to_file(gdir.get_filepath('intersects'))
    else:
        # Sanity check
        if cfg.PARAMS['use_intersects']:
            raise RuntimeError('You seem to have forgotten to set the '
                               'intersects file for this run. OGGM works '
                               'better with such a file. If you know what '
                               'your are doing, set '
                               "cfg.PARAMS['use_intersects'] = False to "
                               "suppress this error.")

    # Open DEM
    source = entity.DEM_SOURCE if hasattr(entity, 'DEM_SOURCE') else None
    if (not os.path.exists(gdir.get_filepath('dem_ts'))) or reset_dems:
        log.info('Assembling local DEMs for {}...'.format(gdir.id))

        # Here: open Swiss Alti DEMs
        utils.get_local_dems(gdir)


    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ulx, uly, dx, dx  # sign change (2nd dx) is done by rasterio.transform
    )

    # Could be extended so that the cfg file takes all Resampling.* methods
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        resampling = Resampling.bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        resampling = Resampling.cubic
    else:
        raise ValueError('{} interpolation not understood'
                         .format(cfg.PARAMS['topo_interp']))

    dem_source_list = [cfg.NAMES['SWISSALTI2010']]
    oggm_dem = False
    homo_dems = []
    homo_dates = []
    for demtype in dem_source_list:
        try:
            data = xr.open_dataset(gdir.get_filepath('dem_ts'), group=demtype)
        except OSError:  # group not found
            print('group {} not found'.format(demtype))
            continue

        # check latitude order (latitude needs to be decreasing as we have
        # to create own transform and dy is automatically negative in rasterio)
        if data.coords['y'][0].item() < data.coords['y'][-1].item():
            data = data.sel(y=slice(None, None, -1))

        for t in data.time:
            dem = data.sel(time=t)

            dem_arr = dem.height.values

            src_transform = rasterio.transform. \
                from_origin(min(dem.coords['x'].values),  # left
                            max(dem.coords['y'].values),  # upper
                            np.abs(
                                dem.coords['x'][1].item() - dem.coords['x'][
                                    0].item()),
                            np.abs(
                                dem.coords['y'][1].item() - dem.coords['y'][
                                    0].item()))

            # Set up profile for writing output
            profile = {'crs': proj4_str,
                       'nodata': dem.height.encoding['_FillValue'],
                       'dtype': str(dem.height.encoding['dtype']),
                       'count': 1,
                       'transform': dst_transform,
                       'interleave': 'band',
                       'driver': 'GTiff',
                       'width': nx,
                       'height': ny,
                       'tiled': False}
            base, ext = os.path.splitext(gdir.get_filepath('dem'))
            dem_reproj = base + str(t.item()) + ext
            with rasterio.open(dem_reproj, 'w', **profile) as dest:
                dst_array = np.empty((ny, nx),
                                     dtype=str(dem.height.encoding['dtype']))
                dst_array[:] = np.nan

                reproject(
                    # Source parameters
                    source=dem_arr,
                    src_crs=dem.pyproj_srs,
                    src_transform=src_transform,
                    # Destination parameters
                    destination=dst_array,
                    dst_transform=dst_transform,
                    dst_crs=proj4_str,
                    dst_nodata=np.nan,
                    # Configuration
                    resampling=resampling)

                dest.write(dst_array, 1)

                homo_dems.append(dst_array)
                homo_dates.append(t.time.values)

    # Stupid, but we need it until we are able to fill the whole galcier grid with valid DEM values/take care of NaNs
    # Open DEM
    source = entity.DEM_SOURCE if hasattr(entity,
                                          'DEM_SOURCE') else None
    dem_list, dem_source = get_topo_file((minlon, maxlon),
                                         (minlat, maxlat),
                                         rgi_region=gdir.rgi_region,
                                         rgi_subregion=gdir.rgi_subregion,
                                         source=source)

    log.debug('(%s) DEM source: %s', gdir.rgi_id, dem_source)

    # A glacier area can cover more than one tile:
    if len(dem_list) == 1:
        dem_dss = [rasterio.open(
            dem_list[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        if LooseVersion(rasterio.__version__) >= LooseVersion(
                '1.0'):
            src_transform = dem_dss[0].transform
        else:
            src_transform = dem_dss[0].affine
    else:
        dem_dss = [rasterio.open(s) for s in
                   dem_list]  # list of rasters
        dem_data, src_transform = merge_tool(
            dem_dss)  # merged rasters

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ulx, uly, dx, dx
        # sign change (2nd dx) is done by rasterio.transform
    )

    # Set up profile for writing output
    profile = dem_dss[0].profile
    profile.update({
        'crs': proj4_str,
        'transform': dst_transform,
        'width': nx,
        'height': ny
    })

    try:
        resampling = Resampling[cfg.PARAMS['topo_interp'].lower()]
    except ValueError:
        raise ValueError(
            '{} interpolation not understood. Must be a '
            'rasterio.Resampling method string supported by '
            'rasterio.warp.reproject).'
            .format(cfg.PARAMS['topo_interp']))
    ## Once there is a SUPPORTED_RESAMPLING constant in rasterio.warp (with 1.0 release)
    # if resampling not in SUPPORTED_RESAMPLING:
    #     raise ValueError()

    dem_reproj = gdir.get_filepath('dem')
    with rasterio.open(dem_reproj, 'w', **profile) as dest:
        dst_array = np.empty((ny, nx), dtype=dem_dss[0].dtypes[0])
        reproject(
            # Source parameters
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            # Destination parameters
            destination=dst_array,
            dst_transform=dst_transform,
            dst_crs=proj4_str,
            # Configuration
            resampling=resampling)

        dest.write(dst_array, 1)

    for dem_ds in dem_dss:
        dem_ds.close()

    oggm_dem = True

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=proj_out, nxny=(nx, ny),  dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))
    gdir.write_pickle(dem_source_list, 'dem_source')

    # write homo dem time series
    homo_dem_ts = xr.Dataset({'height': (['time', 'y', 'x'],
                                     np.array(homo_dems))},
                         coords={
                             'x': np.linspace(dst_transform[2],
                                              dst_transform[2] + nx *
                                              dst_transform[0], nx),
                             'y': np.linspace(dst_transform[5],
                                              dst_transform[5] + ny *
                                              dst_transform[4], ny),
                             'time': homo_dates},
                         attrs={'id': gdir.rgi_id, 'name': gdir.name,
                                'res': dx})
    homo_dem_ts = homo_dem_ts.sortby('time')
    homo_dem_ts.to_netcdf(gdir.get_filepath('homo_dem_ts'))


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
    slope, aspect, hillshade = calc_slope_aspect_hillshade(gdir)


def calc_slope_aspect_hillshade(gdir):
    """
    Reads dem_ts group('alti') to xarray, then
    converts to data_array, calculate slope, aspect and
    hillshade

    :param gdirs: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    :return: slope, aspect: 3-D numpy array
    """

    dem_ts = xr.open_dataset(gdir.get_filepath('dem_ts'),
                             group="alti")
    elevation_grid = dem_ts.isel(time=0).height.values
    # Resample DEM to 10 Meter Resolution:
    dx = dem_ts.isel(time=0).height.attrs['res'][0]
    print(elevation_grid.shape)

    # hillshade requires solar angles:
    solar_angles = xr.open_dataset(gdir.get_filepath('solar_angles'))
    solar_azimuth = solar_angles.sel(time=cfg.PARAMS['date'][0], band='solar_azimuth').solar_angles.values
    solar_zenith = solar_angles.sel(time=cfg.PARAMS['date'][0], band='solar_zenith').solar_angles.values
    print(solar_azimuth.shape, solar_zenith.shape)

    z_bc = assign_bc(elevation_grid)
    # Compute finite differences
    slope_x = (z_bc[1:-1, 2:] - z_bc[1:-1, :-2]) / (2 * dx)
    slope_y = (z_bc[2:, 1:-1] - z_bc[:-2, 1:-1]) / (2 * dx)

    # Magnitude of slope in radians
    slope = np.arctan(np.sqrt(slope_x ** 2 + slope_y ** 2))
    # Apsect ratio in radians
    aspect = np.arctan2(slope_y, slope_x)

    # Convert solar angles from deg to rad:
    azimuth_rad, elevation_rad = (360 - solar_azimuth + 90) * np.pi / 180, (90 - solar_zenith) * np.pi / 180
    hillshade = ((np.cos(elevation_rad) * np.cos(slope)) + ...
            (np.sin(elevation_rad)* np.sin(slope) * np.cos(azimuth_rad - aspect)))

    plt.imshow(hillshade)
    plt.show()



    return slope, aspect, hillshade



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

