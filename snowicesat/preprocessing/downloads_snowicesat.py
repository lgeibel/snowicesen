from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import zipfile
import geopandas as gpd
import rasterio
import fiona
from shapely.geometry import shape, LineString, box
from shapely.geometry import Polygon
from salem import Grid, wgs84
import os
import numpy as np
import pyproj
import logging
import xarray as xr
from crampon import entity_task
import snowicesat.utils as utils
import snowicesat.cfg as cfg
import snowicesat.utils as utils


log = logging.getLogger(__name__)

@entity_task(log)
def download_sentinel(gdir):
    """Creates Download request for Sentinel-2 images for a given date for an
    entry in the glacier directory,
    reproject to local grid, cut to glacier extent, reproject to local grid write all bands
    into the GlacierDirectory folder as netcdf file

    Parameters
    ----------
    gdirs: Glacier Dirctory
    Returns
    -------
    sentinel_path: (list with path to the sentinel netcdf file)
    """
    glacier = gpd.read_file(gdir.get_filepath('outlines'))

    products, api = utils.get_sentinelsat_query(glacier)
    # Check if data tile for given date has already been downloaded:
    # TODO: What do we return if we have more than one tile?
    safe_name = next(iter(products.values()))['filename']

    if not os.path.isdir(os.path.join(cfg.PATHS['working_dir'], safe_name)):
       #  if not downloaded: downloading all products
        download_zip = api.download_all(products, directory_path=cfg.PATHS['working_dir'])

        # Unzip files into .safe directory, delete .zip folder
        for key in download_zip[0].keys():
            with zipfile.ZipFile(download_zip[0][key]['path']) as zip_file:
                print(zip_file)
                zip_file.extractall(cfg.PATHS['working_dir'])
        os.remove(download_zip[0][key]['path'])

        # TODO: Extract from SAFE to

    else:
        print("Tile is downloaded already")

    # Read glacier outline in local grid
    glacier = gpd.read_file(gdir.get_filepath('outlines'))
    # 3. read bands from .SAFE directory
    # TODO: move into if-array so its only executed when new tile is downloaded
    #utils.read_safe_to_cache(glacier, safe_name)

    # Create netcdf file where I read entire tiles?
    utils.crop_sentinel_to_glacier(glacier, gdir, safe_name)



