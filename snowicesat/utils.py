"""
A collection of some useful miscellaneous functions.
"""

from __future__ import absolute_import, division

from joblib import Memory
import posixpath
import salem
import os
import pandas as pd
import numpy as np
import logging
import paramiko as pm
import xarray as xr
import rasterio
import subprocess
from rasterio.merge import merge as merge_tool
from rasterio.warp import transform as transform_tool
from rasterio.mask import mask as riomask
import geopandas as gpd
import shapely
import datetime as dt
from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import fiona
import geopandas as gpd
from itertools import product
import dask
import sys
import glob
import fnmatch
import netCDF4
from scipy import stats
from salem import lazy_property, read_shapefile
from functools import partial, wraps
from oggm.utils import *
# Locals
import snowicesat.cfg as cfg
from pathlib import Path
from osgeo import ogr


log = logging.getLogger(__name__)


def parse_credentials_file_snowicesat(credfile=None):
    """ Reads .credential file for sentinelhub login, username and password

    Parameters: credfile: path to .credentials file
    Returns: cr: list credentials for different platforms
    """
    if credfile is None:
        credfile = os.path.join(os.path.abspath(os.path.dirname(
            os.path.dirname(__file__))), '.credentials')

    try:
        cr = ConfigObj(credfile, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Credentials file could not be parsed (%s): %s',
                     credfile, e)
        sys.exit()

    return cr

def get_sentinelsat_query(gdir):
    """
    Gets query/list of products for sentinelsat
    - Reads outlines into geodata frame
    - Transforms local to UTM 32 grid
    - create bounding box around outline (full outline to complex to be processed by server)
    - files request with sentinelsat package
    :param: gdir: GlacierDirectory
    :return: products:
    """

    # 1. Read glacier outline in local grid:
    glacier = gpd.read_file(gdir.get_filepath('outlines'))
    # Reproject to Sentinel-2 Grid:
    glacier = glacier.to_crs({'init': 'epsg:4326'})

    # Create bounding box/envelope as polygon, safe to geogjson file
    bbox = glacier.envelope
    bbox_filename = 'bbox.geojson'
    # Avoid Fiona bug with overwriting geojson file: https://github.com/Toblerity/Fiona/issues/438
    try:
        os.remove(bbox_filename)
    except OSError:
        pass
    bbox.to_file(filename=bbox_filename, driver='GeoJSON')

    # 2.  use sentinelsat package to request which tiles intersect with glacier outlines: create an api
    # Read credentials file:
    cr = parse_credentials_file_snowicesat(os.path.join(os.path.abspath(os.path.dirname(
        os.path.dirname(__file__))), 'snowicesat.credentials'))

    api = SentinelAPI(
        cr['sentinel']['user'],
        cr['sentinel']['password'],
        api_url="https://scihub.copernicus.eu/apihub/")

    # Search for products matching query
    products = api.query(area=geojson_to_wkt(read_geojson(bbox_filename)),
                         date=(tuple(str(item) for item in cfg.PARAMS['date'])),
                         platformname="Sentinel-2",
                         producttype="S2MSI1C",
                         cloudcoverpercentage="[{} TO {}]".format(cfg.PARAMS['cloudcover'][0],cfg.PARAMS['cloudcover'][1]))

    # count number of products matching query and their size
    print("Sentinel Tiles found:", api.count(area=geojson_to_wkt(read_geojson(bbox_filename)),
                                             date=(tuple(str(item) for item in cfg.PARAMS['date'])),
                                             platformname="Sentinel-2",
                                             producttype="S2MSI1C",
                                             cloudcoverpercentage="[{} TO {}]".format(cfg.PARAMS['cloudcover'][0],cfg.PARAMS['cloudcover'][1])),
          ", Total size: ", api.get_products_size(products),
          "GB.")
    # TODO: Read into xarray, safe somewhere? Does it make sense to write a new format?
    # TODO: if more than 1 tile: merge tiles! ... probably write a new function for that?


    return products, api

def crop_sentinel_to_glacier(gdir, filename):
    """Reads Sentinel Data from .SAFE File in Working Directory
    Crops Sentinel Data to glacier outline
    Params: img_path: filename of .SAFE folder
    gdir: Glacier Directory

    Returns: sen_dat: netcdf 4 file containing all bands for all times ever added for the glacier
    """

    # TODO: Check if netcdf file already exists in glacier directory, if not, create it:
    # Create cfg.PARAMS or PATHS for sentinel.nc (similar to DEM_ts)



    img_path = [x[0] for x in os.walk(os.path.join(cfg.PATHS['working_dir'], filename))][8]

    # Read glacier outline in local grid
    glacier = gpd.read_file(gdir.get_filepath('outlines'))

    # Reproject to Sentinel-2 (UTM zone 32) Grid:
    glacier = glacier.to_crs({'init': 'epsg:32632'})
    glacier.to_file(driver='ESRI Shapefile', filename='outline_UTM32.shp')

    # Open outline in UTM zone 32 Grid
    # TODO: check if glacier is on more than one tile....problem with filename?
    # TODO:
    with fiona.open('outline_UTM32.shp', "r") as glacier_reprojected:
        # Read local geometry
        features = [feature["geometry"] for feature in glacier_reprojected]

    # iterate over all bands
    for band in os.listdir(img_path):
        with rasterio.open(os.path.join(img_path, band)) as src:
                #   Open Sentinel file: Apply glacier outline
            out_image, out_transform = rasterio.mask.mask(src, features,
                                                              crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})

                #  Write reprojected and cropped image to .tif file
            with rasterio.open('sentinel_cropped.tif', "w", **out_meta) as dest:
                dest.write(out_image)
                #  Better: write to netcdf-file!



