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

from rasterio.merge import merge
from rasterio.plot import show
import geopandas as gpd
import shapely
import datetime as dt
from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import fiona
import xarray
from oggm.utils import *
# Locals
import snowicesat.cfg as cfg
from pathlib import Path
from osgeo import ogr
from datetime import datetime, timedelta
from datetime import date


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

def int_to_datetime(date_str):
    """
    Reads cfg.PARAMS('date') in format 20170219, 20170227, returns datetime
    :param date_str = cfg.PARAMS('date')
    :return: start_date, end_date
    """
    start_date = date(int(str(cfg.PARAMS['date'][0])[0:4]), int(str(cfg.PARAMS['date'][0])[4:6]),
                          int(str(cfg.PARAMS['date'][0])[6:8]))
    end_date = date(int(str(cfg.PARAMS['date'][1])[0:4]), int(str(cfg.PARAMS['date'][1])[4:6]),
                    int(str(cfg.PARAMS['date'][1])[6:8]))

    return start_date, end_date

def datetime_to_int(start_date, end_date):
    """
    Converts datetime to list of two integers in format 20170219, 20170227

    :param start_date:
    :param end_date:
    :return: date_int
    """
    date_int = int(start_date.strftime("%Y%m%d")), int(end_date.strftime("%Y%m%d"))

    return date_int

def get_sentinelsat_query(glacier):
    """
    Gets query/list of products for sentinelsat
    - Reads outlines into geodata frame
    - Transforms local to UTM 32 grid
    - create bounding box around outline (full outline to complex to be processed by server)
    - files request with sentinelsat package
    :param: glacier: GeoDataFrame containing all GlacierOutlines
    """
    # 1.  use sentinelsat package to request which tiles intersect with glacier outlines: create an api
    # Read credentials file:
    cr = parse_credentials_file_snowicesat(os.path.join(os.path.abspath(os.path.dirname(
        os.path.dirname(__file__))), 'snowicesat.credentials'))

    api = SentinelAPI(
        cr['sentinel']['user'],
        cr['sentinel']['password'],
        api_url="https://scihub.copernicus.eu/apihub/")

    # 2. Geodataframe containing all glaciers:
    # Reproject to  WGS 84 Grid for query:
    glacier = glacier.to_crs({'init': 'epsg:4326'})
    # Create bounding box/envelope as polygon, safe to geojson file
    bbox = glacier.envelope

    for index in bbox:
    #TODO: adjust date
    # iterate over each item in the glacier list,
    # Search for products matching query
        products = api.query(area=index, #geojson_to_wkt(read_geojson(bbox_filename))
                         date=(tuple(str(item) for item in cfg.PARAMS['date'])),
                         platformname="Sentinel-2",
                         producttype="S2MSI1C",
                         cloudcoverpercentage="[{} TO {}]".format(cfg.PARAMS['cloudcover'][0],cfg.PARAMS['cloudcover'][1]))

    # count number of products matching query and their size
        print("Sentinel Tiles found:", api.count(area=index,
                                             date=(tuple(str(item) for item in cfg.PARAMS['date'])),
                                             platformname="Sentinel-2",
                                             producttype="S2MSI1C",
                                             cloudcoverpercentage="[{} TO {}]".format(cfg.PARAMS['cloudcover'][0],cfg.PARAMS['cloudcover'][1])),
          ", Total size: ", api.get_products_size(products),
          "GB.")

        if not len(products) is 0: # If products are available, download them:
            product_id = list(products.keys())
            for index in product_id:
                safe_name = products[index]['filename']
                if not os.path.isdir(os.path.join(cfg.PATHS['working_dir'], safe_name)):
                    #  if not downloaded: downloading all products
                    download_zip = api.download(index, directory_path=cfg.PATHS['working_dir'])
                    # Unzip files into .safe directory, delete .zip folder
                    with zipfile.ZipFile(download_zip['path']) as zip_file:
                        print(zip_file)
                        zip_file.extractall(cfg.PATHS['working_dir'])
                    os.remove(download_zip['path'])
                else:
                    print("Tile is downloaded already")

    print("Merging all downloaded tiles")
    # Check if file already exists:

    # Merging downloaded tiles to Mosaic: read in all band tiles, merge spatially per band, write out each tile per band
    # find all file that end with B01.jp2, B02.jp2, etc.
    s = time.time()
    # go Through all bands
    band_list = ["B{:02d}".format(i) for i in range(1, 13)]
    for band in band_list:
        if not os.path.isfile(os.path.join(cfg.PATHS['working_dir'],'cache\mosaic'+str(cfg.PARAMS['date'][0])+band+'.tif')):
            # list of filenames of same band in all .safe tiles
            file_list = [filename for filename in glob.glob(cfg.PATHS['working_dir']+'/**/**/**/**/*'+band+'.jp2', recursive=False)]

            #create empty list for datafile
            src_files_to_mosaic = []
            # Open File, append to list
            for fp in file_list:
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)
            # Merge all tiles together
            mosaic, out_trans = merge(src_files_to_mosaic)
            #show(mosaic, cmap='terrain')
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                            "height": mosaic.shape[1],
                            "width": mosaic.shape[2],
                            "transform": out_trans,
                            "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "})
            with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cache\mosaic'+str(cfg.PARAMS['date'][0])+band+'.tif'), "w", **out_meta) as dest:
                print('Writing mosaic to file...', band)
                dest.write(mosaic)
        e = time.time()
        print(e - s)