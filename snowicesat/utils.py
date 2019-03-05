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
from rasterio.warp import calculate_default_transform, reproject, Resampling
import time
import shutil
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.plot import show
import geopandas as gpd
import shapely
import datetime as dt
from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import fiona
import xarray
import sys
from osgeo import gdal
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

def get_sentinelsat_query(glacier):
    """
    Gets query/list of products for sentinelsat
    - Reads outlines into geodata frame
    - Transforms local to UTM 32 grid
    - create bounding box around outline (full outline to complex to be processed by server)
    - files request with sentinelsat package
    :param: glacier: GeoDataFrame of GlacierOutlines
    :return: products:
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

    bbox_filename = 'bbox.geojson'
    # Avoid Fiona bug with overwriting geojson file: https://github.com/Toblerity/Fiona/issues/438
    try:
        os.remove(bbox_filename)
    except OSError:
        pass

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
        # TODO: what if length of products is more than 1?


        if not len(products) is 0: # If products are available, download them:
            product_id = list(products.keys())
            # TODO: if more than 1 tile, it only checks if first tile exists
            for index in product_id:
                safe_name = products[index]['filename']
                print('safe_name', safe_name)
                if not os.path.isdir(os.path.join(cfg.PATHS['working_dir'], safe_name)):
                    #  if not downloaded: downloading all products
                    download_zip = api.download(index, directory_path=cfg.PATHS['working_dir'])
                    print(download_zip)

                    # Unzip files into .safe directory, delete .zip folder
                    with zipfile.ZipFile(download_zip['path']) as zip_file:
                        print(zip_file)
                        zip_file.extractall(cfg.PATHS['working_dir'])
                    os.remove(download_zip['path'])
                else:
                    print("Tile is downloaded already")

    print("Now Merging all downloaded tiles")

    # Merging downloaded tiles: read in all band tiles, merge spatially per band

    # walk all .safe directories to get list of band_01 names
    # list all safe directories:
    safe_dirs = os.listdir(cfg.PATHS['working_dir'])

    #get a list of all bands in all directories:

    img_path = [x[0] for x in os.walk(os.path.join(cfg.PATHS['working_dir'], safe_name))][8]
    cache_name = os.path.join(cfg.PATHS['working_dir'], 'band_01_' + str(cfg.PARAMS['date'][0]) + '.tif')

    file_list = os.listdir(img_path)
    file_list = [img_path + "\\" + s for s in file_list]




    return products, api

def read_safe_to_cache(glacier, filename):
    """Reads Sentinel Data from .SAFE File into Working Directory .tif wit
    If cache already has a file for given date, merge them together
    Params: filename of .SAFE folder
    glacier: Glacier outline DataFrame in local grid

    Returns: sen_cache: netcdf 4 file containing all bands
    """
    print("In read_safe_to_cache")
    # Filename of new tile
    img_path = [x[0] for x in os.walk(os.path.join(cfg.PATHS['working_dir'], filename))][8]
    cache_name = os.path.join(cfg.PATHS['working_dir'], 'cache_'+ str(cfg.PARAMS['date'][0])+ '.nc')

    #TODO: Handle different bands, read in with xarray!
    file_list = os.listdir(img_path)
    file_list =[img_path + "\\" + s for s in file_list]

    # #####################     TEST READ/MERGE XARRAY  #####################
    # b_sub = []
    # for band in file_list:
    #     print(band)
    #     band_array = xarray.open_rasterio(band)
    #     band_array.attrs['pyproj_srs'] = band_array.crs
    #     b_sub.append(band_array.salem.subset(margin=0))
    #
    # all_bands = xr.concat((b_sub[0], b_sub[1], b_sub[2], b_sub[3], b_sub[4], b_sub[5]), dim='band')
    # print('all_bands after concat', all_bands)
    # all_bands['band'] = list(range(1,7))
    # print('all_bands after band',all_bands)
    # all_bands.name = 'img_values'
    # print('all_bands after bands.name',all_bands)
    # all_bands = all_bands.assign_coords(time=cfg.PARAMS['date'][0])
    # print('all_bands after assign_coords', all_bands)
    # all_bands = all_bands.expand_dims('time')
    # print('all_bands after expand_dims',all_bands)
    # ########################################################################




    # #######################################  TEST READ/MERGE RASTERIO
    file_list = os.listdir(img_path)
    file_list =[img_path + "\\" + s for s in file_list]
    # Read metadata of second band: first band has lower resolution
    with rasterio.open(file_list[1]) as src0:
        meta = src0.meta
        # # Update meta to reflect the number of bands
    meta.update(count=len(file_list))
        # # Read each band and write it to cache_name_new
    print("Merging Bands into one file")
    with rasterio.open(cache_name, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            print("Write Band #", id)
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1).astype(rasterio.uint16))
    e = time.time()
    print(e - s)
    ################################################################

    # print("done writing")
    #  # Read new file into DataArray
    # new = rasterio.open(cache_name_new)
    #

    # ##############      Merge Tiles #################
    # # Check if old cache file exists for this date already (with other tiles in it)
    # if os.path.isfile(cache_name):
    #          # If exits, read old tile, concat to old tile
    #     old = rasterio.open(cache_name)
    #     new = [old, new]
    #
    #  # The merge function returns a single array and the affine transform info
    #     new, out_trans = merge(new)
    #     out_meta = new.meta.copy()
    #     out_meta.update({"driver": "GTiff",
    #                           "height": new.shape[1],
    #                           "width": new.shape[2],
    #                           "transform": out_trans})
    #
    #  # Write merged tile to file:
    #     with rasterio.open(cache_name_new, 'w', **out_meta) as dst:
    #        dst.write(new.astype(rasterio.uint8), 1)
    #
    ############################################      END Merge Tiles


    #move cache_name_new to cache
#    shutil.move(cache_name_new, cache_name)

    #remove .SAFE directory
    print(os.path.join(cfg.PATHS['working_dir'], filename))
    shutil.rmtree(os.path.join(cfg.PATHS['working_dir'], filename))
    print('done with reading cache')


def crop_sentinel_to_glacier(glacier, gdir, filename):
    """
    Reads Sentinel Data from .SAFE File in Working Directory
    Crops Sentinel Data to glacier outline
    Reprojects Raster of Glacier to local grid
    Writes local raster of all bands to netcdf file with dimensions 'bands' and 'time'

    Params: filename: filename of .SAFE folder where glacier is located
    glacier: Glacier Outline dataFrame in local grid
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns: sen_dat: netcdf 4 file containing all bands for all times ever added for the glacier
    """

    # TODO: Check if netcdf file already exists in glacier directory, if not, create it:
    # Create cfg.PARAMS or PATHS for sentinel.nc (similar to DEM_ts)
    print("In crop_sentinel_to_glacier")

    img_path = [x[0] for x in os.walk(os.path.join(cfg.PATHS['working_dir'], filename))][8]

    # Reproject to Sentinel-2 (UTM zone 32) Grid:
    local_crs = glacier.crs
    glacier = glacier.to_crs({'init': 'epsg:32632'})
    glacier.to_file(driver='ESRI Shapefile', filename='outline_UTM32.shp')

    # Open outline in UTM zone 32 Grid
    s = time.time()

    with fiona.open('outline_UTM32.shp', "r") as glacier_reprojected:
        # Read local geometry
        features = [feature["geometry"] for feature in glacier_reprojected]

    # iterate over all bands
    b_sub = []
    for band in os.listdir(img_path)[:-2]: # Ignore TCL band and band 08A
        with rasterio.open(os.path.join(img_path, band)) as src:
            #   Open Sentinel file: Apply glacier outline
            out_image, out_transform = rasterio.mask.mask(src, features,
                                                              crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                #  Write cropped image to .tif file --> necessary?
        with rasterio.open('cropped_not_reprojected.tif','w', **out_meta) as src:
            src.write(out_image)
        with rasterio.open('cropped_not_reprojected.tif', 'r', **out_meta) as src:

            # REPROJECT to local grid: we want to project out_image with out_meta to local crs of glacier
                # Calculate Transform
            dst_transform, width, height = calculate_default_transform(
                    src.crs, local_crs, out_image.shape[1], out_image.shape[2], *src.bounds)

            out_meta.update({
                    'crs': local_crs,
                    'transform': dst_transform,
                    'width': width,
                    'height': height
                })
            with rasterio.open('cropped_reprojected_band.tif', 'w', **out_meta) as dst:
                reproject(
                    source=out_image,
                    destination=rasterio.band(dst,1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=local_crs,
                    resampling=Resampling.nearest)
                # Write to geotiff in cache
                dst.write(out_image)

                # Open with xarray into DataArray
            band_array = xarray.open_rasterio('cropped_reprojected_band.tif')
            # TODO: KeyError...
            band_array.attrs['pyproj_srs'] = band_array.crs
            b_sub.append(band_array.salem.subset(margin=0))
                #

    # Merge all subbands to write into netcdf file!
    all_bands = xr.concat(b_sub, dim='band')
    all_bands['band'] = list(range(len(b_sub)))
    all_bands.name = 'img_values'
    #print('all_bands after bands.name',all_bands)
    all_bands = all_bands.assign_coords(time=cfg.PARAMS['date'][0])
    #print('all_bands after assign_coords', all_bands)
    all_bands = all_bands.expand_dims('time')
    #print('all_bands after expand_dims',all_bands)
    all_bands.to_netcdf(gdir.get_filepath('sentinel'), 'w', format='NETCDF4')

    # TODO: something wrong with crs...... not sure what
    # TODO: Deal with different dates! Append new day to existing netcdf

    e = time.time()
    print(e - s)

