"""
A collection of some useful miscellaneous functions.
"""

from __future__ import absolute_import, division

from rasterio.merge import merge
import xml.etree.ElementTree as ET
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np
from rasterio.plot import show
import geopandas as gpd
import shapely
import datetime as dt
from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from scipy.ndimage.interpolation import map_coordinates
import fiona
import xarray
from oggm.utils import *
# Locals
import snowicesat.cfg as cfg
from pathlib import Path
from osgeo import ogr
from datetime import datetime, timedelta
from datetime import date
from math import sqrt, cos, sin, tan, pi, asin, acos, atan, atan2, isnan
import math
import sys
import re
import os
import struct
import xml.etree.ElementTree as ET
import numpy as np
from osgeo import gdal

from pathlib import Path

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

def download_all_tiles(glacier, clear_cache = False, clear_safe = False):
    """
    Function to download all available Sentinel-2 Tiles
    to cache for a given set of Glaciers (stored in "glacier") for
    two consecutive days. Extracts the .SAFE directories
    into 12 geoTiff mosaics of the entire scene for each band
    (stored in working_dir/cache/date/mosaic/B01.tif)

    Options: deleting .SAFE files after extracting data,
    deleting mosaics from previous time steps

    Structure
    - Reads outlines into GeoDataFrame
    - Transforms local to WGS 84 grid
    - create bounding box around outline
        (full outline is too complex to be processed by server)
    - files request with sentinelsat package
    - downloads new data for given date if available
    - unzip folder into .SAFE directory
    - reads all tiles bandwise, merges them, writes into working_dir/cache as GeoTiff

    WARNING: downloading and merging is very time-consuming, so be careful what you delete.
     Tiles can be very big,
    so storing too many tiles risks in running out of memory

    Parameters:
    ------------
    glacier: GeoDataFrame containing all GlacierOutlines
    clear_cache: boolean: clearing merged GeoTiff from previous step before starting new download, default: False
    clear_safe:boolean:  deleting .SAFE directories after reading/merging tiles into Geotiffs, default: False

    Returns:
    -------------
         tiles_downloaded: how many tiles were downloaded for this date

    """
    # 1.  Use sentinelsat package to request which tiles intersect with glacier outlines: create an api
    # Read credentials file:
    cr = parse_credentials_file_snowicesat(os.path.join(os.path.abspath(os.path.dirname(
        os.path.dirname(__file__))), 'snowicesat.credentials'))

    api = SentinelAPI(
        cr['sentinel']['user'],
        cr['sentinel']['password'],
        api_url="https://scihub.copernicus.eu/apihub/")

    # Create cache directory for this date:
    if not os.path.exists(os.path.join(cfg.PATHS['working_dir'], 'cache', str(cfg.PARAMS['date'][0]))):
        print('creating new folder for this date')
        os.makedirs(os.path.join(cfg.PATHS['working_dir'], 'cache', str(cfg.PARAMS['date'][0])))
        os.makedirs(os.path.join(cfg.PATHS['working_dir'], 'cache', str(cfg.PARAMS['date'][0]), 'mosaic'))

    # 2. Geodataframe containing all glaciers:
    # Reproject to  WGS 84 Grid for query (requested by sentinelsat module):
    glacier = glacier.to_crs({'init': 'epsg:4326'})
    # Create bounding box/envelope as polygon, safe to geojson file
    bbox = glacier.envelope

    for index in bbox:
    # Iterate over each item in the glacier list,
    # Search for products matching query
        products = api.query(area=index, #geojson_to_wkt(read_geojson(bbox_filename))
                         date=(tuple(str(item) for item in cfg.PARAMS['date'])),
                         platformname="Sentinel-2",
                         producttype="S2MSI1C",
                         cloudcoverpercentage="[{} TO {}]".format(cfg.PARAMS['cloudcover'][0],
                                                                  cfg.PARAMS['cloudcover'][1]))

    # Count number of products matching query and their size
        print("Sentinel Tiles found:", api.count(area=index,
                                             date=(tuple(str(item) for item in cfg.PARAMS['date'])),
                                             platformname="Sentinel-2",
                                             producttype="S2MSI1C",
                                             cloudcoverpercentage="[{} TO {}]".
                                                 format(cfg.PARAMS['cloudcover'][0],
                                                 cfg.PARAMS['cloudcover'][1])),
          ", Total size: ", api.get_products_size(products),
          "GB.")
        tiles_downloaded = 0
        if not len(products) is 0: # If products are available, download them:
            product_id = list(products.keys())
            tiles_downloaded += 1
            print('Downloaded Tiles: ',tiles_downloaded)
            for index in product_id:
                safe_name = products[index]['filename']
                print(safe_name)
                if not os.path.isdir(os.path.join(cfg.PATHS['working_dir'],'cache',
                                                  str(cfg.PARAMS['date'][0]), safe_name)):
                    #  If not downloaded: downloading all products
                    print("exists:",os.path.join(cfg.PATHS['working_dir'], 'cache',
                                                 str(cfg.PARAMS['date'][0]),safe_name))
                    download_zip = api.download(index,
                                                directory_path=os.path.join(
                                                    cfg.PATHS['working_dir'],
                                                    'cache',str(cfg.PARAMS['date'][0])))
                    # Unzip files into .safe directory, delete .zip folder
                    with zipfile.ZipFile(os.path.join(cfg.PATHS['working_dir'],'cache',
                                                      str(cfg.PARAMS['date'][0]),
                                                      download_zip['path'])) \
                            as zip_file:
                        zip_file.extractall(os.path.join(cfg.PATHS['working_dir'],
                                                         'cache', str(cfg.PARAMS['date'][0])))
                    os.remove(os.path.join(cfg.PATHS['working_dir'],'cache',
                                           str(cfg.PARAMS['date'][0]), download_zip['path']))
                else:
                    print("Tile is downloaded already")

    # Check if file already exists:

    # Merging downloaded tiles to Mosaic: read in all band tiles,
    # Merge spatially per band, write out each tile per band
    # Find all files that end with B01.jp2, B02.jp2, etc.

    # Iterate through all bands
    band_list = ["B{:02d}".format(i) for i in range(1, 13)]
    band_list = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
         'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
    if tiles_downloaded > 0:
        if clear_cache:
            print("Removing old merged tiles from cache")
            tif_list = os.listdir(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                               str(cfg.PARAMS['date'][0]), 'mosaic'))
            for f in tif_list:
                os.remove(f)
        for band in band_list:
            if not os.path.isfile(os.path.join(cfg.PATHS['working_dir'],'cache',
                                               str(cfg.PARAMS['date'][0]),'mosaic',
                                               str(band+'.tif'))):
                if not os.path.exists(os.path.join(cfg.PATHS['working_dir'],'cache',
                        str(cfg.PARAMS['date'][0]))):
                    os.makedirs(os.path.join(cfg.PATHS['working_dir'],'cache',
                                             str(cfg.PARAMS['date'][0])))

                # List of filenames of same band of this date in all .safe tiles
                file_list = [filename for filename in
                             glob.glob(os.path.join(
                                 cfg.PATHS['working_dir'],
                                 'cache',str(cfg.PARAMS['date'][0]),
                                  '**','**','**','**',
                                  str('*'+band+'.jp2')),
                                  recursive=False)]

                # Create empty list for datafile
                src_files_to_mosaic = []
                # Open File, append to list
                for fp in file_list:
                    src = rasterio.open(fp)
                    src_files_to_mosaic.append(src)
                # Merge all tiles together
                mosaic, out_trans = merge(src_files_to_mosaic)
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                                "height": mosaic.shape[1],
                                "width": mosaic.shape[2],
                                "transform": out_trans})

                #----- Reproject to 10 Meter resolution:
                bands_60m = ['B01.tif', 'B09.tif', 'B10.tif']
                bands_20m = ['B05.tif', 'B06.tif', 'B07.tif', 'B11.tif', 'B12.tif', 'B8A.tif']
                print("Current band is ",band)
                if band+".tif" in bands_60m or band+".tif" in bands_20m:
                    if band+".tif" in bands_60m:
                        res_factor = 6
                    elif band+".tif" in bands_20m:
                        res_factor = 2
                    arr = mosaic
                    newarr = np.empty(shape=(arr.shape[0],  # same number of bands
                                             (arr.shape[1] * res_factor),
                                             (arr.shape[2] * res_factor)), dtype='uint16')
                    print(band, arr.shape)
                    print(band, newarr.shape)

                    # adjust the new affine transform to the smaller cell size
                    old_transform = out_trans
                    new_transform = Affine(old_transform.a / res_factor, old_transform.b,
                                           old_transform.c, old_transform.d,
                                           old_transform.e / res_factor, old_transform.f)
                    out_meta['transform'] = new_transform
                    out_meta['width'] = out_meta['width'] * res_factor
                    out_meta['height'] = out_meta['height'] * res_factor
                #--------- End reproject

                # Now Write to file: B01. tif, B02.tif, etc.

                    with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cache',
                                                    str(cfg.PARAMS['date'][0]),'mosaic',
                                                    str(band+'.tif')), "w", **out_meta) \
                            as dest:
                        reproject(
                            source=arr, destination=newarr,
                            src_transform=old_transform,
                            dst_transform=new_transform,
                            src_crs=dest.crs,
                            dst_crs=dest.crs,
                            resampling=Resampling.nearest)
                        dest.write(mosaic)
                else:
                    with rasterio.open(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                                str(cfg.PARAMS['date'][0]), 'mosaic',
                                                str(band + '.tif')), "w", **out_meta) \
                            as dest:
                        dest.write(mosaic)

        # ------ Metadata -----
        if not os.path.exists(os.path.join(cfg.PATHS['working_dir'],
                                           'cache', str(cfg.PARAMS['date'][0]),
                                           'meta', "solar_zenith_angles.tif")):
            # Extract Metadata for each Tile
            # list of all metadata files for this date
            meta_list = glob.glob(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                               str(cfg.PARAMS['date'][0]), '**',
                                               'GRANULE', '**', 'MTD_TL.xml'),
                                  recursive=False)
            id = 0
            for meta_name in meta_list:
                # Read eaach tile from .xml to GeoTIff and reproject to
            #  10 Meter resolution
                solar_zenith, solar_azimuth = extract_metadata(meta_name)
                resample_meta(solar_zenith, solar_azimuth, id)
                id = id+1

            # Create empty list for datafile
            solar_zenith_to_mosaic = []
            solar_azimuth_to_mosaic =[]

            # List all geotiffs containing solar zenith angles
            zenith_list = glob.glob(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                               str(cfg.PARAMS['date'][0]), 'meta',
                                                '*zenith.tif'),
                                    recursive=False)
            for fp in zenith_list:
                # Open File, append to list
                 with rasterio.open(fp) as src:
                    solar_zenith_to_mosaic.append(src)
                    # Merge all tiles of zenith angles together
                    zenith_mosaic, out_trans = merge(solar_zenith_to_mosaic)

            # Update metadata for merges mosaic
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                  "height": zenith_mosaic.shape[1],
                                  "width": zenith_mosaic.shape[2],
                                  "transform": out_trans})

            # Open new file to write merged mosaic
            with rasterio.open(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                             str(cfg.PARAMS['date'][0]), 'meta',
                                             "solar_zenith_angles.tif"), "w", **out_meta) \
                 as dest:
                dest.write(zenith_mosaic)


            # List all GeoTiffs conatining solar azimuth angles
            azimuth_list = glob.glob(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                               str(cfg.PARAMS['date'][0]), 'meta',
                                                '*azimuth.tif'),
                                    recursive=False)
            for fp in azimuth_list:
                # Open File, append to list
                with rasterio.open(fp) as src:
                    solar_azimuth_to_mosaic.append(src)
                    # Merge all tiles with solar azimuth angles together
                    azimuth_mosaic, out_trans = merge(solar_azimuth_to_mosaic)

            # Open file to write merged mosaic of solar azimuth angles
            with rasterio.open(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                             str(cfg.PARAMS['date'][0]), 'meta',
                                             "solar_azimuth_angles.tif"), "w", **out_meta) \
                 as dest:
                dest.write(azimuth_mosaic)

            for fp in zenith_list:
                os.remove(fp)

            for fp in azimuth_list:
                os.remove(fp)


            # -------  end of Metadata

        if clear_safe:
            print("Deleting downloaded .SAFE directories...")
            safe_list = [filename for filename in glob.glob(os.path.join(cfg.PATHS['working_dir'],
                                                            'cache','**','*.SAFE'),
                                                            recursive=False)]
            for f in safe_list:
                shutil.rmtree(f)



    return tiles_downloaded

def extract_metadata(XML_File):
    """
    Extracts solar zenith and azimuth angle from GRANULE/MTD_TL.xml metadata file of .safe directory
    - main structure after s2a_angle_bands_mod.py module by
    Sentinel-2 MultiSpectral Instrument (MSI) data processing for aquatic science applications:
     Demonstrations and validations
    N.Pahlevan et al., Appendix B:


    :param XML_file: Metadata located in GRANULE/**/MDT_TL.xml file:
    :return: zcoord: solar zenith angle in array of 5x5 km resolution
    acoord: solar azimuth angle in array of 10x10m resolution
    """
    # Parse the XML file
    print("Extracting Metadata now")
    tree = ET.parse(XML_File)
    root = tree.getroot()

    # Find the angles
    for child in root:
        if child.tag[-12:] == 'General_Info':
            geninfo = child
        if child.tag[-14:] == 'Geometric_Info':
            geoinfo = child

    # Find the angles
    for segment in geoinfo:
        if segment.tag == 'Tile_Geocoding':
            frame = segment
        if segment.tag == 'Tile_Angles':
            angles = segment

    for angle in angles:
        if angle.tag == 'Sun_Angles_Grid':
            for bset in angle:
                if bset.tag == 'Zenith':
                    zenith = bset
                if bset.tag == 'Azimuth':
                    azimuth = bset
            for field in zenith:
                if field.tag == 'Values_List':
                    zvallist = field
            for field in azimuth:
                if field.tag == 'Values_List':
                    avallist = field
            zcoord =[]
            acoord= []

            for rindex in range(len(zvallist)):
                zvalrow = zvallist[rindex]
                avalrow = avallist[rindex]
                zvalues=[float(i) for i in zvalrow.text.split(' ')]
                avalues=[float(i) for i in avalrow.text.split(' ')]
                zcoord.append(zvalues)
                acoord.append(avalues)

            zcoord = np.asarray(zcoord)
            acoord = np.asarray(acoord)

    return zcoord, acoord



def resample_meta(solar_zenith, solar_azimuth, index):
    """
    Resamples Solar Zenith and solar azimuth angle from 5x5 km
    to 10 m Grid (nearest neighbor) and writes into GeoTIFF

    :params:
    ------------
    solar_zenith, solar_azimuth:
    np.array float64 containing solar zenith and azimuth
    angles of tile in 5 km x 5 km grid


    Returns:
    ------------
    None
    """

    # Open 10 m resolution tile to read size and dimension of final tile:
    with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cache',
                                                    str(cfg.PARAMS['date'][0]),'mosaic',
                                                    str('B02.tif'))) as src:
        out_meta = src.meta.copy()
        newarr = src.read()
        newarr = np.squeeze(newarr)
        newarr = newarr.astype(float)

        new_transform = src.transform
        old_transform = Affine(new_transform.a * 500, new_transform.b,
                               new_transform.c, new_transform.d,
                               new_transform.e * 500, new_transform.f)
        out_meta['dtype'] = 'float64'

    angles = [solar_zenith, solar_azimuth]
    angles_str = ["solar_zenith.tif", "solar_azimuth.tif"]
    id = 0
    if not os.path.exists(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                       str(cfg.PARAMS['date'][0]), 'meta')):
        os.makedirs(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                 str(cfg.PARAMS['date'][0]), 'meta'))

    for angle in angles:
        with rasterio.open(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                         str(cfg.PARAMS['date'][0]), 'meta',
                                         str(index)+angles_str[id]), "w", **out_meta) \
             as dest:
            reproject(
                     source=angle, destination=newarr,
                     src_transform=old_transform,
                     dst_transform=src.transform,
                     src_crs=src.crs,
                     dst_crs=src.crs,
                     resampling=Resampling.nearest)
            dest.write(newarr, indexes=1)
        id = id + 1

