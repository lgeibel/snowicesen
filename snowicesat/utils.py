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
import xml.etree.ElementTree as ET
from rasterio.warp import calculate_default_transform, reproject, Resampling
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
                                                      str(cfg.PARAMS['date'][0]), download_zip['path'])) \
                            as zip_file:
                        zip_file.extractall(os.path.join(cfg.PATHS['working_dir'],
                                                         'cache', str(cfg.PARAMS['date'][0])))
                    os.remove(os.path.join(cfg.PATHS['working_dir'],'cache',
                                           str(cfg.PARAMS['date'][0]), download_zip['path']))
                else:
                    print("Tile is downloaded already")

    print("Merging all downloaded tiles")
    # Check if file already exists:

    # Merging downloaded tiles to Mosaic: read in all band tiles,
    # Merge spatially per band, write out each tile per band
    # Find all files that end with B01.jp2, B02.jp2, etc.

    # Iterate through all bands
    band_list = ["B{:02d}".format(i) for i in range(1, 13)]
    if tiles_downloaded > 0:
        if clear_cache:
            print("Removing old merged tiles from cache")
            #TODO: something is weird here with the date...which cache gets removed?
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
                file_list = [filename for filename in glob.glob(os.path.join(
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
                # Show(mosaic, cmap='terrain')
                out_meta = src.meta.copy()
                out_meta.update({"driver": "GTiff",
                                "height": mosaic.shape[1],
                                "width": mosaic.shape[2],
                                "transform": out_trans})
                with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cache',
                                                str(cfg.PARAMS['date'][0]),'mosaic',
                                                str(band+'.tif')), "w", **out_meta) as dest:
                    print('Writing mosaic to file...', band)
                    dest.write(mosaic)

        # Extract Metadata for Tile

        meta_name = glob.glob(os.path.join(cfg.PATHS['working_dir'], 'cache',
                                           str(cfg.PARAMS['date'][0]), '**',
                                           'GRANULE', '**', 'MTD_TL.xml'),
                              recursive=False)[0]

        if clear_safe:
            print("Deleting downloaded .SAFE directories...")
            safe_list = [filename for filename in glob.glob(cfg.PATHS['working_dir'] +
                                                            '/cache/**/*.SAFE',
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
    :param xml_file: :
    :return: solar zenith angle
    solar azimuth angle
    """
    # Parse the XML file
    tree = ET.parse(XML_File)
    root = tree.getroot()

    # Find the angles
    for child in root:
        if child.tag[-12:] == 'General_Info':
            geninfo = child
        if child.tag[-14:] == 'Geometric_Info':
            geoinfo = child

    for segment in geninfo:
        if segment.tag == 'TILE_ID':
            tile_id = segment.text.strip()

    for segment in geoinfo:
        if segment.tag == 'Tile_Geocoding':
            frame = segment
        if segment.tag == 'Tile_Angles':
            angles = segment

    for box in frame:
        if box.tag == 'HORIZONTAL_CS_NAME':
            czone = box.text.strip()[-3:]
            hemis = czone[-1:]
            zone = int(czone[:-1])
        if box.tag == 'Size' and box.attrib['resolution'] == '60':
            for field in box:
                if field.tag == 'NROWS':
                    nrows = int(field.text)
                if field.tag == 'NCOLS':
                    ncols = int(field.text)
        if box.tag == 'Geoposition' and box.attrib['resolution'] == '60':
            for field in box:
                if field.tag == 'ULX':
                    ulx = float(field.text)
                if field.tag == 'ULY':
                    uly = float(field.text)
    if hemis == 'S':
        lzone = -zone
    else:
        lzone = zone
    AngleObs = {'zone': zone, 'hemis': hemis, 'nrows': nrows, 'ncols': ncols, 'ul_x': ulx, 'ul_y': uly, 'obs': []}

    for angle in angles:
        if angle.tag == 'Sun_Angles_Grids':
#            bandId = int(angle.attrib['bandId'])
#            detectorId = int(angle.attrib['detectorId'])
            for bset in angle:
                if bset.tag == 'Zenith':
                    zenith = bset
                if bset.tag == 'Azimuth':
                    azimuth = bset
            for field in zenith:
                if field.tag == 'COL_STEP':
                    col_step = int(field.text)
                if field.tag == 'ROW_STEP':
                    row_step = int(field.text)
                if field.tag == 'Values_List':
                    zvallist = field
            for field in azimuth:
                if field.tag == 'Values_List':
                    avallist = field
            for rindex in range(len(zvallist)):
                zvalrow = zvallist[rindex]
                avalrow = avallist[rindex]
                zvalues = float(zvalrow.text.split(' '))
                avalues = float(avalrow.text.split(' '))
                values = zip(zvalues, avalues)
                ycoord = uly - rindex * row_step
                id = 0
                for cindex in zvalues:
                    #TODO: think about how to write 5x5 km values into a grid....
                    # we have: xcoord, ycoord, zvalue (which form?) avalue...shouldn't be too tricky?
#                    print("Cindex",cindex)
#                    print(id)
                    xcoord = ulx + id * col_step
                    id = id+1
#                    (lat, lon) = utm_inv(lzone, xcoord, ycoord)
                        #zen = float(cindex[0])
                        #az = float(cindex[1])
                       # observe = [xcoord, ycoord, Sat, Gx]
                       # AngleObs['obs'].append(observe)

    #return (AngleObs)

def utm_inv(Zone, X, Y, a=6378137.0, b=6356752.31414):
    """ From s2a_angle_bands_mod.py module"""
    if Zone < 0:
        FNorth = 10000000.0  # Southern hemisphere False Northing
    else:
        FNorth = 0.0  # Northern hemisphere False Northing
    FEast = 500000.0  # UTM False Easting
    Scale = 0.9996  # Scale at CM (UTM parameter)
    LatOrigin = 0.0  # Latitude origin (UTM parameter)
    CMDeg = -177 + (abs(int(Zone)) - 1) * 6
    CM = float(CMDeg) * pi / 180.0  # Central meridian (based on zone)
    ecc = 1.0 - b / a * b / a
    ep = ecc / (1.0 - ecc)
    M0 = a * ((1.0 - ecc * (0.25 + ecc * (3.0 / 64.0 + ecc * 5.0 / 256.0))) * LatOrigin
                 - ecc * (0.375 + ecc * (3.0 / 32.0 + ecc * 45.0 / 1024.0)) * sin(2.0 * LatOrigin)
                  + ecc * ecc * (15.0 / 256.0 + ecc * 45.0 / 1024.0) * sin(4.0 * LatOrigin)
                  - ecc * ecc * ecc * 35.0 / 3072.0 * sin(6.0 * LatOrigin))
    M = M0 + (Y - FNorth) / Scale
    Mu = M / (a * (1.0 - ecc * (0.25 + ecc * (3.0 / 64.0 + ecc * 5.0 / 256.0))))
    e1 = (1.0 - sqrt(1 - ecc)) / (1.0 + sqrt(1.0 - ecc))
    Phi1 = Mu + (e1 * (1.5 - 27.0 / 32.0 * e1 * e1) * sin(2.0 * Mu)
                  + e1 * e1 * (21.0 / 16.0 - 55.0 / 32.0 * e1 * e1) * sin(4.0 * Mu)
                     + 151.0 / 96.0 * e1 * e1 * e1 * sin(6.0 * Mu)
                     + 1097.0 / 512.0 * e1 * e1 * e1 * e1 * sin(8.0 * Mu))
    slat = sin(Phi1)
    clat = cos(Phi1)
    Rn1 = a / sqrt(1.0 - ecc * slat * slat)
    T1 = slat * slat / clat / clat
    C1 = ep * clat * clat
    R1 = Rn1 * (1.0 - ecc) / (1.0 - ecc * slat * slat)
    D = (X - FEast) / Rn1 / Scale
      # Calculate Lat/Lon
    Lat = Phi1 - (Rn1 * slat / clat / R1 * (D * D / 2.0
                                                - (
                                                            5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * ep) * D * D * D * D / 24.0
                                                + (
                                                            61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1 * T1 - 252.0 * ep - 3.0 * C1 * C1) * D * D * D * D * D * D / 720.0))
    Lon = CM + (D - (1.0 + 2.0 * T1 + C1) * D * D * D / 6.0 + (
                    5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1 * C1 + 8.0 * ep + 24.0 * T1 * T1)
                    * D * D * D * D * D / 120.0) / clat

    return (Lat, Lon)

def LOSVec( Lat, Lon, Zen, Az ):
    """ From s2a_angle_bands_mod.py module
    :param Lat:
    :param Lon:
    :param Zen:
    :param Az:
    :return:
    """
    a = 6378137.0  # WGS 84 semi-major axis in meters
    b = 6356752.314  # WGS 84 semi-minor axis in meters
    ecc = 1.0 - b / a * b / a  # WGS 84 ellipsoid eccentricity
    LSRx = ( -sin(Lon), cos(Lon), 0.0 )
    LSRy = ( -sin(Lat)*cos(Lon), -sin(Lat)*sin(Lon), cos(Lat) )
    LSRz = ( cos(Lat)*cos(Lon), cos(Lat)*sin(Lon), sin(Lat) )
    LOS = ( sin(Zen)*sin(Az), sin(Zen)*cos(Az), cos(Zen) )
    Sat = ( LOS[0]*LSRx[0] + LOS[1]*LSRy[0] + LOS[2]*LSRz[0],
	        LOS[0]*LSRx[1] + LOS[1]*LSRy[1] + LOS[2]*LSRz[1],
	        LOS[0]*LSRx[2] + LOS[1]*LSRy[2] + LOS[2]*LSRz[2] )
    Rn = a / sqrt( 1.0 - ecc *sin(Lat)*sin(Lat))
    Gx = ( Rn*cos(Lat)*cos(Lon),
               Rn*cos(Lat)*sin(Lon),
               Rn*(1-ecc)*sin(Lat) )
    return (Sat, Gx)




