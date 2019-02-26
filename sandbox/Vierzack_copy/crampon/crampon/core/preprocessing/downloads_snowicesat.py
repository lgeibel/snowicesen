from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import zipfile
import geopandas as gpd
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
from crampon import utils
import crampon.cfg as cfg


log = logging.getLogger(__name__)

def download_sentinel(lon_ex, lat_ex):
    """Creates Download request for Sentinel-2 images for a given date for an
    entry in the glacier directory,
    TODO: reproject to local grid, cut to glacier extent, reproject to local grid write all bands into the GlacierDirectory folder

    Parameters
    ----------
    lon_ex : tuple, required
        a (min_lon, max_lon) tuple delimiting the requested area longitudes
    lat_ex : tuple, required
        a (min_lat, max_lat) tuple delimiting the requested area latitudes
    Returns
    -------
    tuple: (list with path(s) to the DEM file, data source)

    Workflow:
    We need: user, password, area_polygon, datum, cloudcover, downloadpath

    1. Read lat_extend, lon_extend fo glacier outline, create bounding box as wkt
    2. Read date, user, password, cloudcover from cfg.PARAMs

    """
    user = "lgeibel"  # Sentinelhub user name
    password = "snowicesat"  # Sentinelhub password
    datum = ("20181001", "20181003")  # date /time frame of interest, format ("20181001", "20181002")
    cloudcover = "[0 TO 40]"  # cloud cover percentage of interest, format [0 TO 30]
    downloadpath = cfg.PATHS['working_dir']

    # 1. Create bounding box as polygon from lat/lon extend of glacier:
    b = box(lon_ex[0], lat_ex[0], lon_ex[1], lat_ex[1], ccw=True)
    crs = {'init': 'epsg:4326'}
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[b])

    bbox_filename = 'bbox.geojson'
    # Avoid Fiona bug with overwriting geojson file: https://github.com/Toblerity/Fiona/issues/438
    try:
        os.remove(bbox_filename)
    except OSError:
        pass
    polygon.to_file(filename=bbox_filename, driver='GeoJSON')

    #2.  use sentinelsat package to request which tiles intersect with glacier outlines: create an api
    api = SentinelAPI(
        user,
        password,
        api_url="https://scihub.copernicus.eu/apihub/")

    # Search for products matching query
    products = api.query(
        area= geojson_to_wkt(read_geojson(bbox_filename)),
        date=datum,
        platformname="Sentinel-2",
        producttype="S2MSI1C",
        cloudcoverpercentage=cloudcover)

    # count number of products matching query and their size
    print("Tiles found:", api.count(area= geojson_to_wkt(read_geojson(bbox_filename)),
                                    date=datum,
                                    platformname="Sentinel-2",
                                    producttype="S2MSI1C",
                                    cloudcoverpercentage=cloudcover),
          ", Total size: ", api.get_products_size(products),
          "GB. Now downloading those tiles")
    print(next(iter(products.values()))['filename'])
    # Product to value in liste umwandeln:
    #print(products['filename'])
    # Check if data tile for given date has already been downloaded:
    #if os.path.isdir(prod)


    # downloading all products
    download_zip = api.download_all(products, directory_path=downloadpath)

    # Unzip files into .safe directory, delete .zip folder
    for key in download_zip[0].keys():
        with zipfile.ZipFile(download_zip[0][key]['path']) as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = zip_file.open(member)
            source.close()

        os.remove(download_zip[0][key]['path'])

