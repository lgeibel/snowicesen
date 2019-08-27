"""
=====
utils
=====
A collection of the functions used to automatically find, download 
and preprocess the sentinel data as well as some other general functions

"""

from __future__ import absolute_import, division

from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio import Affine
import shapely
from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI
import fiona
from oggm.utils import *
# Locals
import snowicesen.cfg as cfg
from datetime import date
import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
from crampon import GlacierDirectory
from retrying import retry
import matplotlib.pyplot as plt
import matplotlib
log = logging.getLogger(__name__)
print(cfg.PATHS)

def parse_credentials_file_snowicesen(credfile=None):
    """ Reads .credential file for sentinelhub login, username and password

    Parameters 
    ----------
    credfile: str
        path to .credentials file

    Returns
    -------
    cr: list
        list of  credentials for different platforms
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
    Reads cfg.PARAMS('date') in format 20170219, 20170227, returns
    datetime

    Parameters
    ----------
    date_str: tuple
        two integers in format yyyymmdd, yyyymmdd

    Returns
    -------
    start_date, end_date: datetime.date
        start and end date
    """
    start_date = date(int(str(cfg.PARAMS['date'][0])[0:4]), int(str(cfg.PARAMS['date'][0])[4:6]),
                          int(str(cfg.PARAMS['date'][0])[6:8]))
    end_date = date(int(str(cfg.PARAMS['date'][1])[0:4]), int(str(cfg.PARAMS['date'][1])[4:6]),
                    int(str(cfg.PARAMS['date'][1])[6:8]))

    return start_date, end_date

def datetime_to_int(start_date, end_date):
    """
    Converts datetime to tuple  of two integers in format 20170219, 20170227
        
    Parameters
    ----------
     start_date: datetime.date
        Format  yyyy-mm-dd
     end_date: datetime.date
        Formate yyyy-mm-dd

    Returns
    -------
    date_int: tuple
        two ints with start and end date 
                    as yyyymmdd, yyyyymmdd
            
    """
    date_int = int(start_date.strftime("%Y%m%d")), int(end_date.strftime("%Y%m%d"))

    return date_int

def download_all_tiles(glacier, tile, use_tiles = True,  clear_cache = False, clear_safe = False):
    """ Download, exctract and merge all Sentinrl data to a mosaic

    Function to download all available Sentinel-2 Tiles
    to cache for a given set of Glaciers (stored in "glacier") for
    two consecutive days. Extracts the .SAFE directories
    into 12 GeoTiff mosaics of the entire scene for each band
    (stored in working_dir/cache/date/mosaic/B01.tif).


    Structure
    - Reads outlines into GeoDataFrame
    - Transforms local to WGS 84 grid
    - create bounding box around outline
        (full outline is too complex to be processed by 
        sentinelsat-server)
    - files a request with sentinelsat package with the 
      outlines of the bounding box
    - if use_tiles = True, products are selected to keep 
        only results that intersect with tiles of given ID.
     ( direct search by file ID is only possible for dates
     newer than 2017). By using this, downloading of doubles
     of areas in different tiles or weird, oversized tiles 
     is avoided and the amount of downloaded data can be
     considerably reduced, making the function considerably 
     faster

    The function then
    - downloads new data for given date if available
    - unzip folder into .SAFE directory
    - reads all tiles bandwise, reprojects all bands 
        to 10 Meter resolution,
        merges them, writes tile into 
        working_dir/cache/date/mosaic as GeoTiff

    Then the metadata/solar angles (solar zenith and solar azimuth) are
    extracted from the MTD_TML.xml file (on a 5x 5km grid)
    and written into tiles, reprojected and then merged.
    The output is then stored in
    working_dir/cache/date/meta/solar_zenith.tif and
    working_dir/cache/date/meta/solar_azimuth.tif

    WARNING: downloading and merging is very time-consuming, 
    so deleting safe and cache file might be difficult when
    having to re-run an analysis.
    Tiles can be very big,
    so storing too many tiles risks in running out of memory

    Parameters
    ----------
    glacier: GeoDataFrame
        containing all GlacierOutlines
    tile: list
        List of al tile IDs that  intersect with Shapefile of Glaciers
    use_tiles: boolesn, default: True
        if products should be checked to download only main tiles
        that are specified via tileID. If set to False, al
    clear_cache: boolean
        clearing merged GeoTiff from previous 
         step before starting new download, default: False
    clear_safe:boolean
        deleting .SAFE directories after
        reading/merging tiles into Geotiffs, default: False

    Returns
    -------
    tiles_downloaded: int
        how many tiles were downloaded for this date

    """
    #  Use sentinelsat package to request which tiles intersect 
    # with glacier outlines:
    # 
    # 1. Geodataframe containing all glaciers:
    # Reproject to  WGS 84 Grid for query (requested by sentinelsat
    # module):
    glacier = glacier.to_crs({'init': 'epsg:4326'})

    # Create bounding box/envelope as polygon, safe to geojson file
    bbox = glacier.envelope

    bbox = bbox.total_bounds
    p1 = shapely.geometry.Point(bbox[0], bbox[1])
    p2 = shapely.geometry.Point(bbox[2], bbox[1])
    p3 = shapely.geometry.Point(bbox[2], bbox[3])
    p4 = shapely.geometry.Point(bbox[0], bbox[3])

    pointList = [p1, p2, p3, p4, p1]
    # Area as shapely Geometry:
    index = shapely.geometry.Polygon([[p.x, p.y] for p in pointList])
    
    products, api = get_sentinel_products(index)

    # Count number of products matching query and their size
    print("Sentinel Tiles found:",len(products),
      ", Total size: ", api.get_products_size(products),
      "GB.")
    tiles_downloaded = 0
    if not len(products) is 0: # If products are available, download them:
            # Create cache directory for this date:
        if not os.path.exists(os.path.join(cfg.PATHS['working_dir'],
                 'cache', str(cfg.PARAMS['date'][0]))):
             os.makedirs(os.path.join(cfg.PATHS['working_dir'], 
                 'cache', str(cfg.PARAMS['date'][0])))
             os.makedirs(os.path.join(cfg.PATHS['working_dir'],
                 'cache', str(cfg.PARAMS['date'][0]), 'mosaic'))

        product_id = list(products.keys())

        tiles_downloaded += len(products)
        print('Downloaded Tiles: ',tiles_downloaded)
        for index in product_id:
            product_tile_id = products[index]['tileid']
            # check if tile_id of product is in tile_id list
            # that is specified by user:
            if use_tiles:
                if product_tile_id in tile:
                    print('Product_Id is in list of desired tiles')
                    #  we want to download this file
                else:
                    print('Product Id not in list of tiles - not\
                        downloading this one')
                    tiles_downloaded -=1
                    # continue to next product
                    continue
            safe_name = products[index]['filename']
            if not os.path.isdir(os.path.join(cfg.PATHS['working_dir'],
                    'cache', str(cfg.PARAMS['date'][0]), safe_name)):
                #  If not downloaded: downloading all products
                try:
                   download_zip = api.download(index,
                                 directory_path=os.path.join(
                                    cfg.PATHS['working_dir'],
                                    'cache',
                                    str(cfg.PARAMS['date'][0])))
                except:
                    # In case downloading fails: 
                    # different kinds of corrupt files in Copernicus Hub:
                    # Just skip this file and cycle ahead

                    continue

                # Unzip files into .safe directory, delete .zip folder
                with zipfile.ZipFile(os.path.join(cfg.PATHS['working_dir'],
                                             'cache',
                                              str(cfg.PARAMS['date'][0]),
                                              download_zip['path'])) \
                        as zip_file:
                    zip_file.extractall(os.path.join(
                            cfg.PATHS['working_dir'],
                                    'cache', str(cfg.PARAMS['date'][0])))
                os.remove(os.path.join(cfg.PATHS['working_dir'],
                    'cache',str(cfg.PARAMS['date'][0]), 
                    download_zip['path']))
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
    print('Tiles Downloaded', tiles_downloaded)
    if tiles_downloaded > 0:
        if clear_cache:
            print("Removing old merged tiles from cache")
            tif_list = os.listdir(os.path.join(
                cfg.PATHS['working_dir'], 'cache',
                                 str(cfg.PARAMS['date'][0]), 'mosaic'))
            for f in tif_list:
                os.remove(f)
        for band in band_list:
            print('Merging tiles: Band', band)
            if not os.path.isfile(os.path.join(
                        cfg.PATHS['working_dir'],'cache',
                        str(cfg.PARAMS['date'][0]),'mosaic',
                        str(band+'.tif'))):
                if not os.path.exists(os.path.join(
                        cfg.PATHS['working_dir'],'cache',
                        str(cfg.PARAMS['date'][0]))):
                    os.makedirs(os.path.join(cfg.PATHS['working_dir'],
                        'cache', str(cfg.PARAMS['date'][0])))

                # List of filenames of same band of this date in 
                # all .safe tiles
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
                bands_20m = ['B05.tif', 'B06.tif', 'B07.tif', 
                            'B11.tif', 'B12.tif', 'B8A.tif']
                if band+".tif" in bands_60m or band+".tif" in bands_20m:
                    if band+".tif" in bands_60m:
                        res_factor = 6
                    elif band+".tif" in bands_20m:
                        res_factor = 2
                    arr = mosaic
                    newarr = np.empty(shape=(
                                    arr.shape[0], # same number of bands
                                    (arr.shape[1] * res_factor),
                                    (arr.shape[2] * res_factor)),
                                    dtype='uint16')

                    # adjust the new affine transform 
                    # to the smaller cell size
                    old_transform = out_trans
                    new_transform = Affine(old_transform.a / res_factor,
                                           old_transform.b,
                                           old_transform.c,
                                           old_transform.d,
                                           old_transform.e / res_factor,
                                           old_transform.f)
                    out_meta['transform'] = new_transform
                    out_meta['width'] = out_meta['width'] * res_factor
                    out_meta['height'] = out_meta['height'] * res_factor
                #--------- End reproject

                # Now Write to file: B01. tif, B02.tif, etc.

                    with rasterio.open(os.path.join(
                                 cfg.PATHS['working_dir'],'cache',
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
                    with rasterio.open(os.path.join(
                                 cfg.PATHS['working_dir'], 'cache',
                                 str(cfg.PARAMS['date'][0]), 'mosaic',
                                 str(band + '.tif')), "w", **out_meta) \
                            as dest:
                        dest.write(mosaic)


        # ------ Metadata -----
        if not os.path.isfile(os.path.join(
                            cfg.PATHS['working_dir'],
                            'cache', str(cfg.PARAMS['date'][0]),
                            'meta', "solar_zenith_angles.tif")):
            print('Extracting SolarAngles')
            # Extract Metadata for each Tile
            # list of all metadata files for this date
#            meta_list = glob.glob(os.path.join(
#                            cfg.PATHS['working_dir'], 'cache',
#                             str(cfg.PARAMS['date'][0]), '**',
#                                         'GRANULE', '**', 'MTD_TL.xml'),
#                                  recursive=False)
#            id = 0
#            for meta_name in meta_list:
#                # Read eaach tile from .xml to GeoTIff and reproject to
#            #  10 Meter resolution
#                solar_zenith, solar_azimuth = extract_metadata(meta_name)
#                
#                resample_meta(solar_zenith, solar_azimuth, meta_name, id)
#                id += 1
#            print(solar_zenith.shape)
#            mean_zenith = np.mean(solar_zenith, axis = (0,1))
#            mean_azimuth = np.mean(solar_azimuth, axis = (0,1))
#            cfg.PARAMS['zenith_mean'] = mean_zenith
#            cfg.PARAMS['azimuth_mean'] = mean_azimuth


            ## Create empty list for datafile
            #solar_zenith_to_mosaic = []
            #solar_azimuth_to_mosaic =[]

            ## List all geotiffs containing solar zenith angles
            #zenith_list = glob.glob(os.path.join(
            #                     cfg.PATHS['working_dir'], 'cache',
            #                     str(cfg.PARAMS['date'][0]), 'meta',
            #                                    '*zenith.tif'),
            #                        recursive=False)
#
#            for fp in zenith_list:
#                # Open File, append to list
#                src = rasterio.open(fp)
#                solar_zenith_to_mosaic.append(src)
#             # Merge all tiles of zenith angles together
#            zenith_mosaic, out_trans = merge(solar_zenith_to_mosaic)
#
#            # Update metadata for merges mosaic
#            out_meta = src.meta.copy()
#            out_meta.update({"driver": "GTiff",
#                                  "height": zenith_mosaic.shape[1],
#                                  "width": zenith_mosaic.shape[2],
#                                  "transform": out_trans})
#
#            # Open new file to write merged mosaic
#            with rasterio.open(os.path.join(
#                                    cfg.PATHS['working_dir'], 'cache',
#                                    str(cfg.PARAMS['date'][0]), 'meta',
#                                    "solar_zenith_angles.tif"), "w",
#                                        **out_meta) \
#                 as dest:
#                # Take average of mosaic here:
#                mean_zenith = np.mean(zenith_mosaic, axis = (0,1,2))
#                dest.write(zenith_mosaic)
#
#
#            # List all GeoTiffs conatining solar azimuth angles
#            azimuth_list = glob.glob(os.path.join(
#                            cfg.PATHS['working_dir'], 'cache',
#                            str(cfg.PARAMS['date'][0]), 'meta',
#                                                '*azimuth.tif'),
#                                    recursive=False)
#            for fp in azimuth_list:
#                # Open File, append to list
#                src =  rasterio.open(fp)
#                solar_azimuth_to_mosaic.append(src)
#            # Merge all tiles with solar azimuth angles together
#            azimuth_mosaic, out_trans = merge(solar_azimuth_to_mosaic)
#
#            # Open file to write merged mosaic of solar azimuth angles
#            with rasterio.open(os.path.join(cfg.PATHS['working_dir'],
#                'cache', str(cfg.PARAMS['date'][0]), 'meta',
#                     "solar_azimuth_angles.tif"), "w", **out_meta) \
#                 as dest:
#                # Take average of azimuth here:
#                mean_azimuth = np.mean(azimuth_mosaic, axis = (0,1,2))
#                dest.write(azimuth_mosaic)
#
#
#            for fp in zenith_list:
#                os.remove(fp)
#
#            for fp in azimuth_list:
#                os.remove(fp)

            #### Sometimes we have problems with Metadata georeferencing- 
            # Quick workaround: write out average value as global variable
            # to use for Ekstrand correction (barely any influence on 
            # corrected values since variations over Switzerland are very 
            # small anyways)
 #           cfg.PARAMS['zenith_mean'] = mean_zenith
 #           cfg.PARAMS['azimuth_mean'] = mean_azimuth
 #           print(cfg.PARAMS['zenith_mean'])
 #           print(cfg.PARAMS['azimuth_mean'])


            # -------  end of Metadata

        if clear_safe:
            print("Deleting downloaded .SAFE directories...")
            safe_list = [filename for filename in glob.glob(
                os.path.join(cfg.PATHS['working_dir'],
                'cache','**','*.SAFE'),recursive=False)]
            for f in safe_list:
                shutil.rmtree(f)


    return tiles_downloaded

@retry(stop_max_attempt_number = 100, wait_fixed = 10000)
def get_sentinel_products(index):
    """
    Function to set API for sentinel Copernicus hub with 
    credentials from the .snowicesat_credentials file
    and get products for request.
    Used with retry decorator to account for connectivity
    problems of the Copernicus hub

    Parameters:
    -----------
    index: shapely.geometry.Polygon
            Shape of Bounding box around Area of area of interest

    Returns:
    --------
    products: sentinelsata.Products
        Ordered Dict of products available for request
    api: 
        api of Copernicus Hub with sentinelsat
    """
    cr = parse_credentials_file_snowicesen(os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(
            __file__))), 'snowicesen.credentials'))
    print("Connecting to server..")
    api = SentinelAPI(
            cr['sentinel']['user'],
            cr['sentinel']['password'],
            api_url = "https://scihub.copernicus.eu/apihub/")
    print("Getting products....", api)
    products = api.query(area = index,
                        date = (tuple(str(item) for item in cfg.PARAMS['date'])),
                        platformname = "Sentinel-2",
                        producttype = "S2MSI1C",
                        cloudcoverpercentage="[{} TO {}]".format(
                                cfg.PARAMS['cloudcover'][0],
                                cfg.PARAMS['cloudcover'][1]))

    return products, api


def extract_metadata(XML_File):
    """ Extract solar angles from .xml file
    Extracts solar zenith and azimuth angle from GRANULE/MTD_TL.xml
    metadata file of .safe directory
    - main structure after s2a_angle_bands_mod.py module by
    Sentinel-2 MultiSpectral Instrument (MSI) data processing for
    aquatic science applications:
     Demonstrations and validations
    N.Pahlevan et al., Appendix B:

    Parameters
    ----------
    XML_file:str: Name of Metadata file located in
                GRANULE/**/MDT_TL.xml 

    Returns
    -------
    zcoord: np.array: solar zenith angle in array of 5x5 km resolution
    acoord: np.array: solar azimuth angle in array of 5x5 km resolution
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



def resample_meta(solar_zenith, solar_azimuth,meta_name, index):
    """
    Resamples Solar Zenith and solar azimuth angle from 5x5 km
    to 10 m Grid (nearest neighbor) and writes into GeoTIFF

    Parameters
    ----------
    solar_zenith: np.array float64
        data in 5km x 5 km grid
    solar_azimuth: np.array float64
        data in 5 km x 5 km grid
    meta_name: str
        Name of .xml file containing metadata
    index: int
        number of current tile (just used for naming convention)


    Returns
    -------
    None
    """
    reference_tile = glob.glob(os.path.join(meta_name[:-10],'IMG_DATA','*_B02.jp2'))[0]

    # Open 10 m resolution tile to read size and dimension of final tile:
    with rasterio.open(reference_tile) as src:
        
        out_meta = src.meta.copy()
        newarr = src.read()
        newarr = np.squeeze(newarr)
        newarr = newarr.astype(float)

        new_transform = src.transform
        old_transform = Affine(new_transform.a * 500, new_transform.b,
                               new_transform.c, new_transform.d,
                               new_transform.e * 500, new_transform.f)
        out_meta['dtype'] = 'float64'
        out_meta['driver'] = "GTiff"

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

def assign_bc(elev_grid):
    """ Pads the boundaries of a grid
    Boundary condition pads the boundaries with equivalent values
    to the data margins, e.g. x[-1,1] = x[1,1]
    This creates a grid 2 rows and 2 columns larger than the input
    which is necessary when computing 2-D slopes

    Paramaters
    ----------
    elev_grid: np.array: grid (e.g. DEM)

    Returns
    ------
    z_bc: np.array: grid with 2 extra rows and columns
    """
    ny, nx = elev_grid.shape  # size of array
    z_bc = np.zeros((ny+2, nx +2)) # Create BC

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




def two_d_scatter(x_as, x_na, x_nai,y_as, y_na, y_nai,z_as, z_na, z_nai, x_name, y_name, z_name, figure_id):
    """
    Create 2 - d Scatter Plots with third dimension in color scale.
    Used to plot validation results (kappa, SLA) against snow_cover, area,
    cloud_cover, etc. of the validation data set in different combinations

    Parameters:
    -----------
    x_as: list: x-values returned by the ASMAG-Algorithm
    x_na: list: x-values returned by the algorithm by naegeli
    x_nai: list: x- values returned by the improved version of the naegeli-algorithm
    .
    .
    .
    x_name: str: name of x-variable
    .
    .
    figure_id: int: id of figure to be plotted
    """
    plt.figure(figure_id,figsize=(15,5))
    plt.rcParams.update({"font.size": 12})
    plt.suptitle(str(z_name+ 'in dependence on '+ x_name+ ' and '+ y_name))
    plt.subplot(1,3,1)

   
    cmap = matplotlib.cm.get_cmap('YlGnBu')
    normalize = matplotlib.colors.Normalize(vmin=min(z_as),
            vmax = max(z_as))
    colors = [cmap(normalize(value)) for value in z_as]
    plt.scatter(x_as, y_as, s=20,color=colors,linewidth=0.5, edgecolor='blue')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label(z_name)
    #plt.xlim([0.01, 1.1])
    plt.xlim(0,1)
   # plt.xscale('log')
    plt.ylim(0,1)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    plt.title('ASMAG')
   
    plt.subplot(1,3,2)
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    normalize = matplotlib.colors.Normalize(vmin=min(z_na),
            vmax = max(z_na))
    colors = [cmap(normalize(value)) for value in z_na]
    plt.scatter(x_na, y_na, s=20,color=colors,linewidth=0.5, edgecolor='red')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label(z_name)
    plt.xlim(0,1)
    #plt.xscale('log')
    
    plt.ylim(0,1)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    plt.title('Naegeli')
   
    plt.subplot(1,3,3)
   
    cmap = matplotlib.cm.get_cmap('YlGn')
    normalize = matplotlib.colors.Normalize(vmin=min(z_nai),
             vmax = max(z_nai))
    colors = [cmap(normalize(value)) for value in z_nai]
    plt.scatter(x_na, y_na, s=20,color=colors,linewidth=0.5, edgecolor='green')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label(z_name)
    plt.xlim(0,1)
   # plt.xscale('log')
    
    plt.ylim(0,1)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    plt.title('Naegeli Improved')
    plt.tight_layout()
   
