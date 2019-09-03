""" 
==============
use_validation_datset.py
==============
Script to call the "create_manual_snow_map" entity task to
Read ín manual validation data set, transform from polygon to raster and write output into snow_cover_man.nc NETCDF file
"""


import warnings
warnings.filterwarnings('ignore')
from snowicesen import cfg
from snowicesen import tasks
import geopandas as gpd
import logging
from snowicesen.workflow import init_glacier_regions
from snowicesen.utils import download_all_tiles
from oggm.workflow import execute_entity_task
from datetime import timedelta
from snowicesen.utils import datetime_to_int, int_to_datetime, extract_metadata


logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize(r"/scratch_net/vierzack03_third/geibell/snowicesen/snowicesen_params.cfg")
# Caution: In crampon.utils.GlacierDirectory._init cfg.initialize is
# called again --> change path there as well! EDIT: only necessary in Windows
    
if __name__ == '__main__':
    # Shapefile with Glacier Geometries:
    rgidf = gpd.read_file(r"/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")
    # Shapefile with Sentinel -2 tiles
    tiles = gpd.read_file(r"/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/sentinel2_tiles_world.shp")
    # List of TileIDs that intersect with Swiss Glacier Inventory
    # (needs to be adjusted if using other region)
    tiles = ['32TLS', '32TLR', '32TMS', '32TMR',
                    '32TNT', '32TNS']


    # Ignore all glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]
#    rgidf = rgidf.sample(n=10)

    # Only keep those glaciers to have smaller dataset
#    rgidf = rgidf[rgidf.RGIId.isin([
#        'RGI50-11.A10G05'])]
#        'RGI50-11.A54L36n', # Fiescher (Shaded)
#        'RGI50-11.B4312n-1'  # Rhone,
#        'RGI50-11.B8315n' # Corbassiere
#        'RGI50-11.B5616n-1',  # Findelen
#        'RGI50-11.A55F03',  # Plaine Morte
#        'RGI50-11.C1410',  # Basodino
#        'RGI50-11.A10G05',  # Silvretta
#        'RGI50-11.B3626-1'  # Gr. Aletsch
#       ])]


 #   log.info('Number of glaciers: {}'.format(len(rgidf)))

    #Go - initialize working directories
    gdirs = init_glacier_regions(rgidf, reset= False, force=True)
    print("Done with init_glacier_regions")

    # Read RGB Folder, list all files in directory

    # Entity task:
    cfg.PARAMS['count'] = 0

    task_list = [tasks.create_manual_snow_map # output:
                        ]
    for task in task_list:
        execute_entity_task(task, gdirs)


