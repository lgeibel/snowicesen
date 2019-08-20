""" 
==============
setup_gdirs.py
==============

This is the main file to excute snowicesen: 
It inititalizes the Configuration/Parameters file, 
reads the outlines of the region and creates the Glacier Directories
(see OGGM: Open Global Glacier Model for Documentation of the Class).
It then iterates over the given time period and downloads
all Sentinel-2 images available for this region & current date, 
crops the big tile to sentinel.nc file for each glacier and 
then performs the entity tasks for preprocessing (Ekstrand Correction,
Cloud Masking, thresholding to remove dark, debris-covered areas).
After the preprocessing, 3 snow mapping algorithms are performed so
retrieve the snow covered area and the SLA  on each glacier.

The parameters file is inititalized in as snowicesen_params.cfg. 
For any other file, the path in cfg.initialite needs to be adapted

The rgidf contains all glaciers on which the snow mapping should be
performed. Initially, those are all polygons in the .shp file that 
determines the region of interest (usually based on a glacier
inventory).
For snow  classification, it is advised to focus on areas larger 
than 0.1 km^2. 
Additionally, any subset of glaciers can be specified (e.g. by
their RGIId)
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

cfg.initialize("/scratch_net/vierzack03_third/geibell/snowicesen/snowicesen_params.cfg")
# Caution: In crampon.utils.GlacierDirectory._init cfg.initialize is
# called again --> change path there as well! EDIT: only necessary in Windows
    
if __name__ == '__main__':
    # Shapefile with Glacier Geometries:
    rgidf = gpd.read_file("/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")
    # Shapefile with Sentinel -2 tiles
    tiles = gpd.read_file('/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/sentinel2_tiles_world.shp')
    # List of TileIDs that intersect with Swiss Glacier Inventory
    # (needs to be adjusted if using other region)
    tiles = ['32TLS', '32TLR', '32TMS', '32TMR',
                    '32TNT', '32TNS']


    # Ignore all glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]
#    rgidf = rgidf.sample(n=10)

    # Only keep those glaciers to have smaller dataset
#    rgidf = rgidf[~rgidf.RGIId.isin([
#        'RGI50-11.E4508-1'])]
#        'RGI50-11.A54L36n', # Fiescher (Shaded)
#        'RGI50-11.B4312n-1',  # Rhone
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


# Here start iterating for date: for date start to date end:
    start_date_var, end_date = int_to_datetime(cfg.PARAMS['date'])
    end_date_var = start_date_var + timedelta(days=1)
    # Iterate over all days, checking between start and enddate
    while end_date_var <= end_date:
        cfg.PARAMS['date'] = datetime_to_int(start_date_var,
                                                end_date_var)
        print("Date = ", cfg.PARAMS['date'][0])
       # Download data for given glaciers for this date
        tiles_downloaded = download_all_tiles(rgidf, tiles, use_tiles = True,
                                              clear_cache = False,
                                              clear_safe= False) 
                                            # Function in utils
        # move one day ahead
        start_date_var = start_date_var+timedelta(days=1)
        end_date_var = start_date_var+timedelta(days=1)
        if tiles_downloaded > 0:
            # Processing tasks: only execute when new files 
            # were downloaded!
            task_list = [tasks.crop_satdata_to_glacier, # output:
                                                        # sentinel.nc,
                                                        #solar_angles.nc
                                                        # dem_ts.nc
                        # tasks.ekstrand_correction, # output: ekstrand.nc
                         tasks.cloud_masking, # ouput: cloud_masked.nc
                         tasks.remove_sides, # output: sentinel_temp.nc
                         tasks.asmag_snow_mapping,
                         tasks.naegeli_snow_mapping,
                         tasks.naegeli_improved_snow_mapping,
                        # tasks.plot_results
                        ]
            for task in task_list:
                execute_entity_task(task, gdirs)


