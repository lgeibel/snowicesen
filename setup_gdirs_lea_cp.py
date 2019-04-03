import warnings
warnings.filterwarnings('ignore')
from snowicesat import cfg
from crampon import utils
from snowicesat import tasks
import geopandas as gpd
import logging
from snowicesat.workflow import init_glacier_regions_snowicesat
from snowicesat.utils import download_all_tiles
from crampon.workflow import execute_entity_task
from datetime import timedelta
from snowicesat.utils import datetime_to_int, int_to_datetime, extract_metadata
import glob
import os


logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\snowicesat_params.cfg")
cfg.PATHS['working_dir'] = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\SWISS"
cfg.PATHS['dem_dir'] = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\data\DEM\SWISSALTI3D_2018"


    # something new in OGGM that we are not yet able to handle
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['mp_processes'] = 4

if __name__ == '__main__':
    rgidf = gpd.read_file(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\data\outlines\rgi_copy.shp")

    # Only keep those glaciers to have smaller dataset
    rgidf = rgidf[rgidf.RGIId.isin([
 #       'RGI50-11.B4504',  # Gries
 #       'RGI50-11.A54L36n', # Fiescher (Shaded)
        'RGI50-11.B4312n-1',  # Rhone
 #       'RGI50-11.B5616n-1',  # Findelen
 #       'RGI50-11.A55F03',  # Plaine Morte
 #       'RGI50-11.C1410',  # Basodino
 #       'RGI50-11.A10G05',  # Silvretta
        'RGI50-11.B3626-1'  # Gr. Aletsch
        ])]

    log.info('Number of glaciers: {}'.format(len(rgidf)))

    #Go - initialize working directories
    gdirs = init_glacier_regions_snowicesat(rgidf, reset=False, force=True)
    print("Done with init_glacier_regions")


# Here start iterating for date: for date start to date end:
    start_date_var, end_date = int_to_datetime(cfg.PARAMS['date'])
    end_date_var = start_date_var + timedelta(days=1)
    # Iterate over all days, checking between start and enddate
    while end_date_var <= end_date:
        cfg.PARAMS['date'] = datetime_to_int(start_date_var, end_date_var)
       # Download data for given glaciers for this date
        tiles_downloaded = download_all_tiles(rgidf,
                                              clear_cache=False,
                                              clear_safe=False)  # Function in utils
       # tiles_downloaded = 1
        # move one day ahead
        start_date_var = start_date_var+timedelta(days=1)
        end_date_var = start_date_var+timedelta(days=1)
        if tiles_downloaded > 0:
            # Preprocessing tasks: only execute when new files were downloaded!
            task_list = [#tasks.crop_sentinel_to_glacier,
                         #tasks.crop_metadata_to_glacier,
                         #tasks.crop_dem_to_glacier,
                         #tasks.ekstrand_correction,
                         #tasks.cloud_masking,
                         #tasks.remove_sides,
                         #tasks.otsu_tresholding,
                         tasks.naegeli_snow_mapping
                        ]
            for task in task_list:
                execute_entity_task(task, gdirs)

