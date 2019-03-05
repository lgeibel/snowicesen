import warnings
warnings.filterwarnings('ignore')
from snowicesat import cfg
from crampon import utils
from snowicesat import tasks
import geopandas as gpd
import logging
from snowicesat.workflow import init_glacier_regions_snowicesat, download_all_tiles
from crampon.workflow import execute_entity_task
from datetime import timedelta
from snowicesat.utils import datetime_to_int, int_to_datetime



logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\snowicesat_params.cfg")
cfg.PATHS['working_dir'] = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\SWISS"
cfg.PATHS['dem_dir'] = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\data\DEM"


    # something new in OGGM that we are not yet able to handle
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['mp_processes'] = 4

if __name__ == '__main__':
    rgidf = gpd.read_file(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\data\outlines\rgi_copy.shp")

    # Only keep those glaciers to have smaller dataset
    rgidf = rgidf[rgidf.RGIId.isin([
        'RGI50-11.B4504',  # Gries
        'RGI50-11.B4312n-1',  # Rhone
        'RGI50-11.B5616n-1',  # Findelen
        'RGI50-11.A55F03',  # Plaine Morte
        'RGI50-11.C1410',  # Basodino
        'RGI50-11.A10G05',  # Silvretta
        'RGI50-11.B3626-1'  # Gr. Aletsch
        ])]

    log.info('Number of glaciers: {}'.format(len(rgidf)))

    #Go - initialize working directories
    gdirs = init_glacier_regions_snowicesat(rgidf, reset=False, force=False)
    print("Done with init_glacier_regions")


# Here start iterating for date: for date start to date end:
    end_date, start_date_var = int_to_datetime(cfg.PARAMS['date'])
    end_date_var = start_date_var + timedelta(days=1)
    # Iterate over all days, checking between start and enddate
    while end_date_var<=end_date:
        print(end_date_var)
        # TODO: why is it not iterating over more than one day? 
        cfg.PARAMS['date']= datetime_to_int(start_date_var, end_date_var)
        # Download data for given glaciers for this date
        download_all_tiles(rgidf) # Function in snowicesat/workflow
        #move one day ahead
        start_date_var = start_date_var+timedelta(days=1)
        end_date_var = start_date_var+timedelta(days=1)



    # Preprocessing tasks
    task_list = [tasks.crop_sentinel_to_glacier]
    for task in task_list:
            execute_entity_task(task, gdirs)

