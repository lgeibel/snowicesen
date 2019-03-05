import warnings
warnings.filterwarnings('ignore')
from snowicesat import cfg
from crampon import utils
from snowicesat import tasks
import geopandas as gpd
import logging
from snowicesat.workflow import init_glacier_regions_snowicesat, download_all_tiles
from crampon.workflow import execute_entity_task

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
        'RGI50-11.B5616n-1',  # FIndelen
        'RGI50-11.A55F03',  # Plaine Morte
        'RGI50-11.C1410',  # Basodino
        'RGI50-11.A10G05',  # Silvretta
        'RGI50-11.B3626-1'  # Gr. Aletsch
        ])]

    log.info('Number of glaciers: {}'.format(len(rgidf)))

    #Go - initialize working directories, downloads all past time steps
    # TODO: figure out a way to download only a given time period when initializing Glacier Directories
    gdirs = init_glacier_regions_snowicesat(rgidf, reset=False, force=False)
    print("Done with init_glacier_regions")

# Here start iterating for date: for date 1 to date end:
    # Now get all available tiles for all of Switzerland (bounding box of .shape)
    download_all_tiles(rgidf) # Function in snowicesat/workflow


    # Preprocessing tasks
#    task_list = [tasks.download_sentinel]
#    for task in task_list:
#            execute_entity_task(task, gdirs)

