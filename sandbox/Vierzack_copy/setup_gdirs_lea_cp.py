import warnings
warnings.filterwarnings('ignore')
from crampon import cfg
from crampon import utils
from crampon import tasks
import geopandas as gpd
import logging
from crampon import workflow
from crampon.workflow import execute_entity_task

logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')
if __name__ == '__main__':


    cfg.initialize(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\sandbox\Vierzack_copy\CH_params.cfg")
    print("cfg.initialize= ", cfg.PARAMS['grid_dx_method'])
    cfg.PATHS['working_dir'] = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\sandbox\Vierzack_copy\SWISS"
    #cfg.PATHS['dem_dir'] = '/scratch_net/vierzack03_second/geibell/crampon/data/DEM/'

    # something new in OGGM that we are not yet able to handle
    cfg.PARAMS['use_tar_shapefiles'] = False

    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 4

    rgidf = gpd.read_file(r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\sandbox\Vierzack_copy\crampon\data\rgi_copy.shp")

    # Only keep those glaciers to have smaller dataset
    rgidf = rgidf[rgidf.RGIId.isin([
        'RGI50-11.B4504',  # Gries
#        'RGI50-11.B4312n-1',  # Rhone
#        'RGI50-11.B5616n-1',  # FIndelen
#        'RGI50-11.A55F03',  # Plaine Morte
#        'RGI50-11.C1410',  # Basodino
#        'RGI50-11.A10G05',  # Silvretta
#        'RGI50-11.B3626-1'  # Gr. Aletsch
        ])]

    log.info('Number of glaciers: {}'.format(len(rgidf)))

    #Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)


    utils.joblib_read_climate_crampon.clear()
    # Preprocessing tasks
    task_list = [
    tasks.glacier_masks,
    ]
    for task in task_list:
            execute_entity_task(task, gdirs)

