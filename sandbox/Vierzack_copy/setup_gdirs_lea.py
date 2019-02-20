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

cfg.initialize('/scratch_net/vierzack03_second/geibell/crampon/sandbox/CH_params.cfg')

cfg.PATHS['working_dir'] = '/scratch_net/vierzack03_second/geibell/modelruns/CH'

cfg.PATHS['climate_file'] = '/scratch_net/vierzack03_second/geibell/crampon/data/meteo/climate_all.nc'

cfg.PATHS['lfi_dir'] = '/scratch_net/vierzack03_second/geibell/crampon/data/DEM/LFI'

cfg.PATHS['firncore_dir'] = '/scratch_net/vierzack03_second/geibell/crampon/data/firncores/'

cfg.PATHS['dem_dir'] = '/scratch_net/vierzack03_second/geibell/crampon/data/DEM/'

cfg.PATHS['mb_dir'] = '/scratch_net/vierzack03_second/geibell/crampon/data/MB/'

cfg.PATHS['modelrun_backup_dir_1'] = '/scratch_net/vierzack03_second/geibell/run_backup'

cfg.PATHS['hfile'] = '/scratch_net/vierzack03_second/geibell/crampon/data/DEM/hgt.nc'

cfg.PATHS['climate_dir'] = '/scratch_net/vierzack03_second/geibell/crampon/data/meteo'

cfg.PATHS['lfi_worksheet'] = '/scratch_net/vierzack03_second/geibell/crampon/data/meteo/ginzler_ws.shp'

# something new in OGGM that we are not yet able to handle
cfg.PARAMS['use_tar_shapefiles'] = False


cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['mp_processes'] = 4

rgidf = gpd.read_file('/scratch_net/vierzack03_second/geibell/crampon/data/outlines/mauro_sgi_merge.shp')
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


# Go - initialize working directories
gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=False)
utils.joblib_read_climate_crampon.clear()
# Preprocessing tasks
task_list = [
tasks.glacier_masks,
tasks.compute_centerlines,
tasks.initialize_flowlines,
tasks.compute_downstream_line,
tasks.catchment_area,
tasks.catchment_intersections,
tasks.catchment_width_geom,
tasks.catchment_width_correction,
tasks.process_custom_climate_data,

]
for task in task_list:
        execute_entity_task(task, gdirs)

