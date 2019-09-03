"""
Make_plot_results.py
====================
Script to call plotting entity tasks (plot_results). See there for further documentation
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
#fg.PARAMS['working_dir'] = "/scratch_net/vierzack03_third/geibell/snowicesen/SWISS"
# Caution: In crampon.utils.GlacierDirectory._init cfg.initialize is
# called again --> change path there as well!

if __name__ == '__main__':
    rgidf = gpd.read_file("/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")
    # Ignore all values glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]
#    rgidf = rgidf.sample(n=10)

    # Only keep those glaciers to have smaller dataset
    rgidf = rgidf[rgidf.RGIId.isin([
#        'RGI50-11.B9004'])]
#        'RGI50-11.A54L36n', # Fiescher (Shaded)
       'RGI50-11.B4312n-1',  # Rhone
#        'RGI50-11.B5622n'
#        'RGI50-11.B5616n-1',  # Findelen
#        'RGI50-11.A55F03',  # Plaine Morte
#        'RGI50-11.C1410',  # Basodino
#        'RGI50-11.A10G05',  # Silvretta
#        'RGI50-11.B3626-1'  # Gr. Aletsch
       ])]


 #   log.info('Number of glaciers: {}'.format(len(rgidf)))

    #Go - initialize working directories
    gdirs = init_glacier_regions(rgidf, reset= False, force=True)

    
    task_list = [#tasks.crop_satdata_to_glacier, # output:
                                                        # sentinel.nc,
                                                        #solar_angles.nc
                                                        # dem_ts.nc
                 #tasks.ekstrand_correction, # output: ekstrand.nc
                 #tasks.cloud_masking, # ouput: cloud_masked.nc
                 #tasks.remove_sides, # output: sentinel_temp.nc
                 #tasks.asmag_snow_mapping,
                 #tasks.naegeli_snow_mapping,
                 #tasks.naegeli_improved_snow_mapping,
                 tasks.plot_results
                 ]
    for task in task_list:
        execute_entity_task(task, gdirs)


