"""
==============
recreate_snow_map.py
==============
This is a modified version of setup_gdirs.py used to re-run some sections of the processing line after some bugs in the code and data output were resolved (be careful to be in the correct working directory to avoid overwriting data)^

"""

import warnings

warnings.filterwarnings('ignore')
from snowicesen import cfg
from snowicesen import tasks
import geopandas as gpd
import logging
from snowicesen.workflow import init_glacier_regions
from snowicesen.utils import download_all_tiles
import glob
from oggm.workflow import execute_entity_task
from datetime import timedelta
from snowicesen.utils import datetime_to_int, int_to_datetime, extract_metadata

logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize(r"/scratch_net/vierzack03_third/geibell/snowicesen/snowicesen_params.cfg")
cfg.PATHS['working_dir'] = r"/scratch_net/vierzack03_third/geibell/snowicesen/SWISS_NEW"
cfg.PARAMS['date'] =  20150508, 20191117
# Caution: In crampon.utils.GlacierDirectory._init cfg.initialize is
# called again --> change path there as well! EDIT: only necessary in Windows

if __name__ == '__main__':
    # Shapefile with Glacier Geometries:
    rgidf = gpd.read_file(
        r"/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")

    # Ignore all glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]
    #    rgidf = rgidf.sample(n=10)

    # Only keep those glaciers for which we have validation dataset:

    # Get list of dates:
    #
    # .....
    # List all .shp file in RGB folder
    file_list = glob.glob(r"/scratch_net/vierzack03_third/geibell/snowicesen/RGB_Polygons/*.shp")
    print(file_list)
    RGB_list =  glob.glob(r"/scratch_net/vierzack03_third/geibell/snowicesen/RGB_Polygons/*.tif")
    # write all dates into list
    date_list = [item.split("/")[-1].split("_")[1].split(".")[0] for item in file_list][:-1]
    # write all RGIIds for which we have snow maps into list
    RGIId_list = [item.split("/")[-1].split("_")[0] for item in file_list][:-1]

    # Here start iterating for date: for date start to date end:
    start_date_var, end_date = int_to_datetime(cfg.PARAMS['date'])
    end_date_var = start_date_var + timedelta(days=1)
    cfg.PARAMS['count'] = 0
    #gdirs = init_glacier_regions(rgidf, reset=False, force=False)
    # Iterate over all days, checking between start and enddate
    while end_date_var <= end_date:
        cfg.PARAMS['date'] = datetime_to_int(start_date_var,
                                             end_date_var)
       # print(cfg.PARAMS['date'])
        # Only process days that are in list
        if str(cfg.PARAMS['date'][0]) in date_list:
           # print("Date = ", cfg.PARAMS['date'][0])
            
            # Adjust gdirs: for which glacier is the current date in the
            # validation set?
            # Get index of glacier with this date:
            RGIId_set = set(RGIId_list)
            RGIId_list_curr = list(RGIId_set)
            #print(RGIId_list_curr)
            RGIId_list_curr = RGIId_list[date_list.index(str(cfg.PARAMS['date'][0]))]
            RGIId_indices = [i for i,val in enumerate(date_list) if val == str(cfg.PARAMS['date'][0])] 
            RGIId_list_curr = [RGIId_list[i] for i in RGIId_indices]
            print(RGIId_list_curr)
           # print(hello)
          

            rgidf_curr = rgidf[rgidf.RGIId.isin(RGIId_list_curr)]
           # print(rgidf_curr)

            # Go - initialize working directories for this date:
            gdirs = init_glacier_regions(rgidf_curr, reset=False, force=True)
            cfg.PARAMS['count']=cfg.PARAMS['count']+ len(RGIId_list_curr)
            #cfg.PARAMS['count'] = 0
            print("Done with init_glacier_regions")
            task_list = [
                 #   tasks.crop_satdata_to_glacier,
                    tasks.cloud_masking,  # ouput: cloud_masked.nc
                    tasks.remove_sides,  # output: sentinel_temp.nc
                    tasks.asmag_snow_mapping,
                    tasks.naegeli_snow_mapping,
                    tasks.naegeli_improved_snow_mapping,
                    #tasks.plot_results
                ]
            for task in task_list:
                execute_entity_task(task, gdirs)
            
        start_date_var = start_date_var + timedelta(days=1)
        end_date_var = start_date_var + timedelta(days=1)
        print(cfg.PARAMS['count'])


