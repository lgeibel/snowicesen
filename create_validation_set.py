import warnings
warnings.filterwarnings('ignore')
from snowicesen import cfg
from snowicesen import tasks
import geopandas as gpd
import logging
from snowicesen.workflow import init_glacier_regions
import xarray as xr
import random
from snowicesen.utils import download_all_tiles
from oggm.workflow import execute_entity_task
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os
from snowicesen.utils import datetime_to_int, int_to_datetime, extract_metadata


logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize("/scratch_net/vierzack03_third/geibell/snowicesen/snowicesat_params.cfg")
# Caution: In crampon.utils.GlacierDirectory._init cfg.initialize is called again --> change path there as well!
#cfg.PATHS['working_dir'] = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\snowicesat\SWISS"
cfg.PATHS['dem_dir'] = "/scratch_net/vierzack03_third/geibell/snowicesen/data/DEM/SWISSALTI3D_2018"

    # something new in OGGM that we are not yet able to handle
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['mp_processes'] = 4

if __name__ == '__main__':
    samples = 30 # number of samples that we want
    rgidf = gpd.read_file("/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")
    # Ignore all values glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]

    rgidf = rgidf.sample(n=100)
    gdirs = init_glacier_regions(rgidf, reset= False, force=False)
    # Total area:
    area_tot = sum(rgidf.Area)
    # Weight by area:
    weight = (rgidf.Area/area_tot).tolist()
    rgidf.RGIId

    my_list = []
    for num, name in enumerate(rgidf.RGIId):
        my_list.append([name] * round(weight[num] * samples))
    # flatten list of lists:
    my_list = [item for sublist in my_list for item in sublist]

    file_list = []
    for i in range(0,samples):
        rand_RGIId = random.choice(my_list)
        # get index of rgIId in gdirs list:
        index = [gdir.id for gdir in gdirs].index(rand_RGIId)
        # get gdir for given index:
        gdir = gdirs[index]

        # Read file, get list of all available dates:
        snow = xr.open_dataset(gdir.get_filepath('snow_cover'))

        # get a random date from all dates for this glacier:
        rand_date  = random.choice(snow.time.values)
        print(snow.time.values)
        print(rand_date)

        # Read corresponding sentinel file:
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel_temp'))

        # write into RGB image
        b04 = sentinel.sel(band = 'B04', time = rand_date).img_values.values/10000
        b03 = sentinel.sel(band = 'B03', time = rand_date).img_values.values/10000
        b02 = sentinel.sel(band = 'B02', time = rand_date).img_values.values/10000

        rgb_image = np.array([b04, b03, b02]).transpose((1,2,0))
        rgb_image[rgb_image > 1] = 1

        # RGB to file:
        filename = os.path.join(cfg.PATHS['working_dir'], 'RGB', str(rand_RGIId) +'_' + str(rand_date)+'.jpg')
        scipy.misc.imsave(filename, rgb_image)




