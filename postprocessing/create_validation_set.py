"""
Create Validtaion Data Set:
===========================
Script to read  all available Glaciers, picks a sample weighted by glacier area and from this sample picks a random date for which data is available. Th output is then stored in a GeoTiff file with the filename as glacier_id_date.tif in the specified output directory
"""
import warnings
warnings.filterwarnings('ignore')
from snowicesen import cfg
import geopandas as gpd
import logging
from snowicesen.workflow import init_glacier_regions
import xarray as xr
import random
import rasterio
import os
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize("/scratch_net/vierzack03_third/geibell/snowicesen/snowicesen_params.cfg")
cfg.PATHS['dem_dir'] = "/scratch_net/vierzack03_third/geibell/snowicesen/data/DEM/SWISSALTI3D_2018"

    # something new in OGGM that we are not yet able to handle
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['continue_on_error'] = False
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['mp_processes'] = 4

if __name__ == '__main__':
    rgidf = gpd.read_file("/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")
    # Ignore all values glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]
   # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B8315n'])]  ##########
    gdirs = init_glacier_regions(rgidf, reset= False, force=False)
    # Total area
    area_tot = sum(rgidf.Area)
    # Weight by area:
    weight = (rgidf.Area/area_tot).tolist()
    rgidf.RGIId

    samples = 600  # number of samples for this list #########
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

        #read
        sentinel = xr.open_dataset(gdir.get_filepath('sentinel'))
        snow = xr.open_dataset(gdir.get_filepath('snow_cover'))
        # chose random date 
        rand_date = random.choice(snow.time.values) 
        if rand_date > 20171129:
            glacier = gpd.read_file(gdir.get_filepath('outlines'))
            DEM = rasterio.open(os.path.join(cfg.PATHS['dem_dir'], 'swissAlti3D_lv03_ln02_r2018_bilinear10m_UTM32.tif'))
            out_meta = DEM.profile
            print(gdir)
            #print(sentinel.img_values.attrs)
            #sentinel.sel(time=rand_date, band= 'B08').img_values.plot
            #plt.imshow(sentinel.sel(time=rand_date, band='B08').img_values.values)
            #plt.show()
        

            # Update Metadata
            out_meta['crs'] = sentinel.img_values.attrs['crs']
            out_meta['count'] = 3
            try:
                out_meta['transform'] = rasterio.transform.guard_transform(sentinel.img_values.attrs['transform'])
            except:
                # Transform does not exist: smething wrong with glacier, has black stripes
                # --> dont use this data for validation
                continue
            out_meta['height'] = sentinel.img_values.y.values.size
            out_meta['width'] = sentinel.img_values.x.values.size
            out_meta['dtype'] = 'uint16'
            out_meta['nodata'] = 65535

            file_list = ['B02','B8A', 'B08']

            # Write to GeoTiff
            with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'RGB', str(rand_RGIId+'_'+ str(rand_date)+'.tif')), 'w', **out_meta) as dst:
                for id,layer in enumerate(file_list, start=1):
                    dst.write_band(id, sentinel.sel(time=rand_date, band=layer).img_values.values)


        #print(sentinel.time)





