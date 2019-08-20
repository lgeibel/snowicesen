"""
=====================
validate_snow_mapping
=====================
Entity task to read in manually retrieved snow lines
where available and create snow maps to compare with the
snow maps retrieved by the snowicesen algorithm
"""

from __future__ import absolute_import, division

import os
import shutil
import numpy as np
import logging
import xarray as xr
from crampon import entity_task
import snowicesen.cfg as cfg
from snowicesen.snow_mapping import get_SLA_asmag
import geopandas
import fiona
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool
import glob
import matplotlib.pyplot as plt
import rasterio.mask
from rasterio import features

# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def create_manual_snow_map(gdir):
    """
    Create snow map from manually delined snow lines
    by intersecting them with the glacier outlines
    Output is then stores in manual_snow_map.nc

    :param gdir:
    :return:
    """
    # List all .shp file in RGB folder
    file_list = glob.glob(r"/scratch_net/vierzack03_third/geibell/snowicesen/RGB_Polygons/*.shp")
    RGB_list =  glob.glob(r"/scratch_net/vierzack03_third/geibell/snowicesen/RGB_Polygons/*.tif")
   # print(RGB_list)
   # print(file_list)
    # write all dates into list
    date_list = [item.split("/")[-1].split("_")[1].split(".")[0] for item in file_list]
    # write all RGIIds for which we have snow maps into list
    RGIId_list = [item.split("/")[-1].split("_")[0] for item in file_list]
    print(len(RGIId_list))
    # write all RGIIds for which we have RGBs into list
    RGIId_list_RGB = [item.split("/")[-1].split("_")[0] for item in RGB_list]
    #print(gdir.id)
    if gdir.id in RGIId_list:
        #print(gdir.id)
        # read glacier outlines from GlacierDirectory
        glacier = geopandas.read_file(gdir.get_filepath('outlines'))
        # list of indices in list for this glacier:
        RGB_index = [i for i, e in enumerate(RGIId_list_RGB) if e == gdir.id]
        index = [i for i, e in enumerate(RGIId_list) if e == gdir.id]

        # iterate over all dates that have snow line information
        for ind, item in enumerate([date_list[i] for i in index]):
            # Read snow line for this date:
            try:
                snow_area = geopandas.read_file(file_list[index[ind]])
            except:
                continue
            # Intersect outlines with polygon from hand-edited
            try:
                res_intersection = geopandas.overlay(glacier, snow_area, how='intersection')
            except KeyError:
                print('Empty Overlay')
                snow_area.plot()
                plt.show()
                continue

            # Safe to shapefile:
            #print(gdir, item)

            inras = RGB_list[RGB_index[ind]]
            shapefile_name = r"/scratch_net/vierzack03_third/geibell/snowicesen/shapefile.shp"
            try:
                res_intersection.to_file(shapefile_name)
                outras = r"/scratch_net/vierzack03_third/geibell/snowicesen/RGB_Polygons/rasterized.tif"
                with fiona.open(shapefile_name, "r") as shapefile:
                    features = [feature["geometry"] for feature in shapefile]

                with rasterio.open(inras) as src:
                    out_image, out_transform = rasterio.mask.mask(src, features,
                                                              filled=True, nodata=0)
                print(cfg.PARAMS['count'])
            except:
                print("VALUE ERROR! - skip this file")
                continue
                            
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform,
                             "count": 1})
            # transform to raster with snow = 1, ice = 0
            out_image[out_image == 65535] = 0
            out_image[out_image > 0] = 1
            snow_man_ras = out_image[1,:,:]
            SLA_man = get_SLA_asmag(gdir, snow_man_ras)
            if SLA_man is None:
                SLA_man = 0

            # safe to netCDF file:
            # Check if netCDF already exists for this glacier.
           # if not os.path.exists(gdir.get_filepath('snow_cover_man')):
            if not 'snow_man_new' in locals():
                print("snow_cover_man.nc does not exist, creating new file:")
                # Open snow_cover.nc and use it as template for file structure
                try:
                    snow_auto = xr.open_dataset(gdir.get_filepath('snow_cover'))
                    print("Copying snow_cover.nc structure:")
                except FileNotFoundError:
                    # No file after re-running the code with correct cloud cover:
                    print("No file after re-running code")
                    return
                # Remove double/non-unique dates:
                _, i = np.unique(snow_auto['time'], return_index=True)
                snow_auto = snow_auto.isel(time=i)
                # Drop time entries
                snow_auto = snow_auto.drop([time_id for time_id in snow_auto['time'].values][:-1],
                                       dim='time').squeeze('time', drop='True')
                # Drop Model Coordinates:
                snow_auto = snow_auto.drop(['naegeli_orig', 'naegeli_improv'],
                               dim='model').squeeze('model', drop=True)

                # Add time label:
                snow_auto = snow_auto.assign_coords(time=int(item))
                snow_man = snow_auto.copy()
                snow_man = snow_man.expand_dims('time')
                #print("New file: ", snow_man)

            else:
                # nc. already exists, create new xarray Dataset and concat
                # to obtain new time dimension
                #snow_man = xr.open_dataset(gdir.get_filepath('snow_cover_man'))
                snow_man = snow_man_new
                print('Existing file has the following dates: ' ,snow_man['time'].values)
                if int(item) not in snow_man['time'].values:
                    snow_new_ds = snow_man.copy()
                    snow_new_ds = snow_new_ds.isel(time=0)
                    snow_new_ds.coords['time'] = np.array(int(item))
                   # print("Now assining new date: snow_new_ds = ", snow_new_ds)
                    snow_man = xr.concat([snow_man, snow_new_ds], dim='time')
                    #print("Append to existing file: ", snow_man)

            cfg.PARAMS['count']=cfg.PARAMS['count']+1
                # Assign snow map to ASMAG model snow_map variable
            snow_man_new = snow_man.copy(deep = True)
            #print("Copy of snow_man = ",snow_man_new)
            snow_man_new['snow_map'].loc[dict(time=int(item))] = \
                np.zeros([snow_man_ras.shape[0], snow_man_ras.shape[1]])
            #snow_man_new.sel(time=int(item)).snow_map.plot()
            #plt.show()
                # weird bug, first need to assign zeros array...
            snow_man_new['snow_map'].loc[dict(time=int(item))] = \
                snow_man_ras

                # Assign NaN Value for cloudy/non-glacier pixels:
            snow_man_new = snow_man_new.where(snow_man_new['snow_map'] != 0)
            #print('After assigning / Maskin NaN values', snow_man_new)

            # Assign SLA:
            snow_man_new['SLA'].loc[dict(time=int(item))] = SLA_man
            #print("After assigning SLA: ", snow_man_new)

            #snow_man_new.sel(time=int(item)).snow_map.plot()
            #plt.show()

            ## Safe file:
            #snow_man.close()
           # snow_man_new.close()
            print("Snow_man_new at the end of the iteration: ",snow_man_new)

        
        #if os.path.isfile(gdir.get_filepath('snow_cover_man')):
        #    os.remove(gdir.get_filepath('snow_cover_man'))
        print("this is what we are saving to the snow_cover.nc file: ", snow_man_new)
        snow_man_new.to_netcdf(gdir.get_filepath('snow_cover_man'), 'w')
            #snow_man_new.to_netcdf('snow_cover_test.nc', 'w')
        snow_man_new.close()
        snow_man.close()

        test = xr.open_dataset(gdir.get_filepath('snow_cover_man'))
            #test = xr.open_dataset('snow_cover_test.nc')
        print("Test look at snow_cover_man.nc: ", test)
        print("date = ", int(item))
        #test.sel(time=int(item)).snow_map.plot()
        #plt.show()

    print("continue")


@entity_task(log)
def create_confusion_matrix(gdir):
    """
    Create confusion matrix for each date of this glacier

    :param gdir:
    :return:
    """
    # Check if validation dataset is available:
    # Read Validation data set
    try:
        snow = xr.open_dataset(gdir.get_filepath('snow_cover'))
        time_cnt = snow.time.values.size 
        #fg.PARAMS['count'] = cfg.PARAMS['count']+time_cnt
        #rint(cfg.PARAMS['count'])
    except FileNotFoundError:
        # Skip this glacier if no validation data set is available
        print("No snow_cover.nc available")
        return

    # Read Original dataset:
    try:
        snow_man = xr.open_dataset(gdir.get_filepath('snow_cover_man'))
        time_cnt = snow_man.time.values.size
        cfg.PARAMS['count'] = cfg.PARAMS['count'] + time_cnt
        print(cfg.PARAMS['count'])
    except FileNotFoundError:
        print("No snow_cover_man.nc file")
        return
    # Read Glacier outlines:
    ########  !!!! TODO !!! ########
    # Use sentinel_temp instead of sentinel to get cloud_corrected scene already   
    try: 
        sentinel = xr.open_dataset(gdir.get_filepath('cloud_masked'))
    except FileNotFoundError:
        print("No cloud_masked.nc file")
        return
    # Remove double/non-unique dates:
    _, i = np.unique(snow_man['time'], return_index=True)
    snow_man = snow_man.isel(time=i)
    _, i = np.unique(snow['time'], return_index=True)
    snow = snow.isel(time=i)
    _, i = np.unique(sentinel['time'], return_index=True)
    sentinel = sentinel.isel(time=i)

    # Iterate over Dates for which we have validation data:
    for day in snow_man.time.values:
        snow_day = snow.sel(time=day)
        try:
            snow_man_day = snow_man.sel(time=day)
        except KeyError:
            print("No Automatic snow map for this scene")
            continue
        #plt.subplot(1,3,1)
        #snow_day.sel(model='asmag').snow_map.plot()
        #plt.subplot(1,3,2)
        #snow_man_day.snow_map.plot()
        #plt.subplot(1,3,3)
        sentinel_day = sentinel.sel(time=day)
        #sentinel_day.sel(band = 'B08').img_values.plot()
        #plt.show()

        # ASMAG algorithm:
        cohen_asmag, C_asmag = get_cohens_kappa(
            snow_day, snow_man_day, sentinel_day, 'asmag')
        cohen_naegeli_orig, C_naegeli_orig = get_cohens_kappa(
            snow_day, snow_man_day, sentinel_day, 'naegeli_orig')
        cohen_naegeli_improv, C_naegeli_improv = get_cohens_kappa(
            snow_day, snow_man_day, sentinel_day, 'naegeli_improv')


        ###### Check if variables already exist, if not, create them:
        #  Cohen's Kappa:
        try:
            snow_man['kappa_asmag'].loc[dict(time=int(day))] = cohen_asmag

        except KeyError:
            # Key doesn't exist yet, create new:
            snow_man['kappa_asmag'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['kappa_naegeli_orig'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['kappa_naegeli_improv'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))

            # Confusion matrix (entries a,b,c,d seperately):
            snow_man['asmag_a'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_orig_a'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_improv_a'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))

            snow_man['asmag_b'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_orig_b'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_improv_b'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))

            snow_man['asmag_c'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_orig_c'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_improv_c'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))

            snow_man['asmag_d'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_orig_d'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))
            snow_man['naegeli_improv_d'] = (['time'], np.zeros((snow_man.time.size), dtype=np.float))

        # Assign Values:
        # Cohen's Kappa
        snow_man['kappa_asmag'].loc[dict(time=int(day))] = cohen_asmag
        snow_man['kappa_naegeli_orig'].loc[dict(time=int(day))] = cohen_naegeli_orig
        snow_man['kappa_naegeli_improv'].loc[dict(time=int(day))] = cohen_naegeli_improv

        # Confusion Matrix:
        snow_man['asmag_a'].loc[dict(time=int(day))] = C_asmag[0]
        snow_man['naegeli_orig_a'].loc[dict(time=int(day))] = C_naegeli_orig[0]
        snow_man['naegeli_improv_a'].loc[dict(time=int(day))] = C_naegeli_improv[0]

        snow_man['asmag_b'].loc[dict(time=int(day))] = C_asmag[1]
        snow_man['naegeli_orig_b'].loc[dict(time=int(day))] = C_naegeli_orig[1]
        snow_man['naegeli_improv_b'].loc[dict(time=int(day))] = C_naegeli_improv[1]

        snow_man['asmag_c'].loc[dict(time=int(day))] = C_asmag[2]
        snow_man['naegeli_orig_c'].loc[dict(time=int(day))] = C_naegeli_orig[2]
        snow_man['naegeli_improv_c'].loc[dict(time=int(day))] = C_naegeli_improv[2]

        snow_man['asmag_d'].loc[dict(time=int(day))] = C_asmag[3]
        snow_man['naegeli_orig_d'].loc[dict(time=int(day))] = C_naegeli_orig[3]
        snow_man['naegeli_improv_d'].loc[dict(time=int(day))] = C_naegeli_improv[3]

    # Make sure 'snow_cover_man_full.nc' is only written when there is statistics data:
    try:
        print(snow_man.kappa_asmag)
        # Safe file (strange permission error, different than the "normal
        #  one", so we have to save to a new file....:
        snow_man.to_netcdf(gdir.get_filepath('snow_cover_man_full'), 'w')
        snow_man.close()
    except AttributeError:
        print('Attribute Error, no data for the whole glacier')
        return

def get_cohens_kappa(snow_day, snow_man_day, sentinel, model):
    """
    Calculate Confusion Matrix and Cohen's Kappa
    from two vectors of observed and modeled snow
    cover (snow = 1, ice =0)

    Parameters
    ----------
    snow_day: xarray Dataset: modeled snow cover for current date
    snow_man_day: xarray Dataset: Manually delineated snow cover for
                current date
    sentinel: xarray Dataset: Cloud masked, preprocessed
            Sentinel Images for current day
    model: str:  'asmag', 'naegeli_orig' or 'naegeli_improv'

    Returns
    -------
    cohen: int: Cohen's Kappa of the scene
    C: Confusion matrix of the scene
    """
    # as np.array
    outline = sentinel.sel(band = 'B02').img_values.values
    #print(outline)
    #plt.imshow(outline)
    #plt.show()
    #set all values > 0 to 1 (to extract glacier outlines and 
    # cloud-free pixels)
    outline[outline > 0] = 1
    outline = np.reshape(outline, outline.size)

    snow_obs = snow_man_day.snow_map.values
   # plt.imshow(snow_obs)
   # plt.show()
    snow_obs = np.reshape(np.nan_to_num(snow_obs), snow_obs.size)
    snow_obs = snow_obs[outline == 1]
   # print("snow_ops = ", np.unique(snow_obs))
    snow_asmag = snow_day.sel(model=model).snow_map.values
   # plt.imshow(snow_asmag)
    snow_asmag = np.reshape(np.nan_to_num(snow_asmag), snow_asmag.size)
    snow_model = snow_asmag[outline == 1]
   # print(" snow_model = ", np.unique(snow_model))
    
    

    C = confusion_matrix(snow_obs, snow_model).ravel()
    a = C[0]
    b = C[1]
    c = C[2]
    d = C[3]
    sum_C = a + b + c + d
    p_0 = (a + d) / sum_C  # Accuracy
    p_snow = (a + b) / sum_C * (a + c) / sum_C
    p_ice = (c + d) / sum_C * (b + d) / sum_C
    p_e = p_snow + p_ice
    cohen = (p_0 - p_e) / (1 - p_e)

    return cohen, C





