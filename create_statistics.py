"""
==============
create_statistics.py
==============
Create Statistical Evaluation (Confusion Matrix, Kappa, etc.) and plot results
"""

import warnings

warnings.filterwarnings('ignore')
from snowicesen import cfg
from snowicesen import tasks
from snowicesen.utils import two_d_scatter
import geopandas as gpd
import logging
from snowicesen.workflow import init_glacier_regions
import xarray as xr
import numpy as np
import snowicesen.utils as utils
from snowicesen.utils import download_all_tiles
from oggm.workflow import execute_entity_task
from datetime import timedelta, date
from datetime import datetime
from snowicesen.utils import datetime_to_int, int_to_datetime, extract_metadata
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import pandas as pd
import numpy.ma as ma
import richdem as rd
import itertools
import pickle
from scipy import stats

logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

cfg.initialize(r"/scratch_net/vierzack03_third/geibell/snowicesen/snowicesen_params.cfg")
# Caution: In crampon.utils.GlacierDirectory._init cfg.initialize is
# called again --> change path there as well! EDIT: only necessary in Windows

if __name__ == '__main__':
    # Shapefile with Glacier Geometries:
    rgidf = gpd.read_file(
            r"/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/rgi_copy_status.shp")
    # Shapefile with Sentinel -2 tiles
    tiles = gpd.read_file(
        r"/scratch_net/vierzack03_third/geibell/snowicesen/data/outlines/sentinel2_tiles_world.shp")
    # List of TileIDs that intersect with Swiss Glacier Inventory
    # (needs to be adjusted if using other region)
    tiles = ['32TLS', '32TLR', '32TMS', '32TMR',
             '32TNT', '32TNS']

    # Ignore all glaciers smaller than 0.1 km^2
    rgidf = rgidf.loc[rgidf['Area'] > 0.1]
    #    rgidf = rgidf.sample(n=10)

    # Only keep those glaciers to have smaller dataset
#    rgidf = rgidf[rgidf.RGIId.isin([
#        'B8529n-3'
#        #        'RGI50-11.B9004'])]
#        #        'RGI50-11.A54L36n', # Fiescher (Shaded)
#        'RGI50-11.B4312n-1'  # Rhone,
#        #        'RGI50-11.B8315n' # Corbassiere
#        #        'RGI50-11.B5616n-1',  # Findelen
#        #        'RGI50-11.A55F03',  # Plaine Morte
#        #        'RGI50-11.C1410',  # Basodino
#        #        'RGI50-11.A10G05',  # Silvretta
#        #        'RGI50-11.B3626-1'  # Gr. Aletsch
#    ])]

    #   log.info('Number of glaciers: {}'.format(len(rgidf)))

    # Go - initialize working directories
    gdirs = init_glacier_regions(rgidf, reset=False, force=True)
    print("Done with init_glacier_regions")
    cfg.PARAMS['count'] = 0
    # Entity task:
#   Uncomment, only needs to be run once
#    task_list = [tasks.create_confusion_matrix
#                 ]
#    for task in task_list:
#        execute_entity_task(task, gdirs)

#    print(cfg.PARAMS['count'])
    # Loop over all gdirs for which we have validation data:
    kappa_asmag_sum = 0
    kappa_naegeli_sum = 0
    kappa_naegeli_improv_sum = 0
    C_asmag = np.zeros((4,1))
    C_naegeli = np.zeros((4,1))
    C_naegeli_improv = np.zeros((4,1))
    kappa_as_arr = []
    kappa_na_arr = []
    kappa_nai_arr = []
    kappa_as_arr_cloud = []
    kappa_na_arr_cloud = []
    kappa_nai_arr_cloud = []

    snow_cover_arr = []
    cloud_cover_arr = []
    slope_arr = []
    SLA_man_arr =[]
    SLA_as_arr = []
    SLA_na_arr = []
    SLA_nai_arr = []

    area_arr =[]
    extent_arr =[]
    aspect_arr = []
    doy_arr = []
    vert_ext_arr =[]
    scenes_arr =[]
    entries = 0

    for glacier in gdirs:
        try:
            snow_man = xr.open_dataset(glacier.get_filepath('snow_cover_man_full'))
            snow = xr.open_dataset(glacier.get_filepath('snow_cover'))
            #print(snow_man.time.values)
            cloud_masked = xr.open_dataset(glacier.get_filepath('cloud_masked'))
            #print(cloud_masked.time.values)
            sentinel = xr.open_dataset(glacier.get_filepath('sentinel'))
            dem = xr.open_dataset(glacier.get_filepath('dem_ts'))
            
        except FileNotFoundError:
            # Skip this glacier if no validation data set is available
            print('No data for comparison')
            continue
        
        # sum up all kappas
        kappa_asmag_sum = kappa_asmag_sum+ snow_man.kappa_asmag.sum(dim='time').values
        kappa_naegeli_sum = kappa_naegeli_sum+ snow_man.kappa_naegeli_orig.sum(dim='time').values
        kappa_naegeli_improv_sum = kappa_naegeli_improv_sum+ snow_man.kappa_naegeli_improv.sum(dim='time').values

        _,i = np.unique(cloud_masked['time'], return_index = True)
        cloud_masked = cloud_masked.isel(time=i)
        

        # Get snow_cover, cloud_cover, date and slope information for each day:
        for day in snow_man.time.values:
            # take outline, reshape to 1-D Array
            outline = sentinel.sel(time=day, band='B08').img_values.values.flatten()

            try:
                cloud_cover = cloud_masked.sel(time=day, band= 'B08').img_values.values.flatten()
                cloud_cover = 1-(cloud_cover[cloud_cover >0].size/outline[outline>0].size)
                cloud_cover_arr.append(cloud_cover)
                kappa_as_arr_cloud.append(snow_man.sel(time=day).kappa_asmag.values)
                kappa_na_arr_cloud.append(snow_man.sel(time=day).kappa_naegeli_orig.values)
                kappa_nai_arr_cloud.append(snow_man.sel(time=day).kappa_naegeli_improv.values)
            except KeyError:
                continue

        # write Day of Year (doy) into array:
            end_date = date(int(str(day)[0:4]), int(str(day)[4:6]),
                    int(str(day)[6:8]))
            doy = end_date.timetuple().tm_yday
            doy_arr.append(doy)
        # Write all kappas into an array:
            kappa_as_arr.append(snow_man.sel(time=day).kappa_asmag.values)
            kappa_na_arr.append(snow_man.sel(time=day).kappa_naegeli_orig.values)
            kappa_nai_arr.append(snow_man.sel(time=day).kappa_naegeli_improv.values)

           #print(snow_man.sel(time=day).SLA.values[0][0], snow.sel(time=day, 
           #    model = 'asmag').SLA.values[0][0], snow.sel(time=day, model= 'naegeli_orig').SLA.values[0][0], snow.sel(time=day, model = 'naegeli_improv').SLA.values[0][0])
           #   
            SLA_man = snow_man.sel(time=day).SLA.values[0][0]
            SLA_as = snow.sel(time=day, model='asmag').SLA.values[0][0]
            SLA_na = snow.sel(time=day, model='naegeli_orig').SLA.values[0][0]
            SLA_nai = snow.sel(time=day, model='naegeli_improv').SLA.values[0][0]
            
            print(glacier, glacier.rgi_area_km2)
            area_arr.append(glacier.rgi_area_km2)
            
            if SLA_as.size > 1:
                #print("We have a problem here :", SLA_as, SLA_na,SLA_nai)
                SLA_as = SLA_as[0]
                SLA_na = SLA_na[0]
                SLA_nai = SLA_nai[0]
            SLA_man_arr.append(SLA_man)
            SLA_as_arr.append(SLA_as)
            SLA_na_arr.append(SLA_na)
            SLA_nai_arr.append(SLA_nai)

            snow_cover = snow_man.sel(time=day).snow_map.values.flatten()

            # Snow Covered area = Size of snow covered array/Size of Glacier
            snow_cover = snow_cover[snow_cover==1].size/outline[outline>0].size
            
            #print(snow_cover)
            snow_cover_arr.append(snow_cover)

            # Get average slope
            dem_ts = dem.isel(time=0,band=0).height_in_m.values
            # Get altitude extent:
            print(dem_ts[dem_ts < 5000].max(), dem_ts[dem_ts>0].min())
            extent = dem_ts[dem_ts < 5000].max()-dem_ts[dem_ts > 0].min()
            print(extent)
            z_bc = utils.assign_bc(dem_ts)
            dx = 10
            # Compute finite differences
            slope_x = (z_bc[1:-1, 2:]-z_bc[1:-1,:-2])/(2*dx)
            slope_y = (z_bc[2:, 1:-1]-z_bc[:-2,1:-1])/(2*dx)
            slope = np.arctan(np.sqrt(slope_x**2+slope_y**2))
            aspect = np.arctan2(slope_y, slope_x)

            # Average slope in degrees
            slope = np.degrees(np.mean(slope[slope>0]))
            # Average Aspect in degreed:
            aspect = np.degrees(np.mean(aspect[aspect>0]))
            #rint(slope)
            slope_arr.append(slope)
            extent_arr.append(extent)
            aspect_arr.append(aspect)
        


        # Calculate Confusion Matrix of all scenes:
        C_asmag[0]=C_asmag[0]+ snow_man.asmag_a.sum(dim='time').values
        C_asmag[1]=C_asmag[1]+ snow_man.asmag_b.sum(dim='time').values
        C_asmag[2]=C_asmag[2]+ snow_man.asmag_c.sum(dim='time').values
        C_asmag[3]=C_asmag[3]+ snow_man.asmag_d.sum(dim='time').values

        C_naegeli[0]=C_naegeli[0]+ snow_man.naegeli_orig_a.sum(dim='time').values
        C_naegeli[1]=C_naegeli[1]+ snow_man.naegeli_orig_b.sum(dim='time').values
        C_naegeli[2]=C_naegeli[2]+ snow_man.naegeli_orig_c.sum(dim='time').values
        C_naegeli[3]=C_naegeli[3]+ snow_man.naegeli_orig_d.sum(dim='time').values

        C_naegeli_improv[0]=C_naegeli_improv[0]+ snow_man.naegeli_improv_a.sum(dim='time').values
        C_naegeli_improv[1]=C_naegeli_improv[1]+ snow_man.naegeli_improv_b.sum(dim='time').values
        C_naegeli_improv[2]=C_naegeli_improv[2]+ snow_man.naegeli_improv_c.sum(dim='time').values
        C_naegeli_improv[3]=C_naegeli_improv[3]+ snow_man.naegeli_improv_d.sum(dim='time').values

    #####   PLOTS: #####


    # Snow Cover vs. Kappa:
    #print("manual :",SLA_man_arr)
    #print("ASMAG  :", SLA_as_arr)
    plt.figure(1, figsize=(15,5))
    plt.suptitle('Cohens Kappa vs. Snow Cover and Glacier Area')
    plt.subplot(1,3,1)

    cmap = matplotlib.cm.get_cmap('YlGnBu')
    normalize = matplotlib.colors.Normalize(vmin=min(area_arr),
            vmax = max(area_arr))
    colors = [cmap(normalize(value)) for value in area_arr]
    plt.scatter(snow_cover_arr, kappa_as_arr, s=10,color=colors,linewidth=0.5, edgecolor='blue')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Glacier Area')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Snow Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('ASMAG')

    plt.subplot(1,3,2)
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    normalize = matplotlib.colors.Normalize(vmin=min(area_arr),
            vmax = max(area_arr))
    colors = [cmap(normalize(value)) for value in area_arr]
    plt.scatter(snow_cover_arr, kappa_na_arr, s=10,color=colors,linewidth=0.5, edgecolor='red')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Glacier Area')
    plt.xlim(0,1)

    plt.ylim(0,1)
    plt.xlabel('Snow Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('Naegeli')

    plt.subplot(1,3,3)

    cmap = matplotlib.cm.get_cmap('YlGn')
    normalize = matplotlib.colors.Normalize(vmin=min(area_arr),
            vmax = max(area_arr))
    colors = [cmap(normalize(value)) for value in area_arr]
    plt.scatter(snow_cover_arr, kappa_nai_arr, s=10,color=colors,linewidth=0.5, edgecolor='green')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Glacier Area')
    plt.xlim(0,1)

    plt.ylim(0,1)
    plt.xlabel('Snow Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('Naegeli_Improved')
    plt.tight_layout(rect = [0,0.03,1, 0.95])
    plt.savefig('PLOTS/kappa_vs_snowcover_area.png')

    # Slope vs. Kappa:
    # Snow Cover vs. Kappa:
    plt.figure(2, figsize=(15,5))
    plt.suptitle('Cohens Kappa vs. Snow Cover and Cloud Cover')
    plt.subplot(1,3,1)

    cmap = matplotlib.cm.get_cmap('YlGnBu')
    normalize = matplotlib.colors.Normalize(vmin=min(cloud_cover_arr),
            vmax = max(cloud_cover_arr))
    colors = [cmap(normalize(value)) for value in cloud_cover_arr]
    plt.scatter(snow_cover_arr, kappa_as_arr, s=10,color=colors,linewidth=0.5, edgecolor='blue')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Cloud Cover')
    plt.xlim(0,1)

    plt.ylim(0,1)
    plt.xlabel('Snow Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('ASMAG')

    plt.subplot(1,3,2)
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    normalize = matplotlib.colors.Normalize(vmin=min(cloud_cover_arr),
            vmax = max(cloud_cover_arr))
    colors = [cmap(normalize(value)) for value in cloud_cover_arr]
    plt.scatter(snow_cover_arr, kappa_na_arr, s=10,color=colors,linewidth=0.5, edgecolor='red')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Cloud Cover')
    plt.xlim(0,1)

    plt.ylim(0,1)
    plt.xlabel('Snow Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('Naegeli')

    plt.subplot(1,3,3)

    cmap = matplotlib.cm.get_cmap('YlGn')
    normalize = matplotlib.colors.Normalize(vmin=min(cloud_cover_arr),
            vmax = max(cloud_cover_arr))
    colors = [cmap(normalize(value)) for value in cloud_cover_arr]
    plt.scatter(snow_cover_arr, kappa_nai_arr, s=10,color=colors,linewidth=0.5, edgecolor='green')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Cloud Cover')
    plt.xlim(0,1)

    plt.ylim(0,1)
    plt.xlabel('Snow Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('Naegeli_Improved')
    plt.tight_layout(rect = [0,0.03,1, 0.95])
    plt.savefig('PLOTS/kappa_vs_snowcover_slope.png')



    # Cloud Cover vs. Kappa
 #   print(len(cloud_cover_arr), len(kappa_as_arr))
 #   z_as = np.polyfit(cloud_cover_arr, kappa_as_arr_cloud,1)
 #   p_as = np.poly1d(z_as)
 #   z_na = np.polyfit(cloud_cover_arr, kappa_na_arr_cloud,1)
 #   p_na = np.poly1d(z_na)
 #   z_nai = np.polyfit(cloud_cover_arr, kappa_nai_arr_cloud,1)
 #   p_nai = np.poly1d(z_nai)


    plt.figure(3, figsize=(15,5))
    plt.suptitle('Cohens Kappa vs Cloud Cover')
    plt.title('Cohens Kappa vs Cloud Cover')
    plt.subplot(1,3,1)
    plt.scatter(cloud_cover_arr, kappa_as_arr_cloud, s=10)
#    plt.plot(cloud_cover_arr, p_as(cloud_cover_arr), 'black')
    plt.ylim(0,1)
    plt.xlim(left=0)
    plt.xlabel('Cloud Cover')
    plt.yticks(np.arange(0,1,0.1))
    plt.ylabel('Cohens Kappa')
    plt.title('ASMAG')
    plt.grid()
    plt.subplot(1,3,2)
#    plt.plot(cloud_cover_arr, p_na(cloud_cover_arr), 'black')
    plt.scatter(cloud_cover_arr, kappa_na_arr_cloud, s=10, color= 'red')
    plt.xlim(left =0)
    plt.ylim(0,1)
    plt.yticks(np.arange(0,1,0.1))
    plt.xlabel('Cloud Cover')
    plt.ylabel('Cohens Kappa')
    plt.grid()
    plt.title('Naegeli')
    plt.subplot(1,3,3)
#    plt.plot(cloud_cover_arr, p_nai(cloud_cover_arr), 'black')
    plt.scatter(cloud_cover_arr, kappa_nai_arr_cloud, s=10,color= 'green')
    plt.xlim(left =0)
    plt.ylim(0,1)
    plt.yticks(np.arange(0,1,0.1))
    plt.xlabel('Cloud Cover')
    plt.grid()
    plt.ylabel('Cohens Kappa')
    plt.title('Naegeli_Improved')
    plt.tight_layout(rect = [0,0.3,1, 0.95])
    plt.savefig('PLOTS/kappa_vs_cloudcover.png')

    # Area vs. Kappa
    plt.figure(4, figsize=(15,5))
    plt.suptitle('Cohens Kappa vs Area')
    plt.title('Cohens Kappa vs Area')
    plt.subplot(1,3,1)
    plt.scatter(area_arr, kappa_as_arr, s=10)
    plt.ylim(bottom=0)
    plt.xlabel('Area')
    plt.ylabel('Cohens Kappa')
    plt.title('ASMAG')
    plt.subplot(1,3,2)
    plt.scatter(area_arr, kappa_na_arr, s=10, color= 'red')
    plt.ylim(bottom = 0)
    plt.xlabel('Area')
    plt.ylabel('Cohens Kappa')
    plt.title('Naegeli')
    plt.subplot(1,3,3)
    plt.scatter(area_arr, kappa_nai_arr, s=10,color= 'green')
    plt.ylim(bottom=0)
    plt.xlabel('Area')
    plt.ylabel('Cohens Kappa')
    plt.title('Naegeli_Improved')
    plt.tight_layout(rect = [0,0.3,1, 0.95])
    plt.savefig('PLOTS/kappa_vs_area.png')

    
    # SLA vs. SLA_man
    z_as, intercept =  np.polyfit(SLA_man_arr, SLA_as_arr,1)
    _,_,r_val_as,_,_ = stats.linregress(SLA_man_arr, SLA_as_arr)
    p_as = np.poly1d(z_as)
    z_na, intercept = np.polyfit(SLA_man_arr, SLA_na_arr,1)
    _,_,r_val_na,_,_ = stats.linregress(SLA_man_arr, SLA_na_arr)
    p_na = np.poly1d(z_na)
    z_nai, intercept = np.polyfit(SLA_man_arr, SLA_nai_arr,1)
    _,_,r_val_nai,_,_ = stats.linregress(SLA_man_arr, SLA_nai_arr)
    p_nai = np.poly1d(z_nai)


    plt.figure(5, figsize=(15,5))
    plt.suptitle('Snow Line Altitude Observed vs. Modeled')
    plt.subplot(1,3,1)
    cmap = matplotlib.cm.get_cmap('YlGnBu')
    normalize = matplotlib.colors.Normalize(vmin=min(kappa_as_arr),
            vmax = max(kappa_as_arr))
    colors = [cmap(normalize(value)) for value in kappa_as_arr]
    plt.scatter(SLA_man_arr, SLA_as_arr, s=10,color=colors,linewidth=0.5, edgecolor='blue')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Kappa')


    plt.plot([0,4000],[0,4000]) 
    print(p_as(SLA_man_arr))
    plt.plot( SLA_man_arr, p_as(SLA_man_arr), 'black', label ='R2 = %.3f'.format(r_val_as**2))
    plt.ylim((2000,3600))
    plt.xlim((2000,3600))
    plt.xlabel('SLA Manual')
    plt.ylabel('SLA Calculated')
    plt.title('ASMAG')

    plt.subplot(1,3,2)
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    normalize = matplotlib.colors.Normalize(vmin=min(kappa_na_arr),
            vmax = max(kappa_na_arr))
    colors = [cmap(normalize(value)) for value in kappa_na_arr]
    plt.scatter(SLA_man_arr, SLA_na_arr, s=10,color=colors,linewidth=0.5, edgecolor='red')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Kappa')
    
    plt.plot([0,4000],[0,4000], color = 'red')  
    plt.plot([0,4000], p_na([0,4000]), 'black')
    plt.legend("R2 = %.3f"%r_val_na**2)

    plt.ylim((2000,3600))
    plt.xlim((2000,3600))
    plt.xlabel('SLA Manual')
    plt.ylabel('SLA Calculated')
    plt.title('Naegeli')

    plt.subplot(1,3,3)
    cmap = matplotlib.cm.get_cmap('YlGn')
    normalize = matplotlib.colors.Normalize(vmin=min(kappa_nai_arr),
            vmax = max(kappa_nai_arr))
    colors = [cmap(normalize(value)) for value in kappa_nai_arr]
    plt.scatter(SLA_man_arr, SLA_nai_arr, s=10,color=colors,linewidth=0.5, edgecolor='green')
    sm = matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Kappa')
    
    plt.plot([0,4000],[0,4000], color = 'green')  
    plt.plot([0,4000], p_nai([0, 4000]), 'black')
    plt.legend("R2 = %.3f"%r_val_nai**2)
    
    
    plt.ylim((2000,3600))
    plt.xlim((2000,3600)) 
    plt.xlabel('SLA Manual')
    plt.ylabel('SLA calculated')
    plt.title('Naegeli_Improved')
    plt.tight_layout(rect = [0,0.3,1, 0.95])
    plt.savefig('PLOTS/SLA_man_vs_calc.png')

    ######   Overall  Confusion Matrix and Kappa ######
        # Count number of entries:
    entries = entries + snow_man['time'].size
    C_asmag_norm = C_asmag/np.sum(C_asmag)
    C_naegeli_norm = C_naegeli/np.sum(C_naegeli)
    C_naegeli_improv_norm = C_naegeli_improv/np.sum(C_naegeli_improv)
    
    print(C_asmag_norm, C_naegeli_norm, C_naegeli_improv_norm)

    
    kappa_asmag_tot = kappa_asmag_sum/entries
    kappa_naegeli_tot = kappa_naegeli_sum/entries
    kappa_naegeli_improv_tot = kappa_naegeli_improv_sum/entries

    print(kappa_asmag_tot, kappa_naegeli_tot, kappa_naegeli_improv_tot)
    print("Entries = ", entries)
     
    # Plot Confusion Matrix:
    print(C_asmag)
    cm = C_naegeli_improv
    cm = [[cm[0][0], cm[1][0]],[ cm[2][0], cm[3][0]]]

    cm = np.asarray(cm)
    print(cm)
    title = "Confusion Matrix Naegeli Improved"
    target_names = ['Ice', 'Snow']

    accuracy = np.trace(cm)/float(np.sum(cm))
    misclass = 1 - accuracy
    cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

    cmap = plt.get_cmap('Greens')
    plt.figure(figsize=(8,6))
    plt.rcParams.update({"font.size": 22})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    tresh = cm.max() / 1.5 

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,"{:0.4f}".format(cm[i,j]),
                horizontalalignment = "center",
                color = "white" if cm[i,j] > tresh else "black")

   # plt.tight_layout()
    plt.ylabel('True Value')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass= {:0.4f}'.format(accuracy, misclass))
    plt.rcParams.update({"font.size": 22})
    plt.tight_layout()
    plt.savefig('PLOTS/confusion_matrix_naegeli_improved.png')



    two_d_scatter(cloud_cover_arr, cloud_cover_arr, cloud_cover_arr, 
        snow_cover_arr, snow_cover_arr, snow_cover_arr, 
        kappa_as_arr, kappa_na_arr, kappa_nai_arr,
        'Cloud Cover', 'Snow Cover', 'Cohens Kappa', 7)

    #two_d_scatter
    cloud_cover = np.asarray(cloud_cover_arr)
    snow_cover = np.asarray(snow_cover_arr)
    kappa_as = np.asarray(kappa_as_arr)
    kappa_na = np.asarray(kappa_na_arr)
    kappa_nai = np.asarray(kappa_nai_arr)
    slope = np.asarray(slope_arr)
    aspect = np.asarray(aspect_arr)
    extent = np.asarray(extent_arr)
    area = np.asarray(area_arr)
    doy = np.asarray(doy_arr)
    SLA_man = np.asarray(SLA_man_arr)
    SLA_as = np.asarray(SLA_as_arr)
    SLA_na = np.asarray(SLA_na_arr)
    SLA_nai = np.asarray(SLA_nai_arr)


    x = cloud_cover+0.1
    y = snow_cover
    z = kappa_as
    print(snow_cover)
    print(cloud_cover)
    pickle.dump([
        cloud_cover, 
        snow_cover,
        slope,
        aspect,
        area, 
        extent,
        doy,
        kappa_as,
        kappa_na,
        kappa_nai,
        SLA_man,
        SLA_as,
        SLA_na,
        SLA_nai
        ],open('variables.p', 'wb'))

    
    
                     

    #f, ax  = plt.subplots(1,2,sharex=True, sharey=True)
    #ax[0].tripcolor(triang, z)
    #ax[1].tripcolor(x,y,Z,20)
    #ax[1].plot(x,y,'ko')
    #ax[0].plot(x,y,'ko')


    plt.show()
