######    Crops glacier outline from scene file that was cropped with cloud mask --> glacier mask is not on glaciers -> where does offset come from?...
######

import rasterio
from rasterio.mask import mask
import geopandas
import fiona
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

###############################
#    1. Load glacier outlines (.shp file)
###############################
scene_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\GRANULE\L2A_T32TMS_A008428_20181017T103202\IMG_DATA\R10m\T32TMS_20181017T103019_B08_10m.jp2"
glacier_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\R20180321_1500\Gletscher_TLMNRelease_Edit.shp"     # 2018
glacier_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\SGI_2008\SGI2010.shp'                              # 2010

glacier_reprojected_file = 'glacier_mask_reprojected.shp'

######   Reproject glacier file from swiss to 32632, write to glacier_mask_reprojected.shp
glacier_mask = geopandas.read_file(glacier_file)
print('Original Glacier mask crs:',glacier_mask.crs)
glacier_mask = glacier_mask.to_crs({'init': 'epsg:4326'})
print('reprojected glacier mask crs',glacier_mask.crs)
#shape=gpd.read_file('shapefile')
#shape.plot()
glacier_mask.to_file(driver='ESRI Shapefile',filename=glacier_reprojected_file)

with fiona.open(glacier_reprojected_file, "r") as glacier_reprojected:
    features = [feature["geometry"] for feature in glacier_reprojected]


##########################
#   2.  Open cloud cropped scene file
##########################
with rasterio.open('reprojected_and_cropped_scene_file.tif') as src:
    ######   Apply mask
    out_image, out_transform = rasterio.mask.mask(src, features,
                                                        invert=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    #####  Write reprojected and cropped image to .tif file
    with rasterio.open('final_scene_file.tif', "w", **out_meta) as dest:
         dest.write(out_image)

#####  Plot results:
img = mpimg.imread('final_scene_file.tif')
imgplot = plt.imshow(img)
plt.show()
