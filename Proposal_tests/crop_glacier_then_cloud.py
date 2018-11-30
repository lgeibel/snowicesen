#####   Project Glacier file to WGS 32, crop from scene file, save in WGS 32, crop cloud file from

import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas
import fiona
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Input files:   Notes: scene and cloud file are in 32632, glacier file in Swiss CS, EPSG:21781 - CH1903 / LV03 --> code needs to be adjusted when using diferent input
scene_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\GRANULE\L2A_T32TMS_A008428_20181017T103202\IMG_DATA\R10m\T32TMS_20181017T103019_B08_10m.jp2"
cloud_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\clouds.shp"  # Manually converted from  .gml to shp in

glacier_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\R20180321_1500\Gletscher_TLMNRelease_Edit.shp"     # 2018 (?)
glacier_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\SGI_2008\SGI2010.shp'                              # 2010


# Outout files
glacier_reprojected_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_mask_reprojected_32632.shp'
glacier_cropped_scene_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_cropped_scene_file_32632.tif'
glacier_cloud_cropped_scene_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_cloud_cropped_scene_file.tif'   # Final output in WGS84


###############################
#    1. Load glacier outlines (.shp file)
###############################
print('Loading glacier outlines...')

######   Reproject glacier file from swiss to 32632, write to glacier_mask_reprojected.shp
glacier_mask = geopandas.read_file(glacier_file)
print('Original Glacier mask crs:',glacier_mask.crs)
glacier_mask = glacier_mask.to_crs({'init': 'epsg:32632'})
print('Reprojected  glacier mask crs',glacier_mask.crs)
glacier_mask.to_file(driver='ESRI Shapefile',filename=glacier_reprojected_file)

with fiona.open(glacier_reprojected_file, "r") as glacier_reprojected:
    features = [feature["geometry"] for feature in glacier_reprojected]

##########################
#   2.  Open scene file
##########################
with rasterio.open(scene_file) as src:
    ######   Apply mask
    out_image, out_transform = rasterio.mask.mask(src, features,
                                                        crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    #####  Write reprojected and cropped image to .tif file
    with rasterio.open(glacier_cropped_scene_file, "w", **out_meta) as dest:
         dest.write(out_image)

###########################
#   2. apply cloud mask:
##########################

with fiona.open(cloud_file, "r") as cloud_mask:
    features = [feature["geometry"] for feature in cloud_mask]


##########################
#   2.1  Open glacier cropped scene file
##########################
with rasterio.open(glacier_cropped_scene_file) as src:
    ######   Apply cloud mask
    out_image, out_transform = rasterio.mask.mask(src, features,
                                                        invert=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    #######  Reproject from 32632 to 4326
    dst_crs = 'EPSG:4326'
    print('CRS of Glacier cropped Scene file:', src.crs, '...now reprojecting to WGS 84')
    dst_transform, width, height = calculate_default_transform(
       src.crs, dst_crs, out_image.shape[1], out_image.shape[2], *src.bounds)
    kwargs = out_meta
    kwargs.update({
         'crs': dst_crs,
         'transform': dst_transform,
         'width': width,
         'height': height
     })
    with rasterio.open(glacier_cloud_cropped_scene_file, 'w', **kwargs) as dst:
            reproject(
                source=out_image,
                destination=rasterio.band(dst, 1),
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
            print('New CRS of Scene file;', dst.crs)
    #####  Write reprojected and cropped image to .tif file
            dst.write(out_image)


#####  Plot results:
img = mpimg.imread(glacier_cloud_cropped_scene_file)
imgplot = plt.imshow(img)
plt.show()

