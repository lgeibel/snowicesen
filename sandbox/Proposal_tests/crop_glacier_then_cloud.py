"""
Project Glacier file to UTM 32, crop glacier from scene file, save in UTM 32, crop cloud file from result, safe as TIF, Plot

Created October 2018 by lgeibel
""""
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas
import fiona
import numpy as np
import affine
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Input files:   Notes: scene and cloud file are in 32632, glacier file in Swiss CS, EPSG:21781 - CH1903 / LV03 --> code needs to be adjusted when using diferent input

# To do: Loop over different bands. band needded: b2, b3, b4, b8, b11, b12 --> Resolution? 10 m for everything but 20 m for 11, 12.  For knap we only need b3, b8
band = '11'

#specify if cloud mask is polygon ('.shp') or raster file ('.tif')
cloud_form = '.tif'

scene_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\GRANULE\L2A_T32TMS_A008428_20181017T103202\IMG_DATA\R10m\T32TMS_20181017T103019_B"+band+"_10m.jp2"
#scene_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\Rabatel_tests\paul_2016_applieb_B08.tif"

print(scene_file)
cloud_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\clouds.shp"  # Manually converted from  .gml to shp in QGIS
#cloud_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\cloud_mask_based_on_scene_classification.shp"  # extracted values 8, 9 (medium&high cloud probability) and 10 (thin cirrus) from 20 m scene classification
cloud_raster = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\cloud_mask_based_on_scene_classification_raster.tif"  # extracted values 8, 9 (medium&high cloud probability) and 10 (thin cirrus) from 20 m scene classification


glacier_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\R20180321_1500\Gletscher_TLMNRelease_Edit.shp"     # 2018 (?)
#glacier_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\SGI_2008\SGI2010.shp'                              # 2010

# Outout files
glacier_reprojected_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_mask_reprojected_32632_B'+band+'.shp'
glacier_cropped_scene_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_cropped_scene_file_32632_B'+band+'.tif'
glacier_cloud_cropped_scene_file = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_cloud_cropped_scene_file_B'+band+'.tif'   # Final output in WGS84

###############################
#    1. Load glacier outlines (.shp file)
###############################
print('Loading glacier outlines...')

######   Reproject glacier file from swiss to 32632, write to glacier_mask_reprojected.shp
glacier_mask = geopandas.read_file(glacier_file)
print('Original Glacier mask crs:',glacier_mask.crs)
glacier_mask.crs = {'init' :'epsg:21781'}  # for SGI 2010
glacier_mask.crs = {'init' :'epsg:2056'}   # for SGI 2018
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

if cloud_form == '.shp':
    print('cloud_form = .shp')
    with fiona.open(cloud_file, "r") as cloud_mask:
        features = [feature["geometry"] for feature in cloud_mask]
    with rasterio.open(glacier_cropped_scene_file) as src:
        out_meta = src.meta.copy()
        out_image, out_transform = rasterio.mask.mask(src, features,
                                                          invert=True)
        out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

elif cloud_form == '.tif':
    print('cloud_form = .tif')
    with rasterio.open(cloud_raster) as src:
        cloud_raster = src.read()
        affine_ = src.transform
    ######  First Resample resolution of cloud raster from 20m to 10 m
        # create empty np.array with new resolution
        cloud_raster_new = np.empty(shape=(cloud_raster.shape[0],  # same number of bands
                                  out_image.shape[1],  # same resolution as scene file: 10 meters
                                  out_image.shape[2]), dtype = 'uint16')

    with rasterio.open(r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\cropped_but_not_reprojected.tif', 'w', **out_meta) as dst:
        # Resample
        reproject(source=cloud_raster, destination=cloud_raster_new,
                src_transform=src.transform,
                dst_transform=dst.transform,
                src_crs=src.crs,
                dst_crs = src.crs,
                resampling=Resampling.nearest)
        # Invert cloud mask:
        cloud_raster_new = (cloud_raster_new == 0)
        # Apply resampled cloud raster to scene file
        out_image = cloud_raster_new * out_image
        #####  Write reprojected and cropped image to .tif file
        dst.write(out_image)



    #######  Reproject from 32632 to 4326
    dst_crs = 'EPSG:4326'
    print('CRS of Glacier cropped Scene file:', src.crs, '...now reprojecting to WGS 84')

    #Calculate Transform
    dst_transform, width, height = calculate_default_transform(
       src.crs, dst_crs, out_image.shape[1], out_image.shape[2], *src.bounds)
    kwargs = out_meta
    kwargs.update({
         'crs': dst_crs,
         'transform': dst_transform,
         'width': width,
         'height': height
     })
    print('src.transform=', src.transform)
    print('dst_transform=', dst_transform)

    with rasterio.open(glacier_cloud_cropped_scene_file, 'w', **kwargs) as dst:
            # Apply reprojection and create new file
            reproject(
                source=out_image,
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
            print('New CRS of Scene file;', dst.crs)
    #####  Write reprojected and cropped image to .tif file
            dst.write(out_image)


#####  Plot results:
#img = mpimg.imread(glacier_cloud_cropped_scene_file)
#imgplot = plt.imshow(img)
#plt.show()

