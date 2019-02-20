#def crop_mask_reproj( names, etc):
   ## loads .shp shapefile and .jp2 sentinel-2 scene(in epsg:32632), crops file and

import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona

### files:

# manually converted tp shp file with QGIS
cloud_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\clouds.shp"
scene_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2B_MSIL2A_20181017T103019_N0209_R108_T32TMS_20181017T152001.SAFE\GRANULE\L2A_T32TMS_A008428_20181017T103202\IMG_DATA\R10m\T32TMS_20181017T103019_B08_10m.jp2"
glacier_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\R20180321_1500\Gletscher_TLMNRelease_Edit.shp"
#########################
#  1. Open cloud mask:
########################

with fiona.open(cloud_file, "r") as cloud_mask:
    features = [feature["geometry"] for feature in cloud_mask]


##########################
#   2.  Open scene file
##########################
with rasterio.open(scene_file) as src:
    ######   Apply mask
    out_image, out_transform = rasterio.mask.mask(src, features,
                                                        invert=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    #######  Reproject
    dst_crs = 'EPSG:4326'
    print('CRS of Scene file:', src.crs, '...now reprojecting to WGS 84')
    dst_transform, width, height = calculate_default_transform(
       src.crs, dst_crs, out_image.shape[1], out_image.shape[2], *src.bounds)
    kwargs = out_meta
    kwargs.update({
         'crs': dst_crs,
         'transform': dst_transform,
         'width': width,
         'height': height
     })
    with rasterio.open('reprojected_and_cropped_scene_file.tif', 'w', **kwargs) as dst:
     #############################################################
     ####
     ####    Issues: combining projecting and cropping in one open comman (when first cropping then reprojecting it has problems to open the .tif of the clipped image.
     ####     Otherwise not sure wichinput parameters for reproject fct, especially for source = out_image??
     ####
     ####
     #############################################################
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
#    img = mpimg.imread('reprojected_and_cropped_scene_file.tif')
#    imgplot = plt.imshow(img)
#    plt.show()












# # #
# # ###### Reload scene file and reproject to WGS84 :
# # #
# dst_crs = 'EPSG:4326'
#
#
# with rasterio.open('only_cropped.tif') as src:
#     print('CRS of Scene file:', src.crs, '...now reprojecting to WGS 84')
#     dst_transform, width, height = calculate_default_transform(
#        src.crs, dst_crs, src.width, src.height, *src.bounds)
#     kwargs = src.meta.copy()
#     kwargs.update({
#          'crs': dst_crs,
#          'transform': dst_transform,
#          'width': width,
#          'height': height
#      })
#     with rasterio.open('reprojected_and_cropped_scene_file.tif', 'w', **kwargs) as dst:
#         for i in range(1, src.count + 1):
#             reproject(
#                 source=rasterio.band(src, i),
#                 destination=rasterio.band(dst, i),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=dst_transform,
#                 dst_crs=dst_crs,
#                 resampling=Resampling.nearest)
#         print('New CRS of Scene file;', dst.crs)
#     #####  Write reprojected and cropped image to .tif file
#
