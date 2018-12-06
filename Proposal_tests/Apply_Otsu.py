import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
from skimage.io import imread
import rasterio
import numpy as np
import xarray as xr

band = '08'
glacier_cloud_cropped_scene_file_08 = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_cloud_cropped_scene_file_B'+band+'.tif'   # Final output in WGS84
snow_area_otsu = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\snow_area_otsu_B' +band+ '.tif'
snow_area_naegli = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\snow_area_naegli.tif'


###########
glacier_B08 = imread(glacier_cloud_cropped_scene_file_08, as_gray = True)


########## Apply Otsu:
val = filters.threshold_otsu(glacier_B08)
hist, bins_center = exposure.histogram(glacier_B08)
snow = glacier_B08 < val
snow = snow*1

########################################################################        Problems writing file to GeoTIFF :)
### Write Otsu to file:
with rasterio.open(glacier_cloud_cropped_scene_file_08, mode="r", dtype=rasterio.float64) as raster:
    band1 = raster.read(1)
    kwargs = raster.meta
    kwargs.update(driver='GTiff',
                      dtype=rasterio.int32,
                      count=1,
                      compress='lzw',
                      nodata=0,
                      bigtiff='YES')

    with rasterio.open(snow_area_otsu, "w", **kwargs) as destinationFile:
        destinationFile.write_band(1, snow)



###########   narrow-to broadband-conversion after Naegli et al. also need band 3:
band = '03'
glacier_cloud_cropped_scene_file_03 = r'C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\glacier_cloud_cropped_scene_file_B'+band+'.tif'   # Final output in WGS84
glacier_B03 = imread(glacier_cloud_cropped_scene_file_03, as_gray = True)

######  apply scaling factor of 10000
glacier_B03 = glacier_B03/10000
glacier_B08 = glacier_B08/10000


# aKnap = 0.726b3 + 0.322b3^2 + 0.015b8 + 0.581b8^2
alpha =  0.726*glacier_B03 + 0.322*glacier_B03**2 + 0.015*glacier_B08 + 0.581*glacier_B08**2
# aLiang = 0.356b2 + 0.130b4 + 0.373b8 + 0.085b11 + 0.072b12 + 0.0018  ---> resolution? 10 m vs 20 m? --> reproject everything to 20m I assume

snow_naegli = alpha > 0.55
ice_naegli = alpha < 0.25
bigger_025 = alpha > 0.25

bigger_025 = bigger_025*1
snow_naegli = snow_naegli*1
ice_naegli = ice_naegli*1
print(np.unique(snow_naegli))

# Creating map:  1 for snow (white), 0.5 for critical area, 0 for ice, 0 outside of glacier, '
map_naegli = 0.5*(snow_naegli + bigger_025)
print(np.unique(map_naegli))

### Write Naegli map to file:
with rasterio.open(glacier_cloud_cropped_scene_file_08, mode="r", dtype=rasterio.float64) as raster:
    band1 = raster.read(1)
    kwargs = raster.meta
    kwargs.update(driver='GTiff',
                      dtype=rasterio.float64,
                      count=1,
                      compress='lzw',
                      nodata=0,
                      bigtiff='YES')

    with rasterio.open(snow_area_naegli, "w", **kwargs) as destinationFile:
        destinationFile.write_band(1, map_naegli)


################  Plot Otsu & Naegli
plt.figure(figsize=(9, 4))
plt.subplot(121)
plt.imshow(snow, cmap='gray', interpolation='nearest')

plt.subplot(122)
plt.imshow(map_naegli, cmap='gray', interpolation='nearest')

plt.tight_layout()
plt.show()

# ############## Plot otsu and histogram
# plt.figure(figsize=(9, 4))
# plt.subplot(131)
# plt.imshow(glacier_B08, cmap='gray', interpolation='nearest')
#
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(snow, cmap='gray', interpolation='nearest')
# plt.axis('off')
#
# plt.subplot(133)
# plt.plot(bins_center, hist, lw=2)
# plt.axvline(val, color='k', ls='--')
# plt.xlim(left=2)  # remove black from histogram
# plt.ylim(top=4000, bottom=0)

# plt.tight_layout()
# plt.show()



