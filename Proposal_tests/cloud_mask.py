import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd


### files:

# manually converted tp shp file with QGIS
cloud_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2A_MSIL2A_20180803T103021_N0208_R108_T32TLS_20180803T151712.SAFE\clouds.shp"
scene_file = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\S2A_MSIL2A_20180803T103021_N0208_R108_T32TLS_20180803T151712.SAFE\GRANULE\L2A_T32TLS_A016264_20180803T103239\IMG_DATA\R10m\T32TLS_20180803T103021_B08_10m.jp2"


##### Open cloud mask:
cloud_mask = gpd.read_file(cloud_file)
print(cloud_mask.crs)
    # Change crs of cloud mask
cloud_mask = cloud_mask.to_crs({'init' :'epsg:4326'})
print(cloud_mask.crs)

    # create geojson object from the shapefile to with for mask function
clouds_geojson = mapping(cloud_mask['geometry'][0])
clouds_geojson

##### Open scene file:
with rio.open(scene_file) as src:
    print(src.crs)
    src_crop, src_crop_affine = mask(src, [clouds_geojson],invert=True)