import snowicesat.cfg as cfg
import snowicesat.utils as utils
from oggm.utils import *

from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI
from rasterio.mask import mask as riomask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
import time
import rasterio
import fiona
import xarray
import geopandas as gpd
import netCDF4

log = logging.getLogger(__name__)

@entity_task(log)
def crop_sentinel_to_glacier(gdir):
    """
    Reads Sentinel Data from cache folder in Working Directory (bandwise)
    Crops Sentinel Data to glacier outline
    Reprojects Raster of Glacier to local grid
    Writes local raster of all bands to netcdf file with dimensions 'bands' and 'time'

    Params: filename: filename of .SAFE folder where glacier is located
    glacier: Glacier Outline dataFrame in local grid
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns: sen_dat: netcdf 4 file containing all bands for all times ever added for the glacier
    """

    # TODO: Check if netcdf file already exists in glacier directory, if not, create it:
    print("In crop_sentinel_to_glacier")
    glacier = gpd.read_file(gdir.get_filepath('outlines'))
    #img_path = [x[0] for x in os.walk(os.path.join(cfg.PATHS['working_dir'], filename))][8]
    img_path = os.path.join(cfg.PATHS['working_dir'],'cache')

    # Reproject to Sentinel-2 (UTM zone 32) Grid:
    local_crs = glacier.crs
    glacier = glacier.to_crs({'init': 'epsg:32632'})
    glacier.to_file(driver='ESRI Shapefile', filename='outline_UTM32.shp')

    # Open outline in UTM zone 32 Grid
    s = time.time()

    with fiona.open('outline_UTM32.shp', "r") as glacier_reprojected:
        # Read local geometry
        features = [feature["geometry"] for feature in glacier_reprojected]

    # iterate over all bands
    b_sub = []
    for band in os.listdir(img_path): # Ignore TCL band and band 08A
        with rasterio.open(os.path.join(img_path, band)) as src:
            #   Open Sentinel file: Apply glacier outline
            out_image, out_transform = rasterio.mask.mask(src, features,
                                                              crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                #  Write cropped image to .tif file --> necessary?
        with rasterio.open('cropped_not_reprojected.tif','w', **out_meta) as src:
            src.write(out_image)
        with rasterio.open('cropped_not_reprojected.tif', 'r', **out_meta) as src:

            # REPROJECT to local grid: we want to project out_image with out_meta to local crs of glacier
                # Calculate Transform
            dst_transform, width, height = calculate_default_transform(
                    src.crs, local_crs, out_image.shape[1], out_image.shape[2], *src.bounds)

            out_meta.update({
                    'crs': local_crs,
                    'transform': dst_transform,
                    'width': width,
                    'height': height
                })
            with rasterio.open('cropped_reprojected_band.tif', 'w', **out_meta) as dst:
                reproject(
                    source=out_image,
                    destination=rasterio.band(dst,1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=local_crs,
                    resampling=Resampling.nearest)
                # Write to geotiff in cache
                dst.write(out_image)

                #TODO: remove all funny cropped.tif files

                # Open with xarray into DataArray
            band_array = xarray.open_rasterio('cropped_reprojected_band.tif')
            band_array.attrs['pyproj_srs'] = band_array.crs
            b_sub.append(band_array.salem.subset(margin=0))
                #

    # Merge all subbands to write into netcdf file!
    all_bands = xr.concat(b_sub, dim='band')
    all_bands['band'] = list(range(len(b_sub)))
    all_bands.name = 'img_values'

    all_bands = all_bands.assign_coords(time=cfg.PARAMS['date'][0])
    all_bands = all_bands.expand_dims('time')
    # TODO: Deal with different dates! Append new day to existing netcdf

    #TODO: dimension as  unlimited? Think about form of date

    # check if netcdf file for this glacier already exists, create if not, append if exists
    if not os.path.isfile(gdir.get_filepath('sentinel')):
        print("netcdf does not exist yet, creating new")
        all_bands.to_netcdf(gdir.get_filepath('sentinel'), 'w', format='NETCDF4')
    else:
        # Open file: Append to time dimension
        all_bands.to_netcdf(gdir.get_filepath('sentinel'), 'w', format='NETCDF4')
        existing_netcdf = netCDF4.Dataset(gdir.get_filepath('sentinel'), "r", format="NETCDF4")
        print("dims = ", existing_netcdf.dimensions)

    # TODO: something weird with crs...... not sure what? Check!

    e = time.time()
    print(e - s)

