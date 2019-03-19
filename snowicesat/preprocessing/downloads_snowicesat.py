import snowicesat.cfg as cfg
import snowicesat.utils as utils
from oggm.utils import *

from configobj import ConfigObj, ConfigObjError
from sentinelsat import SentinelAPI
import rasterio
from rasterio.mask import mask as riomask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine
import glob
import time
import rasterio
import fiona
import xarray
import geopandas as gpd
import netCDF4
import snowicesat.utils

log = logging.getLogger(__name__)

@entity_task(log)
def crop_sentinel_to_glacier(gdir):
    """
    A function that reads Sentinel Tiles in .tif format from
    cache folder for different dates
    and processes them to the size of the each glacier.
    The output is then stored in the sentinel.nc file with the
    dimensions of all 12 Sentinel bands with the current time stamp

    Structure of the function:
    - Reading Sentinel Data from cache folder in Working Directory (bandwise)
    - Cropping Sentinel Data to glacier outline
    - Reprojecting Raster of Glacier to local grid
    - Resampling all bands to 10 Meter resolution
    - Writing local raster of all bands to multi-temporal
       netCDF file sentinel.nc

    Params:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns:
    ----------
        None
    """
    print("In crop_sentinel_to_glacier")
    glacier = gpd.read_file(gdir.get_filepath('outlines'))
    img_path = os.path.join(os.path.join(
                        cfg.PATHS['working_dir'],
                        'cache',str(cfg.PARAMS['date'][0]),
                        'mosaic'))

    # iterate over all bands
    b_sub = []
    for band in os.listdir(img_path):
        with rasterio.open(os.path.join(img_path, band)) as src:
            # Read crs from Sentinel-2 Tile:
            if band == "B01.tif":
                local_crs = glacier.crs
                glacier = glacier.to_crs(src.crs)
                glacier.to_file(driver='ESRI Shapefile',
                                filename=os.path.join(cfg.PATHS['working_dir'],
                                                      'outline_sentinel_grid.shp'))

                with fiona.open(os.path.join(cfg.PATHS['working_dir'],
                                             'outline_sentinel_grid.shp'), "r") \
                        as glacier_reprojected:
                    # Read local geometry
                    features = [feature["geometry"] for feature in glacier_reprojected]

            # --- 1.   Open Sentinel file: CROP file to glacier outline
            out_image, out_transform = rasterio.mask.mask(src, features,
                                                              crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})

            with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cache',
                                            str(cfg.PARAMS['date'][0]),
                                            'cropped_cache'+band),'w', **out_meta) \
                    as src1:
                src1.write(out_image)

            # ---  2. REPROJECT to local grid: we want to project out_image with out_meta to local crs of glacier
                # Calculate Transform
            dst_transform, width, height = calculate_default_transform(
                    src1.crs, local_crs, out_image.shape[1],
                    out_image.shape[2], *src1.bounds)

            out_meta.update({
                    'crs': local_crs,
                    'transform': dst_transform,
                    'width': width,
                    'height': height
                })

            with rasterio.open(
                        os.path.join(cfg.PATHS['working_dir'], 'cache',
                                     str(cfg.PARAMS['date'][0]),
                                     'cropped_cache' + band),
                        'w', **out_meta) as src1:

                reproject(
                    source=out_image,
                    destination=rasterio.band(src1,1),
                    src_transform=src1.transform,
                    src_crs=src1.crs,
                    dst_transform=dst_transform,
                    dst_crs=local_crs,
                    resampling=Resampling.nearest)
                # Write to geotiff in cache
                src1.write(out_image)

            # Open with xarray into DataArray
        band_array = xarray.open_rasterio(
            os.path.join(cfg.PATHS['working_dir'],'cache',
            str(cfg.PARAMS['date'][0]),'cropped_cache'+band)
        )

        band_array.attrs['pyproj_srs'] = band_array.crs
        band_array.attrs['pyproj_srs'] = rasterio.crs.CRS.to_proj4(src1.crs)

        # write all bands into list b_sub:
        b_sub.append(band_array)

    # Merge all bands to write into netcdf file!
    all_bands = xr.concat(b_sub, dim='band')
    all_bands['band'] = list(range(1,len(b_sub)+1))
    all_bands.name = 'img_values'

    all_bands = all_bands.assign_coords(time=cfg.PARAMS['date'][0])
    all_bands = all_bands.expand_dims('time')

    # check if netcdf file for this glacier already exists, create if not, append if exists
    if not os.path.isfile(gdir.get_filepath('sentinel')):
        print("netcdf does not exist yet, creating new")
        all_bands.to_netcdf(gdir.get_filepath('sentinel'),
                            'w', format='NETCDF4',
                            unlimited_dims={'time': True})
    else:
        print('Open existing file')
        existing = xr.open_dataset(gdir.get_filepath('sentinel'))
        # Convert all_bands from DataArray to Dataset
        all_bands= all_bands.to_dataset(name = 'img_values')
        if all_bands.time.values in existing.time.values:
            print("date already exists, not writing again")
        else:
            print("New date, appending to netcdf...")
            appended = xr.concat([existing, all_bands], dim='time')
            existing.close()
            #Write to file
            appended.to_netcdf(gdir.get_filepath('sentinel'),
                               'w', format='NETCDF4',
                               unlimited_dims={'time': True})

    # Remove .tif files:
    cache_list = [filename for filename in glob.glob(
                    os.path.join(cfg.PATHS['working_dir'],
                                 'cache',
                                  str(cfg.PARAMS['date'][0]),
                                 'cropped_cache', '*.tif'),
                    recursive=False)]
    print(cache_list)
    for f in cache_list:
        shutil.rmtree(f)




