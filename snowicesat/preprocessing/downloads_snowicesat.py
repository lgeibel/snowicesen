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
    print("In crop_sentinel_to_glacier")
    glacier = gpd.read_file(gdir.get_filepath('outlines'))
    #img_path = [x[0] for x in os.walk(os.path.join(cfg.PATHS['working_dir'], filename))][8]
    img_path = os.path.join(os.path.join(cfg.PATHS['working_dir'],'cache',str(cfg.PARAMS['date'][0]),'mosaic'))

    # Reproject to Sentinel-2 (UTM zone 32) Grid:
    local_crs = glacier.crs
    glacier = glacier.to_crs({'init': 'epsg:32632'})
    glacier.to_file(driver='ESRI Shapefile', filename=os.path.join(cfg.PATHS['working_dir'],'outline_UTM32.shp'))

    # Open outline in UTM zone 32 Grid
    s = time.time()

    with fiona.open(os.path.join(cfg.PATHS['working_dir'],'outline_UTM32.shp'), "r") as glacier_reprojected:
        # Read local geometry
        features = [feature["geometry"] for feature in glacier_reprojected]

    # iterate over all bands
    b_sub = []
    for band in os.listdir(img_path):
        with rasterio.open(os.path.join(img_path, band)) as src:
            # --- 1.   Open Sentinel file: CROP file to glacier outline
            out_image, out_transform = rasterio.mask.mask(src, features,
                                                              crop=True)
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                #  Write cropped image to .tif file --> necessary?
        with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cropped_not_reprojected.tif'),'w', **out_meta) as src:
            src.write(out_image)
        with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cropped_not_reprojected.tif'), 'r', **out_meta) as src:

            # ---  2. REPROJECT to local grid: we want to project out_image with out_meta to local crs of glacier
                # Calculate Transform
            dst_transform, width, height = calculate_default_transform(
                    src.crs, local_crs, out_image.shape[1], out_image.shape[2], *src.bounds)

            out_meta.update({
                    'crs': local_crs,
                    'transform': dst_transform,
                    'width': width,
                    'height': height
                })

            with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cropped_reprojected_band.tif'), 'w', **out_meta) as dst:
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
        tile = rasterio.open(os.path.join(cfg.PATHS['working_dir'], 'cropped_reprojected_band.tif'))
        print(tile.width, tile.height)

        with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cropped_reprojected_band.tif')) as src:
                # --- 3. RESAMPLE all bands to 10 Meter Resolution:
            bands_60m = ['B01.tif', 'B09.tif','B10.tif'];
            bands_20m = ['B05.tif','B06.tif','B07.tif','B11.tif','B12.tif']

            if band in bands_60m or band in bands_20m:
                print(band)
                if band in bands_60m:
                    res_factor=6
                elif band in bands_20m:
                    res_factor = 2
                arr = src.read()
                newarr = np.empty(shape=(arr.shape[0],  # same number of bands
                                         round(arr.shape[1] * res_factor),
                                         round(arr.shape[2] * res_factor)),  dtype = 'uint16')

                # adjust the new affine transform to the smaller cell size
                with rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cropped_reprojected_resampled_band.tif'), 'w', **out_meta) as dst:
                    aff = src.transform
                    newaff = Affine(aff.a / res_factor, aff.b, aff.c,
                                    aff.d, aff.e / res_factor, aff.f)

                    reproject(source = arr, destination = newarr,
                              src_transform=src.transform,
                              dst_transform=newaff,
                              src_crs=src.crs,
                              dst_crs=src.crs,
                              resampling=Resampling.nearest)
                    dst.write(newarr)
                # TODO: what on Earth is wrong with reprojection?
            tile= rasterio.open(os.path.join(cfg.PATHS['working_dir'],'cropped_reprojected_resampled_band.tif'))
            print(tile.width, tile.height)


               #TODO: remove all funny cropped.tif files

            # Open with xarray into DataArray
            band_array = xarray.open_rasterio(os.path.join(cfg.PATHS['working_dir'],'cropped_reprojected_band.tif'))

            band_array.attrs['pyproj_srs'] = band_array.crs
            band_array.attrs['pyproj_srs'] = rasterio.crs.CRS.to_proj4(dst.crs)

            #print(band_array.crs)
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
        all_bands.to_netcdf(gdir.get_filepath('sentinel'), 'w', format='NETCDF4', unlimited_dims={'time': True})
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
            appended.to_netcdf(gdir.get_filepath('sentinel'), 'w', format='NETCDF4', unlimited_dims={'time': True})

    # TODO: Check if the result is actually ok ...
    # TODO: fix: resample all bands to 10 meter resolution
    # TODO: what on earth is wrong?!?

    e = time.time()
    print(e - s)

