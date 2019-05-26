import snowicesat.cfg as cfg
from oggm.utils import *
from rasterio.mask import mask as riomask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import rasterio.plot
import rasterio
import fiona
import xarray
import geopandas as gpd
import matplotlib.pyplot as plt


log = logging.getLogger(__name__)

@entity_task(log)
def crop_satdata_to_glacier(gdir):
    """"Crops sentinel data to glacier grid:
    - all 12 Sentinel Bands in crop_sentinel_to_glacier
    - Solar Zenith and Azimuth Angle in crop_metadata_to_glacier
    - DEM in same projection as Sentinel Tile in crop_dem_to_glacier
    """

    crop_sentinel_to_glacier(gdir)
    crop_metadata_to_glacier(gdir)
    crop_dem_to_glacier(gdir)


def crop_sentinel_to_glacier(gdir):
    """
    Crop 10 Meter resolution Geotiff to glacier extent

    Reads Sentinel Imagery from merged GeoTiff of entinre Area of interest
    , crops to individual glacier, changes projection to local crs and
    saves into netCDF file for current date

    Parameters:
    --------
    gdir:

    Returns:
    --------
    None
    """
    img_path = os.path.join(os.path.join(
                             cfg.PATHS['working_dir'],
                             'cache', str(cfg.PARAMS['date'][0]),
                             'mosaic'))
    dim_name = "band"
    dim_label = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
         'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B8A']
    img_list = [os.path.join(img_path, band+'.tif') for band in dim_label]
    var_name = 'img_values'
    time_stamp = cfg.PARAMS['date'][0]
    file_group = 'sentinel'

    crop_geotiff_to_glacier(gdir, img_list,  dim_name, dim_label,
                            var_name, time_stamp, file_group)


def crop_metadata_to_glacier(gdir):
    """
    Crop 10 Meter resolution Geotiff to glacier extent

    Reads Solar Angles of Sentinel Imagery from merged GeoTiff
    of entire Area of interest, crops to individual glacier,
     changes projection to local crs and
    saves into netCDF file for current date

    Parameters:
    --------
    gdir:

    Returns:
    --------
    None
    """
    img_path = os.path.join(os.path.join(
        cfg.PATHS['working_dir'],
        'cache', str(cfg.PARAMS['date'][0]),
        'meta'))
    img_list = os.listdir(img_path)
    img_list = [os.path.join(img_path, band) for band in img_list]
    dim_name = "band"
    dim_label = ['solar_azimuth', 'solar_zenith']
    var_name = 'angles_in_deg'
    time_stamp = cfg.PARAMS['date'][0]
    file_group = 'solar_angles'

    crop_geotiff_to_glacier(gdir, img_list, dim_name, dim_label,
                            var_name, time_stamp, file_group)

def crop_dem_to_glacier(gdir):
    """
    Crop 10 Meter resolution Geotiff of DEM
    to glacier extent

    Reads DEM of entinre Area of interest
    , crops to individual glacier, changes projection to local crs and
    saves into netCDF file for current date

    Parameters:
    --------
    gdir:

    Returns:
    --------
    None
    """
    img_path = cfg.PATHS['dem_dir']
    img_list = os.listdir(img_path)
    img_list = [os.path.join(img_path, band) for band in img_list]
    dim_name = "band"
    dim_label = ['height_in_m']
    var_name = 'height_in_m'
    time_stamp = 20180101
    file_group = 'dem_ts'

    # check if cropped dem for glacier already exists:
    if os.path.isfile(os.path.join(gdir.get_filepath('dem_ts'))):
        # exit function, no need to read again
        return

    # for first time:
    # Project DEM to same crs as sentinel tiles:

    # get crs from sentinel tile:
    with rasterio.open(os.path.join(os.path.join(os.path.join(
            cfg.PATHS['working_dir'],
            'cache', str(cfg.PARAMS['date'][0]),
            'mosaic')), os.listdir(os.path.join(os.path.join(
                 cfg.PATHS['working_dir'],
                'cache', str(cfg.PARAMS['date'][0]),
            'mosaic')))[1])) as sentinel:
        dst_crs = sentinel.crs
    dst_crs = 'EPSG:32632'

    for band in img_list:
        with rasterio.open(band) as src:
            if dst_crs == src.crs:
                # DEM is already reprojected
                crop_geotiff_to_glacier(gdir, img_list, dim_name, dim_label,
                                        var_name, time_stamp, file_group)

            else: # reproject
                # TODO: something is not right with reprojection...
                print(src.crs)
                src_crs = CRS.to_proj4(src.crs)
                transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(band, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        # safe reprojected file:
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src_crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)
                    print(dst.meta)

    crop_geotiff_to_glacier(gdir, img_list, dim_name, dim_label,
                            var_name, time_stamp, file_group)


def crop_geotiff_to_glacier(gdir, img_list, dim_name, dim_label,
                            var_name, time_stamp, file_group):
    """
    A function that reads Tiles in .tif format from
     folder for different dates
    and processes them to the size of the each glacier.
    The output is then stored in the sentinel.nc file with the
    dimensions of all 12 Sentinel bands with the current time stamp

    Structure of the function:
    - Reading Data from imgfolder
    - Cropping Data to glacier outline
    - Reprojecting Raster of Glacier to local grid
    - Resampling all bands to 10 Meter resolution
    - Writing local raster of all bands to multi-temporal
       netCDF file xxxx.nc

    Params:
    -----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.
    img_list: os.listdir(img_path), list with paths of all
        tiles (need to have same resolution and projection)
        to be processed into netCDF
    dim_name: str: dimension name for variables: e.g. height, bands, angles
    dim_label: list of str: name of e.g. each band or each angle
            ['solar_azimuth','solar_zenith'], range(len(list(range(1,len(b_sub)+1))
    var_name: str: Name of variable: e.g. 'img_values', 'height', 'angles'
    time_stamp: int: date in format yyyymmdd (no datetime.datetime!)
    file_group: str: filepath in cfg.PATH, e.g. 'sentinel', 'solar_angles'


    Returns:
    ----------
        None
    """
    glacier = gpd.read_file(gdir.get_filepath('outlines'))

    # iterate over all bands
    b_sub = []
    band_index = 0
    for band in img_list:
        with rasterio.open(band) as src:
            # Read crs from first Tile of list:
            if band == img_list[0]:
                local_crs = glacier.crs
                src_crs = src.crs
                list_conts = os.listdir(cfg.PATHS['dem_dir'])[0]
                # Problem with Swiss CRS, set crs manually for DEM
                if band == os.path.join(cfg.PATHS['dem_dir'], list_conts):
                    src_crs = CRS.to_proj4(src.crs)
                # Convert outline to projection of tile (src_crs) to crop it out
                glacier = glacier.to_crs(src_crs)
                glacier.to_file(driver='ESRI Shapefile',
                                filename=os.path.join(cfg.PATHS['working_dir'],
                                                      'outline_tile_grid.shp'))
                with fiona.open(os.path.join(cfg.PATHS['working_dir'],
                                             'outline_tile_grid.shp'), "r") \
                        as glacier_reprojected:
                    # Read local geometry
                    features = [feature["geometry"] for feature in glacier_reprojected]


            # --- 1.   Open file: CROP file to glacier outline

            try:
                out_image, out_transform = rasterio.mask.mask(src, features,
                                                              crop=True)
            except ValueError:
                # Glacier not located in tile
                return
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})

            with rasterio.open(gdir.get_filepath('cropped_cache'),'w', **out_meta) \
                    as src1:
                src1.write(out_image)
            # ---  2. REPROJECT to local grid: we want to project out_image with
            # out_meta to local_crs of glacier
                # Calculate Transform
            dst_transform, width, height = calculate_default_transform(
                    src_crs, local_crs, out_image.shape[1],
                    out_image.shape[2], *src1.bounds)

            out_meta.update({
                    'crs': local_crs,
                    'transform': dst_transform,
                    'width': width,
                    'height': height
                })

            with rasterio.open(gdir.get_filepath('cropped_cache'),
                        'w', **out_meta) as src1:

                reproject(
                    source=out_image,
                    destination=rasterio.band(src1,1),
                    src_transform=src1.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=local_crs,
                    resampling=Resampling.nearest)
                # Write to geotiff in cache
                src1.write(out_image)

            # Open with xarray into DataArray
        band_array = xarray.open_rasterio(gdir.get_filepath('cropped_cache'))
        band_array = band_array.squeeze('band').drop('band')
        band_array = band_array.assign_coords(band = dim_label[band_index])
        band_array = band_array.expand_dims('band')

        # write all bands into list b_sub:
        b_sub.append(band_array)
        band_index = band_index + 1


    # Merge all bands to write into netcdf file!
    all_bands = xr.concat(b_sub, dim=dim_name)
   # all_bands[dim_name] = dim_label
    all_bands.name = var_name

    all_bands = all_bands.assign_coords(time=time_stamp)
    all_bands = all_bands.expand_dims('time')
    all_bands_attrs = all_bands.attrs

    # check if netcdf file for this glacier already exists, create if not, append if exists
    if not os.path.isfile(gdir.get_filepath(file_group)):
        all_bands = all_bands.to_dataset(name=var_name)
        all_bands.attrs = all_bands_attrs
        all_bands.attrs['pyproj_srs'] = rasterio.crs.CRS.to_proj4(src1.crs)
        all_bands.to_netcdf(gdir.get_filepath(file_group),
                            'w',
                            unlimited_dims={'time': True})
        all_bands.close()
    else:
        #  Open existing file
        existing = xr.open_dataset(gdir.get_filepath(file_group))
        # Convert all_bands from DataArray to Dataset
        all_bands = all_bands.to_dataset(name=var_name)
        if not all_bands.time.values in existing.time.values:
            appended = xr.concat([existing, all_bands], dim='time')
            existing.close()
            appended.attrs = all_bands_attrs
            appended.attrs['pyproj_srs'] = rasterio.crs.CRS.to_proj4(src1.crs)
            #Write to file
            appended.to_netcdf(gdir.get_filepath(file_group),
                               'w',
                               unlimited_dims={'time': True})
            appended.close()
           # shutil.move(gdir.get_filepath("sentinel_temp"), gdir.get_filepath(file_group))


    # Remove cropped_cache.tif file:
    os.remove(gdir.get_filepath('cropped_cache'))




