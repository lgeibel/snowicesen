================
About Snowicesen
================
Snowicesen is a tool that uses Sentinel-2 Satellite images to automatically detect the snow cover and retrieve the snow line altitude for glaciers in a given region. It currently offers three different methods for snow cover mapping whereas the highest accuracy is achieved with the ASMAG-algorithm that is based on Otsu-Thresholding of the Near-Infrared band.

The downloading and precprocessing is fully automized, the only necessary input is a shapefile containing all glacier outlines and a DEM of the region.


============
Requirements
============
To run the snow mapping tool snowicesen, we first need to clone the github repository into the desired folder by running::

   git clone <url> 

Then change into the repository and create a virtual environment with::

   conda env create -f requirements-py36-snowicesen.yml

(The package availability and dependencies were tested in Python 3.6 on Linux
and Windows)

The Crampon Module that snowicesat is based on is still being developed, we therefore have to clone and 
install the repository manually by running::
    git clone <url>
    cd crampon
    pip install -e.

Now the virtual environment is created. 
It can be activated with::
    conda activate snowicesen_env

We also need to create an account for 
the Copernicus Hub that provides the Sentinel-2 Imagery.

.. _Copernicus Open Access Hub: https://scihub.copernicus.eu/dhus/#self-registration

The username and password then need to be stored in an snowicesen.credentials file as::
   ['sentinel'] 
          'user' = 'username'
          'password' = '********'

(make sure this file included in the .gitignore when pushing back to the remote server so your password does not end up on the remote repository).

====================
Input and Parameters
====================

In the `snowicesen_params.cfg` file, we find the configurations for our setup.
We can define the working directory, the filepath to the DEMs that we use as an input,
the date or time frame of interest and the cloud cover range for which we want to download the data.
The further paramaters are inherited from OGGM (the Open Global Glacier Model) so details about them can be found in the OGGM  documentation.



In the setup.py file, we define the filepath to the parameters (default name: `snowicesen_params.cfg`) file, the filepath to the outlines of the glaciers
and the shapefile containing the tiles of the Sentinel-2 data. 

Some notes about the input data for snowicesen:

- **DEM**: The program was developed with the SWISSTOPO DEM with a 10 m resolution and in the WGS84 projection. It is necessary to use a DEM that has the same resolution as the Sentinel-2 data. This could be changed but the current code therefore only runs with a 10 m DEM. In theory, the projection of the DEM should not matter, it is automatically reprojected to the CRS of the Sentinel-2 tiles. However, there were some problems with reading the Swiss CRS, therefore it is recommended to give the DEM in a WGS 84 projection to avoid those.
 
- **Glacier outlines**: Input is a shapefile of the desired glacier regions. The program was developed with the Swiss Glacier Inventory Data 2018 (SGI 2018) and used a modified version that also contains the Randolph Glacier Inventory (RGI)  ID since this is used in the OGGM GlacierDirectories.
  Since OGGM uses the RGI V6, the code expects a "status" field which currently has to be edited manually (or OGGM can be adapted to also read the file without the status field).

- **Sentinel-2 Tiles**: This shapefile is provided by Sentinel in the `.kml` format and shows the outlines of each Sentinel Tile. In general, the code also works without this file (set the option use_tiles = false). However, there are some inconsistencies in the Copernicus data bank and the tiles often have a large overlap, that can not be indentiefied when only searching by the glacier outlines. In order to avoid downloading more data than necessary (which is very time- and storage consuming), it is recommended to manually check the overlap of the glacier outlines with the Sentinel tiles (e.g. in QGIS) and add the Tile ID of each intersecting tile.

Now the snowicesen_params.cfg file needs to be adjusted for the run. The following parameters need to be set:

- `working_dir`: filepath to the working directory into which all output will be written
- `dem_dir`: filepath to the DEM that is used

- `date`: The start and end date of the period that should be processed. Format: yyyymmdd, yyyymmdd, (e.g. 20160719, 20161009=

- `cloud_cover`: the range of cloud cover percentage for each tile in which the sentinel-data is being checked. Note that is only applies for the whole tile, later another cloud mask is applied individually to each glacier and only scenes with a cover higher than 70 % on a glacier will be processed. A value of 90% was used to get as much datas possible, but usually using tile with a cloud cover higher than 70 does not contain very many usable cloud free scenes . Format: 0, 90

All further parameters are inherited from OGGM and their documentation can be found in the OGGM documentation.

Now that everything  is set up, the Snow-Ice classification can be performed.
This is done by running the `setup.py` file with python.

====================
Structure and Output
====================


The program works the following way:
For each polygon in the shapefile that defines the outlines, a glacier directory is created (Glacier Directories are a OGGM class, see there for further documentation). it contains the outlines of each glacier in a local grid that is defined with the center of the glacier as the center for the projection.

The general processing is done day-wise, so for every day in the given interval, the following functions are performed:
`download_all_tiles`

This function creates a  bounding box around the outlines of the shapefile and uses this as a search mask for a request with the sentinelsat module to find the 100x100 km tiles of Sentinel imagery for the given date, area and cloud cover, that is provided by the Copernicus Open Access Hub.  Those tiles are then downloaded in the *.SAFE format* specified by ESA, the bands and metadata (solar angles) are extracted as GeoTIFFs and then all 100x100km tiles are merged together into a mosaic.

After downloading, all further tasks are performed as Entity Tasks for  each Glacier Directory (see OGGM documentation for more information about EntityTasks). They allow parallel processing of all Glaciers.
The following entity tasks are usually executed for the preprocessing of each scene:

- `crop_satdata_to_glacier`: This task crops the merged GeoTIFF mosaics of satellite data and solar angles as well as the DEM to the extent of each glacier and reprojects them into the local grid. The files are saved as netCDF4 files, containing all bands and dates that have been processed for one glacier.
 Output: *sentinel.nc, solar_angles.nc, dem_ts.nc* 

- `ekstrand_correction`: A terrain Ekstrand- correction of the image is performed to account for topographic efects on the reflectance.
 Output: *ekstrand.nc*

- `cloud_masking`: The machine-learning based cloud-masking algorithm *s2cloudless* provided by Sentinelhub is applied, cloud covered pixels are set to 0 and scenes with a cloud cover higher than 70% are not processed any further. 
 Output: *cloud_masked.nc*

- `remove_sides`: Thresholding with the Normalized Differential Snow Index (NDSI) to remove dark (debris covered or shaded) sides of the glacier.
 Output; *sentinel_temp.nc*

After succesful preprocessing, the actual snow mapping is performed with three different algorithms:

- `asmag_snow_mapping`: This algorithm uses flexible Otsu-Thresholding to create a binary ice-snow map

- `naegeli_snow_mapping`: The algorithm described by `Naegeli, 2019: Change Detection of Bare Ice Albedo in the Swiss Alps` to detect bare ice and snow-covered areas on glaciers
 
- `naegeli_improved_snow_mapping`: An improved version of the original `naegeli_snow_mapping` method with added flexibility to better adjust for different scenarios and glacier sizes

All three snow cover maps are saved in the `snow_cover.nc` file in each Glacier Directory  for the given date.

================
Plotting results
================
After retrieving the snow cover maps, the results can either be processed individually or displayed with some of the pre-defined plotting functions. The desired subset of glaciers can be processed by running the `make_plots.py` file. There, different plotting routines (see documentation of `plots.py` for an overview and more detailed descriptions) can be selected in the tasks list and the plots can be saved individually or automatically.

TODO: Insert examples for different plots





