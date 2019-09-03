## SnowIceSen

SnowIceSen is an automated tool for mapping snow cover on glacier using Sentinel-2 satellite imagery. It includes the data retrieval from the Open Access Copernicus Hub, preprocessing, cloud cover and debris cover masking and three different snow mapping algorithms.
Required input is an ESRI shapefile of the glaciers/region of interest, a DEM of the region and the time period of interest. The output is stored in netCDF file as a binary snow map and an approximation for the snow line altitude (SLA). \
\
<img src="https://user-images.githubusercontent.com/44255527/64164173-fda31e80-ce42-11e9-84c4-0dc1f5465ed5.png" width="90%"></img>
\
Code Documentation, installataion and user guide can be found on http://snowicesen.readthedocs.org. (Autodoc does not always compile correctly on readthedocs, a locally compilable documentation can be found in the docs/ folder of this respository). \
\
A detailed desrciption of the work flow, the implemented snow mapping algorithms and an anaylsis of the performance of the snow mapping algoritm can be found in the( /cite Masterthesis Lea Geibel)
