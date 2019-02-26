"""
Namelist for snowicesat:
adapt parameters for:

created 18.02.2109
"""

# 1. Download / Data retrieval: #

user          = "lgeibel"                # Sentinelhub user name
password      = "snowicesat"        # Sentinelhub password

area_polygon  = "bbox.geojson"        # geojson polygon of area of interest, created on geojson.io. Here: Swiss Alps
datum         = ("20181001", "20181002") # date /time frame of interest, format ("20181001", "20181002")
cloudcover    = "[0 TO 40]"          # cloud cover percentage of interest, format [0 TO 30]
downloadpath  = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data"


from tests_download import search_download_sen2_data

# Search and Download all products matching query to local directory, unzip into .SAFE format
search_download_sen2_data(user, password, area_polygon, datum, cloudcover, downloadpath)


