"""
uses sentinelsat module for automated downloading of the data of the Swiss Alps, searchs for given date, displays results and downloads

created 18.02.2019
"""
def search_download_sen2_data(user, password, area_polygon, datum, cloudcover, downloadpath):
    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    import zipfile
    import os
    # connect to API
    api = SentinelAPI(
    user,
    password,
    api_url="https://scihub.copernicus.eu/apihub/"
    )

    # Search for products matching query
    products = api.query(
        area = geojson_to_wkt(read_geojson(area_polygon)),
        date = datum,
        platformname = "Sentinel-2",
        producttype = "S2MSI1C",
        cloudcoverpercentage=cloudcover
    )

    # count number of products matching query
    print("Tiles found:", api.count(area = geojson_to_wkt(read_geojson(area_polygon)),
        date = datum,
        platformname = "Sentinel-2",
        producttype = "S2MSI1C",
        cloudcoverpercentage=cloudcover) ,", Total size: ", api.get_products_size(products) ,"GB. Now downloading those tiles")

    # downloading all products
    download_zip = api.download_all(products, directory_path=downloadpath)

    # Unzip files, delete

    for key in download_zip[0].keys():
        with zipfile.ZipFile(download_zip[0][key]['path']) as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = zip_file.open(member)
            source.close()

        os.remove(download_zip[0][key]['path'])
