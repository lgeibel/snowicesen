"""
Gives Area classified as snow by Otsu-algorithm and the primary surface type evaluation by Naegeli 2017

Created 22.01.2019 by lgeibel
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt


glacier_area = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\Rabatel_tests\glacier_area.tif"
snow_area_otsu = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\Rabatel_tests\snow_area_otsu_correct.tif"
snow_area_naegeli = r"C:\Users\Lea Geibel\Documents\ETH\MASTERTHESIS\Data\Test 1\output\snow_area_naegli_correct.tif"

with rasterio.open(glacier_area) as src:
    glacier = src.read()

with rasterio.open(snow_area_otsu) as src:
    otsu = src.read()

with rasterio.open(snow_area_naegeli) as src:
    naegeli = src.read()

print(np.count_nonzero(glacier), glacier.shape[1]*glacier.shape[2])
print(np.count_nonzero(otsu), otsu.shape[1]*otsu.shape[2])
print(np.count_nonzero(naegeli), naegeli.shape[1]*naegeli.shape[2])

snow_cover_otsu = np.count_nonzero(otsu)/np.count_nonzero(glacier)

naegeli_ambiguous= (naegeli==0.5).sum()
snow_cover_naegeli_certain = (np.count_nonzero(naegeli)-naegeli_ambiguous)/np.count_nonzero(glacier)
snow_cover_naegeli_ambigous = naegeli_ambiguous/np.count_nonzero(glacier)

print('snow_cover_otsu =',snow_cover_otsu )
print('snow_cover_naegeli_ambiguous=', snow_cover_naegeli_ambigous)
print('snow_cover_naegeli_certain=', snow_cover_naegeli_certain)

