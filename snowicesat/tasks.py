from __future__ import absolute_import, division
import crampon.cfg as cfg

""" Shortcuts to the functions of snowicesat and some Crampon functions
"""

# Entity tasks
from snowicesat.preprocessing.create_gdirs import define_glacier_region_snowicesat
from snowicesat.preprocessing.image_corrections import  ekstrand_correction, cloud_masking, remove_sides
from snowicesat.snow_classification.snow_mapping import otsu_tresholding, naegeli_snow_mapping
from crampon.core.preprocessing.gis import glacier_masks
from snowicesat.preprocessing.geometric_preprocessing import crop_sentinel_to_glacier, crop_metadata_to_glacier, crop_dem_to_glacier