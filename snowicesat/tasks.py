from __future__ import absolute_import, division
import crampon.cfg as cfg

""" Shortcuts to the functions of snowicesat and some Crampon functions
"""

# Entity tasks
from snowicesat.preprocessing.gis import define_glacier_region_snowicesat, ekstrand_correction
from crampon.core.preprocessing.gis import glacier_masks
from snowicesat.preprocessing.preprocessing_data import crop_sentinel_to_glacier, crop_metadata_to_glacier