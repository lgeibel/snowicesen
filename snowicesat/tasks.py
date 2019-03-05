from __future__ import absolute_import, division
import crampon.cfg as cfg

""" Shortcuts to the functions of snowicesat and some Crampon functions
"""

# Entity tasks
from snowicesat.preprocessing.gis import define_glacier_region_snowicesat
from crampon.core.preprocessing.gis import glacier_masks
from snowicesat.preprocessing.downloads_snowicesat import crop_sentinel_to_glacier