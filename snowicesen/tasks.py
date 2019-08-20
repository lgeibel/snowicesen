from __future__ import absolute_import, division
import crampon.cfg as cfg

""" Shortcuts to the functions of snowicesen and some Crampon functions
"""

# Entity tasks
from snowicesen.preprocessing.create_gdirs import define_glacier_region_snowicesen
from snowicesen.preprocessing.image_corrections import  ekstrand_correction, cloud_masking, remove_sides
from snowicesen.snow_mapping import asmag_snow_mapping, naegeli_improved_snow_mapping, naegeli_snow_mapping
from snowicesen.validate_snow_mapping import create_manual_snow_map, create_confusion_matrix
from crampon.core.preprocessing.gis import glacier_masks
from snowicesen.preprocessing.geometric_preprocessing import crop_satdata_to_glacier
from snowicesen.plots import plot_results
