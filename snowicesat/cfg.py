"""  Configuration file and options

A number of globals are defined here to be available everywhere.

COMMENT: the CRAMPON-specific BASENAMES  and PATHS get the prefix "C" in order
to make sure they come from CRAMPON. Later on, they are merged with the
respective OGGM dictionaries.
"""
from __future__ import absolute_import, division

from oggm.cfg import PathOrderedDict, DocumentedDict, set_intersects_db, \
    pack_config, unpack_config, oggm_static_paths, get_lru_handler
from oggm.cfg import initialize as oggminitialize


import logging
import os
import sys
from collections import OrderedDict

import numpy as np
from configobj import ConfigObj, ConfigObjError

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Local logger
log = logging.getLogger(__name__)

# Path to the cache directory
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.oggm')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
# Path to the config file
CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.oggm_config')

# Globals
IS_INITIALIZED = False
CONTINUE_ON_ERROR = False
CPARAMS = OrderedDict()
PARAMS = OrderedDict()
NAMES = OrderedDict()
CPATHS = PathOrderedDict()
PATHS = PathOrderedDict()
CBASENAMES = DocumentedDict()
BASENAMES = DocumentedDict()
RGI_REG_NAMES = False
RGI_SUBREG_NAMES = False
LRUHANDLERS = OrderedDict()

# Constants
SEC_IN_YEAR = 365*24*3600
SEC_IN_DAY = 24*3600
SEC_IN_HOUR = 3600
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SEC_IN_MONTHS = [d * SEC_IN_DAY for d in DAYS_IN_MONTH]
CUMSEC_IN_MONTHS = np.cumsum(SEC_IN_MONTHS)
BEGINSEC_IN_MONTHS = np.insert(CUMSEC_IN_MONTHS[:-1], [0], 0)

RHO = 900.  # ice density
RHO_W = 1000.  # water density
LATENT_HEAT_FUSION_WATER = 334000  #(J kg-1)
HEAT_CAP_ICE = 2050  # (J kg-1 K-1)
R = 8.314  # gas constant (J K-1 mol-1)
E_FIRN = 21400.  # activation energy for firn (see Reeh et al., 2008) (J mol-1)
ZERO_DEG_KELVIN = 273.15

G = 9.81  # gravity
N = 3.  # Glen's law's exponent
A = 2.4e-24  # Glen's default creep's parameter
FS = 5.7e-20  # Default sliding parameter from Oerlemans - OUTDATED
TWO_THIRDS = 2./3.
FOUR_THIRDS = 4./3.
ONE_FIFTH = 1./5.

# Added from CRAMPON:
_doc = 'A time series of all available DEMs for the glacier. Contains groups' \
       ' for different resolutions.'
CBASENAMES['homo_dem_ts'] = ('homo_dem_ts.nc', _doc)
_doc = 'A time series of all available DEMs for the glacier, brought to the ' \
       'minimum common resolution.'
CBASENAMES['dem_ts'] = ('dem_ts.nc', _doc)
_doc = 'A netcdf file containing all sentinel bands for given times for the glacier'
CBASENAMES['sentinel'] =('sentinel.nc', _doc)
_doc = 'A netcdf file containing all sentinel bands for given times for the glacier' \
       'for temporary writing'
CBASENAMES['sentinel_temp'] =('sentinel_temp.nc', _doc)
_doc = 'A netcdf file containing all sentinel bands for given times for the glacier, ' \
       'corrected with the Ekstrand terrain correction'
CBASENAMES['ekstrand'] =('ekstrand.nc', _doc)
_doc = 'A netcdf file containing solar zenith and solar azimuth angle for a glacier'
CBASENAMES['solar_angles'] =('solar_angles.nc', _doc)
_doc = 'A netcdf file containing all sentinel bands for given times for the glacier, ' \
       'after applying the cloud mask'
CBASENAMES['cloud_masked'] =('cloud_masked.nc', _doc)
_doc = 'A netcdf file containing snow cover map of each glacier after ASMAG algorithm' \
       'and after classifcation by Naegeli'
CBASENAMES['snow_cover'] =('snow_cover.nc', _doc)
_doc = 'A temporary geoTIFF file containing cache files while preparing sentinel data' \
       ' for a glacier'
CBASENAMES['cropped_cache'] =('cropped_cache.tif', _doc)
_doc = 'Figure containing mapped snow cover with Otsu_Thresholding'
CBASENAMES['plt_otsu'] = ('plot_otsu.png', _doc)
_doc = 'Figure containing mapped snow cover with improved Nageli Method'
CBASENAMES['plt_impr_naegeli'] = ('plot_impr_naegeli.png', _doc)
_doc = 'Figure containing mapped snow cover with Naegeli Method'
CBASENAMES['plt_naegeli'] = ('plot_naegeli.png', _doc)
_doc = 'Figure showing cloud mask detected by s2cloudless'
CBASENAMES['plt_cloud_mask'] = ('plot_cloud_mask.png', _doc)


CPARAMS['date'] = ['date']
CPARAMS['cloudcover'] = ['cloudcover']
CPARAMS['tile_id'] = ['']


# some more standard names, for less hardcoding
NAMES['DHM25'] = 'dhm25'
NAMES['SWISSALTI2010'] = 'alti'
NAMES['LFI'] = 'lfi'


def initialize(file=None):
    """Read the configuration file containing the run's parameters."""

    global IS_INITIALIZED
    global BASENAMES
    global PARAMS
    global PATHS
    global NAMES
    global CONTINUE_ON_ERROR
    global GRAVITY_CONST
    global E_FIRN
    global ZERO_DEG_KELVIN
    global R
    global LATENT_HEAT_FUSION_WATER
    global HEAT_CAP_ICE
    global N
    global A
    global RHO
    global RHO_W
    global RGI_REG_NAMES
    global RGI_SUBREG_NAMES

    # This is necessary as OGGM still refers to its own initialisation
    oggminitialize()
    import oggm.cfg as oggmcfg

    # Add the CRAMPON-specific keys to the dicts
    oggmcfg.BASENAMES.update(CBASENAMES)

    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')

    log.info('Parameter file: %s', file)

    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s', file, e)
        sys.exit()

    # Default
    oggmcfg.PATHS['working_dir'] = cp['working_dir']
    if not oggmcfg.PATHS['working_dir']:
        oggmcfg.PATHS['working_dir'] = os.path.join(os.path.expanduser('~'),
                                                    'OGGM_WORKING_DIRECTORY')

    # Paths
    oggm_static_paths()
    oggmcfg.PATHS['dem_dir'] = cp['dem_dir']

    # run params
    oggmcfg.PARAMS['run_period'] = [int(vk) for vk in cp.as_list('run_period')]

    CPARAMS['date'] = [int(vk) for vk in cp.as_list('date')]
    CPARAMS['cloudcover'] = [int(vk) for vk in cp.as_list('cloudcover')]

    # Multiprocessing pool
    oggmcfg.PARAMS['use_multiprocessing'] = cp.as_bool('use_multiprocessing')
    oggmcfg.PARAMS['mp_processes'] = cp.as_int('mp_processes')

    # Some non-trivial params
    oggmcfg.PARAMS['grid_dx_method'] = cp['grid_dx_method']
    oggmcfg.PARAMS['topo_interp'] = cp['topo_interp']
    oggmcfg.PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')

    # Make sure we have a proper cache dir
    from oggm.utils import download_oggm_files
    download_oggm_files()

    # Delete non-floats
    ltr = ['working_dir', 'dem_file', 'dem_dir', 'grid_dx_method',
           'mp_processes', 'use_multiprocessing', 'hfile',
            'topo_interp', 'date',
            'continue_on_error', 'cloudcover',
           'run_period', 'auto_skip_task']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        oggmcfg.PARAMS[k] = cp.as_float(k)

    # Empty defaults
    from oggm.utils import get_demo_file
    set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    IS_INITIALIZED = True

    # Update the dicts in case there are changes
    oggmcfg.PATHS.update(CPATHS)
    oggmcfg.PARAMS.update(CPARAMS)

    BASENAMES = oggmcfg.BASENAMES
    PATHS = oggmcfg.PATHS
    PARAMS = oggmcfg.PARAMS

    # Always call this one! Creates tmp_dir etc.
    oggm_static_paths()
