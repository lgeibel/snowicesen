from __future__ import absolute_import, division
import salem
import os
import logging
import snowicesat.cfg as cfg
from crampon import utils
from snowicesat.preprocessing import create_gdirs
from crampon.core.preprocessing import gis
import crampon
from shutil import rmtree
from oggm.workflow import execute_entity_task
from snowicesat.utils import GlacierDirectory
# MPI similar to OGGM - not yet implemented
try:
    import oggm.mpi as ogmpi
    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False

# Module logger
log = logging.getLogger(__name__)


def init_glacier_regions_snowicesat(shapedf=None, reset=False, force=False, all_dates=True):
    """
    Set up or take over GlacierDirectories. The first task (always!).

    The main  function is copied from Crampon that copied it froom OGGM,
    but it has been simplified so that it only uses one input 
    (currently SwissALti3d) for the DEM input and only calculates 
    the necessary parts of it.
    It also uses the define_glacier_regions_snowicesat entity task.

    For a more verstatile use, the Crampon or OGGM function
    init_glacier_regions and define_glacier_regions 
    might has to be employed.

    Set reset=True in order to delete the content of the directories.

    Parameters
    ----------
    shapedf: :obj:`geopandas.GeoDataFrame`, optional
        A geopandas.GeoDataFrame with geometries to use for setting up the
        GlacierDirectories.
    reset: bool, optional
        Whether or not the existing GlacierDirectories and log shall be
        deleted. Default: False.
    force: bool, optional
        Whether or not to ask before deleting GlacierDirectories and log.
        Default: False.

    Returns
    -------
    gdirs: list
        A list of the GlacierDirectory objects.
    """
    print("In init_glacier_regions_snowicesat")
    if reset and not force:
        reset = utils.query_yes_no('Delete all glacier directories?')

    # if reset delete also the log directory
    if reset:
        fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
        if os.path.exists(fpath):
            rmtree(fpath)

    gdirs = []
    new_gdirs = []
    if shapedf is None:
        if reset:
            raise ValueError('Cannot use reset without a rgi file')
        # The dirs should be there already
        gl_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
        for root, _, files in os.walk(gl_dir):
            if files and ('dem.tif' in files):
                gdirs.append(GlacierDirectory(os.path.basename(root)))
    else:
        for _, entity in shapedf.iterrows():
            gdir = GlacierDirectory(entity, reset=reset)
            if not os.path.exists(gdir.get_filepath('dem')):
                new_gdirs.append((gdir, dict(entity=entity)))
            gdirs.append(gdir)
    # If not initialized, run the task in parallel
    execute_entity_task(create_gdirs.define_glacier_region_snowicesat, new_gdirs)


    return gdirs

init_glacier_regions = init_glacier_regions_snowicesat

